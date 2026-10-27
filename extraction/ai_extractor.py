"""VLM branch: send the document (image or PDF) to Claude and read fields back.

PRIVACY: the whole document is uploaded to Anthropic's API. See docs/MEMBER2_EXTRACTION.md.

Uses Claude structured outputs (``messages.parse`` with a pydantic schema) so the reply
is JSON in our shape. Each field is still checked again here. Values are copied as
printed; the model's confidence is kept as ``model_self_report`` (an estimate only).
"""

from __future__ import annotations

import base64
import logging
import os
from pathlib import Path
from typing import Any

import anthropic
from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

from .patterns import canonical_key
from .schemas import (
    CANONICAL_FIELDS,
    DOCUMENT_TYPES,
    Evidence,
    ExtractionResult,
    FieldResult,
    empty_fields,
    failed_result,
)

log = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-opus-5-5"
MAX_IMAGE_BYTES = 5 * 1024 * 1024  # API limit per image
MAX_PDF_BYTES = 30 * 1024 * 1024  # API request limit is 32 MB
FALLBACK_CONFIDENCE = 0.5  # used when the model gives no usable confidence
EMPTY_VALUES = {"", "null", "none", "n/a", "unknown"}

_MEDIA = {
    b"\x89PNG": "image/png",
    b"\xff\xd8\xff": "image/jpeg",
    b"GIF8": "image/gif",
    b"RIFF": "image/webp",
    b"%PDF": "application/pdf",
}

PROMPT = f"""Extract fields from this business document.
Rules:
- Copy every value EXACTLY as printed: same characters, same date order, same number
  format, same currency symbol. Do not convert, reformat, calculate or translate.
- document_type: one of {", ".join(DOCUMENT_TYPES)}, or null if unclear.
- vendor: the company issuing the document. address: the vendor's address.
- date: the document/transaction date (not due or delivery dates).
- currency: the currency code or symbol as printed (e.g. "RM", "MYR", "$"), else null.
- tax_rate: the percentage as printed (e.g. "6%").
- Use null for anything absent, unreadable or uncertain. Never guess.
- evidence: the exact text you read the value from.
- confidence: 0 to 1, how sure you are the value is read correctly."""


class VLMField(BaseModel):
    """One field as returned by the model."""

    model_config = ConfigDict(extra="ignore")
    value: str | None = None
    confidence: float | None = None
    evidence: str | None = None

    @field_validator("value", "evidence", mode="before")
    @classmethod
    def _reject_non_text(cls, v: Any) -> Any:
        if isinstance(v, bool | list | dict):
            raise ValueError("expected text")
        return str(v) if isinstance(v, int | float) else v


class VLMResponse(BaseModel):
    """Schema sent to Claude as the required output format."""

    document_type: VLMField
    vendor: VLMField
    address: VLMField
    invoice_number: VLMField
    order_number: VLMField
    date: VLMField
    currency: VLMField
    subtotal: VLMField
    tax_rate: VLMField
    tax: VLMField
    rounding_adjustment: VLMField
    total: VLMField


def _media_type(data: bytes) -> str | None:
    return next((mt for magic, mt in _MEDIA.items() if data.startswith(magic)), None)


def _load(document: str | Path | bytes) -> bytes:
    if isinstance(document, bytes | bytearray):
        return bytes(document)
    path = Path(document)
    if not path.is_file():
        raise FileNotFoundError(f"file not found: {path.name}")
    return path.read_bytes()


def parse_vlm_output(data: Any, model: str) -> ExtractionResult:
    """Check the model's JSON field by field and convert it to the shared format."""
    if not isinstance(data, dict):
        return failed_result("vlm", model, "vlm_invalid_response", "response is not a JSON object")
    result = ExtractionResult(source="vlm", engine=model, fields=empty_fields("vlm"))
    seen: set[str] = set()
    for key, raw in data.items():
        name = canonical_key(key)
        if name not in CANONICAL_FIELDS or name in seen:
            result.extra_fields[key] = str(raw)[:200]
            continue
        seen.add(name)
        fr = result.fields[name]
        if not isinstance(raw, dict):  # tolerate a bare value instead of {value, ...}
            raw = {"value": raw}
        try:
            vf = VLMField.model_validate(raw)
        except ValidationError:
            fr.warnings.append("invalid_type")
            continue
        value = vf.value.strip() if vf.value else None
        fr.evidence = Evidence(text=vf.evidence) if vf.evidence else None
        if value is None or value.lower() in EMPTY_VALUES:
            fr.warnings.append("not_found")
            continue
        if name == "document_type" and value.lower() not in DOCUMENT_TYPES:
            fr.warnings.append("unsupported_value")
            fr.evidence = Evidence(text=value)
            continue
        fr.value = value.lower() if name == "document_type" else value
        if vf.confidence is not None and 0.0 <= vf.confidence <= 1.0:
            fr.confidence, fr.confidence_method = vf.confidence, "model_self_report"
        else:
            fr.warnings.append("missing_confidence" if vf.confidence is None else "invalid_confidence")
            fr.confidence = FALLBACK_CONFIDENCE if vf.evidence else FALLBACK_CONFIDENCE * 0.6
            fr.confidence_method = "vlm_fallback_heuristic"
    for name in CANONICAL_FIELDS:
        if name not in seen:
            result.fields[name].warnings.append("missing_in_response")
    return result


def make_client() -> anthropic.Anthropic:
    """Credentials come from ANTHROPIC_API_KEY (never hardcoded)."""
    return anthropic.Anthropic(timeout=float(os.getenv("VLM_TIMEOUT_SECONDS", "120")), max_retries=2)


def extract_with_vlm(
    document: str | Path | bytes,
    *,
    client: Any = None,
    model: str | None = None,
    document_type: str | None = None,
) -> ExtractionResult:
    """Public VLM-branch entry point. Accepts PNG/JPEG/GIF/WebP images or a PDF.
    Never raises for API/response problems; see ``result.errors``."""
    model = model or os.getenv("VLM_MODEL") or DEFAULT_MODEL
    try:
        data = _load(document)
    except FileNotFoundError as exc:
        return failed_result("vlm", model, "invalid_input", str(exc))
    media_type = _media_type(data)
    if media_type is None:
        return failed_result("vlm", model, "invalid_input", "unsupported file format")
    limit = MAX_PDF_BYTES if media_type == "application/pdf" else MAX_IMAGE_BYTES
    if len(data) > limit:
        return failed_result("vlm", model, "invalid_input", f"file larger than {limit // (1024 * 1024)} MB")
    if client is None:
        if not os.getenv("ANTHROPIC_API_KEY"):
            return failed_result("vlm", model, "vlm_not_configured", "ANTHROPIC_API_KEY is not set")
        client = make_client()

    block_type = "document" if media_type == "application/pdf" else "image"
    source = {"type": "base64", "media_type": media_type, "data": base64.b64encode(data).decode()}
    content = [{"type": block_type, "source": source}, {"type": "text", "text": PROMPT}]
    try:
        response = client.messages.parse(
            model=model,
            max_tokens=16000,
            output_config={"effort": os.getenv("VLM_EFFORT", "medium")},
            messages=[{"role": "user", "content": content}],
            output_format=VLMResponse,
        )
    except anthropic.APITimeoutError:
        return failed_result("vlm", model, "vlm_timeout", "request timed out")
    except anthropic.APIStatusError as exc:
        return failed_result("vlm", model, "vlm_api_error", f"HTTP {exc.status_code}")
    except anthropic.APIError as exc:
        return failed_result("vlm", model, "vlm_api_error", type(exc).__name__)
    except ValidationError:
        return failed_result("vlm", model, "vlm_invalid_response", "response did not match the schema")

    stop_reason = getattr(response, "stop_reason", None)
    if stop_reason == "refusal":
        return failed_result("vlm", model, "vlm_refused", "model declined the request")
    parsed = getattr(response, "parsed_output", None)
    if parsed is None:
        return failed_result("vlm", model, "vlm_invalid_response", "no structured output")
    result = parse_vlm_output(parsed.model_dump() if isinstance(parsed, BaseModel) else parsed, model)
    if stop_reason == "max_tokens":
        result.warnings.append("response_truncated")
    if document_type:
        result.fields["document_type"] = FieldResult(
            source="vlm", value=document_type, confidence=1.0, confidence_method="caller_provided"
        )
    log.info("vlm finished: %d fields with values", sum(f.value is not None for f in result.fields.values()))
    return result
