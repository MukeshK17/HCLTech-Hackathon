"""Output format shared by the OCR and VLM branches.

Every field = the value exactly as printed on the document + its own confidence.
``ExtractionResult.record()`` gives just the values; ``fields`` adds the metadata.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

SCHEMA_VERSION = "1.0"

Source = Literal["ocr", "vlm"]
ConfidenceMethod = Literal[
    "ocr_heuristic",  # OCR recognition score x label/position/ambiguity factors
    "model_self_report",  # VLM's own estimate, NOT a calibrated probability
    "vlm_fallback_heuristic",  # VLM gave no usable confidence; conservative default
    "caller_provided",  # value supplied by the caller (e.g. known document type)
    "none",  # no value found -> confidence 0.0
]

DOCUMENT_TYPES = ("receipt", "invoice", "delivery_note", "form")
CANONICAL_FIELDS = (
    "document_type",
    "vendor",
    "address",
    "invoice_number",
    "order_number",
    "date",
    "currency",
    "subtotal",
    "tax_rate",
    "tax",
    "rounding_adjustment",
    "total",
)


class Evidence(BaseModel):
    text: str | None = None  # the line/text the value was read from
    bbox: list[float] | None = None  # [x_min, y_min, x_max, y_max] in pixels (OCR only)


class Candidate(BaseModel):
    value: str
    score: float
    evidence: Evidence | None = None


class FieldResult(BaseModel):
    value: str | None = None  # verbatim text from the document, never reformatted
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    confidence_method: ConfidenceMethod = "none"
    source: Source
    evidence: Evidence | None = None
    candidates: list[Candidate] = Field(default_factory=list)  # other possible values
    warnings: list[str] = Field(default_factory=list)


class ExtractionError(BaseModel):
    code: str  # invalid_input | ocr_unavailable | ocr_failed | vlm_not_configured | vlm_timeout
    message: str  #  | vlm_api_error | vlm_refused | vlm_invalid_response


class ExtractionResult(BaseModel):
    schema_version: str = SCHEMA_VERSION
    source: Source
    engine: str  # "paddleocr" or the VLM model id
    fields: dict[str, FieldResult]  # always contains every CANONICAL_FIELDS key
    extra_fields: dict[str, str] = Field(default_factory=dict)  # unmapped fields, kept not dropped
    raw_text: str | None = None  # full OCR text (OCR branch only, in memory)
    warnings: list[str] = Field(default_factory=list)
    errors: list[ExtractionError] = Field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def record(self) -> dict[str, str | None]:
        """Business record: field name -> value as printed (or None)."""
        return {name: self.fields[name].value for name in CANONICAL_FIELDS}

    def to_json_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


def empty_fields(source: Source) -> dict[str, FieldResult]:
    return {name: FieldResult(source=source) for name in CANONICAL_FIELDS}


def failed_result(source: Source, engine: str, code: str, message: str) -> ExtractionResult:
    return ExtractionResult(
        source=source,
        engine=engine,
        fields=empty_fields(source),
        errors=[ExtractionError(code=code, message=message)],
    )
