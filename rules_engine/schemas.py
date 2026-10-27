"""Schemas and input normalization for Member 3's Rules Engine subsystem.

Defines canonical field representations, enums, structured result containers,
and input normalization adapters for handling varied upstream outputs from Member 2.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class ValidationStatus(str, Enum):
    """Status of deterministic field and cross-field checks."""

    VALID = "valid"
    INVALID = "invalid"
    WARNING = "warning"
    NOT_CHECKABLE = "not_checkable"
    MISSING = "missing"


class AgreementStatus(str, Enum):
    """Comparison status between OCR and AI extraction branches."""

    MATCH = "match"
    MISMATCH = "mismatch"
    OCR_MISSING = "ocr_missing"
    AI_MISSING = "ai_missing"
    BOTH_MISSING = "both_missing"
    NOT_COMPARABLE = "not_comparable"


class FieldDecision(str, Enum):
    """Routing decision for an individual field."""

    AUTO_ACCEPT = "auto_accept"
    HUMAN_REVIEW = "human_review"
    RETRY_EXTRACTION = "retry_extraction"
    REJECT = "reject"


class DocumentDecision(str, Enum):
    """Routing decision for an entire document."""

    AUTO_ACCEPTED = "auto_accepted"
    HUMAN_REVIEW = "human_review"
    RETRY_EXTRACTION = "retry_extraction"
    REJECTED = "rejected"


class DocumentType(str, Enum):
    """Supported document types."""

    RECEIPT = "receipt"
    INVOICE = "invoice"
    DELIVERY_NOTE = "delivery_note"
    FORM = "form"
    GENERIC = "generic"


# Standard canonical fields expected in receipts and invoices
CANONICAL_FIELDS: tuple[str, ...] = (
    "vendor_name",
    "date",
    "address",
    "currency",
    "subtotal",
    "tax_amount",
    "tax_rate",
    "discount_amount",
    "rounding_adjustment",
    "total_amount",
)

# Alias map for upstream field name variations from various models or OCR formats
FIELD_ALIASES: dict[str, str] = {
    # Vendor / company
    "company": "vendor_name",
    "company_name": "vendor_name",
    "merchant": "vendor_name",
    "merchant_name": "vendor_name",
    "seller": "vendor_name",
    "store": "vendor_name",
    "store_name": "vendor_name",
    # Total
    "total": "total_amount",
    "grand_total": "total_amount",
    "amount_due": "total_amount",
    "invoice_total": "total_amount",
    "net_total": "total_amount",
    # Subtotal
    "sub_total": "subtotal",
    "taxable_amount": "subtotal",
    "taxable_base": "subtotal",
    # Tax
    "tax": "tax_amount",
    "gst": "tax_amount",
    "vat": "tax_amount",
    "sales_tax": "tax_amount",
    # Tax rate
    "gst_rate": "tax_rate",
    "vat_rate": "tax_rate",
    "tax_percentage": "tax_rate",
    # Discount
    "discount": "discount_amount",
    # Rounding
    "rounding": "rounding_adjustment",
    "round_off": "rounding_adjustment",
    # Date
    "invoice_date": "date",
    "receipt_date": "date",
    "datetime": "date",
    # Address
    "vendor_address": "address",
    "store_address": "address",
}


@dataclass
class CanonicalFieldData:
    """Internal canonical representation of a single field extracted from upstream."""

    field_name: str
    ocr_value: Any = None
    ai_value: Any = None
    normalized_value: Any = None
    raw_ocr_confidence: float | None = None
    normalized_ocr_confidence: float | None = None
    confidence_notes: str | None = None
    is_human_corrected: bool = False
    is_present_in_ocr: bool = False
    is_present_in_ai: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict."""
        return asdict(self)


@dataclass
class NormalizedExtractionData:
    """Normalized extraction inputs containing canonical field data and metadata."""

    fields: dict[str, CanonicalFieldData] = field(default_factory=dict)
    additional_fields: dict[str, Any] = field(default_factory=dict)
    raw_ocr_input: Any = None
    raw_ai_input: Any = None
    raw_confidence_input: Any = None
    quality_warnings: list[str] = field(default_factory=list)
    document_type: str = "generic"
    retry_count: int = 0
    parsing_errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "fields": {k: v.to_dict() for k, v in self.fields.items()},
            "additional_fields": self.additional_fields,
            "quality_warnings": self.quality_warnings,
            "document_type": self.document_type,
            "retry_count": self.retry_count,
            "parsing_errors": self.parsing_errors,
        }


@dataclass
class FieldValidationResult:
    """Result of deterministic validation on a single field."""

    status: ValidationStatus
    reasons: list[str] = field(default_factory=list)
    is_critical_failure: bool = False
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict."""
        return {
            "status": self.status.value,
            "reasons": self.reasons,
            "is_critical_failure": self.is_critical_failure,
            "details": self.details,
        }


@dataclass
class CrossFieldCheckResult:
    """Result of multi-field checks like total reconciliation or tax calculation."""

    check_name: str
    status: ValidationStatus
    expected_value: Any = None
    actual_value: Any = None
    difference: Any = None
    tolerance: Any = None
    reasons: list[str] = field(default_factory=list)
    is_critical_failure: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict."""
        return {
            "check_name": self.check_name,
            "status": self.status.value,
            "expected_value": str(self.expected_value) if self.expected_value is not None else None,
            "actual_value": str(self.actual_value) if self.actual_value is not None else None,
            "difference": str(self.difference) if self.difference is not None else None,
            "tolerance": str(self.tolerance) if self.tolerance is not None else None,
            "reasons": self.reasons,
            "is_critical_failure": self.is_critical_failure,
        }


@dataclass
class ConfidenceBreakdown:
    """Comprehensive breakdown of field reliability score components."""

    final_score: float | None
    raw_ocr_confidence: float | None
    normalized_ocr_confidence: float | None
    agreement_score: float | None
    validation_score: float | None
    components_used: list[str]
    components_missing: list[str]
    weights_used: dict[str, float]
    evidence_sufficient: bool
    explanations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict."""
        return {
            "final_score": round(self.final_score, 4) if self.final_score is not None else None,
            "raw_ocr_confidence": self.raw_ocr_confidence,
            "normalized_ocr_confidence": (
                round(self.normalized_ocr_confidence, 4) if self.normalized_ocr_confidence is not None else None
            ),
            "agreement_score": self.agreement_score,
            "validation_score": self.validation_score,
            "components_used": self.components_used,
            "components_missing": self.components_missing,
            "weights_used": {k: round(v, 4) for k, v in self.weights_used.items()},
            "evidence_sufficient": self.evidence_sufficient,
            "explanations": self.explanations,
        }


@dataclass
class FieldResult:
    """End-to-end processing result for an individual field."""

    field_name: str
    ocr_value: Any
    ai_value: Any
    normalized_value: Any
    confidence_score: float | None
    validation_status: ValidationStatus
    validation_reasons: list[str]
    agreement_status: AgreementStatus
    decision: FieldDecision
    decision_reasons: list[str]
    evidence_missing: list[str]
    evidence_sufficient: bool
    confidence_breakdown: ConfidenceBreakdown
    is_human_corrected: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "field_name": self.field_name,
            "ocr_value": self.ocr_value,
            "ai_value": self.ai_value,
            "normalized_value": (
                str(self.normalized_value)
                if hasattr(self.normalized_value, "is_finite")  # Decimal check
                else self.normalized_value
            ),
            "confidence_score": (round(self.confidence_score, 4) if self.confidence_score is not None else None),
            "validation_status": self.validation_status.value,
            "validation_reasons": self.validation_reasons,
            "agreement_status": self.agreement_status.value,
            "decision": self.decision.value,
            "decision_reasons": self.decision_reasons,
            "evidence_missing": self.evidence_missing,
            "evidence_sufficient": self.evidence_sufficient,
            "confidence_breakdown": self.confidence_breakdown.to_dict(),
            "is_human_corrected": self.is_human_corrected,
        }


@dataclass
class DocumentResult:
    """End-to-end processing result for an entire document."""

    document_type: str
    document_decision: DocumentDecision
    decision_reasons: list[str]
    fields: dict[str, FieldResult]
    cross_field_checks: list[CrossFieldCheckResult]
    critical_failures: list[str]
    is_auto_acceptable: bool
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary for Member 4 review interface."""
        return {
            "document_type": self.document_type,
            "document_decision": self.document_decision.value,
            "decision_reasons": self.decision_reasons,
            "fields": {k: v.to_dict() for k, v in self.fields.items()},
            "cross_field_checks": [c.to_dict() for c in self.cross_field_checks],
            "critical_failures": self.critical_failures,
            "is_auto_acceptable": self.is_auto_acceptable,
            "metadata": self.metadata,
        }


def _safe_parse_json(raw_input: Any, context_name: str, errors: list[str]) -> Any:
    """Safely parse dict or JSON string without throwing unhandled exceptions."""
    if raw_input is None:
        return None
    if isinstance(raw_input, (dict, list)):
        return raw_input
    if isinstance(raw_input, str):
        trimmed = raw_input.strip()
        if not trimmed:
            return None
        try:
            return json.loads(trimmed)
        except json.JSONDecodeError as exc:
            msg = f"Failed to parse JSON string for {context_name}: {exc}"
            logger.warning(msg)
            errors.append(msg)
            return None
    msg = f"Unexpected type for {context_name}: {type(raw_input).__name__}, expected dict or JSON string"
    logger.warning(msg)
    errors.append(msg)
    return None


def _normalize_confidence_value(raw_val: Any) -> tuple[float | None, str | None]:
    """Normalize raw OCR confidence value to float in [0.0, 1.0].

    Returns (normalized_score, explanation_note).
    """
    if raw_val is None:
        return None, "OCR confidence is missing"
    if isinstance(raw_val, bool):
        return None, "Boolean is not a valid confidence value"
    if isinstance(raw_val, (int, float)):
        val = float(raw_val)
        if val < 0.0:
            return None, f"Negative confidence {val} is invalid"
        if val > 100.0:
            return None, f"Confidence {val} exceeds 100 scale"
        if val > 1.0:
            # Assumed 0-100 percentage scale
            return val / 100.0, "Normalized from 0-100 scale"
        # Already 0.0 - 1.0 scale
        return val, "Standard 0.0-1.0 scale"
    if isinstance(raw_val, str):
        cleaned = raw_val.strip().rstrip("%")
        try:
            num = float(cleaned)
            return _normalize_confidence_value(num)
        except ValueError:
            return None, f"Could not parse confidence string '{raw_val}'"
    return None, f"Unsupported confidence type: {type(raw_val).__name__}"


def _canonicalize_field_name(name: str) -> str:
    """Map field name alias to canonical field name."""
    norm = name.strip().lower()
    return FIELD_ALIASES.get(norm, norm)


def normalize_extraction_inputs(
    ocr_json: Any,
    ai_json: Any,
    ocr_confidence: Any = None,
    document_type: str = "generic",
    quality_warnings: list[str] | None = None,
    retry_count: int = 0,
) -> NormalizedExtractionData:
    """Normalize and adapt Member 2's OCR, AI, and confidence outputs into canonical structures.

    Handles:
    - Dicts, JSON strings, None, or empty values.
    - Field alias resolution.
    - Explicit tracking of missing branches (OCR missing vs AI missing).
    - OCR confidence format detection (field-level vs segment-level without mapping).
    - Preserving raw inputs for evidence traceability.
    """
    parsing_errors: list[str] = []
    parsed_ocr = _safe_parse_json(ocr_json, "ocr_json", parsing_errors)
    parsed_ai = _safe_parse_json(ai_json, "ai_json", parsing_errors)
    parsed_conf = _safe_parse_json(ocr_confidence, "ocr_confidence", parsing_errors)

    warnings: list[str] = list(quality_warnings) if quality_warnings else []

    # Map upstream dicts
    ocr_dict: dict[str, Any] = {}
    ocr_segment_level = False
    if isinstance(parsed_ocr, dict):
        # Check if this is a segment/line list (e.g. {"lines": [...], "boxes": [...]})
        if "lines" in parsed_ocr or "segments" in parsed_ocr or "boxes" in parsed_ocr:
            ocr_segment_level = True
        for k, v in parsed_ocr.items():
            ocr_dict[_canonicalize_field_name(k)] = v
    elif isinstance(parsed_ocr, list):
        # List of segment tokens or lines: mark as segment-level
        ocr_segment_level = True

    ai_dict: dict[str, Any] = {}
    if isinstance(parsed_ai, dict):
        for k, v in parsed_ai.items():
            ai_dict[_canonicalize_field_name(k)] = v

    conf_dict: dict[str, Any] = {}
    conf_is_segment = False
    if isinstance(parsed_conf, dict):
        # Check if keys look like segment IDs or field names
        for k, v in parsed_conf.items():
            canon_k = _canonicalize_field_name(k)
            conf_dict[canon_k] = v
    elif isinstance(parsed_conf, (int, float)):
        # Scalar document-level confidence provided
        warnings.append(f"Scalar OCR confidence {parsed_conf} provided; not mapped to specific fields")
    elif isinstance(parsed_conf, list):
        conf_is_segment = True
        warnings.append("Segment-level OCR confidence list provided without reliable field mapping")

    # Collect all field names
    all_keys = set(CANONICAL_FIELDS) | set(ocr_dict.keys()) | set(ai_dict.keys())
    # Exclude structural metadata keys if OCR was segment-level
    if ocr_segment_level:
        all_keys.discard("lines")
        all_keys.discard("segments")
        all_keys.discard("boxes")

    canonical_fields: dict[str, CanonicalFieldData] = {}
    additional_fields: dict[str, Any] = {}

    for key in all_keys:
        is_in_ocr = key in ocr_dict
        is_in_ai = key in ai_dict
        ocr_val = ocr_dict.get(key)
        ai_val = ai_dict.get(key)

        # Handle OCR confidence mapping
        raw_conf: Any = None
        norm_conf: float | None = None
        conf_note: str | None = None

        if conf_is_segment or ocr_segment_level:
            conf_note = "Segment-level OCR confidence provided without reliable field mapping"
        elif key in conf_dict:
            raw_conf = conf_dict[key]
            norm_conf, conf_note = _normalize_confidence_value(raw_conf)
        else:
            conf_note = "OCR confidence unavailable for this field"

        field_data = CanonicalFieldData(
            field_name=key,
            ocr_value=ocr_val,
            ai_value=ai_val,
            normalized_value=ai_val if ai_val is not None else ocr_val,
            raw_ocr_confidence=raw_conf if isinstance(raw_conf, (int, float)) else None,
            normalized_ocr_confidence=norm_conf,
            confidence_notes=conf_note,
            is_present_in_ocr=is_in_ocr,
            is_present_in_ai=is_in_ai,
        )

        if key in CANONICAL_FIELDS:
            canonical_fields[key] = field_data
        else:
            # Preserve additional fields
            canonical_fields[key] = field_data
            additional_fields[key] = {
                "ocr": ocr_val,
                "ai": ai_val,
            }

    return NormalizedExtractionData(
        fields=canonical_fields,
        additional_fields=additional_fields,
        raw_ocr_input=ocr_json,
        raw_ai_input=ai_json,
        raw_confidence_input=ocr_confidence,
        quality_warnings=warnings,
        document_type=document_type,
        retry_count=retry_count,
        parsing_errors=parsing_errors,
    )
