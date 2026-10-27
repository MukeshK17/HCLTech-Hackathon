"""Rules Engine, Confidence Scoring, and Review Router for Messy Document Extraction.

Subsystem for Member 3:
- Deterministic validation rules (dates, currencies, monetary Decimal arithmetic, tax, reconciliation).
- OCR-AI consistency checks.
- Field-level heuristic reliability scoring.
- Intelligent routing to auto-acceptance, human review, or extraction retry.
- Revalidation of human corrections from Member 4's review interface.
- Offline evaluation utilities for benchmark datasets.
"""

from __future__ import annotations

import logging
from typing import Any

from rules_engine.confidence import (
    calculate_all_confidences,
    calculate_field_confidence,
    derive_validation_component_score,
)
from rules_engine.config import (
    ConfidenceWeights,
    DateConfig,
    DocumentTypeConfig,
    MonetaryConfig,
    RoutingThresholds,
    RulesEngineConfig,
)
from rules_engine.evaluation import evaluate_predictions
from rules_engine.router import is_retry_warranted, route_document, route_field
from rules_engine.schemas import (
    AgreementStatus,
    CanonicalFieldData,
    ConfidenceBreakdown,
    CrossFieldCheckResult,
    DocumentDecision,
    DocumentResult,
    DocumentType,
    FieldDecision,
    FieldResult,
    FieldValidationResult,
    NormalizedExtractionData,
    ValidationStatus,
    normalize_extraction_inputs,
)
from rules_engine.validator import (
    compare_ocr_and_ai,
    parse_and_validate_date,
    parse_monetary_amount,
    reconcile_total,
    validate_currency,
    validate_document,
    validate_field,
    validate_tax_calculation,
    validate_text_field,
)

logger = logging.getLogger(__name__)

__all__ = [
    # Main public functions
    "process_extraction",
    "revalidate_corrected_fields",
    "normalize_extraction_inputs",
    "evaluate_predictions",
    # Validation helpers
    "validate_document",
    "validate_field",
    "compare_ocr_and_ai",
    "reconcile_total",
    "validate_tax_calculation",
    "parse_and_validate_date",
    "parse_monetary_amount",
    "validate_currency",
    "validate_text_field",
    # Confidence helpers
    "calculate_all_confidences",
    "calculate_field_confidence",
    "derive_validation_component_score",
    # Routing helpers
    "route_document",
    "route_field",
    "is_retry_warranted",
    # Config
    "RulesEngineConfig",
    "ConfidenceWeights",
    "RoutingThresholds",
    "MonetaryConfig",
    "DateConfig",
    "DocumentTypeConfig",
    # Schemas & Enums
    "ValidationStatus",
    "AgreementStatus",
    "FieldDecision",
    "DocumentDecision",
    "DocumentType",
    "CanonicalFieldData",
    "NormalizedExtractionData",
    "FieldValidationResult",
    "CrossFieldCheckResult",
    "ConfidenceBreakdown",
    "FieldResult",
    "DocumentResult",
]


def process_extraction(
    ocr_json: Any,
    ai_json: Any,
    ocr_confidence: Any = None,
    document_type: str = "generic",
    quality_warnings: list[str] | None = None,
    retry_count: int = 0,
    config: RulesEngineConfig | None = None,
) -> DocumentResult:
    """Primary public entry point to process upstream extraction outputs through the rules engine.

    Orchestrates:
    1. Input normalization (handles dicts, JSON strings, missing branches, confidence scales).
    2. Deterministic validation (dates, monetary Decimal amounts, currency, required fields).
    3. OCR-AI consistency checks.
    4. Heuristic reliability score calculation with component decomposition.
    5. Field-level and document-level routing with human-readable explanations.

    Returns:
        DocumentResult: Structured, JSON-serializable output ready for Member 4 review interface.
    """
    engine_config = config or RulesEngineConfig()

    # 1. Normalize upstream inputs
    normalized_data = normalize_extraction_inputs(
        ocr_json=ocr_json,
        ai_json=ai_json,
        ocr_confidence=ocr_confidence,
        document_type=document_type,
        quality_warnings=quality_warnings,
        retry_count=retry_count,
    )

    # 2. Deterministic validation
    field_validations, normalized_values, cross_checks, critical_failures = validate_document(
        normalized_data=normalized_data,
        engine_config=engine_config,
    )

    # 3. OCR-AI comparison for every field
    agreement_results: dict[str, tuple[AgreementStatus, float | None, list[str]]] = {}
    for f_name, f_data in normalized_data.fields.items():
        agr_res = compare_ocr_and_ai(
            field_name=f_name,
            ocr_val=f_data.ocr_value,
            ai_val=f_data.ai_value,
            monetary_config=engine_config.monetary,
        )
        agreement_results[f_name] = agr_res

        # If a critical field has an OCR-AI mismatch, record critical failure
        doc_type_cfg = engine_config.get_document_config(normalized_data.document_type)
        if f_name in doc_type_cfg.required_fields and agr_res[0] == AgreementStatus.MISMATCH:
            crit_msg = f"Critical field '{f_name}' has OCR-AI disagreement"
            if crit_msg not in critical_failures:
                critical_failures.append(crit_msg)

    # 4. Calculate field reliability confidences
    confidence_breakdowns = calculate_all_confidences(
        fields_data=normalized_data.fields,
        validations=field_validations,
        agreement_results=agreement_results,
        config=engine_config,
    )

    # 5. Route fields and document
    return route_document(
        normalized_data=normalized_data,
        field_validations=field_validations,
        normalized_values=normalized_values,
        agreement_results=agreement_results,
        confidence_breakdowns=confidence_breakdowns,
        cross_checks=cross_checks,
        critical_failures=critical_failures,
        config=engine_config,
    )


def revalidate_corrected_fields(
    original_document: DocumentResult | dict[str, Any],
    corrected_fields: dict[str, Any],
    config: RulesEngineConfig | None = None,
) -> DocumentResult:
    """Revalidate document after human corrections in Member 4's review interface.

    Preserves original OCR and AI values, re-runs deterministic validation on corrected
    values, re-executes cross-field arithmetic checks (reconciliation, tax), and updates
    routing decisions and provenance without recalculating OCR confidence.
    """
    engine_config = config or RulesEngineConfig()

    # Convert to dict if DocumentResult instance
    doc_dict: dict[str, Any] = (
        original_document.to_dict() if hasattr(original_document, "to_dict") else original_document
    )

    doc_type = doc_dict.get("document_type", "generic")
    existing_fields = doc_dict.get("fields", {})

    # Reconstruct CanonicalFieldData with updated human corrections
    updated_fields: dict[str, CanonicalFieldData] = {}
    for f_name, f_info in existing_fields.items():
        ocr_val = f_info.get("ocr_value")
        ai_val = f_info.get("ai_value")
        conf_breakdown = f_info.get("confidence_breakdown", {})
        raw_ocr_conf = conf_breakdown.get("raw_ocr_confidence")
        norm_ocr_conf = conf_breakdown.get("normalized_ocr_confidence")

        # Check if human corrected this field
        if f_name in corrected_fields:
            human_val = corrected_fields[f_name]
            updated_fields[f_name] = CanonicalFieldData(
                field_name=f_name,
                ocr_value=ocr_val,
                ai_value=ai_val,
                normalized_value=human_val,
                raw_ocr_confidence=raw_ocr_conf,
                normalized_ocr_confidence=norm_ocr_conf,
                confidence_notes="Preserved from original extraction; field human-corrected",
                is_human_corrected=True,
                is_present_in_ocr=ocr_val is not None,
                is_present_in_ai=ai_val is not None,
            )
        else:
            updated_fields[f_name] = CanonicalFieldData(
                field_name=f_name,
                ocr_value=ocr_val,
                ai_value=ai_val,
                normalized_value=f_info.get("normalized_value"),
                raw_ocr_confidence=raw_ocr_conf,
                normalized_ocr_confidence=norm_ocr_conf,
                confidence_notes=None,
                is_human_corrected=f_info.get("is_human_corrected", False),
                is_present_in_ocr=ocr_val is not None,
                is_present_in_ai=ai_val is not None,
            )

    # Include newly added fields from corrections if any
    for corr_name, corr_val in corrected_fields.items():
        if corr_name not in updated_fields:
            updated_fields[corr_name] = CanonicalFieldData(
                field_name=corr_name,
                ocr_value=None,
                ai_value=None,
                normalized_value=corr_val,
                raw_ocr_confidence=None,
                normalized_ocr_confidence=None,
                confidence_notes="Added via human correction",
                is_human_corrected=True,
                is_present_in_ocr=False,
                is_present_in_ai=False,
            )

    # Build updated normalized data
    norm_doc = NormalizedExtractionData(
        fields=updated_fields,
        document_type=doc_type,
        quality_warnings=doc_dict.get("metadata", {}).get("quality_warnings", []),
        retry_count=doc_dict.get("metadata", {}).get("retry_count", 0),
    )

    # Re-run deterministic validation
    field_validations, normalized_values, cross_checks, critical_failures = validate_document(
        normalized_data=norm_doc,
        engine_config=engine_config,
    )

    # For agreement and confidence:
    # Human-corrected fields are marked valid if deterministic checks pass.
    # Preserve original confidence breakdowns or assign high reliability for human-verified fields
    updated_confidences: dict[str, ConfidenceBreakdown] = {}
    updated_agreements: dict[str, tuple[AgreementStatus, float | None, list[str]]] = {}

    for f_name, f_data in updated_fields.items():
        v_res = field_validations.get(
            f_name,
            FieldValidationResult(
                status=ValidationStatus.NOT_CHECKABLE,
                reasons=["Not validated"],
            ),
        )

        if f_data.is_human_corrected:
            # Human verified value
            if v_res.status == ValidationStatus.VALID:
                updated_confidences[f_name] = ConfidenceBreakdown(
                    final_score=1.0,
                    raw_ocr_confidence=f_data.raw_ocr_confidence,
                    normalized_ocr_confidence=f_data.normalized_ocr_confidence,
                    agreement_score=1.0,
                    validation_score=1.0,
                    components_used=["human_verified", "validation"],
                    components_missing=[],
                    weights_used={"human_verified": 0.5, "validation": 0.5},
                    evidence_sufficient=True,
                    explanations=["Field verified and corrected by human reviewer; deterministic validation passed"],
                )
                updated_agreements[f_name] = (
                    AgreementStatus.MATCH,
                    1.0,
                    ["Human verified value"],
                )
            else:
                # Human entered an invalid value! (e.g. malformed date or amount)
                updated_confidences[f_name] = ConfidenceBreakdown(
                    final_score=0.0,
                    raw_ocr_confidence=f_data.raw_ocr_confidence,
                    normalized_ocr_confidence=f_data.normalized_ocr_confidence,
                    agreement_score=0.0,
                    validation_score=0.0,
                    components_used=["validation"],
                    components_missing=[],
                    weights_used={"validation": 1.0},
                    evidence_sufficient=True,
                    explanations=[f"Human entered value failed validation: {'; '.join(v_res.reasons)}"],
                )
                updated_agreements[f_name] = (
                    AgreementStatus.NOT_COMPARABLE,
                    None,
                    ["Human entered invalid value"],
                )
        else:
            # Field was not modified by human; keep original confidence or recalculate
            agr_res = compare_ocr_and_ai(
                f_name,
                f_data.ocr_value,
                f_data.ai_value,
                engine_config.monetary,
            )
            updated_agreements[f_name] = agr_res
            updated_confidences[f_name] = calculate_field_confidence(
                field_data=f_data,
                validation_result=v_res,
                agreement_status=agr_res[0],
                agreement_score=agr_res[1],
                config=engine_config,
            )

    # Route updated document
    result = route_document(
        normalized_data=norm_doc,
        field_validations=field_validations,
        normalized_values=normalized_values,
        agreement_results=updated_agreements,
        confidence_breakdowns=updated_confidences,
        cross_checks=cross_checks,
        critical_failures=critical_failures,
        config=engine_config,
    )

    result.metadata["revalidated_at_review"] = True
    result.metadata["corrected_fields_count"] = len(corrected_fields)
    return result
