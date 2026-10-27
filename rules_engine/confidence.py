"""Field-level heuristic reliability scoring module for Member 3's Rules Engine.

Implements the configurable heuristic reliability score formula:
    C_f = w_ocr * O + w_agreement * A + w_validation * V

Provides transparent decomposition of:
- O: Normalized OCR confidence from Member 2.
- A: Agreement score between OCR and AI branches.
- V: Deterministic validation score.

Handles missing evidence via weight renormalization while strictly enforcing
evidence sufficiency to prevent premature automatic acceptance.
"""

from __future__ import annotations

import logging

from rules_engine.config import RulesEngineConfig
from rules_engine.schemas import (
    AgreementStatus,
    CanonicalFieldData,
    ConfidenceBreakdown,
    FieldValidationResult,
    ValidationStatus,
)

logger = logging.getLogger(__name__)


def derive_validation_component_score(
    validation_result: FieldValidationResult,
) -> tuple[float | None, str]:
    """Map deterministic validation status to numerical validation score V in [0.0, 1.0] or None.

    Rules:
    - VALID: 1.0 (all checks pass with positive evidence)
    - WARNING: 0.5 (noncritical warning only)
    - INVALID or Critical Failure: 0.0 (material failure or contradiction)
    - NOT_CHECKABLE / MISSING: None (insufficient evidence to evaluate check)
    """
    if validation_result.is_critical_failure or validation_result.status == ValidationStatus.INVALID:
        return 0.0, "Validation failed or critical check triggered"
    if validation_result.status == ValidationStatus.VALID:
        return 1.0, "Deterministic validation passed"
    if validation_result.status == ValidationStatus.WARNING:
        return 0.5, "Passed with noncritical warnings"
    if validation_result.status in (ValidationStatus.NOT_CHECKABLE, ValidationStatus.MISSING):
        return None, f"Validation evidence unavailable ({validation_result.status.value})"
    return None, f"Unknown validation status: {validation_result.status}"


def calculate_field_confidence(
    field_data: CanonicalFieldData,
    validation_result: FieldValidationResult,
    agreement_status: AgreementStatus,
    agreement_score: float | None,
    config: RulesEngineConfig,
) -> ConfidenceBreakdown:
    """Calculate field-level heuristic reliability score and generate component breakdown.

    All weights, thresholds, and evidence sufficiency rules are configurable.
    """
    explanations: list[str] = []
    components_used: list[str] = []
    components_missing: list[str] = []

    # 1. Component O: OCR Confidence
    o_val: float | None = field_data.normalized_ocr_confidence
    if o_val is not None:
        components_used.append("ocr_confidence")
        if field_data.confidence_notes:
            explanations.append(f"OCR confidence ({o_val:.3f}): {field_data.confidence_notes}")
    else:
        components_missing.append("ocr_confidence")
        note = field_data.confidence_notes or "OCR confidence unavailable"
        explanations.append(f"OCR confidence missing: {note}")

    # 2. Component A: Agreement Score
    a_val: float | None = agreement_score
    if a_val is not None:
        components_used.append("agreement")
        explanations.append(f"OCR-AI agreement ({agreement_status.value}): {a_val:.2f}")
    else:
        components_missing.append("agreement")
        explanations.append(f"OCR-AI agreement missing: {agreement_status.value}")

    # 3. Component V: Validation Score
    v_val, v_note = derive_validation_component_score(validation_result)
    if v_val is not None:
        components_used.append("validation")
        explanations.append(f"Validation score ({validation_result.status.value}): {v_val:.2f} ({v_note})")
    else:
        components_missing.append("validation")
        explanations.append(f"Validation score missing: {v_note}")

    # If all components are missing
    if not components_used:
        explanations.append("All confidence components missing; score is None")
        return ConfidenceBreakdown(
            final_score=None,
            raw_ocr_confidence=field_data.raw_ocr_confidence,
            normalized_ocr_confidence=field_data.normalized_ocr_confidence,
            agreement_score=None,
            validation_score=None,
            components_used=[],
            components_missing=["ocr_confidence", "agreement", "validation"],
            weights_used={},
            evidence_sufficient=False,
            explanations=explanations,
        )

    # 4. Weight renormalization for available components
    raw_weights: dict[str, tuple[float, float]] = {
        "ocr_confidence": (config.weights.w_ocr, o_val if o_val is not None else 0.0),
        "agreement": (config.weights.w_agreement, a_val if a_val is not None else 0.0),
        "validation": (config.weights.w_validation, v_val if v_val is not None else 0.0),
    }

    sum_available_weights = sum(raw_weights[c][0] for c in components_used)
    if sum_available_weights <= 0.0:
        explanations.append("Sum of available weights is zero; score cannot be computed")
        return ConfidenceBreakdown(
            final_score=None,
            raw_ocr_confidence=field_data.raw_ocr_confidence,
            normalized_ocr_confidence=field_data.normalized_ocr_confidence,
            agreement_score=a_val,
            validation_score=v_val,
            components_used=components_used,
            components_missing=components_missing,
            weights_used={},
            evidence_sufficient=False,
            explanations=explanations,
        )

    weights_used: dict[str, float] = {}
    weighted_sum = 0.0
    for comp in components_used:
        nominal_w, val = raw_weights[comp]
        renorm_w = nominal_w / sum_available_weights
        weights_used[comp] = renorm_w
        weighted_sum += renorm_w * val

    # Ensure score is strictly bounded in [0.0, 1.0]
    final_score = max(0.0, min(1.0, weighted_sum))

    # Evidence sufficiency evaluation
    evidence_sufficient = len(components_used) >= config.min_components_for_sufficiency
    if not evidence_sufficient:
        explanations.append(
            f"Evidence insufficient: only {len(components_used)} component(s) available "
            f"({', '.join(components_used)}), minimum required is {config.min_components_for_sufficiency}"
        )

    return ConfidenceBreakdown(
        final_score=final_score,
        raw_ocr_confidence=field_data.raw_ocr_confidence,
        normalized_ocr_confidence=field_data.normalized_ocr_confidence,
        agreement_score=a_val,
        validation_score=v_val,
        components_used=components_used,
        components_missing=components_missing,
        weights_used=weights_used,
        evidence_sufficient=evidence_sufficient,
        explanations=explanations,
    )


def calculate_all_confidences(
    fields_data: dict[str, CanonicalFieldData],
    validations: dict[str, FieldValidationResult],
    agreement_results: dict[str, tuple[AgreementStatus, float | None, list[str]]],
    config: RulesEngineConfig,
) -> dict[str, ConfidenceBreakdown]:
    """Calculate confidence breakdowns for all fields in the document."""
    breakdowns: dict[str, ConfidenceBreakdown] = {}
    for f_name, f_data in fields_data.items():
        v_res = validations.get(
            f_name,
            FieldValidationResult(
                status=ValidationStatus.NOT_CHECKABLE,
                reasons=["Validation not performed"],
            ),
        )
        agr_res = agreement_results.get(
            f_name,
            (AgreementStatus.NOT_COMPARABLE, None, ["Not compared"]),
        )
        agr_status, agr_score, _ = agr_res
        breakdowns[f_name] = calculate_field_confidence(
            field_data=f_data,
            validation_result=v_res,
            agreement_status=agr_status,
            agreement_score=agr_score,
            config=config,
        )
    return breakdowns
