"""Review router module for Member 3's Rules Engine.

Evaluates field-level reliability scores, evidence sufficiency, validation statuses,
critical overrides, and upstream quality warnings to produce:
1. Field-level decisions (AUTO_ACCEPT, HUMAN_REVIEW, RETRY_EXTRACTION, REJECT)
2. Document-level decisions (AUTO_ACCEPTED, HUMAN_REVIEW, RETRY_EXTRACTION, REJECTED)
3. Transparent, human-readable explanations for every decision.
"""

from __future__ import annotations

import logging
from typing import Any

from rules_engine.config import RulesEngineConfig
from rules_engine.schemas import (
    AgreementStatus,
    ConfidenceBreakdown,
    CrossFieldCheckResult,
    DocumentDecision,
    DocumentResult,
    FieldDecision,
    FieldResult,
    FieldValidationResult,
    NormalizedExtractionData,
    ValidationStatus,
)

logger = logging.getLogger(__name__)

# Keywords in quality warnings that suggest extraction retry is warranted
RETRY_WARNING_KEYWORDS: tuple[str, ...] = (
    "unusable ocr",
    "empty ocr",
    "severe blur",
    "severe skew",
    "truncated image",
    "partial scan",
    "unreadable",
    "ocr failed",
)


def is_retry_warranted(quality_warnings: list[str], retry_count: int, max_retries: int) -> tuple[bool, str]:
    """Check if upstream quality warnings warrant an extraction retry."""
    if retry_count >= max_retries:
        return (
            False,
            f"Retry limit ({max_retries}) reached; routing to human review instead of retrying",
        )

    for warn in quality_warnings:
        warn_lower = warn.lower()
        if any(kw in warn_lower for kw in RETRY_WARNING_KEYWORDS):
            return True, f"Extraction retry recommended due to upstream warning: '{warn}'"

    return False, ""


def route_field(
    field_name: str,
    field_data: Any,
    norm_val: Any,
    validation_res: FieldValidationResult,
    agreement_res: tuple[AgreementStatus, float | None, list[str]],
    confidence: ConfidenceBreakdown,
    is_required: bool,
    config: RulesEngineConfig,
    retry_warranted: bool,
    retry_reason: str,
) -> FieldResult:
    """Determine the routing decision and readable explanations for a single field."""
    agr_status, _, agr_reasons = agreement_res
    reasons: list[str] = []
    score = confidence.final_score

    # Check 1: Retry extraction
    if retry_warranted:
        decision = FieldDecision.RETRY_EXTRACTION
        reasons.append(retry_reason)
        return FieldResult(
            field_name=field_name,
            ocr_value=field_data.ocr_value,
            ai_value=field_data.ai_value,
            normalized_value=norm_val,
            confidence_score=score,
            validation_status=validation_res.status,
            validation_reasons=validation_res.reasons,
            agreement_status=agr_status,
            decision=decision,
            decision_reasons=reasons,
            evidence_missing=confidence.components_missing,
            evidence_sufficient=confidence.evidence_sufficient,
            confidence_breakdown=confidence,
            is_human_corrected=getattr(field_data, "is_human_corrected", False),
        )

    # Check 2: Critical validation failure overrides any confidence score
    if validation_res.is_critical_failure:
        decision = FieldDecision.HUMAN_REVIEW
        reasons.append("Critical validation failure forces human review")
        reasons.extend(validation_res.reasons)
        return FieldResult(
            field_name=field_name,
            ocr_value=field_data.ocr_value,
            ai_value=field_data.ai_value,
            normalized_value=norm_val,
            confidence_score=score,
            validation_status=validation_res.status,
            validation_reasons=validation_res.reasons,
            agreement_status=agr_status,
            decision=decision,
            decision_reasons=reasons,
            evidence_missing=confidence.components_missing,
            evidence_sufficient=confidence.evidence_sufficient,
            confidence_breakdown=confidence,
            is_human_corrected=getattr(field_data, "is_human_corrected", False),
        )

    # Check 3: Missing required field
    if is_required and validation_res.status == ValidationStatus.MISSING:
        decision = FieldDecision.HUMAN_REVIEW
        reasons.append(f"Required field '{field_name}' is missing")
        return FieldResult(
            field_name=field_name,
            ocr_value=field_data.ocr_value,
            ai_value=field_data.ai_value,
            normalized_value=norm_val,
            confidence_score=score,
            validation_status=validation_res.status,
            validation_reasons=validation_res.reasons,
            agreement_status=agr_status,
            decision=decision,
            decision_reasons=reasons,
            evidence_missing=confidence.components_missing,
            evidence_sufficient=confidence.evidence_sufficient,
            confidence_breakdown=confidence,
            is_human_corrected=getattr(field_data, "is_human_corrected", False),
        )

    # Check 4: Insufficient evidence blocks automatic acceptance
    if not confidence.evidence_sufficient:
        decision = FieldDecision.HUMAN_REVIEW
        reasons.append(
            f"Evidence is insufficient for safe auto-acceptance "
            f"({len(confidence.components_used)} of minimum {config.min_components_for_sufficiency} components)"
        )
        if score is not None:
            reasons.append(f"Provisional reliability score was {score:.2f}")
        return FieldResult(
            field_name=field_name,
            ocr_value=field_data.ocr_value,
            ai_value=field_data.ai_value,
            normalized_value=norm_val,
            confidence_score=score,
            validation_status=validation_res.status,
            validation_reasons=validation_res.reasons,
            agreement_status=agr_status,
            decision=decision,
            decision_reasons=reasons,
            evidence_missing=confidence.components_missing,
            evidence_sufficient=confidence.evidence_sufficient,
            confidence_breakdown=confidence,
            is_human_corrected=getattr(field_data, "is_human_corrected", False),
        )

    # Check 5: No score available
    if score is None:
        decision = FieldDecision.HUMAN_REVIEW
        reasons.append("Reliability score could not be calculated due to missing evidence")
        return FieldResult(
            field_name=field_name,
            ocr_value=field_data.ocr_value,
            ai_value=field_data.ai_value,
            normalized_value=norm_val,
            confidence_score=None,
            validation_status=validation_res.status,
            validation_reasons=validation_res.reasons,
            agreement_status=agr_status,
            decision=decision,
            decision_reasons=reasons,
            evidence_missing=confidence.components_missing,
            evidence_sufficient=confidence.evidence_sufficient,
            confidence_breakdown=confidence,
            is_human_corrected=getattr(field_data, "is_human_corrected", False),
        )

    # Check 6: Score-based routing
    if score >= config.thresholds.auto_accept_threshold:
        # Eligible for auto-accept if validation passed without critical errors
        if validation_res.status in (ValidationStatus.VALID, ValidationStatus.WARNING):
            decision = FieldDecision.AUTO_ACCEPT
            reasons.append(
                f"High reliability score ({score:.2f} >= {config.thresholds.auto_accept_threshold:.2f}) "
                f"with valid deterministic checks"
            )
        else:
            decision = FieldDecision.HUMAN_REVIEW
            reasons.append(
                f"Score ({score:.2f}) meets auto-accept threshold, but validation "
                f"status is {validation_res.status.value}"
            )
    elif score >= config.thresholds.human_review_threshold:
        decision = FieldDecision.HUMAN_REVIEW
        reasons.append(
            f"Moderate reliability score ({score:.2f}) falls between review threshold "
            f"({config.thresholds.human_review_threshold:.2f}) and auto-accept threshold "
            f"({config.thresholds.auto_accept_threshold:.2f})"
        )
    else:
        decision = FieldDecision.HUMAN_REVIEW
        reasons.append(f"Low reliability score ({score:.2f} < {config.thresholds.human_review_threshold:.2f})")

    if agr_reasons:
        reasons.extend(agr_reasons)

    return FieldResult(
        field_name=field_name,
        ocr_value=field_data.ocr_value,
        ai_value=field_data.ai_value,
        normalized_value=norm_val,
        confidence_score=score,
        validation_status=validation_res.status,
        validation_reasons=validation_res.reasons,
        agreement_status=agr_status,
        decision=decision,
        decision_reasons=reasons,
        evidence_missing=confidence.components_missing,
        evidence_sufficient=confidence.evidence_sufficient,
        confidence_breakdown=confidence,
        is_human_corrected=getattr(field_data, "is_human_corrected", False),
    )


def route_document(
    normalized_data: NormalizedExtractionData,
    field_validations: dict[str, FieldValidationResult],
    normalized_values: dict[str, Any],
    agreement_results: dict[str, tuple[AgreementStatus, float | None, list[str]]],
    confidence_breakdowns: dict[str, ConfidenceBreakdown],
    cross_checks: list[CrossFieldCheckResult],
    critical_failures: list[str],
    config: RulesEngineConfig,
) -> DocumentResult:
    """Determine document-level decision and consolidate all field-level results."""
    type_config = config.get_document_config(normalized_data.document_type)

    retry_warranted, retry_reason = is_retry_warranted(
        normalized_data.quality_warnings,
        normalized_data.retry_count,
        config.thresholds.max_retries,
    )

    field_results: dict[str, FieldResult] = {}
    for f_name, f_data in normalized_data.fields.items():
        v_res = field_validations.get(
            f_name,
            FieldValidationResult(
                status=ValidationStatus.NOT_CHECKABLE,
                reasons=["Not validated"],
            ),
        )
        norm_val = normalized_values.get(f_name)
        agr_res = agreement_results.get(
            f_name,
            (AgreementStatus.NOT_COMPARABLE, None, []),
        )
        conf = confidence_breakdowns.get(
            f_name,
            ConfidenceBreakdown(
                final_score=None,
                raw_ocr_confidence=None,
                normalized_ocr_confidence=None,
                agreement_score=None,
                validation_score=None,
                components_used=[],
                components_missing=["ocr_confidence", "agreement", "validation"],
                weights_used={},
                evidence_sufficient=False,
                explanations=["Confidence breakdown missing"],
            ),
        )
        is_req = f_name in type_config.required_fields
        field_results[f_name] = route_field(
            field_name=f_name,
            field_data=f_data,
            norm_val=norm_val,
            validation_res=v_res,
            agreement_res=agr_res,
            confidence=conf,
            is_required=is_req,
            config=config,
            retry_warranted=retry_warranted,
            retry_reason=retry_reason,
        )

    # Document-level aggregation
    doc_reasons: list[str] = []
    is_auto_acceptable = True

    # 1. Retry extraction
    if retry_warranted:
        doc_decision = DocumentDecision.RETRY_EXTRACTION
        doc_reasons.append(retry_reason)
        return DocumentResult(
            document_type=normalized_data.document_type,
            document_decision=doc_decision,
            decision_reasons=doc_reasons,
            fields=field_results,
            cross_field_checks=cross_checks,
            critical_failures=critical_failures,
            is_auto_acceptable=False,
            metadata={
                "retry_count": normalized_data.retry_count,
                "quality_warnings": normalized_data.quality_warnings,
            },
        )

    # 2. Critical failures check
    if critical_failures:
        is_auto_acceptable = False
        doc_reasons.append(f"Document has {len(critical_failures)} critical validation failure(s) forcing human review")
        doc_reasons.extend(critical_failures)

    # 3. Cross-field check failures
    for c_check in cross_checks:
        if c_check.is_critical_failure or c_check.status == ValidationStatus.INVALID:
            is_auto_acceptable = False
            doc_reasons.extend(c_check.reasons)

    # 4. Check required fields for auto-accept eligibility
    missing_required: list[str] = []
    review_required: list[str] = []
    for req_f in type_config.required_fields:
        f_res = field_results.get(req_f)
        if not f_res or f_res.validation_status == ValidationStatus.MISSING:
            missing_required.append(req_f)
            is_auto_acceptable = False
        elif f_res.decision != FieldDecision.AUTO_ACCEPT:
            review_required.append(req_f)
            is_auto_acceptable = False

    if missing_required:
        doc_reasons.append(f"Required field(s) missing: {', '.join(missing_required)}")
    if review_required:
        doc_reasons.append(f"Required field(s) require human review: {', '.join(review_required)}")

    # 5. Final document decision
    if is_auto_acceptable:
        doc_decision = DocumentDecision.AUTO_ACCEPTED
        doc_reasons.append("All required fields and cross-field checks passed automatic acceptance criteria")
    else:
        doc_decision = DocumentDecision.HUMAN_REVIEW
        if not doc_reasons:
            doc_reasons.append("One or more fields require human review")

    return DocumentResult(
        document_type=normalized_data.document_type,
        document_decision=doc_decision,
        decision_reasons=doc_reasons,
        fields=field_results,
        cross_field_checks=cross_checks,
        critical_failures=critical_failures,
        is_auto_acceptable=is_auto_acceptable,
        metadata={
            "retry_count": normalized_data.retry_count,
            "quality_warnings": normalized_data.quality_warnings,
        },
    )
