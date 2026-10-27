# ruff: noqa: S101
"""Unit tests for review router and decision logic in Member 3's Rules Engine."""

from __future__ import annotations

from rules_engine.config import RulesEngineConfig
from rules_engine.router import is_retry_warranted, route_document, route_field
from rules_engine.schemas import (
    AgreementStatus,
    CanonicalFieldData,
    ConfidenceBreakdown,
    CrossFieldCheckResult,
    DocumentDecision,
    FieldDecision,
    FieldValidationResult,
    NormalizedExtractionData,
    ValidationStatus,
)


def _make_breakdown(
    score: float | None,
    sufficient: bool = True,
    missing: list[str] | None = None,
) -> ConfidenceBreakdown:
    return ConfidenceBreakdown(
        final_score=score,
        raw_ocr_confidence=score,
        normalized_ocr_confidence=score,
        agreement_score=1.0 if score else None,
        validation_score=1.0 if score else None,
        components_used=["ocr_confidence", "agreement", "validation"] if sufficient else ["agreement"],
        components_missing=missing or [],
        weights_used={},
        evidence_sufficient=sufficient,
        explanations=[],
    )


def test_route_field_auto_accept() -> None:
    config = RulesEngineConfig()
    field_data = CanonicalFieldData(field_name="total_amount", ai_value="100.00")
    v_res = FieldValidationResult(status=ValidationStatus.VALID)
    conf = _make_breakdown(score=0.95, sufficient=True)

    res = route_field(
        field_name="total_amount",
        field_data=field_data,
        norm_val="100.00",
        validation_res=v_res,
        agreement_res=(AgreementStatus.MATCH, 1.0, []),
        confidence=conf,
        is_required=True,
        config=config,
        retry_warranted=False,
        retry_reason="",
    )

    assert res.decision == FieldDecision.AUTO_ACCEPT
    assert any("high reliability" in r.lower() for r in res.decision_reasons)


def test_route_field_critical_override_forces_human_review() -> None:
    config = RulesEngineConfig()
    field_data = CanonicalFieldData(field_name="total_amount", ai_value="100.00")
    # Even if score is 0.99, is_critical_failure is True
    v_res = FieldValidationResult(
        status=ValidationStatus.INVALID,
        reasons=["Total reconciliation mismatch"],
        is_critical_failure=True,
    )
    conf = _make_breakdown(score=0.99, sufficient=True)

    res = route_field(
        field_name="total_amount",
        field_data=field_data,
        norm_val="100.00",
        validation_res=v_res,
        agreement_res=(AgreementStatus.MATCH, 1.0, []),
        confidence=conf,
        is_required=True,
        config=config,
        retry_warranted=False,
        retry_reason="",
    )

    assert res.decision == FieldDecision.HUMAN_REVIEW
    assert any("critical validation failure" in r.lower() for r in res.decision_reasons)


def test_route_field_medium_score_human_review() -> None:
    config = RulesEngineConfig()
    field_data = CanonicalFieldData(field_name="vendor_name", ai_value="ABC Store")
    v_res = FieldValidationResult(status=ValidationStatus.VALID)
    # 0.78 is between 0.70 and 0.90
    conf = _make_breakdown(score=0.78, sufficient=True)

    res = route_field(
        field_name="vendor_name",
        field_data=field_data,
        norm_val="ABC Store",
        validation_res=v_res,
        agreement_res=(AgreementStatus.MATCH, 1.0, []),
        confidence=conf,
        is_required=True,
        config=config,
        retry_warranted=False,
        retry_reason="",
    )

    assert res.decision == FieldDecision.HUMAN_REVIEW


def test_route_field_insufficient_evidence_blocks_auto_accept() -> None:
    config = RulesEngineConfig()
    field_data = CanonicalFieldData(field_name="address", ai_value="123 Main St")
    v_res = FieldValidationResult(status=ValidationStatus.VALID)
    # High score (0.95), but evidence is NOT sufficient
    conf = _make_breakdown(score=0.95, sufficient=False, missing=["ocr_confidence", "validation"])

    res = route_field(
        field_name="address",
        field_data=field_data,
        norm_val="123 Main St",
        validation_res=v_res,
        agreement_res=(AgreementStatus.MATCH, 1.0, []),
        confidence=conf,
        is_required=False,
        config=config,
        retry_warranted=False,
        retry_reason="",
    )

    assert res.decision == FieldDecision.HUMAN_REVIEW
    assert any("insufficient" in r.lower() for r in res.decision_reasons)


def test_route_field_retry_extraction() -> None:
    config = RulesEngineConfig()
    field_data = CanonicalFieldData(field_name="vendor_name", ai_value=None)
    v_res = FieldValidationResult(status=ValidationStatus.MISSING)
    conf = _make_breakdown(score=None, sufficient=False)

    res = route_field(
        field_name="vendor_name",
        field_data=field_data,
        norm_val=None,
        validation_res=v_res,
        agreement_res=(AgreementStatus.BOTH_MISSING, None, []),
        confidence=conf,
        is_required=True,
        config=config,
        retry_warranted=True,
        retry_reason="Empty OCR output warrants retry",
    )

    assert res.decision == FieldDecision.RETRY_EXTRACTION
    assert any("retry" in r.lower() for r in res.decision_reasons)


def test_is_retry_warranted_keywords_and_limits() -> None:
    # Triggered by severe blur keyword
    warranted, reason = is_retry_warranted(["Severe blur detected on scan"], retry_count=0, max_retries=2)
    assert warranted is True
    assert "severe blur" in reason.lower()

    # Max retries reached -> does NOT retry
    warranted_limit, reason_limit = is_retry_warranted(["Severe blur detected on scan"], retry_count=2, max_retries=2)
    assert warranted_limit is False
    assert "retry limit" in reason_limit.lower()


def test_route_document_auto_accepted() -> None:
    config = RulesEngineConfig()
    fields = {
        "vendor_name": CanonicalFieldData(field_name="vendor_name", ai_value="Shoprite"),
        "date": CanonicalFieldData(field_name="date", ai_value="2024-05-01"),
        "total_amount": CanonicalFieldData(field_name="total_amount", ai_value="50.00"),
    }
    norm_doc = NormalizedExtractionData(fields=fields, document_type="receipt")

    field_validations = {
        "vendor_name": FieldValidationResult(status=ValidationStatus.VALID),
        "date": FieldValidationResult(status=ValidationStatus.VALID),
        "total_amount": FieldValidationResult(status=ValidationStatus.VALID),
    }
    norm_values = {"vendor_name": "Shoprite", "date": "2024-05-01", "total_amount": "50.00"}
    agreements = {k: (AgreementStatus.MATCH, 1.0, []) for k in fields}
    confidences = {k: _make_breakdown(score=0.95, sufficient=True) for k in fields}
    cross_checks = [CrossFieldCheckResult(check_name="total_reconciliation", status=ValidationStatus.VALID)]

    doc_res = route_document(
        normalized_data=norm_doc,
        field_validations=field_validations,
        normalized_values=norm_values,
        agreement_results=agreements,
        confidence_breakdowns=confidences,
        cross_checks=cross_checks,
        critical_failures=[],
        config=config,
    )

    assert doc_res.document_decision == DocumentDecision.AUTO_ACCEPTED
    assert doc_res.is_auto_acceptable is True


def test_route_document_human_review_due_to_critical_failure() -> None:
    config = RulesEngineConfig()
    fields = {
        "vendor_name": CanonicalFieldData(field_name="vendor_name", ai_value="Shoprite"),
        "date": CanonicalFieldData(field_name="date", ai_value="2024-05-01"),
        "total_amount": CanonicalFieldData(field_name="total_amount", ai_value="50.00"),
    }
    norm_doc = NormalizedExtractionData(fields=fields, document_type="receipt")

    field_validations = {k: FieldValidationResult(status=ValidationStatus.VALID) for k in fields}
    norm_values = {"vendor_name": "Shoprite", "date": "2024-05-01", "total_amount": "50.00"}
    agreements = {k: (AgreementStatus.MATCH, 1.0, []) for k in fields}
    confidences = {k: _make_breakdown(score=0.95, sufficient=True) for k in fields}
    cross_checks = [
        CrossFieldCheckResult(
            check_name="total_reconciliation",
            status=ValidationStatus.INVALID,
            reasons=["Total reconciliation mismatch"],
            is_critical_failure=True,
        )
    ]

    doc_res = route_document(
        normalized_data=norm_doc,
        field_validations=field_validations,
        normalized_values=norm_values,
        agreement_results=agreements,
        confidence_breakdowns=confidences,
        cross_checks=cross_checks,
        critical_failures=["Total reconciliation mismatch"],
        config=config,
    )

    assert doc_res.document_decision == DocumentDecision.HUMAN_REVIEW
    assert doc_res.is_auto_acceptable is False
    assert any("critical" in r.lower() for r in doc_res.decision_reasons)
