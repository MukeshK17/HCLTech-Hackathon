# ruff: noqa: S101
"""Unit tests for offline evaluation utility in Member 3's Rules Engine."""

from __future__ import annotations

import pytest

from rules_engine.evaluation import evaluate_predictions
from rules_engine.schemas import (
    ConfidenceBreakdown,
    FieldDecision,
    FieldResult,
    ValidationStatus,
)


def _make_field_result(name: str, val: str, decision: FieldDecision, conf: float) -> FieldResult:
    return FieldResult(
        field_name=name,
        ocr_value=val,
        ai_value=val,
        normalized_value=val,
        confidence_score=conf,
        validation_status=ValidationStatus.VALID,
        validation_reasons=[],
        agreement_status=None,  # type: ignore[arg-type]
        decision=decision,
        decision_reasons=[],
        evidence_missing=[],
        evidence_sufficient=True,
        confidence_breakdown=ConfidenceBreakdown(
            final_score=conf,
            raw_ocr_confidence=conf,
            normalized_ocr_confidence=conf,
            agreement_score=1.0,
            validation_score=1.0,
            components_used=[],
            components_missing=[],
            weights_used={},
            evidence_sufficient=True,
            explanations=[],
        ),
    )


def test_evaluate_predictions_synthetic_metrics() -> None:
    # 2 documents, 2 fields each: total_amount and vendor_name (4 fields total)
    # Doc 1:
    #   total_amount: pred="100.00", ref="100.00" -> MATCH, auto_accept
    #   vendor_name: pred="Acme Corp", ref="Acme Corp" -> MATCH, auto_accept
    # Doc 2:
    #   total_amount: pred="50.00", ref="60.00" -> MISMATCH, human_review
    #   vendor_name: pred="Best Foods", ref="Best Foods Ltd" -> MATCH (partial/normalized), human_review

    doc1_fields = {
        "total_amount": _make_field_result("total_amount", "100.00", FieldDecision.AUTO_ACCEPT, 0.95),
        "vendor_name": _make_field_result("vendor_name", "Acme Corp", FieldDecision.AUTO_ACCEPT, 0.92),
    }
    doc2_fields = {
        "total_amount": _make_field_result("total_amount", "50.00", FieldDecision.HUMAN_REVIEW, 0.65),
        "vendor_name": _make_field_result("vendor_name", "Best Foods", FieldDecision.HUMAN_REVIEW, 0.75),
    }

    preds = [
        {"fields": doc1_fields},
        {"fields": doc2_fields},
    ]

    ground_truth = [
        {"total_amount": "100.00", "vendor_name": "Acme Corp"},
        {"total_amount": "60.00", "vendor_name": "Best Foods Ltd"},
    ]

    report = evaluate_predictions(preds, ground_truth)

    assert report.total_documents == 2
    assert report.total_fields_evaluated == 4

    # Correct: Doc 1 total, Doc 1 vendor, Doc 2 vendor (partial match). Total correct = 3 / 4 = 75%
    assert pytest.approx(report.overall_accuracy, abs=1e-4) == 0.75

    # Auto-accepted: 2 fields (both in Doc 1), both correct -> error rate = 0.0
    assert report.auto_accepted_fields_error_rate == 0.0
    assert report.overall_auto_acceptance_coverage == 0.50  # 2 / 4
    assert report.overall_human_review_rate == 0.50  # 2 / 4

    # Calibration warning for small N
    assert not report.calibration.is_statistically_sufficient
    assert "too small" in report.calibration.notes.lower()


def test_evaluate_predictions_empty_dataset() -> None:
    report = evaluate_predictions([], [])
    assert report.total_documents == 0
    assert report.total_fields_evaluated == 0
    assert report.overall_accuracy == 0.0
