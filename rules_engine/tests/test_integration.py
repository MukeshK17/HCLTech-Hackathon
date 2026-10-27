# ruff: noqa: S101
"""End-to-end integration tests for Member 3's Rules Engine pipeline."""

from __future__ import annotations

import json
from decimal import Decimal

from rules_engine import (
    DocumentDecision,
    FieldDecision,
    ValidationStatus,
    process_extraction,
    revalidate_corrected_fields,
)


def test_full_pipeline_happy_path_auto_accept() -> None:
    ocr_json = {
        "company": "Sunrise Grocery",
        "date": "2024-06-15",
        "sub_total": "100.00",
        "tax": "10.00",
        "total": "110.00",
        "currency": "USD",
    }
    ai_json = {
        "vendor_name": "Sunrise Grocery",
        "date": "2024-06-15",
        "subtotal": "100.00",
        "tax_amount": "10.00",
        "total_amount": "110.00",
        "currency": "USD",
    }
    ocr_confidence = {
        "vendor_name": 0.98,
        "date": 0.95,
        "subtotal": 0.94,
        "tax_amount": 0.93,
        "total_amount": 0.97,
        "currency": 0.99,
    }

    doc_result = process_extraction(
        ocr_json=ocr_json,
        ai_json=ai_json,
        ocr_confidence=ocr_confidence,
        document_type="receipt",
    )

    # Document should be auto-accepted
    assert doc_result.document_decision == DocumentDecision.AUTO_ACCEPTED
    assert doc_result.is_auto_acceptable is True
    assert len(doc_result.critical_failures) == 0

    # Every required field should be auto-accepted
    assert doc_result.fields["vendor_name"].decision == FieldDecision.AUTO_ACCEPT
    assert doc_result.fields["date"].decision == FieldDecision.AUTO_ACCEPT
    assert doc_result.fields["total_amount"].decision == FieldDecision.AUTO_ACCEPT

    # JSON serialization check for Member 4
    json_dict = doc_result.to_dict()
    assert isinstance(json_dict, dict)
    serialized = json.dumps(json_dict)
    assert len(serialized) > 0


def test_full_pipeline_mismatch_forces_human_review_then_revalidated() -> None:
    ocr_json = {
        "company": "Mega Supermarket",
        "date": "2024-07-20",
        "subtotal": "200.00",
        "tax": "20.00",
        "total": "220.00",  # OCR saw 220
    }
    ai_json = {
        "vendor_name": "Mega Supermarket",
        "date": "2024-07-20",
        "subtotal": "200.00",
        "tax_amount": "20.00",
        "total_amount": "250.00",  # AI hallucinated 250! Total does not reconcile (200 + 20 = 220 != 250)
    }
    ocr_confidence = {
        "vendor_name": 0.95,
        "date": 0.95,
        "subtotal": 0.95,
        "tax_amount": 0.95,
        "total_amount": 0.95,
    }

    initial_result = process_extraction(
        ocr_json=ocr_json,
        ai_json=ai_json,
        ocr_confidence=ocr_confidence,
        document_type="receipt",
    )

    # Mismatch forces human review
    assert initial_result.document_decision == DocumentDecision.HUMAN_REVIEW
    assert initial_result.is_auto_acceptable is False
    assert len(initial_result.critical_failures) >= 1
    assert any("reconciliation" in f.lower() for f in initial_result.critical_failures)

    # Member 4 review interface: human reviews and corrects total_amount to 220.00
    corrected_fields = {"total_amount": "220.00"}
    revalidated = revalidate_corrected_fields(initial_result, corrected_fields)

    # After correction: total reconciles!
    assert revalidated.document_decision == DocumentDecision.AUTO_ACCEPTED
    assert revalidated.is_auto_acceptable is True
    assert revalidated.fields["total_amount"].is_human_corrected is True
    assert revalidated.fields["total_amount"].normalized_value == Decimal("220.00")
    assert revalidated.to_dict()["fields"]["total_amount"]["normalized_value"] == "220.00"
    # Original OCR and AI values are preserved for provenance
    assert revalidated.fields["total_amount"].ocr_value == "220.00"
    assert revalidated.fields["total_amount"].ai_value == "250.00"


def test_human_correction_with_invalid_value_fails_validation() -> None:
    ocr_json = {"company": "Test Store", "total": "50.00"}
    ai_json = {"vendor_name": "Test Store", "total_amount": "50.00"}
    initial_result = process_extraction(ocr_json=ocr_json, ai_json=ai_json, document_type="receipt")

    # Human accidentally enters an invalid malformed amount
    corrected_fields = {"total_amount": "not_an_amount"}
    revalidated = revalidate_corrected_fields(initial_result, corrected_fields)

    assert revalidated.fields["total_amount"].validation_status == ValidationStatus.INVALID
    assert revalidated.document_decision == DocumentDecision.HUMAN_REVIEW
    assert revalidated.is_auto_acceptable is False
