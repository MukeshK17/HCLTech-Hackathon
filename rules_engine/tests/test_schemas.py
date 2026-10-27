# ruff: noqa: S101
"""Unit tests for schemas and input normalization in Member 3's Rules Engine."""

from __future__ import annotations

import json

from rules_engine.schemas import (
    normalize_extraction_inputs,
)


def test_normalize_extraction_inputs_dict_success() -> None:
    ocr_dict = {
        "company": "Tesco Stores",
        "date": "2024-05-10",
        "total": "125.50",
    }
    ai_dict = {
        "vendor_name": "Tesco Stores Ltd",
        "date": "2024-05-10",
        "total_amount": "125.50",
        "tax_amount": "12.55",
    }
    conf_dict = {
        "vendor_name": 0.95,
        "date": 92.0,  # 0-100 scale
        "total_amount": 0.88,
    }

    norm = normalize_extraction_inputs(
        ocr_json=ocr_dict,
        ai_json=ai_dict,
        ocr_confidence=conf_dict,
        document_type="receipt",
    )

    assert norm.document_type == "receipt"
    assert "vendor_name" in norm.fields
    assert norm.fields["vendor_name"].ocr_value == "Tesco Stores"
    assert norm.fields["vendor_name"].ai_value == "Tesco Stores Ltd"
    assert norm.fields["vendor_name"].normalized_ocr_confidence == 0.95

    # Date confidence was 92.0 -> normalized to 0.92
    assert norm.fields["date"].normalized_ocr_confidence == 0.92
    assert norm.fields["date"].raw_ocr_confidence == 92.0

    # Total amount alias resolved
    assert norm.fields["total_amount"].ocr_value == "125.50"
    assert norm.fields["total_amount"].ai_value == "125.50"


def test_normalize_json_strings() -> None:
    ocr_str = json.dumps({"merchant": "Speedway", "total": "45.00"})
    ai_str = json.dumps({"vendor_name": "Speedway", "total_amount": "45.00"})
    conf_str = json.dumps({"vendor_name": 0.99, "total_amount": 0.98})

    norm = normalize_extraction_inputs(
        ocr_json=ocr_str,
        ai_json=ai_str,
        ocr_confidence=conf_str,
    )

    assert norm.fields["vendor_name"].ocr_value == "Speedway"
    assert norm.fields["total_amount"].ocr_value == "45.00"
    assert norm.fields["vendor_name"].normalized_ocr_confidence == 0.99


def test_missing_ocr_output() -> None:
    ai_dict = {"vendor_name": "Cafe Corner", "total_amount": "15.00"}
    norm = normalize_extraction_inputs(ocr_json=None, ai_json=ai_dict)

    field_data = norm.fields["vendor_name"]
    assert field_data.ocr_value is None
    assert field_data.is_present_in_ocr is False
    assert field_data.is_present_in_ai is True
    assert field_data.ai_value == "Cafe Corner"


def test_missing_ai_output() -> None:
    ocr_dict = {"vendor_name": "Cafe Corner", "total_amount": "15.00"}
    norm = normalize_extraction_inputs(ocr_json=ocr_dict, ai_json=None)

    field_data = norm.fields["vendor_name"]
    assert field_data.ai_value is None
    assert field_data.is_present_in_ai is False
    assert field_data.is_present_in_ocr is True
    assert field_data.ocr_value == "Cafe Corner"


def test_missing_confidence_is_none_not_one() -> None:
    ai_dict = {"vendor_name": "Store A"}
    norm = normalize_extraction_inputs(ocr_json=None, ai_json=ai_dict, ocr_confidence=None)

    assert norm.fields["vendor_name"].normalized_ocr_confidence is None
    assert norm.fields["vendor_name"].raw_ocr_confidence is None
    assert "unavailable" in norm.fields["vendor_name"].confidence_notes.lower()


def test_genuine_zero_confidence_preserved() -> None:
    ai_dict = {"vendor_name": "Store A"}
    conf_dict = {"vendor_name": 0.0}
    norm = normalize_extraction_inputs(ocr_json=None, ai_json=ai_dict, ocr_confidence=conf_dict)

    assert norm.fields["vendor_name"].normalized_ocr_confidence == 0.0
    assert norm.fields["vendor_name"].raw_ocr_confidence == 0.0


def test_out_of_range_confidence() -> None:
    conf_dict = {"vendor_name": -0.5, "date": 150.0}
    norm = normalize_extraction_inputs(ocr_json=None, ai_json={}, ocr_confidence=conf_dict)

    assert norm.fields["vendor_name"].normalized_ocr_confidence is None
    assert norm.fields["date"].normalized_ocr_confidence is None


def test_segment_level_confidence_without_mapping() -> None:
    ocr_dict = {"vendor_name": "Store A"}
    segment_confs = [{"box": [0, 0, 10, 10], "conf": 0.85}]

    norm = normalize_extraction_inputs(ocr_json=ocr_dict, ai_json={}, ocr_confidence=segment_confs)

    assert norm.fields["vendor_name"].normalized_ocr_confidence is None
    assert "segment-level" in norm.fields["vendor_name"].confidence_notes.lower()
    assert any("segment-level" in w.lower() for w in norm.quality_warnings)


def test_additional_unmapped_fields_preserved() -> None:
    ocr_dict = {"custom_field": "12345", "vendor_name": "ABC"}
    ai_dict = {"custom_field": "12345", "vendor_name": "ABC"}

    norm = normalize_extraction_inputs(ocr_json=ocr_dict, ai_json=ai_dict)

    assert "custom_field" in norm.fields
    assert "custom_field" in norm.additional_fields
    assert norm.fields["custom_field"].ai_value == "12345"


def test_malformed_json_strings_handled_gracefully() -> None:
    norm = normalize_extraction_inputs(
        ocr_json="{malformed json",
        ai_json="null",
        ocr_confidence="",
    )

    assert len(norm.parsing_errors) >= 1
    assert norm.raw_ocr_input == "{malformed json"
