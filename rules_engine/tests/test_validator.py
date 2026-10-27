# ruff: noqa: S101
"""Unit tests for deterministic validation rules in Member 3's Rules Engine."""

from __future__ import annotations

from decimal import Decimal

from rules_engine.config import MonetaryConfig, RulesEngineConfig
from rules_engine.schemas import (
    AgreementStatus,
    CanonicalFieldData,
    NormalizedExtractionData,
    ValidationStatus,
)
from rules_engine.validator import (
    compare_ocr_and_ai,
    parse_and_validate_date,
    parse_monetary_amount,
    reconcile_total,
    validate_currency,
    validate_document,
    validate_tax_calculation,
    validate_text_field,
)


def test_parse_and_validate_date_iso() -> None:
    status, norm_val, reasons, is_crit = parse_and_validate_date("2024-05-15")
    assert status == ValidationStatus.VALID
    assert norm_val == "2024-05-15"
    assert not is_crit
    assert len(reasons) == 0


def test_parse_and_validate_date_slash_unambiguous() -> None:
    # Day is 25, month is 12 (must be DMY)
    status, norm_val, _, is_crit = parse_and_validate_date("25/12/2023")
    assert status == ValidationStatus.VALID
    assert norm_val == "2023-12-25"
    assert not is_crit


def test_parse_and_validate_date_ambiguous_no_preference() -> None:
    # 03/04/2024 is ambiguous (March 4 vs April 3)
    status, norm_val, reasons, is_crit = parse_and_validate_date("03/04/2024", preferred_order=None)
    assert status == ValidationStatus.WARNING
    assert norm_val is None
    assert is_crit is True  # Ambiguous date forces review
    assert any("ambiguous" in r.lower() for r in reasons)


def test_parse_and_validate_date_ambiguous_with_preference() -> None:
    # DMY -> 3 April 2024
    status_dmy, norm_dmy, _, _ = parse_and_validate_date("03/04/2024", preferred_order="DMY")
    assert status_dmy == ValidationStatus.VALID
    assert norm_dmy == "2024-04-03"

    # MDY -> 4 March 2024
    status_mdy, norm_mdy, _, _ = parse_and_validate_date("03/04/2024", preferred_order="MDY")
    assert status_mdy == ValidationStatus.VALID
    assert norm_mdy == "2024-03-04"


def test_parse_and_validate_date_invalid_calendar() -> None:
    # February 30 does not exist
    status, norm_val, reasons, is_crit = parse_and_validate_date("2024-02-30")
    assert status == ValidationStatus.INVALID
    assert norm_val is None
    assert is_crit is True
    assert any("calendar" in r.lower() for r in reasons)


def test_parse_monetary_amount_standard() -> None:
    status, dec_val, reasons, is_crit = parse_monetary_amount("1,180.50")
    assert status == ValidationStatus.VALID
    assert dec_val == Decimal("1180.50")
    assert not is_crit


def test_parse_monetary_amount_european_comma() -> None:
    status, dec_val, _, _ = parse_monetary_amount("1.180,50")
    assert status == ValidationStatus.VALID
    assert dec_val == Decimal("1180.50")


def test_parse_monetary_amount_with_currency_symbol() -> None:
    status, dec_val, _, _ = parse_monetary_amount("$ 45.99")
    assert status == ValidationStatus.VALID
    assert dec_val == Decimal("45.99")


def test_parse_monetary_amount_malformed() -> None:
    status, dec_val, reasons, is_crit = parse_monetary_amount("12.3.4")
    assert status == ValidationStatus.INVALID
    assert dec_val is None
    assert is_crit is True
    assert any("multiple decimal points" in r.lower() for r in reasons)


def test_parse_monetary_amount_negative() -> None:
    # Disallowed by default
    status, dec_val, reasons, is_crit = parse_monetary_amount("-15.00", allow_negative=False)
    assert status == ValidationStatus.INVALID
    assert is_crit is True

    # Allowed when explicitly configured
    status_ok, dec_ok, _, is_crit_ok = parse_monetary_amount("-15.00", allow_negative=True)
    assert status_ok == ValidationStatus.VALID
    assert dec_ok == Decimal("-15.00")
    assert not is_crit_ok


def test_validate_currency_supported() -> None:
    config = MonetaryConfig()
    status, code, _, is_crit = validate_currency("USD", config)
    assert status == ValidationStatus.VALID
    assert code == "USD"
    assert not is_crit


def test_validate_currency_ambiguous_symbol() -> None:
    config = MonetaryConfig()
    status, code, reasons, is_crit = validate_currency("$", config)
    assert status == ValidationStatus.WARNING
    assert code is None
    assert is_crit is True  # Ambiguous currency forces review
    assert any("ambiguous" in r.lower() for r in reasons)


def test_validate_currency_unambiguous_symbol() -> None:
    config = MonetaryConfig()
    status, code, _, is_crit = validate_currency("RM", config)
    assert status == ValidationStatus.VALID
    assert code == "MYR"
    assert not is_crit


def test_validate_vendor_name_whitespace() -> None:
    status, norm, _, _ = validate_text_field("   Supermarket   XYZ  \n", "vendor_name", is_required=True)
    assert status == ValidationStatus.VALID
    assert norm == "Supermarket XYZ"


def test_validate_vendor_name_missing_required() -> None:
    status, norm, reasons, is_crit = validate_text_field(None, "vendor_name", is_required=True)
    assert status == ValidationStatus.MISSING
    assert is_crit is True
    assert any("missing" in r.lower() for r in reasons)


def test_reconcile_total_valid() -> None:
    res = reconcile_total(
        subtotal=Decimal("100.00"),
        tax=Decimal("10.00"),
        discount=Decimal("5.00"),
        rounding=Decimal("0.02"),
        extracted_total=Decimal("105.02"),
        tolerance=Decimal("0.05"),
    )
    assert res.status == ValidationStatus.VALID
    assert not res.is_critical_failure
    assert res.difference == Decimal("0.00")


def test_reconcile_total_mismatch_critical() -> None:
    res = reconcile_total(
        subtotal=Decimal("100.00"),
        tax=Decimal("10.00"),
        discount=Decimal("0.00"),
        rounding=Decimal("0.00"),
        extracted_total=Decimal("150.00"),  # Mismatch by 40.00
        tolerance=Decimal("0.05"),
    )
    assert res.status == ValidationStatus.INVALID
    assert res.is_critical_failure is True
    assert res.difference == Decimal("40.00")


def test_reconcile_total_missing_inputs() -> None:
    res = reconcile_total(
        subtotal=None,
        tax=Decimal("10.00"),
        discount=Decimal("0.00"),
        rounding=Decimal("0.00"),
        extracted_total=Decimal("110.00"),
    )
    assert res.status == ValidationStatus.NOT_CHECKABLE
    assert not res.is_critical_failure


def test_validate_tax_calculation_valid() -> None:
    res = validate_tax_calculation(
        subtotal=Decimal("200.00"),
        tax_amount=Decimal("16.00"),
        tax_rate=Decimal("8.00"),  # 8% of 200 is 16.00
        tolerance=Decimal("0.05"),
    )
    assert res.status == ValidationStatus.VALID


def test_validate_tax_calculation_mismatch() -> None:
    res = validate_tax_calculation(
        subtotal=Decimal("200.00"),
        tax_amount=Decimal("25.00"),  # Expected 16.00
        tax_rate=Decimal("8.00"),
        tolerance=Decimal("0.05"),
    )
    assert res.status == ValidationStatus.WARNING
    assert res.difference == Decimal("9.00")


def test_compare_ocr_and_ai_amounts() -> None:
    status_match, score, _ = compare_ocr_and_ai("total_amount", "100.00", "100.00")
    assert status_match == AgreementStatus.MATCH
    assert score == 1.0

    status_diff, score_diff, _ = compare_ocr_and_ai("total_amount", "118.00", "1180.00")
    assert status_diff == AgreementStatus.MISMATCH
    assert score_diff == 0.0


def test_compare_ocr_and_ai_missing_branches() -> None:
    status_ocr_missing, score_none, _ = compare_ocr_and_ai("vendor_name", None, "ABC Corp")
    assert status_ocr_missing == AgreementStatus.OCR_MISSING
    assert score_none is None

    status_both, score_both, _ = compare_ocr_and_ai("vendor_name", None, None)
    assert status_both == AgreementStatus.BOTH_MISSING
    assert score_both is None


def test_validate_document_end_to_end() -> None:
    fields = {
        "vendor_name": CanonicalFieldData(field_name="vendor_name", ai_value="Fresh Mart"),
        "date": CanonicalFieldData(field_name="date", ai_value="2024-04-10"),
        "subtotal": CanonicalFieldData(field_name="subtotal", ai_value="50.00"),
        "tax_amount": CanonicalFieldData(field_name="tax_amount", ai_value="5.00"),
        "total_amount": CanonicalFieldData(field_name="total_amount", ai_value="55.00"),
        "currency": CanonicalFieldData(field_name="currency", ai_value="USD"),
    }
    norm_doc = NormalizedExtractionData(fields=fields, document_type="receipt")
    config = RulesEngineConfig()

    field_vals, norm_vals, cross_checks, crit_failures = validate_document(norm_doc, config)

    assert len(crit_failures) == 0
    assert norm_vals["total_amount"] == Decimal("55.00")
    assert field_vals["vendor_name"].status == ValidationStatus.VALID
    assert any(c.check_name == "total_reconciliation" and c.status == ValidationStatus.VALID for c in cross_checks)
