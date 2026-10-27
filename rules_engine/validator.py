"""Deterministic validation rules for Member 3's Rules Engine.

Implements pure deterministic validation for:
- Dates (parsing, calendar validation, ambiguity detection, normalization to YYYY-MM-DD).
- Monetary amounts (Decimal arithmetic, locale-aware parsing, precision).
- Currencies (ISO codes, ambiguous symbols like '$', conflicts).
- Text fields (vendor name, address whitespace normalization).
- Cross-field arithmetic (tax calculation, total reconciliation).
- OCR-AI consistency checks.
"""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta
from decimal import Decimal, InvalidOperation
from typing import Any

from rules_engine.config import MonetaryConfig, RulesEngineConfig
from rules_engine.schemas import (
    AgreementStatus,
    CanonicalFieldData,
    CrossFieldCheckResult,
    FieldValidationResult,
    NormalizedExtractionData,
    ValidationStatus,
)

# Regex patterns for dates
ISO_DATE_PATTERN = re.compile(r"^(\d{4})[-/](\d{1,2})[-/](\d{1,2})$")
SLASH_DATE_PATTERN = re.compile(r"^(\d{1,2})[-/.](\d{1,2})[-/.](\d{2,4})$")
MONTH_NAME_PATTERN = re.compile(
    r"^(\d{1,2})[-/\s]+([a-zA-Z]{3,9})[-/,\s]+(\d{2,4})$|"
    r"^([a-zA-Z]{3,9})[-/\s]+(\d{1,2})[-/,\s]+(\d{2,4})$"
)

MONTH_MAP = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}


def parse_and_validate_date(
    raw_val: Any,
    preferred_order: str | None = None,
    max_future_days: int = 1,
) -> tuple[ValidationStatus, str | None, list[str], bool]:
    """Parse, validate, and normalize a date value.

    Returns:
        (ValidationStatus, normalized_date_YYYY-MM-DD, reasons_list, is_critical_failure)
    """
    if raw_val is None:
        return ValidationStatus.MISSING, None, ["Date is missing"], False

    if isinstance(raw_val, (date, datetime)):
        norm = raw_val.strftime("%Y-%m-%d")
        return ValidationStatus.VALID, norm, [], False

    if not isinstance(raw_val, str):
        return (
            ValidationStatus.INVALID,
            None,
            [f"Invalid date type: {type(raw_val).__name__}"],
            True,
        )

    val_str = raw_val.strip()
    if not val_str:
        return ValidationStatus.MISSING, None, ["Date string is empty"], False

    # 1. Try ISO YYYY-MM-DD or YYYY/MM/DD
    iso_match = ISO_DATE_PATTERN.match(val_str)
    if iso_match:
        y, m, d = (int(iso_match.group(1)), int(iso_match.group(2)), int(iso_match.group(3)))
        try:
            parsed_dt = date(y, m, d)
            norm = parsed_dt.strftime("%Y-%m-%d")
            return _check_date_recency(parsed_dt, norm, max_future_days)
        except ValueError as exc:
            return ValidationStatus.INVALID, None, [f"Invalid calendar date: {exc}"], True

    # 2. Try month name patterns (e.g. 12-Jan-2024 or Jan 12, 2024)
    month_match = MONTH_NAME_PATTERN.match(val_str)
    if month_match:
        g = month_match.groups()
        if g[0] and g[1] and g[2]:
            day_str, m_str, y_str = g[0], g[1], g[2]
        else:
            m_str, day_str, y_str = g[3], g[4], g[5]

        m_key = m_str.lower()
        if m_key in MONTH_MAP:
            m_int = MONTH_MAP[m_key]
            d_int = int(day_str)
            y_int = int(y_str)
            if y_int < 100:
                y_int = 2000 + y_int if y_int <= 50 else 1900 + y_int
            try:
                parsed_dt = date(y_int, m_int, d_int)
                norm = parsed_dt.strftime("%Y-%m-%d")
                return _check_date_recency(parsed_dt, norm, max_future_days)
            except ValueError as exc:
                return ValidationStatus.INVALID, None, [f"Invalid calendar date: {exc}"], True

    # 3. Try slash/dash/dot patterns: XX/YY/ZZZZ
    slash_match = SLASH_DATE_PATTERN.match(val_str)
    if slash_match:
        p1 = int(slash_match.group(1))
        p2 = int(slash_match.group(2))
        y_int = int(slash_match.group(3))
        if y_int < 100:
            y_int = 2000 + y_int if y_int <= 50 else 1900 + y_int

        # Check ambiguity: both p1 and p2 <= 12 and p1 != p2
        if 1 <= p1 <= 12 and 1 <= p2 <= 12 and p1 != p2:
            if not preferred_order:
                reason = (
                    f"Date '{val_str}' is ambiguous between day-first and month-first "
                    f"({p1:02d}/{p2:02d} vs {p2:02d}/{p1:02d})"
                )
                return ValidationStatus.WARNING, None, [reason], True  # Critical override for review

            order = preferred_order.strip().upper()
            if order == "DMY":
                d_int, m_int = p1, p2
            elif order == "MDY":
                m_int, d_int = p1, p2
            else:
                return (
                    ValidationStatus.INVALID,
                    None,
                    [f"Unsupported preferred_date_order: {preferred_order}"],
                    True,
                )
        else:
            # Unambiguous by range
            if p1 > 12 >= p2 >= 1:
                # p1 must be day, p2 is month (DMY)
                d_int, m_int = p1, p2
            elif p2 > 12 >= p1 >= 1:
                # p2 must be day, p1 is month (MDY)
                m_int, d_int = p1, p2
            elif p1 == p2 and 1 <= p1 <= 12:
                # Day and month are identical
                d_int, m_int = p1, p2
            else:
                return ValidationStatus.INVALID, None, [f"Invalid calendar values in date '{val_str}'"], True

        try:
            parsed_dt = date(y_int, m_int, d_int)
            norm = parsed_dt.strftime("%Y-%m-%d")
            return _check_date_recency(parsed_dt, norm, max_future_days)
        except ValueError as exc:
            return ValidationStatus.INVALID, None, [f"Invalid calendar date: {exc}"], True

    return ValidationStatus.INVALID, None, [f"Date '{val_str}' does not match supported date formats"], True


def _check_date_recency(
    parsed_dt: date, norm: str, max_future_days: int
) -> tuple[ValidationStatus, str | None, list[str], bool]:
    """Check if date is reasonably in the past / near present."""
    today = date.today()
    if parsed_dt > today + timedelta(days=max_future_days):
        return (
            ValidationStatus.WARNING,
            norm,
            [f"Date {norm} is in the future beyond {max_future_days} day(s)"],
            False,
        )
    # Check if date is older than 50 years (likely OCR misread of year)
    if parsed_dt < date(today.year - 50, 1, 1):
        return (
            ValidationStatus.WARNING,
            norm,
            [f"Date {norm} is unusually far in the past (>50 years)"],
            False,
        )
    return ValidationStatus.VALID, norm, [], False


def parse_monetary_amount(
    raw_val: Any,
    allow_negative: bool = False,
) -> tuple[ValidationStatus, Decimal | None, list[str], bool]:
    """Parse and validate a monetary amount using Decimal arithmetic.

    Rejects malformed amounts (e.g. '12.3.4', 'abc').
    Returns (status, Decimal_value, reasons, is_critical_failure).
    """
    if raw_val is None:
        return ValidationStatus.MISSING, None, ["Amount is missing"], False

    if isinstance(raw_val, (int, float)):
        # Convert via string representation to avoid float representation artifacts
        val_str = str(raw_val)
    elif isinstance(raw_val, Decimal):
        val_str = str(raw_val)
    elif isinstance(raw_val, str):
        val_str = raw_val.strip()
    else:
        return (
            ValidationStatus.INVALID,
            None,
            [f"Invalid amount type: {type(raw_val).__name__}"],
            True,
        )

    if not val_str:
        return ValidationStatus.MISSING, None, ["Amount string is empty"], False

    # Clean currency symbols and spaces: e.g. "$ 1,180.00" -> "1,180.00"
    cleaned = re.sub(r"^[^\d\-+]+", "", val_str).strip()
    cleaned = re.sub(r"[^\d.]+$", "", cleaned).strip()

    # Reject if multiple decimal points or commas mixed abnormally
    # Allow commas as thousands separators: "1,180.00"
    if cleaned.count(".") > 1:
        return (
            ValidationStatus.INVALID,
            None,
            [f"Malformed amount '{raw_val}' contains multiple decimal points"],
            True,
        )

    # Check for European format e.g. "1.180,00" vs standard "1,180.00"
    if "," in cleaned and "." in cleaned:
        if cleaned.rfind(",") > cleaned.rfind("."):
            # European: 1.180,00 -> 1180.00
            cleaned = cleaned.replace(".", "").replace(",", ".")
        else:
            # Standard: 1,180.00 -> 1180.00
            cleaned = cleaned.replace(",", "")
    elif "," in cleaned:
        parts = cleaned.split(",")
        cleaned = cleaned.replace(",", ".") if len(parts) == 2 and len(parts[1]) in (1, 2) else cleaned.replace(",", "")

    # Remove internal spaces e.g. "1 180.00"
    cleaned = cleaned.replace(" ", "")

    try:
        dec_val = Decimal(cleaned)
    except (InvalidOperation, ValueError):
        return (
            ValidationStatus.INVALID,
            None,
            [f"Cannot parse '{raw_val}' as valid monetary Decimal"],
            True,
        )

    if dec_val < Decimal("0.00") and not allow_negative:
        return (
            ValidationStatus.INVALID,
            dec_val,
            [f"Negative amount {dec_val} is not permitted for standard document"],
            True,
        )

    # Standardize to 2 decimal places if practical
    return ValidationStatus.VALID, dec_val, [], False


def validate_currency(
    raw_val: Any,
    monetary_config: MonetaryConfig,
) -> tuple[ValidationStatus, str | None, list[str], bool]:
    """Validate currency against supported list and handle ambiguous symbols."""
    if raw_val is None:
        return (
            ValidationStatus.NOT_CHECKABLE,
            None,
            ["Currency not specified on document"],
            False,
        )

    if not isinstance(raw_val, str):
        return (
            ValidationStatus.INVALID,
            None,
            [f"Invalid currency type: {type(raw_val).__name__}"],
            True,
        )

    val_str = raw_val.strip().upper()
    if not val_str:
        return (
            ValidationStatus.NOT_CHECKABLE,
            None,
            ["Currency string is empty"],
            False,
        )

    # Check exact match with supported ISO codes
    if val_str in monetary_config.supported_currencies:
        return ValidationStatus.VALID, val_str, [], False

    # Check symbol mapping
    orig_sym = raw_val.strip()
    if orig_sym in monetary_config.symbol_to_currency:
        resolved = monetary_config.symbol_to_currency[orig_sym]
        if resolved is None:
            return (
                ValidationStatus.WARNING,
                None,
                [f"Currency symbol '{orig_sym}' is ambiguous across multiple currencies (e.g. USD, SGD, AUD, CAD)"],
                True,  # Critical failure for routing
            )
        return ValidationStatus.VALID, resolved, [], False

    return (
        ValidationStatus.WARNING,
        val_str,
        [f"Currency '{val_str}' is not in configured supported currencies"],
        False,
    )


def validate_text_field(
    raw_val: Any,
    field_name: str,
    is_required: bool = False,
    min_length: int = 2,
) -> tuple[ValidationStatus, str | None, list[str], bool]:
    """Validate text fields (vendor_name, address)."""
    if raw_val is None:
        if is_required:
            return ValidationStatus.MISSING, None, [f"Required field '{field_name}' is missing"], True
        return ValidationStatus.NOT_CHECKABLE, None, [f"Optional field '{field_name}' is not present"], False

    val_str = str(raw_val) if not isinstance(raw_val, str) else raw_val

    # Normalize repeated whitespace
    norm = re.sub(r"\s+", " ", val_str).strip()
    if not norm:
        if is_required:
            return ValidationStatus.MISSING, None, [f"Required field '{field_name}' is empty"], True
        return ValidationStatus.NOT_CHECKABLE, None, [f"Field '{field_name}' is empty"], False

    if len(norm) < min_length:
        return (
            ValidationStatus.WARNING,
            norm,
            [f"Field '{field_name}' is unusually short ('{norm}')"],
            False,
        )

    return ValidationStatus.VALID, norm, [], False


def compare_ocr_and_ai(
    field_name: str,
    ocr_val: Any,
    ai_val: Any,
    monetary_config: MonetaryConfig | None = None,
) -> tuple[AgreementStatus, float | None, list[str]]:
    """Compare OCR and AI extracted values.

    Returns:
        (AgreementStatus, agreement_score_0_to_1_or_None, reasons)
    """
    if ocr_val is None and ai_val is None:
        return AgreementStatus.BOTH_MISSING, None, [f"Both OCR and AI values are missing for '{field_name}'"]
    if ocr_val is None:
        return AgreementStatus.OCR_MISSING, None, [f"OCR value is missing for '{field_name}'"]
    if ai_val is None:
        return AgreementStatus.AI_MISSING, None, [f"AI value is missing for '{field_name}'"]

    # Monetary comparison
    if "amount" in field_name or field_name in ("subtotal", "tax_rate", "rounding_adjustment"):
        _, ocr_dec, _, _ = parse_monetary_amount(ocr_val, allow_negative=True)
        _, ai_dec, _, _ = parse_monetary_amount(ai_val, allow_negative=True)
        if ocr_dec is not None and ai_dec is not None:
            tol = monetary_config.reconciliation_tolerance if monetary_config else Decimal("0.05")
            diff = abs(ocr_dec - ai_dec)
            if diff <= tol:
                return AgreementStatus.MATCH, 1.0, []
            return (
                AgreementStatus.MISMATCH,
                0.0,
                [f"Monetary mismatch on {field_name}: OCR={ocr_dec}, AI={ai_dec} (diff={diff})"],
            )

    # Date comparison
    if field_name == "date":
        _, ocr_dt, _, _ = parse_and_validate_date(ocr_val)
        _, ai_dt, _, _ = parse_and_validate_date(ai_val)
        if ocr_dt is not None and ai_dt is not None:
            if ocr_dt == ai_dt:
                return AgreementStatus.MATCH, 1.0, []
            return (
                AgreementStatus.MISMATCH,
                0.0,
                [f"Date mismatch: OCR normalized='{ocr_dt}', AI normalized='{ai_dt}'"],
            )

    # Currency comparison
    if field_name == "currency":
        o_clean = str(ocr_val).strip().upper()
        a_clean = str(ai_val).strip().upper()
        if o_clean == a_clean:
            return AgreementStatus.MATCH, 1.0, []
        return (
            AgreementStatus.MISMATCH,
            0.0,
            [f"Currency mismatch: OCR='{o_clean}', AI='{a_clean}'"],
        )

    # Text fields (vendor_name, address, etc.)
    o_norm = re.sub(r"\s+", " ", str(ocr_val)).strip().lower()
    a_norm = re.sub(r"\s+", " ", str(ai_val)).strip().lower()

    if o_norm == a_norm:
        return AgreementStatus.MATCH, 1.0, []

    # Conservative substring or containment check
    if o_norm in a_norm or a_norm in o_norm:
        len_ratio = min(len(o_norm), len(a_norm)) / max(len(o_norm), len(a_norm))
        if len_ratio >= 0.8:
            return (
                AgreementStatus.MATCH,
                1.0,
                [f"Conservative partial match between OCR ('{ocr_val}') and AI ('{ai_val}')"],
            )

    return (
        AgreementStatus.MISMATCH,
        0.0,
        [f"Value mismatch on '{field_name}': OCR='{ocr_val}', AI='{ai_val}'"],
    )


def reconcile_total(
    subtotal: Decimal | None,
    tax: Decimal | None,
    discount: Decimal | None,
    rounding: Decimal | None,
    extracted_total: Decimal | None,
    tolerance: Decimal = Decimal("0.05"),
) -> CrossFieldCheckResult:
    """Reconcile total amount: expected = subtotal + tax - discount + rounding."""
    if subtotal is None or extracted_total is None:
        missing = []
        if subtotal is None:
            missing.append("subtotal")
        if extracted_total is None:
            missing.append("total_amount")
        return CrossFieldCheckResult(
            check_name="total_reconciliation",
            status=ValidationStatus.NOT_CHECKABLE,
            reasons=[f"Cannot reconcile total: required fields ({', '.join(missing)}) missing"],
            is_critical_failure=False,
        )

    tax_val = tax if tax is not None else Decimal("0.00")
    disc_val = discount if discount is not None else Decimal("0.00")
    round_val = rounding if rounding is not None else Decimal("0.00")

    expected_total = subtotal + tax_val - disc_val + round_val
    diff = abs(expected_total - extracted_total)

    if diff <= tolerance:
        return CrossFieldCheckResult(
            check_name="total_reconciliation",
            status=ValidationStatus.VALID,
            expected_value=expected_total,
            actual_value=extracted_total,
            difference=diff,
            tolerance=tolerance,
            reasons=["Total amount reconciles with subtotal and tax within tolerance"],
            is_critical_failure=False,
        )

    msg = (
        f"Total reconciliation mismatch: expected {expected_total} "
        f"(subtotal {subtotal} + tax {tax_val} - discount {disc_val} + rounding {round_val}), "
        f"extracted {extracted_total}, difference {diff} exceeds tolerance {tolerance}"
    )
    return CrossFieldCheckResult(
        check_name="total_reconciliation",
        status=ValidationStatus.INVALID,
        expected_value=expected_total,
        actual_value=extracted_total,
        difference=diff,
        tolerance=tolerance,
        reasons=[msg],
        is_critical_failure=True,
    )


def validate_tax_calculation(
    subtotal: Decimal | None,
    tax_amount: Decimal | None,
    tax_rate: Decimal | None,
    tolerance: Decimal = Decimal("0.05"),
) -> CrossFieldCheckResult:
    """Validate tax calculation: expected_tax = subtotal * (tax_rate / 100)."""
    if subtotal is None or tax_rate is None or tax_amount is None:
        missing = []
        if subtotal is None:
            missing.append("subtotal")
        if tax_rate is None:
            missing.append("tax_rate")
        if tax_amount is None:
            missing.append("tax_amount")
        return CrossFieldCheckResult(
            check_name="tax_calculation",
            status=ValidationStatus.NOT_CHECKABLE,
            reasons=[f"Cannot verify tax: fields ({', '.join(missing)}) not fully provided"],
            is_critical_failure=False,
        )

    expected_tax = (subtotal * tax_rate) / Decimal("100.0")
    diff = abs(expected_tax - tax_amount)

    if diff <= tolerance:
        return CrossFieldCheckResult(
            check_name="tax_calculation",
            status=ValidationStatus.VALID,
            expected_value=expected_tax,
            actual_value=tax_amount,
            difference=diff,
            tolerance=tolerance,
            reasons=["Tax amount matches calculated tax rate on subtotal"],
            is_critical_failure=False,
        )

    msg = (
        f"Tax calculation mismatch: subtotal {subtotal} at {tax_rate}% yields {expected_tax}, "
        f"but extracted tax is {tax_amount} (diff {diff} exceeds tolerance {tolerance})"
    )
    return CrossFieldCheckResult(
        check_name="tax_calculation",
        status=ValidationStatus.WARNING,
        expected_value=expected_tax,
        actual_value=tax_amount,
        difference=diff,
        tolerance=tolerance,
        reasons=[msg],
        is_critical_failure=False,
    )


def validate_field(
    field_data: CanonicalFieldData,
    doc_config: Any,
    engine_config: RulesEngineConfig,
) -> tuple[FieldValidationResult, Any]:
    """Validate a single canonical field using deterministic rules.

    Returns:
        (FieldValidationResult, normalized_value)
    """
    f_name = field_data.field_name
    # Prioritize human corrected value if present, else AI value, else OCR value
    if getattr(field_data, "is_human_corrected", False):
        candidate_val = field_data.normalized_value
    elif field_data.ai_value is not None:
        candidate_val = field_data.ai_value
    elif field_data.ocr_value is not None:
        candidate_val = field_data.ocr_value
    else:
        candidate_val = field_data.normalized_value

    is_required = f_name in doc_config.required_fields

    if candidate_val is None:
        if is_required:
            return (
                FieldValidationResult(
                    status=ValidationStatus.MISSING,
                    reasons=[f"Required field '{f_name}' is missing"],
                    is_critical_failure=True,
                ),
                None,
            )
        return (
            FieldValidationResult(
                status=ValidationStatus.NOT_CHECKABLE,
                reasons=[f"Optional field '{f_name}' is absent"],
                is_critical_failure=False,
            ),
            None,
        )

    if f_name == "date":
        status, norm_val, reasons, is_crit = parse_and_validate_date(
            candidate_val,
            preferred_order=engine_config.date.preferred_date_order,
            max_future_days=engine_config.date.max_future_days,
        )
        return FieldValidationResult(status=status, reasons=reasons, is_critical_failure=is_crit), norm_val

    if f_name in (
        "total_amount",
        "subtotal",
        "tax_amount",
        "tax_rate",
        "discount_amount",
        "rounding_adjustment",
    ):
        allow_neg = engine_config.monetary.allow_negative_amounts or (
            f_name in ("discount_amount", "rounding_adjustment")
        )
        status, dec_val, reasons, is_crit = parse_monetary_amount(candidate_val, allow_negative=allow_neg)
        if status == ValidationStatus.MISSING and is_required:
            is_crit = True
        return FieldValidationResult(status=status, reasons=reasons, is_critical_failure=is_crit), dec_val

    if f_name == "currency":
        status, norm_val, reasons, is_crit = validate_currency(
            candidate_val,
            engine_config.monetary,
        )
        return FieldValidationResult(status=status, reasons=reasons, is_critical_failure=is_crit), norm_val

    # Text fields: vendor_name, address, invoice_number, etc.
    status, norm_val, reasons, is_crit = validate_text_field(
        candidate_val,
        field_name=f_name,
        is_required=is_required,
    )
    return FieldValidationResult(status=status, reasons=reasons, is_critical_failure=is_crit), norm_val


def validate_document(
    normalized_data: NormalizedExtractionData,
    engine_config: RulesEngineConfig,
) -> tuple[
    dict[str, FieldValidationResult],
    dict[str, Any],
    list[CrossFieldCheckResult],
    list[str],
]:
    """Validate all fields and cross-field constraints of a normalized document.

    Returns:
        (field_validation_results, normalized_values, cross_field_results, critical_failures)
    """
    doc_config = engine_config.get_document_config(normalized_data.document_type)
    field_validations: dict[str, FieldValidationResult] = {}
    normalized_values: dict[str, Any] = {}
    critical_failures: list[str] = []

    # 1. Field-by-field validation
    for f_name, f_data in normalized_data.fields.items():
        v_res, norm_val = validate_field(f_data, doc_config, engine_config)
        field_validations[f_name] = v_res
        normalized_values[f_name] = norm_val
        if v_res.is_critical_failure:
            critical_failures.extend(v_res.reasons)

    # Check missing required fields that weren't even in normalized_data.fields
    for req_f in doc_config.required_fields:
        if req_f not in field_validations:
            msg = f"Required field '{req_f}' is missing from extraction"
            field_validations[req_f] = FieldValidationResult(
                status=ValidationStatus.MISSING,
                reasons=[msg],
                is_critical_failure=True,
            )
            critical_failures.append(msg)

    # 2. Cross-field checks
    cross_checks: list[CrossFieldCheckResult] = []

    # Total reconciliation
    if doc_config.enable_total_reconciliation:
        subtotal_dec = normalized_values.get("subtotal")
        tax_dec = normalized_values.get("tax_amount")
        disc_dec = normalized_values.get("discount_amount")
        round_dec = normalized_values.get("rounding_adjustment")
        total_dec = normalized_values.get("total_amount")

        # Convert to Decimal if string / numeric
        def _to_dec(val: Any) -> Decimal | None:
            if isinstance(val, Decimal):
                return val
            if val is not None:
                _, d, _, _ = parse_monetary_amount(val, allow_negative=True)
                return d
            return None

        recon_res = reconcile_total(
            subtotal=_to_dec(subtotal_dec),
            tax=_to_dec(tax_dec),
            discount=_to_dec(disc_dec),
            rounding=_to_dec(round_dec),
            extracted_total=_to_dec(total_dec),
            tolerance=engine_config.monetary.reconciliation_tolerance,
        )
        cross_checks.append(recon_res)
        if recon_res.is_critical_failure:
            critical_failures.extend(recon_res.reasons)

    # Tax calculation check
    if doc_config.enable_tax_validation:
        subtotal_dec = normalized_values.get("subtotal")
        tax_dec = normalized_values.get("tax_amount")
        tax_rate_dec = normalized_values.get("tax_rate")
        tax_res = validate_tax_calculation(
            subtotal=_to_dec(subtotal_dec),
            tax_amount=_to_dec(tax_dec),
            tax_rate=_to_dec(tax_rate_dec),
            tolerance=engine_config.monetary.reconciliation_tolerance,
        )
        cross_checks.append(tax_res)
        if tax_res.is_critical_failure:
            critical_failures.extend(tax_res.reasons)

    # Currency consistency check (if both OCR and AI provided currencies)
    curr_data = normalized_data.fields.get("currency")
    if curr_data and curr_data.ocr_value and curr_data.ai_value:
        status, _, reasons = compare_ocr_and_ai(
            "currency", curr_data.ocr_value, curr_data.ai_value, engine_config.monetary
        )
        if status == AgreementStatus.MISMATCH:
            msg = f"Conflicting currency between OCR ('{curr_data.ocr_value}') and AI ('{curr_data.ai_value}')"
            critical_failures.append(msg)

    return field_validations, normalized_values, cross_checks, critical_failures
