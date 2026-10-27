# ruff: noqa: S101
"""Unit tests for field reliability confidence scoring in Member 3's Rules Engine."""

from __future__ import annotations

import pytest

from rules_engine.confidence import (
    calculate_all_confidences,
    calculate_field_confidence,
    derive_validation_component_score,
)
from rules_engine.config import ConfidenceWeights, RulesEngineConfig
from rules_engine.schemas import (
    AgreementStatus,
    CanonicalFieldData,
    FieldValidationResult,
    ValidationStatus,
)


def test_confidence_weights_validation() -> None:
    # Valid weights
    cw = ConfidenceWeights(w_ocr=0.4, w_agreement=0.3, w_validation=0.3)
    assert cw.w_ocr == 0.4

    # Negative weight raises ValueError
    with pytest.raises(ValueError, match="non-negative"):
        ConfidenceWeights(w_ocr=-0.1, w_agreement=0.6, w_validation=0.5)

    # Sum != 1.0 raises ValueError
    with pytest.raises(ValueError, match="must sum to 1.0"):
        ConfidenceWeights(w_ocr=0.5, w_agreement=0.5, w_validation=0.5)


def test_derive_validation_component_score() -> None:
    # Valid -> 1.0
    v_valid = FieldValidationResult(status=ValidationStatus.VALID)
    score, _ = derive_validation_component_score(v_valid)
    assert score == 1.0

    # Warning -> 0.5
    v_warn = FieldValidationResult(status=ValidationStatus.WARNING)
    score, _ = derive_validation_component_score(v_warn)
    assert score == 0.5

    # Invalid -> 0.0
    v_inv = FieldValidationResult(status=ValidationStatus.INVALID)
    score, _ = derive_validation_component_score(v_inv)
    assert score == 0.0

    # Critical failure -> 0.0 even if warning
    v_crit = FieldValidationResult(status=ValidationStatus.WARNING, is_critical_failure=True)
    score, _ = derive_validation_component_score(v_crit)
    assert score == 0.0

    # Not checkable -> None
    v_nc = FieldValidationResult(status=ValidationStatus.NOT_CHECKABLE)
    score, _ = derive_validation_component_score(v_nc)
    assert score is None


def test_calculate_field_confidence_all_components_available() -> None:
    config = RulesEngineConfig()
    field_data = CanonicalFieldData(
        field_name="total_amount",
        normalized_ocr_confidence=0.90,
    )
    v_res = FieldValidationResult(status=ValidationStatus.VALID)

    # O = 0.90, A = 1.0, V = 1.0
    # Expected: 0.4 * 0.90 + 0.3 * 1.0 + 0.3 * 1.0 = 0.36 + 0.30 + 0.30 = 0.96
    conf = calculate_field_confidence(
        field_data=field_data,
        validation_result=v_res,
        agreement_status=AgreementStatus.MATCH,
        agreement_score=1.0,
        config=config,
    )

    assert conf.final_score is not None
    assert pytest.approx(conf.final_score, abs=1e-4) == 0.96
    assert conf.evidence_sufficient is True
    assert set(conf.components_used) == {"ocr_confidence", "agreement", "validation"}
    assert len(conf.components_missing) == 0


def test_calculate_field_confidence_renormalization_missing_ocr() -> None:
    config = RulesEngineConfig()
    field_data = CanonicalFieldData(
        field_name="vendor_name",
        normalized_ocr_confidence=None,  # Missing OCR confidence
    )
    v_res = FieldValidationResult(status=ValidationStatus.VALID)

    # Only Agreement (0.3) and Validation (0.3) available -> sum = 0.6
    # Renormalized weights: 0.3 / 0.6 = 0.5 each
    # Expected: 0.5 * 1.0 + 0.5 * 1.0 = 1.0
    conf = calculate_field_confidence(
        field_data=field_data,
        validation_result=v_res,
        agreement_status=AgreementStatus.MATCH,
        agreement_score=1.0,
        config=config,
    )

    assert conf.final_score is not None
    assert pytest.approx(conf.final_score, abs=1e-4) == 1.0
    assert conf.evidence_sufficient is True  # 2 components used >= min 2
    assert "ocr_confidence" in conf.components_missing
    assert pytest.approx(conf.weights_used["agreement"], abs=1e-4) == 0.5
    assert pytest.approx(conf.weights_used["validation"], abs=1e-4) == 0.5


def test_calculate_field_confidence_insufficient_evidence_single_component() -> None:
    config = RulesEngineConfig()
    field_data = CanonicalFieldData(
        field_name="address",
        normalized_ocr_confidence=None,
    )
    # Validation is not checkable
    v_res = FieldValidationResult(status=ValidationStatus.NOT_CHECKABLE)

    # Only Agreement available (1 component)
    conf = calculate_field_confidence(
        field_data=field_data,
        validation_result=v_res,
        agreement_status=AgreementStatus.MATCH,
        agreement_score=1.0,
        config=config,
    )

    # Provisional score is calculated for diagnostics
    assert conf.final_score == 1.0
    # BUT evidence sufficiency is strictly False!
    assert conf.evidence_sufficient is False
    assert any("insufficient" in exp.lower() for exp in conf.explanations)


def test_calculate_field_confidence_all_missing() -> None:
    config = RulesEngineConfig()
    field_data = CanonicalFieldData(
        field_name="address",
        normalized_ocr_confidence=None,
    )
    v_res = FieldValidationResult(status=ValidationStatus.NOT_CHECKABLE)

    conf = calculate_field_confidence(
        field_data=field_data,
        validation_result=v_res,
        agreement_status=AgreementStatus.BOTH_MISSING,
        agreement_score=None,
        config=config,
    )

    assert conf.final_score is None
    assert conf.evidence_sufficient is False
    assert len(conf.components_missing) == 3


def test_calculate_field_confidence_zero_vs_none() -> None:
    config = RulesEngineConfig()

    # Zero confidence from OCR
    field_zero = CanonicalFieldData(
        field_name="total_amount",
        normalized_ocr_confidence=0.0,
    )
    v_res = FieldValidationResult(status=ValidationStatus.VALID)

    conf_zero = calculate_field_confidence(
        field_data=field_zero,
        validation_result=v_res,
        agreement_status=AgreementStatus.MATCH,
        agreement_score=1.0,
        config=config,
    )

    # 0.4 * 0.0 + 0.3 * 1.0 + 0.3 * 1.0 = 0.60
    assert conf_zero.final_score is not None
    assert pytest.approx(conf_zero.final_score, abs=1e-4) == 0.60
    assert "ocr_confidence" in conf_zero.components_used


def test_calculate_all_confidences_helper() -> None:
    config = RulesEngineConfig()
    fields = {
        "vendor_name": CanonicalFieldData(field_name="vendor_name", normalized_ocr_confidence=0.95),
        "total_amount": CanonicalFieldData(field_name="total_amount", normalized_ocr_confidence=0.85),
    }
    vals = {
        "vendor_name": FieldValidationResult(status=ValidationStatus.VALID),
        "total_amount": FieldValidationResult(status=ValidationStatus.VALID),
    }
    agrs = {
        "vendor_name": (AgreementStatus.MATCH, 1.0, []),
        "total_amount": (AgreementStatus.MATCH, 1.0, []),
    }

    all_confs = calculate_all_confidences(fields, vals, agrs, config)
    assert "vendor_name" in all_confs
    assert "total_amount" in all_confs
    assert all_confs["vendor_name"].final_score is not None
    assert all_confs["total_amount"].final_score is not None
