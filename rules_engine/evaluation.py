"""Offline evaluation utility for labeled datasets in Member 3's Rules Engine.

Evaluates predictions against verified ground-truth reference labels (e.g. SROIE,
invoices) without requiring live OCR or paid AI APIs.

Metrics calculated:
- Field-level accuracy (overall and per field type)
- Auto-accepted field error rate (precision of auto-acceptance decisions)
- Auto-acceptance coverage (% of total fields auto-accepted)
- Human review rate (% of total fields requiring human review)
- Expected Calibration Error (ECE) and Brier Score when sample size permits.

NOTE: Ground-truth reference labels are strictly for offline evaluation and benchmark
tuning. Production processing of new documents does NOT require ground-truth labels.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from rules_engine.schemas import DocumentResult, FieldDecision
from rules_engine.validator import parse_and_validate_date, parse_monetary_amount

logger = logging.getLogger(__name__)


@dataclass
class FieldEvaluationMetric:
    """Evaluation metrics for a specific field name."""

    field_name: str
    total_evaluated: int = 0
    correct_count: int = 0
    accuracy: float = 0.0
    auto_accepted_count: int = 0
    auto_accepted_correct: int = 0
    auto_accepted_error_rate: float = 0.0
    human_review_count: int = 0
    human_review_rate: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict."""
        return {
            "field_name": self.field_name,
            "total_evaluated": self.total_evaluated,
            "correct_count": self.correct_count,
            "accuracy": round(self.accuracy, 4),
            "auto_accepted_count": self.auto_accepted_count,
            "auto_accepted_correct": self.auto_accepted_correct,
            "auto_accepted_error_rate": round(self.auto_accepted_error_rate, 4),
            "human_review_count": self.human_review_count,
            "human_review_rate": round(self.human_review_rate, 4),
        }


@dataclass
class CalibrationMetric:
    """Confidence calibration metrics (ECE, Brier Score, and reliability bins)."""

    brier_score: float | None = None
    expected_calibration_error: float | None = None
    num_samples: int = 0
    num_bins: int = 5
    bins_data: list[dict[str, Any]] = field(default_factory=list)
    is_statistically_sufficient: bool = False
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict."""
        return {
            "brier_score": round(self.brier_score, 4) if self.brier_score is not None else None,
            "expected_calibration_error": (
                round(self.expected_calibration_error, 4) if self.expected_calibration_error is not None else None
            ),
            "num_samples": self.num_samples,
            "num_bins": self.num_bins,
            "bins_data": self.bins_data,
            "is_statistically_sufficient": self.is_statistically_sufficient,
            "notes": self.notes,
        }


@dataclass
class EvaluationReport:
    """Comprehensive offline evaluation report."""

    total_documents: int
    total_fields_evaluated: int
    overall_accuracy: float
    overall_auto_acceptance_coverage: float
    overall_human_review_rate: float
    auto_accepted_fields_error_rate: float
    per_field_metrics: dict[str, FieldEvaluationMetric]
    calibration: CalibrationMetric
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict."""
        return {
            "total_documents": self.total_documents,
            "total_fields_evaluated": self.total_fields_evaluated,
            "overall_accuracy": round(self.overall_accuracy, 4),
            "overall_auto_acceptance_coverage": round(self.overall_auto_acceptance_coverage, 4),
            "overall_human_review_rate": round(self.overall_human_review_rate, 4),
            "auto_accepted_fields_error_rate": round(self.auto_accepted_fields_error_rate, 4),
            "per_field_metrics": {k: v.to_dict() for k, v in self.per_field_metrics.items()},
            "calibration": self.calibration.to_dict(),
            "notes": self.notes,
        }


def _normalize_for_evaluation(field_name: str, val: Any) -> Any:
    """Normalize predicted or reference value for fair comparison."""
    if val is None:
        return None

    if "amount" in field_name or field_name in ("subtotal", "tax_rate", "rounding_adjustment"):
        _, d, _, _ = parse_monetary_amount(val, allow_negative=True)
        return d

    if field_name == "date":
        _, dt_str, _, _ = parse_and_validate_date(val)
        return dt_str if dt_str is not None else str(val).strip()

    if field_name == "currency":
        return str(val).strip().upper()

    # Text fields: lowercase whitespace collapsed
    return re.sub(r"\s+", " ", str(val)).strip().lower()


def _values_match(field_name: str, pred_val: Any, ref_val: Any, tolerance: Decimal = Decimal("0.05")) -> bool:
    """Check if predicted value matches reference ground-truth value."""
    p_norm = _normalize_for_evaluation(field_name, pred_val)
    r_norm = _normalize_for_evaluation(field_name, ref_val)

    if p_norm is None or r_norm is None:
        return p_norm == r_norm

    if isinstance(p_norm, Decimal) and isinstance(r_norm, Decimal):
        return abs(p_norm - r_norm) <= tolerance

    if isinstance(p_norm, str) and isinstance(r_norm, str):
        if p_norm == r_norm:
            return True
        # Partial match for long text (e.g. address)
        if field_name in ("address", "vendor_name") and (p_norm in r_norm or r_norm in p_norm):
            len_r = min(len(p_norm), len(r_norm)) / max(len(p_norm), len(r_norm))
            if len_r >= 0.70:
                return True

    return p_norm == r_norm


def compute_calibration_metrics(
    confidences: list[float],
    correctness_labels: list[int],
    num_bins: int = 5,
) -> CalibrationMetric:
    """Calculate Brier score and Expected Calibration Error (ECE)."""
    n = len(confidences)
    if n == 0:
        return CalibrationMetric(notes="No samples available for calibration")

    # Brier score: mean squared difference between predicted confidence and binary truth
    brier = sum((c - y) ** 2 for c, y in zip(confidences, correctness_labels, strict=True)) / n

    if n < 30:
        return CalibrationMetric(
            brier_score=brier,
            expected_calibration_error=None,
            num_samples=n,
            num_bins=num_bins,
            is_statistically_sufficient=False,
            notes=f"Sample size (N={n}) is too small for reliable Expected Calibration Error (ECE) calculation. "
            f"Heuristic reliability scores are not empirically calibrated probabilities.",
        )

    # Bin samples
    bin_size = 1.0 / num_bins
    bins: list[list[tuple[float, int]]] = [[] for _ in range(num_bins)]

    for c, y in zip(confidences, correctness_labels, strict=True):
        b_idx = min(int(c / bin_size), num_bins - 1)
        bins[b_idx].append((c, y))

    ece = 0.0
    bins_data: list[dict[str, Any]] = []

    for idx, b_samples in enumerate(bins):
        b_min = idx * bin_size
        b_max = (idx + 1) * bin_size
        b_count = len(b_samples)
        if b_count > 0:
            avg_conf = sum(c for c, _ in b_samples) / b_count
            avg_acc = sum(y for _, y in b_samples) / b_count
            ece += (b_count / n) * abs(avg_acc - avg_conf)
            bins_data.append(
                {
                    "bin_range": f"[{b_min:.2f}, {b_max:.2f})",
                    "sample_count": b_count,
                    "avg_confidence": round(avg_conf, 4),
                    "accuracy": round(avg_acc, 4),
                }
            )
        else:
            bins_data.append(
                {
                    "bin_range": f"[{b_min:.2f}, {b_max:.2f})",
                    "sample_count": 0,
                    "avg_confidence": None,
                    "accuracy": None,
                }
            )

    return CalibrationMetric(
        brier_score=brier,
        expected_calibration_error=ece,
        num_samples=n,
        num_bins=num_bins,
        bins_data=bins_data,
        is_statistically_sufficient=True,
        notes="Calibration computed over held-out samples. Note that heuristic scores require empirical calibration.",
    )


def evaluate_predictions(
    predictions: list[DocumentResult | dict[str, Any]],
    ground_truth: list[dict[str, Any]],
    tolerance: Decimal = Decimal("0.05"),
) -> EvaluationReport:
    """Evaluate predicted extraction results against reference ground-truth labels.

    Args:
        predictions: List of DocumentResult instances or equivalent dictionaries.
        ground_truth: List of dictionaries mapping field names to reference values.
        tolerance: Decimal tolerance for monetary comparisons.

    Returns:
        EvaluationReport containing accuracies, auto-accept error rates, coverage, and calibration.
    """
    notes: list[str] = []
    if len(predictions) != len(ground_truth):
        notes.append(
            f"Warning: Count mismatch between predictions ({len(predictions)}) and ground-truth ({len(ground_truth)})"
        )

    num_docs = min(len(predictions), len(ground_truth))
    if num_docs == 0:
        return EvaluationReport(
            total_documents=0,
            total_fields_evaluated=0,
            overall_accuracy=0.0,
            overall_auto_acceptance_coverage=0.0,
            overall_human_review_rate=0.0,
            auto_accepted_fields_error_rate=0.0,
            per_field_metrics={},
            calibration=CalibrationMetric(notes="No documents provided"),
            notes=["Evaluation called with empty dataset"],
        )

    field_stats: dict[str, dict[str, int]] = {}
    total_evaluated = 0
    total_correct = 0
    total_auto_accepted = 0
    total_auto_accepted_correct = 0
    total_human_review = 0

    all_confidences: list[float] = []
    all_correctness: list[int] = []

    for doc_idx in range(num_docs):
        pred_item = predictions[doc_idx]
        ref_dict = ground_truth[doc_idx]

        # Extract fields dictionary
        if hasattr(pred_item, "fields"):
            p_fields = pred_item.fields
        elif isinstance(pred_item, dict) and "fields" in pred_item:
            p_fields = pred_item["fields"]
        elif isinstance(pred_item, dict):
            p_fields = pred_item
        else:
            continue

        for f_name, ref_val in ref_dict.items():
            if ref_val is None:
                continue

            pred_field = p_fields.get(f_name)
            if pred_field is None:
                continue

            # Extract normalized predicted value and decision
            if hasattr(pred_field, "normalized_value"):
                p_val = pred_field.normalized_value
                decision = pred_field.decision
                conf = pred_field.confidence_score
            elif isinstance(pred_field, dict):
                p_val = pred_field.get("normalized_value", pred_field.get("ai_value"))
                dec_str = pred_field.get("decision", "human_review")
                if dec_str in FieldDecision.__members__.values():
                    decision = FieldDecision(dec_str)
                else:
                    decision = FieldDecision.HUMAN_REVIEW
                conf = pred_field.get("confidence_score")
            else:
                p_val = pred_field
                decision = FieldDecision.HUMAN_REVIEW
                conf = None

            is_correct = _values_match(f_name, p_val, ref_val, tolerance)

            if f_name not in field_stats:
                field_stats[f_name] = {
                    "total": 0,
                    "correct": 0,
                    "auto_accept": 0,
                    "auto_accept_correct": 0,
                    "human_review": 0,
                }

            field_stats[f_name]["total"] += 1
            total_evaluated += 1

            if is_correct:
                field_stats[f_name]["correct"] += 1
                total_correct += 1

            if decision == FieldDecision.AUTO_ACCEPT:
                field_stats[f_name]["auto_accept"] += 1
                total_auto_accepted += 1
                if is_correct:
                    field_stats[f_name]["auto_accept_correct"] += 1
                    total_auto_accepted_correct += 1
            else:
                field_stats[f_name]["human_review"] += 1
                total_human_review += 1

            if conf is not None:
                all_confidences.append(float(conf))
                all_correctness.append(1 if is_correct else 0)

    # Compute per-field metrics
    per_field_results: dict[str, FieldEvaluationMetric] = {}
    for f_name, s in field_stats.items():
        tot = s["total"]
        acc = s["correct"] / tot if tot > 0 else 0.0
        aa_tot = s["auto_accept"]
        aa_err = (aa_tot - s["auto_accept_correct"]) / aa_tot if aa_tot > 0 else 0.0
        hr_rate = s["human_review"] / tot if tot > 0 else 0.0

        per_field_results[f_name] = FieldEvaluationMetric(
            field_name=f_name,
            total_evaluated=tot,
            correct_count=s["correct"],
            accuracy=acc,
            auto_accepted_count=aa_tot,
            auto_accepted_correct=s["auto_accept_correct"],
            auto_accepted_error_rate=aa_err,
            human_review_count=s["human_review"],
            human_review_rate=hr_rate,
        )

    # Compute overall metrics
    overall_acc = total_correct / total_evaluated if total_evaluated > 0 else 0.0
    overall_coverage = total_auto_accepted / total_evaluated if total_evaluated > 0 else 0.0
    overall_hr_rate = total_human_review / total_evaluated if total_evaluated > 0 else 0.0
    overall_aa_err = (
        (total_auto_accepted - total_auto_accepted_correct) / total_auto_accepted if total_auto_accepted > 0 else 0.0
    )

    calibration_metric = compute_calibration_metrics(all_confidences, all_correctness)

    return EvaluationReport(
        total_documents=num_docs,
        total_fields_evaluated=total_evaluated,
        overall_accuracy=overall_acc,
        overall_auto_acceptance_coverage=overall_coverage,
        overall_human_review_rate=overall_hr_rate,
        auto_accepted_fields_error_rate=overall_aa_err,
        per_field_metrics=per_field_results,
        calibration=calibration_metric,
        notes=notes,
    )
