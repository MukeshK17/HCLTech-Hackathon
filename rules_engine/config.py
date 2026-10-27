"""Configuration module for the rules engine, confidence scoring, and review routing.

All thresholds, weights, supported currencies, date formats, and document field
requirements are defined here as configurable dataclasses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any


@dataclass(frozen=True)
class ConfidenceWeights:
    """Configurable weights for the field-level confidence scoring formula.

    Formula: C_f = w_ocr * O + w_agreement * A + w_validation * V
    """

    w_ocr: float = 0.4
    w_agreement: float = 0.3
    w_validation: float = 0.3

    def __post_init__(self) -> None:
        """Validate that weights are non-negative and sum to 1.0 within floating tolerance."""
        if any(w < 0.0 for w in (self.w_ocr, self.w_agreement, self.w_validation)):
            raise ValueError(
                f"Confidence weights must be non-negative: "
                f"w_ocr={self.w_ocr}, w_agreement={self.w_agreement}, w_validation={self.w_validation}"
            )
        total = self.w_ocr + self.w_agreement + self.w_validation
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Confidence weights must sum to 1.0, got {total}")


@dataclass(frozen=True)
class RoutingThresholds:
    """Configurable thresholds for routing field-level and document-level decisions."""

    auto_accept_threshold: float = 0.90
    human_review_threshold: float = 0.70
    max_retries: int = 2

    def __post_init__(self) -> None:
        """Validate threshold bounds."""
        if not (0.0 <= self.human_review_threshold <= self.auto_accept_threshold <= 1.0):
            raise ValueError(
                f"Invalid thresholds: human_review ({self.human_review_threshold}) "
                f"must be <= auto_accept ({self.auto_accept_threshold}) within [0.0, 1.0]"
            )
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")


@dataclass(frozen=True)
class MonetaryConfig:
    """Configurable monetary precision, rounding, and reconciliation rules."""

    reconciliation_tolerance: Decimal = Decimal("0.05")
    allow_negative_amounts: bool = False
    supported_currencies: tuple[str, ...] = (
        "USD",
        "EUR",
        "GBP",
        "SGD",
        "MYR",
        "AUD",
        "CAD",
        "INR",
        "JPY",
        "CNY",
        "CHF",
        "NZD",
    )
    # Currency symbols mapped to ambiguous list or unambiguous code
    symbol_to_currency: dict[str, str | None] = field(
        default_factory=lambda: {
            "$": None,  # Ambiguous: USD, SGD, AUD, CAD, etc.
            "€": "EUR",
            "£": "GBP",
            "¥": None,  # Ambiguous: JPY or CNY
            "₹": "INR",
            "RM": "MYR",
            "S$": "SGD",
            "US$": "USD",
            "A$": "AUD",
            "C$": "CAD",
        }
    )


@dataclass(frozen=True)
class DateConfig:
    """Configurable date parsing rules."""

    # Date order hint: None (ambiguous dates flagged), 'DMY', or 'MDY'
    preferred_date_order: str | None = None
    # Maximum future days allowed before date is flagged as unrealistic
    max_future_days: int = 1


@dataclass(frozen=True)
class DocumentTypeConfig:
    """Schema requirements per document type."""

    required_fields: tuple[str, ...]
    optional_fields: tuple[str, ...] = ()
    enable_total_reconciliation: bool = True
    enable_tax_validation: bool = True


DEFAULT_DOCUMENT_CONFIGS: dict[str, DocumentTypeConfig] = {
    "receipt": DocumentTypeConfig(
        required_fields=("vendor_name", "date", "total_amount"),
        optional_fields=(
            "address",
            "currency",
            "subtotal",
            "tax_amount",
            "tax_rate",
            "discount_amount",
            "rounding_adjustment",
        ),
        enable_total_reconciliation=True,
        enable_tax_validation=True,
    ),
    "invoice": DocumentTypeConfig(
        required_fields=("vendor_name", "date", "total_amount"),
        optional_fields=(
            "address",
            "currency",
            "subtotal",
            "tax_amount",
            "tax_rate",
            "discount_amount",
            "rounding_adjustment",
            "invoice_number",
        ),
        enable_total_reconciliation=True,
        enable_tax_validation=True,
    ),
    "delivery_note": DocumentTypeConfig(
        required_fields=("vendor_name", "date"),
        optional_fields=("address", "total_amount", "subtotal"),
        enable_total_reconciliation=False,
        enable_tax_validation=False,
    ),
    "generic": DocumentTypeConfig(
        required_fields=("vendor_name", "date", "total_amount"),
        optional_fields=("address", "currency", "subtotal", "tax_amount", "discount_amount"),
        enable_total_reconciliation=True,
        enable_tax_validation=True,
    ),
}


@dataclass
class RulesEngineConfig:
    """Master configuration for Member 3's Rules Engine subsystem."""

    weights: ConfidenceWeights = field(default_factory=ConfidenceWeights)
    thresholds: RoutingThresholds = field(default_factory=RoutingThresholds)
    monetary: MonetaryConfig = field(default_factory=MonetaryConfig)
    date: DateConfig = field(default_factory=DateConfig)
    document_configs: dict[str, DocumentTypeConfig] = field(default_factory=lambda: dict(DEFAULT_DOCUMENT_CONFIGS))
    # Minimum components required for evidence sufficiency (out of OCR, Agreement, Validation)
    min_components_for_sufficiency: int = 2
    # Critical checks that immediately force human review if failed
    critical_overrides: tuple[str, ...] = (
        "total_reconciliation_mismatch",
        "critical_field_ocr_ai_disagreement",
        "invalid_required_amount",
        "missing_required_field",
        "ambiguous_date",
        "conflicting_currency",
        "insufficient_evidence",
        "unresolved_parsing_error",
    )

    def get_document_config(self, doc_type: str | None) -> DocumentTypeConfig:
        """Lookup document configuration by document type, falling back to 'generic'."""
        if not doc_type:
            return self.document_configs.get("generic", DEFAULT_DOCUMENT_CONFIGS["generic"])
        key = doc_type.strip().lower()
        return self.document_configs.get(key, self.document_configs.get("generic", DEFAULT_DOCUMENT_CONFIGS["generic"]))

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "weights": {
                "w_ocr": self.weights.w_ocr,
                "w_agreement": self.weights.w_agreement,
                "w_validation": self.weights.w_validation,
            },
            "thresholds": {
                "auto_accept_threshold": self.thresholds.auto_accept_threshold,
                "human_review_threshold": self.thresholds.human_review_threshold,
                "max_retries": self.thresholds.max_retries,
            },
            "monetary": {
                "reconciliation_tolerance": str(self.monetary.reconciliation_tolerance),
                "allow_negative_amounts": self.monetary.allow_negative_amounts,
                "supported_currencies": list(self.monetary.supported_currencies),
            },
            "date": {
                "preferred_date_order": self.date.preferred_date_order,
                "max_future_days": self.date.max_future_days,
            },
            "min_components_for_sufficiency": self.min_components_for_sufficiency,
            "critical_overrides": list(self.critical_overrides),
        }
