import re
from datetime import datetime
from decimal import Decimal, InvalidOperation

DEFAULT_AMOUNT_FIELDS = {
    "subtotal",
    "tax",
    "discount",
    "rounding_adjustment",
    "total",
}


def _is_date_field(field_name, definition):
    return definition.get("type") == "date" or field_name == "date" or field_name.endswith("_date")


def _is_amount_field(field_name, definition):
    return definition.get("type") in {"amount", "currency", "number"} or field_name in DEFAULT_AMOUNT_FIELDS


def validate_fields(values, field_definitions=None):
    """Validate dynamic extracted fields without assuming a fixed schema."""
    definitions = field_definitions or {}
    errors = {field_name: [] for field_name in values}
    parsed_amounts = {}

    for field_name, raw_value in values.items():
        definition = definitions.get(field_name, {})
        value = str(raw_value).strip()

        if not value:
            if definition.get("required", True):
                errors[field_name].append("This field is required.")
            continue

        if _is_date_field(field_name, definition):
            if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
                errors[field_name].append("Date must use YYYY-MM-DD format.")
            else:
                try:
                    datetime.strptime(value, "%Y-%m-%d")
                except ValueError:
                    errors[field_name].append("Enter a valid calendar date.")

        if _is_amount_field(field_name, definition):
            try:
                amount = Decimal(value)
                if not amount.is_finite():
                    raise InvalidOperation
                if amount < 0 and field_name != "rounding_adjustment":
                    errors[field_name].append(f"{field_name.replace('_', ' ').title()} cannot be negative.")
                else:
                    parsed_amounts[field_name] = amount
            except InvalidOperation:
                errors[field_name].append(f"{field_name.replace('_', ' ').title()} must be a valid number.")

    arithmetic_fields = {"subtotal", "total"}
    if arithmetic_fields.issubset(values):
        participating = [
            name
            for name in (
                "subtotal",
                "tax",
                "discount",
                "rounding_adjustment",
                "total",
            )
            if name in values
        ]
        if all(name in parsed_amounts for name in participating):
            expected_total = parsed_amounts["subtotal"]
            expected_total += parsed_amounts.get("tax", Decimal("0"))
            expected_total -= parsed_amounts.get("discount", Decimal("0"))
            expected_total += parsed_amounts.get("rounding_adjustment", Decimal("0"))

            if abs(expected_total - parsed_amounts["total"]) > Decimal("0.01"):
                message = "Subtotal + tax - discount + rounding adjustment must equal total."
                for field_name in participating:
                    errors[field_name].append(message)

    return errors
