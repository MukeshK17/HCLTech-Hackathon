import csv
import io
import json
from decimal import Decimal, InvalidOperation

AMOUNT_FIELDS = {"subtotal", "tax", "discount", "rounding_adjustment", "total"}


def prepare_export(values, field_definitions=None, include_confidential=False):
    """
    Convert reviewed field values into structured data.
    Preserve numbers as numbers, not strings.
    """
    result = {}
    excluded_confidential_fields = []
    definitions = field_definitions or {}

    for field_name, value in values.items():
        definition = definitions.get(field_name, {})
        if definition.get("confidential", False) and not include_confidential:
            excluded_confidential_fields.append(field_name)
            continue

        is_amount = field_name in AMOUNT_FIELDS or definition.get("type") in {"amount", "currency", "number"}
        if is_amount:
            try:
                amount = Decimal(str(value).strip())
                result[field_name] = float(amount)
            except (InvalidOperation, ValueError):
                raise ValueError(f"Invalid amount for {field_name}") from None
        else:
            result[field_name] = str(value).strip()

    result["review_status"] = "approved"

    if excluded_confidential_fields:
        result["excluded_confidential_fields"] = excluded_confidential_fields

    return result


def export_json(data):
    return json.dumps(data, indent=2)


def export_csv(data):
    output = io.StringIO()

    writer = csv.DictWriter(output, fieldnames=list(data.keys()))

    writer.writeheader()
    writer.writerow(data)

    return output.getvalue()
