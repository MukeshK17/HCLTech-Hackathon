from rules_engine import process_extraction
from rules_integration import (
    revalidate_review_corrections,
    review_fields_from_rules,
)


def _rules_result():
    return process_extraction(
        ocr_json={
            "company": "Test Store",
            "date": "2024-06-15",
            "subtotal": "100",
            "tax": "10",
            "total": "110",
        },
        ai_json={
            "vendor_name": "Test Store",
            "date": "2024-06-15",
            "subtotal": "100",
            "tax_amount": "10",
            "total_amount": "110",
        },
        ocr_confidence={
            "vendor_name": 0.98,
            "date": 0.95,
            "subtotal": 0.94,
            "tax_amount": 0.93,
            "total_amount": 0.97,
        },
        document_type="receipt",
    )


def test_rules_result_maps_to_review_fields():
    fields = review_fields_from_rules(_rules_result(), confidential_fields={"vendor_name"})

    assert fields["total_amount"]["decision"] == "auto_accept"
    assert fields["vendor_name"]["confidential"] is True
    assert fields["total_amount"]["type"] == "amount"


def test_human_correction_is_revalidated():
    original = _rules_result()
    fields = review_fields_from_rules(original)
    values = {name: str(details["value"]) for name, details in fields.items()}

    values["total_amount"] = "999"
    errors, document_errors, _ = revalidate_review_corrections(original, values, {"total_amount"})
    assert errors["total_amount"] or document_errors

    values["total_amount"] = "110"
    errors, document_errors, updated = revalidate_review_corrections(original, values, {"total_amount"})
    assert not any(errors.values())
    assert not document_errors
    assert updated["fields"]["total_amount"]["is_human_corrected"] is True
