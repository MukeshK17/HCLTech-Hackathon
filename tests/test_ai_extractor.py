import anthropic
import httpx2 as httpx
import pytest

from extraction.ai_extractor import VLMResponse, extract_with_vlm, parse_vlm_output

from .conftest import PNG_BYTES, FakeClient


def f(value, confidence=0.9, evidence="ev"):
    return {"value": value, "confidence": confidence, "evidence": evidence}


VALID = {
    "document_type": f("receipt"),
    "vendor": f("BREWERY TAP"),
    "address": f(None, None, None),
    "invoice_number": f(None, None, None),
    "order_number": f("11649"),
    "date": f("30/03/2018"),
    "currency": f("RM"),
    "subtotal": f("179.00"),
    "tax_rate": f("6%"),
    "tax": f("10.74"),
    "rounding_adjustment": f("0.01"),
    "total": f("189.75", 0.95, "TOTAL MYR 189.75"),
}


def test_valid_response_values_verbatim():
    client = FakeClient(VLMResponse.model_validate(VALID))
    res = extract_with_vlm(PNG_BYTES, client=client)
    assert res.ok and res.source == "vlm"
    rec = res.record()
    assert rec["date"] == "30/03/2018" and rec["currency"] == "RM" and rec["tax_rate"] == "6%"
    assert rec["total"] == "189.75" and rec["address"] is None
    t = res.fields["total"]
    assert t.confidence == 0.95 and t.confidence_method == "model_self_report"
    assert t.evidence.text == "TOTAL MYR 189.75" and t.evidence.bbox is None
    call = client.calls[0]
    assert call["output_format"] is VLMResponse
    assert call["messages"][0]["content"][0]["source"]["media_type"] == "image/png"


def test_pdf_sent_as_document_block():
    client = FakeClient(VLMResponse.model_validate(VALID))
    assert extract_with_vlm(b"%PDF-1.7 fake", client=client).ok
    assert client.calls[0]["messages"][0]["content"][0]["type"] == "document"


def test_missing_keys():
    res = parse_vlm_output({"vendor": f("X"), "total": f("5.00")}, "m")
    assert res.fields["invoice_number"].value is None
    assert "missing_in_response" in res.fields["invoice_number"].warnings


def test_missing_confidence_uses_fallback():
    t = parse_vlm_output({"total": {"value": "5.00", "evidence": "TOTAL 5.00"}}, "m").fields["total"]
    assert t.confidence_method == "vlm_fallback_heuristic" and t.confidence == 0.5
    assert "missing_confidence" in t.warnings
    bad = parse_vlm_output({"total": f("5.00", confidence=7)}, "m").fields["total"]
    assert "invalid_confidence" in bad.warnings and bad.confidence <= 0.5


def test_unexpected_types_and_values():
    res = parse_vlm_output(
        {
            "vendor": f(["a", "b"]),
            "subtotal": 12.5,  # bare number instead of an object
            "total": f("null"),
            "document_type": f("passport"),
            "company_reg": f("123"),
        },
        "m",
    )
    assert res.fields["vendor"].value is None and "invalid_type" in res.fields["vendor"].warnings
    assert res.fields["subtotal"].value == "12.5"
    assert res.fields["total"].value is None and "not_found" in res.fields["total"].warnings
    assert res.fields["document_type"].value is None
    assert "company_reg" in res.extra_fields


def test_aliases_in_response():
    rec = parse_vlm_output({"company": f("UNIHAKKA"), "total": f("$9.90")}, "m").record()
    assert rec["vendor"] == "UNIHAKKA" and rec["total"] == "$9.90"


@pytest.mark.parametrize("data", ["not json", None, [1, 2]])
def test_invalid_output(data):
    assert parse_vlm_output(data, "m").errors[0].code == "vlm_invalid_response"


def test_no_parsed_output_and_refusal():
    assert extract_with_vlm(PNG_BYTES, client=FakeClient(None)).errors[0].code == "vlm_invalid_response"
    refused = FakeClient(VLMResponse.model_validate(VALID), stop_reason="refusal")
    assert extract_with_vlm(PNG_BYTES, client=refused).errors[0].code == "vlm_refused"


REQ = httpx.Request("POST", "https://api.anthropic.com/v1/messages")


@pytest.mark.parametrize(
    ("exc", "code"),
    [
        (anthropic.APITimeoutError(request=REQ), "vlm_timeout"),
        (anthropic.APIConnectionError(request=REQ), "vlm_api_error"),
        (anthropic.InternalServerError("x", response=httpx.Response(500, request=REQ), body=None), "vlm_api_error"),
    ],
)
def test_api_errors(exc, code):
    res = extract_with_vlm(PNG_BYTES, client=FakeClient(exc=exc))
    assert res.errors[0].code == code and all(v is None for v in res.record().values())


def test_input_validation(tmp_path, monkeypatch):
    assert extract_with_vlm(b"not an image", client=FakeClient()).errors[0].code == "invalid_input"
    assert extract_with_vlm(tmp_path / "nope.png", client=FakeClient()).errors[0].code == "invalid_input"
    big = PNG_BYTES + b"\x00" * (5 * 1024 * 1024)
    assert extract_with_vlm(big, client=FakeClient()).errors[0].code == "invalid_input"
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert extract_with_vlm(PNG_BYTES).errors[0].code == "vlm_not_configured"
