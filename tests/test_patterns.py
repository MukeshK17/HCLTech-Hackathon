import pytest

from extraction.patterns import canonical_key, find_amounts, find_currencies, find_dates


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("DATE: 2018-03-30", ["2018-03-30"]),
        ("19 APR 2018", ["19 APR 2018"]),
        ("April 19, 2018", ["April 19, 2018"]),
        ("30/03/2018 21:14", ["30/03/2018"]),
        ("03-30-18", ["03-30-18"]),  # month-first kept as printed
        ("no date here", []),
    ],
)
def test_dates_returned_as_printed(text, expected):
    assert find_dates(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("TOTAL RM9.90", ["9.90"]),
        ("$9.90", ["9.90"]),
        ("1,234.50", ["1,234.50"]),
        ("1.234,56", ["1.234,56"]),
        ("ROUNDING -0.01", ["-0.01"]),
        ("BEER 2 x 89.50   179.00", ["89.50", "179.00"]),
        ("QTY 3", []),
    ],
)
def test_amounts_returned_as_printed(text, expected):
    assert find_amounts(text) == expected


def test_currencies_as_printed():
    assert find_currencies("TOTAL RM9.90") == ["RM"]
    assert find_currencies("Total $9.90 MYR") == ["$", "MYR"]
    assert find_currencies("S$ 5.00") == ["S$"]
    assert find_currencies("nothing") == []


def test_aliases():
    assert canonical_key("company") == "vendor"
    assert canonical_key("Merchant") == "vendor"
    assert canonical_key("Invoice No") == "invoice_number"
    assert canonical_key("custom_field") == "custom_field"
