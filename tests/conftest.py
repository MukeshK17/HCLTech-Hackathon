from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from extraction.ocr_engine import OCRLine, OCRResult

PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32  # magic bytes only; never decoded in tests


def lines(*rows: str | tuple[str, float] | tuple[str, float, list[float]]) -> OCRResult:
    out = []
    for i, r in enumerate(rows):
        if isinstance(r, str):
            r = (r, 0.98)
        text, conf = r[0], r[1]
        bbox = r[2] if len(r) > 2 else [10.0, 20.0 * i, 200.0, 20.0 * i + 15]  # type: ignore[misc]
        out.append(OCRLine(text=text, confidence=conf, bbox=bbox))
    return OCRResult(lines=out)


@pytest.fixture
def receipt() -> OCRResult:
    return lines(
        "BREWERY TAP SDN BHD",
        "12, JALAN TAMPOI 7/4,",
        "81200 JOHOR BAHRU, JOHOR",
        "TEL: 07-1234567",
        "TAX INVOICE",
        "ORDER NO: 11649",
        "DATE: 30/03/2018 21:14",
        "BEER 2 x 89.50   179.00",
        "SUB TOTAL   179.00",
        "GST 6%   10.74",
        "ROUNDING   0.01",
        "TOTAL (MYR)   189.75",
        "CASH   200.00",
        "CHANGE   10.25",
        "THANK YOU",
    )


class FakeOCREngine:
    def __init__(self, raw: Any = None, exc: Exception | None = None) -> None:
        self.raw, self.exc = raw, exc

    def predict(self, img: Any) -> Any:
        if self.exc:
            raise self.exc
        return self.raw


class FakeClient:
    """Mimics anthropic.Anthropic().messages.parse (structured outputs)."""

    def __init__(self, parsed: Any = None, exc: Exception | None = None, stop_reason: str = "end_turn") -> None:
        self.calls: list[dict[str, Any]] = []
        self._parsed, self._exc, self._stop = parsed, exc, stop_reason
        self.messages = SimpleNamespace(parse=self._parse)

    def _parse(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self._exc:
            raise self._exc
        return SimpleNamespace(parsed_output=self._parsed, stop_reason=self._stop)
