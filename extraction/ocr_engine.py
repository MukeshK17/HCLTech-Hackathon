"""PaddleOCR branch: image -> OCR lines (text, bbox, score) -> canonical fields.

PaddleOCR is an optional dependency (``pip install -e ".[ocr]"``) and is imported
lazily, so the rest of the package and the test suite work without it.
"""

from __future__ import annotations

import io
import logging
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from .schemas import ExtractionResult, failed_result

log = logging.getLogger(__name__)

ENGINE_NAME = "paddleocr"

# Input contract (from Member 1): a filesystem path, raw encoded image bytes
# (PNG/JPEG), or a decoded HxWx3 numpy array. Nothing is written to disk.
ImageInput = str | Path | bytes | Any


class OCRLine(BaseModel):
    text: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: list[float] | None = None  # [x_min, y_min, x_max, y_max]


class OCRResult(BaseModel):
    lines: list[OCRLine] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(line.text for line in self.lines)


def _to_bbox(poly: Any) -> list[float] | None:
    """Accept a 4-point polygon or [x1, y1, x2, y2]; return an axis-aligned box."""
    try:
        pts = poly.tolist() if hasattr(poly, "tolist") else list(poly)
        if len(pts) == 4 and all(isinstance(p, int | float) for p in pts):
            x1, y1, x2, y2 = (float(p) for p in pts)
            return [x1, y1, x2, y2]
        xs = [float(p[0]) for p in pts]
        ys = [float(p[1]) for p in pts]
        return [min(xs), min(ys), max(xs), max(ys)] if xs else None
    except (TypeError, ValueError, IndexError):
        return None


def _make_line(text: Any, score: Any, poly: Any, warnings: list[str]) -> OCRLine | None:
    if not isinstance(text, str) or not text.strip():
        return None
    try:
        conf = min(max(float(score), 0.0), 1.0)
    except (TypeError, ValueError):
        warnings.append("ocr_missing_score")
        conf = 0.0
    return OCRLine(text=text.strip(), confidence=conf, bbox=_to_bbox(poly) if poly is not None else None)


def parse_paddle_output(raw: Any) -> OCRResult:
    """Normalize PaddleOCR 3.x (``predict``) or 2.x (``ocr``) output.

    3.x: ``[{"rec_texts": [...], "rec_scores": [...], "rec_polys"|"rec_boxes": [...]}]``
    2.x: ``[[ [polygon, (text, score)], ... ]]`` (one list per page)
    Malformed entries are skipped with a warning instead of raising.
    """
    result = OCRResult()
    if not raw:
        result.warnings.append("ocr_empty")
        return result
    pages = raw if isinstance(raw, list) else [raw]
    for page in pages:
        if page is None:
            continue
        data = getattr(page, "json", None)
        if isinstance(data, dict) and "res" in data:  # 3.x result object
            page = data["res"]
        if hasattr(page, "get") and "rec_texts" in page:
            texts = list(page.get("rec_texts") or [])
            scores = list(page.get("rec_scores") or [])
            polys = page.get("rec_polys")
            if polys is None:
                polys = page.get("rec_boxes")
            polys = list(polys) if polys is not None else []
            for i, text in enumerate(texts):
                score = scores[i] if i < len(scores) else None
                poly = polys[i] if i < len(polys) else None
                line = _make_line(text, score, poly, result.warnings)
                if line:
                    result.lines.append(line)
        elif isinstance(page, list):
            for item in page:
                try:
                    poly, (text, score) = item[0], item[1]
                except (TypeError, ValueError, IndexError):
                    result.warnings.append("ocr_malformed_entry")
                    continue
                line = _make_line(text, score, poly, result.warnings)
                if line:
                    result.lines.append(line)
        else:
            result.warnings.append("ocr_unrecognized_output")
    if not result.lines and "ocr_empty" not in result.warnings:
        result.warnings.append("ocr_empty")
    result.warnings = list(dict.fromkeys(result.warnings))
    return result


_ENGINE: Any = None


def get_engine() -> Any:
    """Create (once) a PaddleOCR instance. Model weights are cached by PaddleOCR
    under the user's home directory on first use (no document data is cached)."""
    global _ENGINE
    if _ENGINE is None:
        from paddleocr import PaddleOCR  # optional heavy dependency

        _ENGINE = PaddleOCR(
            lang=os.getenv("OCR_LANG", "en"),
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            # oneDNN crashes in paddlepaddle 3.3 (NotImplementedError ConvertPirAttribute...).
            enable_mkldnn=os.getenv("OCR_ENABLE_MKLDNN", "0") == "1",
        )
    return _ENGINE


def _prepare_image(image: ImageInput) -> Any:
    if isinstance(image, str | Path):
        path = Path(image)
        if not path.is_file():
            raise FileNotFoundError(f"image not found: {path.name}")
        return str(path)
    if isinstance(image, bytes | bytearray):
        import numpy as np
        from PIL import Image

        with Image.open(io.BytesIO(image)) as img:
            return np.asarray(img.convert("RGB"))[:, :, ::-1].copy()  # RGB -> BGR in memory
    return image  # assume numpy array


def run_ocr(image: ImageInput, engine: Any = None) -> OCRResult:
    engine = engine or get_engine()
    prepared = _prepare_image(image)
    raw = engine.predict(prepared) if hasattr(engine, "predict") else engine.ocr(prepared)
    result = parse_paddle_output(raw)
    log.info("ocr finished: %d lines", len(result.lines))  # counts only, never text
    return result


def extract_with_ocr(
    image: ImageInput,
    *,
    engine: Any = None,
    document_type: str | None = None,
) -> ExtractionResult:
    """Public OCR-branch entry point. Never raises for OCR failures; errors are
    reported in ``ExtractionResult.errors``."""
    from .field_mapper import map_fields

    try:
        ocr = run_ocr(image, engine=engine)
    except ImportError:
        return failed_result("ocr", ENGINE_NAME, "ocr_unavailable", "paddleocr is not installed")
    except FileNotFoundError as exc:
        return failed_result("ocr", ENGINE_NAME, "invalid_input", str(exc))
    except Exception as exc:  # engine errors vary by PaddleOCR version
        log.warning("ocr failed: %s", type(exc).__name__)
        return failed_result("ocr", ENGINE_NAME, "ocr_failed", type(exc).__name__)
    return map_fields(ocr, document_type=document_type)
