# Member 2 — OCR and VLM Extraction

Two independent branches read the **same image** and return the **same contract**:

| Branch | Module | Engine | Image leaves machine? |
|---|---|---|---|
| OCR | `extraction/ocr_engine.py` + `extraction/field_mapper.py` | PaddleOCR (local) | **No** |
| VLM | `extraction/ai_extractor.py` | Anthropic Claude Messages API (`VLM_MODEL`, default `claude-sonnet-5-5`) | **Yes** — uploaded to Anthropic |

Nothing here validates, compares, routes or stores results. That is Member 3 (rules/routing) and Member 4 (review/export).

## 1. Public interface

```python
from extraction import extract_with_ocr, extract_with_vlm, extract_both

ocr = extract_with_ocr(image, document_type=None)  # ExtractionResult
vlm = extract_with_vlm(image_or_pdf, document_type=None)
both = extract_both(image)  # {"ocr": ..., "vlm": ...}, run independently, no comparison
```

**Input (from Member 1):** a file path (`str`/`pathlib.Path`) or the file's bytes.

- **OCR branch:** PNG/JPEG, or a decoded numpy array.
- **VLM branch:** PNG/JPEG/GIF/WebP up to 5 MB, or a PDF up to 30 MB.

These functions do not raise on OCR or API failures; failures are reported in `result.errors`. If Member 1 already knows the type, pass `document_type` and it is recorded with `confidence_method="caller_provided"`.

## 2. Output format (`extraction/schemas.py`)

**Values are copied exactly as printed.** Nothing is reformatted:

- A date stays `30/03/2018`, `03-30-2018` or `2018-03-30`, as printed.
- Amounts keep their format (`1,234.50`, `1.234,56`).
- Currency is the symbol or code as printed (`RM`, `$`, `MYR`).
- Tax rate keeps the percent sign (`6%`).

Every value is a string, or `null` when it is absent. Converting values (ISO dates, `Decimal` amounts) is left to whoever consumes them (Member 3).

```jsonc
{
  "schema_version": "1.0",
  "source": "ocr",                 // "ocr" | "vlm"
  "engine": "paddleocr",           // or the model id
  "fields": {                      // ALWAYS has every field below
    "total": {
      "value": "189.75",           // as printed
      "confidence": 0.79,          // 0..1, see §3
      "confidence_method": "ocr_heuristic",
      "source": "ocr",
      "evidence": {"text": "TOTAL (MYR) | 189.75", "bbox": [10, 220, 200, 235]},
      "candidates": [],            // other possible values (e.g. a second TOTAL line)
      "warnings": []               // not_found, ambiguous, low_confidence, low_ocr_confidence, invalid_type...
    }
  },
  "extra_fields": {},              // fields with no slot below (kept, not dropped)
  "raw_text": "...",               // OCR only; in memory, the CLI hides it unless --include-raw-text
  "warnings": [], "errors": [{"code": "vlm_timeout", "message": "..."}]
}
```

`result.record()` returns the values only:

`document_type, vendor, address, invoice_number, order_number, date, currency, subtotal, tax_rate, tax, rounding_adjustment, total`

- A missing value is always `null`, with confidence `0.0`.
- `document_type` is a label we assign (`receipt | invoice | delivery_note | form`), not text copied from the document.
- Field-name aliases such as `company` or `merchant` → `vendor` are listed in `patterns.ALIASES`.

**Error codes:** `invalid_input`, `ocr_unavailable`, `ocr_failed`, `vlm_not_configured`, `vlm_timeout`, `vlm_api_error`, `vlm_refused`, `vlm_invalid_response`.

**Handing results to Member 3** (`rules_engine/`):

```python
from extraction import extract_both
from extraction.handoff import to_rules_engine_inputs
from rules_engine import process_extraction

results = extract_both(image)
decision = process_extraction(**to_rules_engine_inputs(results["ocr"], results["vlm"]))
```

`to_rules_engine_inputs` only renames fields to the rules engine's names (`vendor` → `vendor_name`, `tax` → `tax_amount`, `total` → `total_amount`). Values are passed exactly as printed; the rules engine converts dates and amounts itself. A failed branch is passed as `None`. VLM confidence isn't passed, because `process_extraction` only takes OCR confidence.

## 3. Confidence

| Method | Meaning |
|---|---|
| `ocr_heuristic` | PaddleOCR recognition score (the lower of the label line and value line) × label weight (e.g. `GRAND TOTAL` 1.0, `TOTAL` 0.85) × pairing factor (same line 1.0, same row by bbox 0.95, next line 0.85) × 0.6 if a different value scores ≥ 80% of the best. Vendor and address use fixed layout heuristics, capped at 0.9 and 0.6× respectively. |
| `model_self_report` | The model's own 0–1 estimate. **Not calibrated**; it is the model's guess about its reading. |
| `vlm_fallback_heuristic` | The model gave no confidence, or an out-of-range one: 0.5 with evidence text, 0.3 without. |
| `caller_provided` | The value came from the caller: 1.0. |
| `none` | No value: 0.0. |

All of these are ranking scores, not accuracy measurements. Check them against labelled samples before choosing router thresholds.

## 4. Field mapping (OCR), and how varied layouts are handled

`patterns.py` regexes only **find** values; `field_mapper.py` decides which value belongs to which field. For each field:

1. **Find a label:** for example `GRAND TOTAL`, `SUB TOTAL`, `GST`, `ROUNDING`, `DATE`, `INVOICE NO`.
2. **Take the value next to it:** the value can be on the same line, on the same row (using the OCR bounding boxes, so a label on the left and an amount on the right still match), or on the next line.
3. **Skip misleading lines:** `SUB TOTAL` doesn't count for total, and `TAX INVOICE` or `GST ID` don't count for tax. A `DUE DATE` or `DELIVERY DATE` scores below a plain `DATE`. The largest number is never assumed to be the total.
4. **Keep the alternatives:** every candidate goes into `candidates`, and close competitors set the `ambiguous` warning.

Vendor is the first text-heavy line near the top, with a bonus for company suffixes (`SDN BHD`, `LTD`, …). Address is the lines just below the vendor that contain address hints. Document type comes from keywords.

This handles common receipt and invoice layouts, but rules can't cover every layout. Unusual labels or table-style invoices will come back empty or `ambiguous`, which is why the VLM branch exists alongside it.

## 5. VLM: where the LLM is used, and which model

The LLM is used **only** in `extraction/ai_extractor.py`; the OCR branch uses no LLM. The VLM is Anthropic Claude, called through the `anthropic` Python SDK. There is **no training or fine-tuning**: the model reads each document fresh with a fixed prompt.

| Setting | Value |
|---|---|
| Model (env `VLM_MODEL`) | `claude-opus-5-5` (default, most accurate). `claude-sonnet-5-5` costs about half. `claude-haiku-5-5` is cheapest, for simple receipts. |
| Price per 1M tokens (input / output) | Opus 5.5: $4 / $20. Sonnet 5.5: $2 / $10. Haiku 5.5: $0.10 / $0.50 |
| Image size | Up to 5 MB per image. Images up to 2576 px on the long edge are read at full detail and cost up to about 4,800 input tokens; larger images are scaled down. |
| PDF | Sent directly, up to 30 MB here (API request limit is 32 MB), with each page read as an image |
| Rough cost | About $0.03–0.06 per receipt on Opus 5.5 (image tokens plus the JSON answer and thinking). The team's 60 samples cost roughly $2–4 on Opus, or about half that on Sonnet. |
| Structured output | `client.messages.parse(output_format=VLMResponse)` makes Claude return JSON in our schema, and every field is checked again in code. (Forced tool calls are rejected by current models.) |
| Effort (env `VLM_EFFORT`) | `medium` by default; `low` is cheaper and faster. Thinking can't be turned off on Opus 5.5. |
| Env vars | `ANTHROPIC_API_KEY` (required), `VLM_MODEL`, `VLM_EFFORT`, `VLM_TIMEOUT_SECONDS` (default 120; the SDK retries twice) |
| Refusals | A declined request is returned as error `vlm_refused`. Anthropic's automatic fallback-to-another-model option is **not** enabled, to keep the call simple. |

**Privacy:** the **full image or PDF is uploaded to Anthropic**. Check your organisation's Anthropic data-retention terms before sending real financial documents. For documents that must stay local, use the OCR branch only.

**Dataset labels (ground truth):** we are not training anything, so labels are never needed to *run* extraction. They are useful only to *measure* how often each branch reads a field correctly, for example to pick confidence thresholds for Member 3. That scoring code is parked on the local branch `wip/evaluation` and is not part of this PR.

## 6. Data lifecycle

| # | Stage | In memory | Temp files | Persistent | External |
|---|---|---|---|---|---|
| 1 | Image from Member 1 | path / bytes / ndarray | none created by us | owned by Member 1 | no |
| 2 | Into PaddleOCR | path or ndarray (bytes are decoded in memory) | none | PaddleOCR model **weights** cached in `~/.paddlex` (no document data) | no; weights download once on first run |
| 3 | Raw OCR text, boxes, scores | `OCRResult` | none | no | no |
| 4 | OCR candidates | `FieldResult.candidates` | none | no | no |
| 5 | Image/PDF to VLM | base64 string | none | at Anthropic, according to its retention policy | **yes, Anthropic API** |
| 6 | VLM response | tool-input dict → `ExtractionResult` | none | no | received from Anthropic |
| 7–8 | Normalized JSON, confidence, evidence | `ExtractionResult` | none | no | no |
| 9 | Handed to Member 3 | returned Python object | none | Member 3/4 decide | no |
| 10 | Logs / artifacts | — | none | logs hold **counts and error class names only**, never document text | no |

Everything can be discarded when the caller drops the result. CI never handles real documents or keys.

## 7. Setup and commands

| Step | Linux (bash) | Windows (PowerShell) |
|---|---|---|
| venv | `python3 -m venv .venv && source .venv/bin/activate` | `py -3.12 -m venv .venv; .\.venv\Scripts\Activate.ps1` |
| dev install | `pip install -e ".[dev]"` | same |
| OCR install (optional, ~1 GB) | `pip install -e ".[ocr]"` | same |
| API key | `cp .env.example .env` then edit, or `export ANTHROPIC_API_KEY=...` | `Copy-Item .env.example .env` or `$env:ANTHROPIC_API_KEY="..."` |
| lint / format | `ruff check . && ruff format --check .` | `ruff check .; ruff format --check .` |
| types | `mypy` | same |
| tests + coverage | `pytest -m "not integration"` | same |
| real PaddleOCR smoke test | `pytest -m integration` | same |
| security | `bandit -c pyproject.toml -r extraction && pip-audit --skip-editable` | `bandit -c pyproject.toml -r extraction; pip-audit --skip-editable` |
| pre-commit | `pre-commit install && pre-commit run --all-files` | same |
| OCR branch | `python -m extraction data/samples/r1.jpg --branch ocr` | `python -m extraction data\samples\r1.jpg --branch ocr` |
| VLM branch | `python -m extraction data/samples/r1.jpg --branch vlm` | same |

Windows notes: if PowerShell blocks activation, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`. PaddlePaddle publishes CPU wheels for Windows x64 on Python 3.9–3.13. If `pip install paddlepaddle` fails, use the official index (`pip install paddlepaddle -i https://www.paddlepaddle.org.cn/packages/stable/cpu/`) or run the OCR branch on Linux/WSL2. The VLM branch and all unit tests work without Paddle.

## 8. CI

- `.github/workflows/ci.yml`: Ruff lint and format, mypy, pytest with coverage, pre-commit.
- `.github/workflows/security.yml`: pip-audit and Bandit, on push, on PR, and weekly.

Both workflows use read-only permissions, pip caching and SHA-pinned actions, and **do not install PaddleOCR or call the API**. There is no Dockerfile, so there is no Docker build job; add one if a Dockerfile is introduced. No coverage threshold is enforced.

## 9. Limitations

- The OCR heuristics are tuned for English/Malay receipts (SROIE style). Invoices with tables, multi-page PDFs (rasterising them is Member 1's job) and handwriting will need more label patterns.
- Vendor and address detection is positional; logos without text, or a header line missed by OCR, will produce the wrong vendor.
- Confidence scores are uncalibrated.
- The VLM can hallucinate. Evidence text is self-reported and not checked against pixels.
- Real PaddleOCR inference and the live Anthropic API have **not** been exercised by the automated tests.
