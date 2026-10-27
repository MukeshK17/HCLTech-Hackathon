# Member 3: Rules Engine, Confidence Scoring, and Review Router

Subsystem documentation for the **Messy Document Extraction with Human Review** project.

---

## 1. Purpose and Scope

Real-world business receipts, invoices, and delivery notes suffer from skew, noise, low resolution, varying layouts, and irregular formats. Downstream enterprise workflows require structured, verified data without incurring the prohibitive cost of 100% manual review or risking silent errors from unverified AI/OCR outputs.

This subsystem (**Member 3**) acts as the deterministic verification layer, reliability scoring mechanism, and intelligent review router between:
- **Upstream extraction** (Member 2's OCR and Vision-Language Models)
- **Downstream human review & export** (Member 4's review interface and persistence)

---

## 2. Member 3 Responsibilities

- **Input Normalization**: Ingest varied upstream representations (dictionaries, JSON strings, partial outputs) and adapt them to canonical fields while preserving raw inputs for evidence traceability.
- **Deterministic Validation**: Rigorous schema, format, calendar date, currency, and monetary Decimal arithmetic validation.
- **Cross-Field Arithmetic**: Total amount reconciliation (`expected = subtotal + tax - discount + rounding`) and tax calculation verification.
- **OCR–AI Consistency Checks**: Conservative multi-branch comparison across text, dates, currencies, and numbers.
- **Field-Level Confidence Scoring**: Transparent, explainable heuristic reliability scoring decomposed into OCR confidence, OCR-AI agreement, and deterministic validation.
- **Intelligent Routing**: Conservative routing into `auto_accept`, `human_review`, `retry_extraction`, or `reject` with critical failure overrides.
- **Human Correction Revalidation**: Re-running deterministic rules and cross-field arithmetic on reviewer corrections without recalculating OCR confidence.
- **Offline Evaluation**: Tooling to evaluate accuracy, auto-acceptance error rates, coverage, and calibration against labeled benchmark datasets (e.g. SROIE).

---

## 3. Directory Structure

All files for Member 3 reside strictly within `rules_engine/`:

```text
rules_engine/
├── __init__.py           # Public integration entry points and exports
├── schemas.py            # Canonical data classes, enums, and normalization adapter
├── config.py             # Configurable weights, thresholds, currency, and schemas
├── validator.py          # Pure deterministic validation rules and cross-field checks
├── confidence.py         # Field-level heuristic reliability score calculation
├── router.py             # Field- and document-level routing and explainability
├── evaluation.py         # Offline ground-truth evaluation and calibration metrics
├── requirements.txt      # Subsystem dependencies
├── README.md             # Subsystem architecture, contract, and usage guide
└── tests/
    ├── __init__.py
    ├── test_schemas.py
    ├── test_validator.py
    ├── test_confidence.py
    ├── test_router.py
    ├── test_evaluation.py
    └── test_integration.py
```

---

## 4. Input Contract

Member 3 expects three inputs from Member 2 (supplied as Python dictionaries or JSON strings):

```python
ocr_json: dict[str, Any] | str | None
ai_json: dict[str, Any] | str | None
ocr_confidence: dict[str, float] | list[Any] | float | str | None
```

### Upstream Interface Status
- Member 2's concrete schemas are unfinalized in the repository.
- Therefore, `rules_engine/schemas.py` provides a fault-tolerant adapter (`normalize_extraction_inputs`) supporting alias mapping, format detection, and segment-level recognition.

---

## 5. Actual and Illustrative Input Schemas

### 5.1 Illustrative Upstream Examples (Unconfirmed)

**`ocr_json`** (may be flat fields or line segments):
```json
{
  "company": "Tesco Stores (Malaysia) Sdn Bhd",
  "date": "25/12/2023",
  "sub_total": "120.00",
  "gst": "7.20",
  "total": "127.20"
}
```

**`ai_json`** (structured output from VLM):
```json
{
  "vendor_name": "Tesco Stores Malaysia",
  "date": "2023-12-25",
  "subtotal": "120.00",
  "tax_amount": "7.20",
  "tax_rate": "6.0",
  "total_amount": "127.20",
  "currency": "MYR"
}
```

**`ocr_confidence`**:
```json
{
  "vendor_name": 0.94,
  "date": 0.96,
  "subtotal": 0.92,
  "tax_amount": 0.91,
  "total_amount": 0.97
}
```

---

## 6. Normalization and Missing-Evidence Behavior

1. **Missing OCR or AI Output**:
   - If `ocr_json` is missing, fields extracted by AI are tracked as `is_present_in_ocr=False`.
   - If `ai_json` is missing, fields extracted by OCR are tracked as `is_present_in_ai=False`.
   - Agreement status is explicitly marked `OCR_MISSING` or `AI_MISSING` with agreement score `None` (never `1.0`).
2. **Missing OCR Confidence**:
   - Missing confidence is recorded as `None`, **never** defaulted to `1.0`.
   - A genuine confidence of `0.0` is preserved as `0.0`.
   - Scaled confidences (e.g. `95.0` on a 0–100 scale) are normalized to `0.95`.
3. **Segment-Level OCR Confidence**:
   - If Member 2 provides bounding box or token-level confidence lists without an unambiguous field-to-segment mapping, field-level OCR confidence is marked `None` with note: `"Segment-level OCR confidence provided without reliable field mapping"`.
   - The subsystem does **not** fabricate field associations.

---

## 7. Supported Canonical Fields

Standard fields supported by the engine:
- `vendor_name`
- `date`
- `address`
- `currency`
- `subtotal`
- `tax_amount`
- `tax_rate`
- `discount_amount`
- `rounding_adjustment`
- `total_amount`

Additional upstream fields are preserved in `additional_fields` for downstream consumers.

Configurable required fields by document type:
- **`receipt`**: `vendor_name`, `date`, `total_amount`
- **`invoice`**: `vendor_name`, `date`, `total_amount` (optional: `invoice_number`, `subtotal`, `tax_amount`, `address`, `currency`)
- **`delivery_note`**: `vendor_name`, `date`
- **`generic`**: `vendor_name`, `date`, `total_amount`

---

## 8. Deterministic Validation Rules

1. **Vendor Name & Address**:
   - Removal of leading/trailing whitespace and collapse of repeated internal whitespace.
   - Required-field presence check.
2. **Date Validation**:
   - Supported formats: ISO (`YYYY-MM-DD`), slash/dash (`DD/MM/YYYY`, `MM/DD/YYYY`), month-name formats (`12-Jan-2024`, `Jan 12, 2024`).
   - Normalization to ISO `YYYY-MM-DD`.
   - Calendar date verification (rejects invalid dates like Feb 30 or month 13).
   - Ambiguous date detection: When both day and month $\le 12$ and day $\ne$ month (e.g. `03/04/2026`), if `preferred_date_order` is not configured, the date is flagged with a **critical failure override** to force human review.
3. **Currency Validation**:
   - Normalized to uppercase 3-letter ISO code (`USD`, `EUR`, `MYR`, `SGD`, `GBP`, etc.).
   - Ambiguous currency symbols (e.g. `$` which could represent USD, SGD, AUD, or CAD) trigger a critical warning/review override unless an unambiguous context or default currency is established.
4. **Monetary Decimal Arithmetic**:
   - All amounts are parsed into Python's `decimal.Decimal`—**never binary floating-point**.
   - Handles comma thousands separators (`1,180.00`) and European decimal commas (`1.180,00`).
   - Rejects malformed values (e.g. `12.3.4`).
   - Negative amounts are rejected for standard receipts/invoices unless explicitly allowed by configuration.

---

## 9. Tax and Total Reconciliation

### Total Reconciliation Baseline Formula
$$\text{expected\_total} = \text{subtotal} + \text{tax\_amount} - \text{discount\_amount} + \text{rounding\_adjustment}$$

- Tolerance: Configurable (default `Decimal("0.05")`).
- A material mismatch (`difference > tolerance`) is marked as a **critical validation failure**, forcing human review regardless of how high the confidence score is.
- If `subtotal` or `total_amount` is absent, status is marked `NOT_CHECKABLE` (never passed).

### Tax Verification Formula
$$\text{expected\_tax} = \frac{\text{subtotal} \times \text{tax\_rate}}{100}$$
- Verified when `subtotal`, `tax_rate`, and `tax_amount` are all present.
- If missing, status is `NOT_CHECKABLE`. The engine never invents tax amounts or rates.

---

## 10. Confidence Scoring Formula & Weights

Field reliability is calculated using a transparent heuristic:

$$C_f = w_O \cdot O + w_A \cdot A + w_V \cdot V$$

Default weights:
- $w_O = 0.4$ (OCR confidence normalized to $[0.0, 1.0]$)
- $w_A = 0.3$ (OCR–AI Agreement: $1.0$ if match, $0.0$ if mismatch, $None$ if either missing)
- $w_V = 0.3$ (Deterministic Validation: $1.0$ for `VALID`, $0.5$ for `WARNING`, $0.0$ for `INVALID`/critical failure, $None$ for `NOT_CHECKABLE`/`MISSING`)

### Weight Renormalization & Evidence Sufficiency
- When components are missing, available weights are renormalized:
  $$w'_i = \frac{w_i}{\sum_{k \in \text{avail}} w_k}$$
- **Sufficiency Guardrail**: If fewer than `min_components_for_sufficiency` (default 2) components are available, `evidence_sufficient` is set to `False`. Insufficient evidence strictly prohibits automatic acceptance.
- If all components are missing, $C_f = None$.

> [!IMPORTANT]
> **Heuristic Reliability Score vs. Calibrated Probability**:
> This score is a heuristic index of evidence consistency and validation success. A score of `0.95` does **not** represent an empirically calibrated 95% Bayesian probability of correctness unless offline calibration has been performed against held-out ground truth.

---

## 11. Review Router & Critical Overrides

Thresholds:
- $C_f \ge 0.90$: Eligible for `auto_accept` **only if** all checks pass and evidence is sufficient.
- $0.70 \le C_f < 0.90$: `human_review`.
- $C_f < 0.70$: `human_review` (or `retry_extraction` if upstream quality warning applies).
- $C_f = None$: `human_review`.

### Critical Overrides (Forced Human Review)
Even if $C_f = 1.0$, the decision is overridden to `human_review` if any of the following occur:
1. Total reconciliation mismatch.
2. OCR–AI disagreement on a required field.
3. Ambiguous date format.
4. Ambiguous or conflicting currency evidence.
5. Malformed monetary value or missing required field.
6. Insufficient evidence (`evidence_sufficient = False`).

### Extraction Retry
The router returns `retry_extraction` when upstream quality warnings contain keywords like `"severe blur"`, `"severe skew"`, or `"empty ocr"`, provided `retry_count < max_retries` (default 2). Once the retry limit is reached, routing falls back to `human_review`.

---

## 12. Human Corrections and Revalidation

When a human reviewer modifies values in Member 4's interface:
```python
revalidate_corrected_fields(
    original_document=doc_result,
    corrected_fields={"total_amount": "220.00", "vendor_name": "Acme Corp"},
    config=custom_config,
)
```

1. **Preserves Provenance**: Retains original `ocr_value` and `ai_value`.
2. **Re-validates Corrected Values**: Human inputs are strictly validated (e.g. malformed dates or numbers are still rejected).
3. **Re-runs Cross-Field Checks**: Total reconciliation and tax checks are re-evaluated with the corrected amounts.
4. **No OCR Confidence Recalculation**: Human-verified fields receive a validation score of `1.0` if deterministic checks pass, leaving OCR metrics intact.

---

## 13. Public Integration Interface for Member 4

```python
from rules_engine import process_extraction, revalidate_corrected_fields, RulesEngineConfig

# 1. Process extraction results from Member 2
doc_result = process_extraction(
    ocr_json=ocr_output,
    ai_json=ai_output,
    ocr_confidence=ocr_conf,
    document_type="receipt",
    quality_warnings=["Possible camera tilt"],
)

# 2. Check document decision
print(doc_result.document_decision.value)  # 'auto_accepted' | 'human_review' | 'retry_extraction'
print(doc_result.decision_reasons)

# 3. Inspect individual fields
for field_name, f_res in doc_result.fields.items():
    print(f"{field_name}: {f_res.normalized_value} -> {f_res.decision.value} (Score: {f_res.confidence_score})")
    if f_res.decision_reasons:
        print(f"  Reasons: {f_res.decision_reasons}")

# 4. Convert to JSON-serializable dictionary for UI rendering
payload = doc_result.to_dict()

# 5. Revalidate after human reviewer updates a field
if doc_result.document_decision.value == "human_review":
    corrected = {"total_amount": "125.00"}
    revalidated_doc = revalidate_corrected_fields(doc_result, corrected)
```

---

## 14. Installation and Testing

### Install Subsystem Dependencies
```bash
pip install -r rules_engine/requirements.txt
```

### Run Tests
```bash
# Standalone execution
pytest -o addopts="" rules_engine/tests -v

# Code formatting and linting
ruff check rules_engine
ruff format --check rules_engine
```

All 54 tests run completely offline without requiring live PaddleOCR, Claude API keys, or paid services.

---

## 15. Offline Ground-Truth Evaluation Utility

> [!NOTE]
> **Ground-truth labels are for offline evaluation only and are NOT required to process new receipts at runtime.**

To evaluate extraction and routing accuracy on labeled datasets (such as SROIE or internal benchmarks):

```python
from rules_engine.evaluation import evaluate_predictions

predictions = [doc_result_1, doc_result_2]
ground_truth = [
    {"vendor_name": "Tesco Stores", "date": "2023-12-25", "total_amount": "127.20"},
    {"vendor_name": "Speedway", "date": "2024-01-10", "total_amount": "45.00"},
]

report = evaluate_predictions(predictions, ground_truth)

print(f"Overall Accuracy: {report.overall_accuracy:.2%}")
print(f"Auto-Accept Error Rate: {report.auto_accepted_fields_error_rate:.2%}")
print(f"Auto-Accept Coverage: {report.overall_auto_acceptance_coverage:.2%}")
print(f"Human Review Rate: {report.overall_human_review_rate:.2%}")

if report.calibration.is_statistically_sufficient:
    print(f"Expected Calibration Error (ECE): {report.calibration.expected_calibration_error:.4f}")
    print(f"Brier Score: {report.calibration.brier_score:.4f}")
```

---

## 16. Security, Reliability, and Known Limitations

- **Untrusted Input**: Extracted OCR text and VLM outputs are treated as untrusted strings. Values are never passed to `eval()` or executed as code.
- **Privacy**: Logging avoids printing sensitive personal identifiable information or full document payloads.
- **Ambiguous Dates**: Without explicit locale or document date order hints (`preferred_date_order="DMY"` or `"MDY"`), dates with ambiguous month/day are routed to human review.
- **Tax Conventions**: Tax-inclusive vs. tax-exclusive pricing varies across jurisdictions. When tax rate is omitted, tax verification is marked `not_checkable`.
- **Segment OCR Mapping**: PaddleOCR line segments must be associated with specific fields by Member 2; without a mapping, OCR confidence remains `None` to prevent spurious scores.
