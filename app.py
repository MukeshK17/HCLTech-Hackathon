"""
app.py - Streamlit frontend for Messy Document Extraction
=========================================================
This is the main application file for the Messy Document Extraction tool.

Features
--------
* Upload PDF / PNG / JPG documents with validation (size, type, corruption,
  password-protected PDFs, page cap); rejected files are listed clearly
* Exact-duplicate detection via SHA-256
* Image cleanup: orientation (auto via Tesseract OSD, or manual), noise
  reduction, deskew, resize - each toggleable, each reported
* Before / after previews, quality metrics, PDF page previews
* Hand-off to the OCR module through ``PreparedDocument.ocr_payload()``,
  published in ``st.session_state["ocr_payloads"]``
* Review contract for the dashboard: ``document_bytes``, ``document_type``,
  ``document_id``, ``review_run_id``, ``extraction_result``
"""

from __future__ import annotations

import html
import time
import uuid

import pandas as pd
import streamlit as st

from preprocess import (
    ALLOWED_EXTENSIONS,
    DEFAULT_MAX_SIZE_MB,
    MAX_PDF_PAGES,
    PreparedDocument,
    process_upload,
    render_pdf_preview,
)


def publish_extraction_result(
    document_id: str,
    validated_fields: dict,
) -> None:
    """Publish validated OCR output for the human-review dashboard."""
    st.session_state["extraction_result"] = {
        "document_id": document_id,
        "fields": validated_fields,
    }


def stretch(fn, *args, **kwargs):
    """Call a Streamlit widget at full width on old and new Streamlit versions."""
    try:
        return fn(*args, width="stretch", **kwargs)
    except Exception:
        return fn(*args, use_container_width=True, **kwargs)


# ---------------------------------------------------------------------------
# Page config and styling (uses Streamlit theme variables: light + dark mode)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="DocExtract - Messy Document Extraction",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.6rem; max-width: 1250px; }
    div[data-testid="stMetric"] {
        background: var(--secondary-background-color);
        border-radius: 10px; padding: 12px 16px;
    }
    .hero { padding: 6px 0 2px 0; }
    .hero h1 { margin-bottom: 0; }
    .hero p { opacity: 0.75; margin-top: 4px; }

    .steps { display: flex; gap: 8px; margin: 10px 0 18px 0; flex-wrap: wrap; }
    .step { flex: 1; min-width: 150px; padding: 10px 14px; border-radius: 10px;
            background: var(--secondary-background-color);
            border: 1px solid rgba(128,128,128,0.25); font-size: 0.9rem; }
    .step b { display: block; font-size: 0.78rem; opacity: 0.65;
              text-transform: uppercase; letter-spacing: 0.04em; }
    .step.done { border-color: rgba(40,167,69,0.7); background: rgba(40,167,69,0.10); }
    .step.active { border-color: var(--primary-color); }

    .chip { display: inline-block; padding: 3px 10px; border-radius: 999px;
            font-size: 0.8rem; margin: 0 6px 6px 0;
            border: 1px solid rgba(128,128,128,0.35); }
    .chip.ok   { background: rgba(40,167,69,0.14);  border-color: rgba(40,167,69,0.6); }
    .chip.warn { background: rgba(255,193,7,0.18);  border-color: rgba(255,193,7,0.7); }
    .chip.dup  { background: rgba(220,53,69,0.12);  border-color: rgba(220,53,69,0.6); }
    .chip.info { background: rgba(23,162,184,0.12); border-color: rgba(23,162,184,0.6); }
    .note { font-size: 0.88rem; opacity: 0.85; margin: 2px 0; }
    </style>
    """,
    unsafe_allow_html=True,
)


def chip(text: str, kind: str = "info") -> str:
    return f'<span class="chip {kind}">{html.escape(text)}</span>'


# ---------------------------------------------------------------------------
# Sidebar - grouped settings
# ---------------------------------------------------------------------------
ROTATION_CHOICES = {"Off (auto-detect)": 0, "90° clockwise": 90, "180°": 180, "270° clockwise": 270}

with st.sidebar:
    st.header("⚙️ Settings")

    with st.expander("📁 File limits", expanded=False):
        max_size = st.number_input(
            "Max file size (MB)", min_value=1, max_value=100,
            value=DEFAULT_MAX_SIZE_MB, step=1,
        )
        st.caption(f"Types: {', '.join(sorted(ALLOWED_EXTENSIONS))} · PDFs up to {MAX_PDF_PAGES} pages")

    with st.expander("🖼️ Image cleanup", expanded=True):
        long_edge = st.slider(
            "Resize long edge (px)", min_value=640, max_value=3840, value=1920, step=64,
        )
        auto_orient = st.checkbox(
            "Auto-fix page orientation", value=True,
            help="Uses Tesseract OSD to fix 90/180/270° rotation. "
                 "If it is unavailable or unsure, nothing is rotated.",
        )
        rotation_label = st.selectbox(
            "Manual rotation override", list(ROTATION_CHOICES),
            help="Use when a page is still sideways or upside down. Overrides auto-orientation.",
        )
        manual_rotation = ROTATION_CHOICES[rotation_label]
        do_denoise = st.checkbox(
            "Reduce noise", value=True,
            help="Only applied when noise is detected, so clean scans are left untouched.",
        )
        do_deskew = st.checkbox("Straighten slight tilt (deskew)", value=True)

    st.divider()
    if st.button("🗑️ Clear processed cache"):
        for key in ("processed_cache", "review_runs", "timings"):
            st.session_state[key] = {}
        st.cache_data.clear()
        st.rerun()

# ---------------------------------------------------------------------------
# Header and upload
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
      <h1>📄 DocExtract</h1>
      <p>Messy documents in, structured fields out. Only the uncertain fields go to a human.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

uploaded_files = st.file_uploader(
    "Upload document(s)",
    type=["pdf", "png", "jpg", "jpeg"],
    accept_multiple_files=True,
    help="Supported: PDF, PNG, JPG/JPEG. Scans, photos and PDFs all work.",
)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
for key in ("processed_cache", "document_ids", "review_runs", "timings"):
    st.session_state.setdefault(key, {})

extraction_result = st.session_state.get("extraction_result") or {"fields": {}}
has_fields = bool(extraction_result.get("fields"))


def render_stepper(uploaded: bool, processed: bool) -> None:
    steps = [
        ("Step 1", "Upload", "done" if uploaded else "active"),
        ("Step 2", "Preprocess", "done" if processed else ("active" if uploaded else "")),
        ("Step 3", "OCR + extract", "done" if has_fields else ""),
        ("Step 4", "Validate", "done" if has_fields else ""),
        ("Step 5", "Human review", "active" if has_fields else ""),
    ]
    boxes = "".join(
        f'<div class="step {state}"><b>{html.escape(a)}</b>{html.escape(b)}</div>'
        for a, b, state in steps
    )
    st.markdown(f'<div class="steps">{boxes}</div>', unsafe_allow_html=True)


if not uploaded_files:
    render_stepper(uploaded=False, processed=False)
    st.info("👆 Upload a document to get started.")
    st.stop()

# ---------------------------------------------------------------------------
# Process uploads (cached per file AND per setting, so toggles take effect)
# ---------------------------------------------------------------------------
results: list[tuple[str, PreparedDocument, str]] = []  # (label, document, cache_key)
rejected: list[tuple[str, str]] = []

for idx, uf in enumerate(uploaded_files):
    label = f"{idx + 1}. {uf.name}"
    cache_key = (
        f"{uf.name}_{uf.size}_{getattr(uf, 'file_id', '')}_{max_size}_{long_edge}_"
        f"{do_deskew}_{do_denoise}_{auto_orient}_{manual_rotation}"
    )

    if cache_key in st.session_state.processed_cache:
        results.append((label, st.session_state.processed_cache[cache_key], cache_key))
        continue

    try:
        t0 = time.perf_counter()
        prepared = process_upload(
            uf.getvalue(),
            uf.name,
            max_size_mb=max_size,
            long_edge=long_edge,
            deskew=do_deskew,
            denoise=do_denoise,
            auto_orient=auto_orient,
            manual_rotation=manual_rotation,
        )
        elapsed = time.perf_counter() - t0
        document_id = st.session_state.document_ids.setdefault(
            prepared.sha256, str(uuid.uuid4())
        )
        st.session_state.review_runs[cache_key] = {
            "document_id": document_id,
            "review_run_id": str(uuid.uuid4()),
        }
        st.session_state.timings[cache_key] = elapsed
        st.session_state.processed_cache[cache_key] = prepared
        results.append((label, prepared, cache_key))
        st.toast(f"{uf.name} processed in {elapsed:.2f}s", icon="✅")
    except ValueError as exc:  # expected rejections: corrupt, protected, too big...
        rejected.append((uf.name, str(exc)))
    except Exception as exc:  # unexpected: never crash the whole page
        rejected.append((uf.name, f"Unexpected error while processing ({exc})."))

render_stepper(uploaded=True, processed=bool(results))

if rejected:
    with st.expander(f"⛔ {len(rejected)} file(s) rejected", expanded=True):
        for name, message in rejected:
            st.error(f"**{name}**: {message}")

if not results:
    st.warning("No valid documents to display.")
    st.stop()

# ---------------------------------------------------------------------------
# Exact-duplicate detection (first occurrence wins)
# ---------------------------------------------------------------------------
first_seen: dict[str, str] = {}
duplicates: dict[str, str] = {}  # label -> label of first copy
for label, prepared, _ in results:
    original_label = first_seen.setdefault(prepared.sha256, label)
    if original_label != label:
        duplicates[label] = original_label

# Publish the hand-off for the OCR module (skip exact duplicates).
st.session_state["ocr_payloads"] = [
    prepared.ocr_payload() for label, prepared, _ in results if label not in duplicates
]

# Publish the first unique upload as the single-document review contract.
# The dashboard reuses these bytes; it must not ask the user to upload again.
primary = next(((l, p, k) for l, p, k in results if l not in duplicates), None)
if primary is not None:
    _, primary_document, primary_key = primary
    primary_run = st.session_state.review_runs[primary_key]
    st.session_state["document_bytes"] = primary_document.original_bytes
    st.session_state["document_type"] = (
        "application/pdf"
        if primary_document.is_pdf
        else ("image/jpeg" if primary_document.extension in {".jpg", ".jpeg"} else "image/png")
    )
    st.session_state["document_id"] = primary_run["document_id"]
    st.session_state["review_run_id"] = primary_run["review_run_id"]
    existing = st.session_state.get("extraction_result")
    if not isinstance(existing, dict) or existing.get("document_id") != primary_run["document_id"]:
        publish_extraction_result(primary_run["document_id"], {})
        extraction_result = st.session_state["extraction_result"]
        has_fields = bool(extraction_result["fields"])


# ---------------------------------------------------------------------------
# PDF preview helper (cached; display only, OCR still gets the original bytes)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Rendering PDF preview...")
def _pdf_preview_images(pdf_bytes: bytes):
    return render_pdf_preview(pdf_bytes, dpi=110)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
n_unique = len(results) - len(duplicates)
n_warn = sum(1 for _, p, _ in results if p.warnings)
total_pages = sum(p.page_count for _, p, _ in results)

summary = st.columns(4)
summary[0].metric("Documents accepted", len(results))
summary[1].metric("Unique (sent to OCR)", n_unique)
summary[2].metric("With quality warnings", n_warn)
summary[3].metric("Total pages", total_pages)

# ---------------------------------------------------------------------------
# Per-document cards
# ---------------------------------------------------------------------------
st.subheader("🖼️ Documents")

for position, (label, prepared, cache_key) in enumerate(results):
    is_dup = label in duplicates
    icon = "🔁" if is_dup else ("⚠️" if prepared.warnings else "✅")
    with st.expander(f"{icon} {label}  -  {prepared.page_count} page(s)", expanded=(position == 0)):
        # Status chips
        chips = []
        if is_dup:
            chips.append(chip(f"Duplicate of {duplicates[label]}", "dup"))
        elif prepared.warnings:
            chips.append(chip(f"{len(prepared.warnings)} quality warning(s)", "warn"))
        else:
            chips.append(chip("No quality issues", "ok"))
        chips.append(chip(prepared.extension.upper().lstrip("."), "info"))
        if not prepared.is_pdf:
            pg = prepared.pages[0]
            if pg.rotation_applied:
                chips.append(chip(f"Rotated {pg.rotation_applied}°", "info"))
            if pg.denoised:
                chips.append(chip("Noise reduced", "info"))
            if pg.deskew_angle:
                chips.append(chip(f"Deskewed {pg.deskew_angle:.1f}°", "info"))
        st.markdown("".join(chips), unsafe_allow_html=True)
        if is_dup:
            st.caption("Excluded from the OCR hand-off.")

        tab_preview, tab_quality, tab_handoff = st.tabs(["Preview", "Quality report", "OCR hand-off"])

        # ---- Preview ----------------------------------------------------
        with tab_preview:
            if prepared.is_pdf:
                try:
                    pages = _pdf_preview_images(prepared.original_bytes)
                    if len(pages) == 1:
                        stretch(st.image, pages[0])
                    else:
                        for tab, img, no in zip(
                            st.tabs([f"Page {n}" for n in range(1, len(pages) + 1)]),
                            pages,
                            range(1, len(pages) + 1),
                        ):
                            with tab:
                                stretch(st.image, img)
                except Exception as exc:
                    st.warning(f"Could not render a preview ({exc}). The PDF is still valid.")
                st.caption("Preview only. The original PDF bytes are sent unchanged to OCR.")
            else:
                pg = prepared.pages[0]
                col_orig, col_proc = st.columns(2)
                with col_orig:
                    st.markdown("**Original**")
                    stretch(st.image, pg.original_image)
                with col_proc:
                    st.markdown("**Processed - what OCR receives**")
                    stretch(st.image, pg.processed_image)

        # ---- Quality ----------------------------------------------------
        with tab_quality:
            if prepared.warnings:
                st.markdown(
                    "".join(chip(w, "warn") for w in prepared.warnings),
                    unsafe_allow_html=True,
                )
            for note in prepared.notes:
                st.markdown(f'<div class="note">• {html.escape(note)}</div>', unsafe_allow_html=True)
            if not (prepared.warnings or prepared.notes):
                st.caption("Nothing to report.")

            cols = st.columns(4)
            cols[0].metric("Size", f"{prepared.file_size / (1024 * 1024):.2f} MB")
            cols[1].metric("Processing time", f"{st.session_state.timings.get(cache_key, 0):.2f} s")
            if prepared.is_pdf:
                cols[2].metric("Pages", prepared.page_count)
                cols[3].metric("Deskew / noise", "n/a (PDF)")
                if prepared.metadata:
                    with st.expander("PDF metadata"):
                        st.json(prepared.metadata)
            else:
                pg = prepared.pages[0]
                cols[2].metric("Sharpness", f"{pg.sharpness:.0f}")
                cols[3].metric("Noise level", f"{pg.noise_sigma:.1f}")
                cols2 = st.columns(4)
                cols2[0].metric("Resolution", f"{pg.width}×{pg.height}")
                cols2[1].metric("Rotation applied", f"{pg.rotation_applied}°")
                cols2[2].metric("Orientation check", pg.orientation_method)
                cols2[3].metric("Deskew applied", f"{pg.deskew_angle:.1f}°" if pg.deskew_angle else "none")

        # ---- Hand-off ---------------------------------------------------
        with tab_handoff:
            payload = prepared.ocr_payload()
            st.json({k: v for k, v in payload.items() if k != "content"}
                    | {"content_bytes": len(payload["content"])})
            if prepared.is_pdf:
                st.download_button(
                    "⬇️ Download original PDF (as sent to OCR)",
                    data=prepared.ocr_bytes,
                    file_name=prepared.ocr_filename,
                    mime=prepared.mime_type,
                    key=f"pdf_{position}",
                )
            else:
                st.download_button(
                    "⬇️ Download processed image (as sent to OCR)",
                    data=prepared.pages[0].processed_bytes("PNG"),
                    file_name=prepared.ocr_filename,
                    mime="image/png",
                    key=f"dl_{position}",
                )

# ---------------------------------------------------------------------------
# Extraction results (filled by Members 2 and 3 through the contract)
# ---------------------------------------------------------------------------
st.divider()
st.subheader("📋 Extracted and validated fields")


def render_fields(fields: dict) -> None:
    """Table for {name: {value, confidence, status}}; falls back to raw JSON."""
    if all(isinstance(v, dict) and "value" in v for v in fields.values()):
        rows = [
            {
                "Field": name,
                "Value": "" if v.get("value") is None else str(v["value"]),
                "Confidence": v.get("confidence"),
                "Status": v.get("status", ""),
            }
            for name, v in fields.items()
        ]
        stretch(
            st.dataframe,
            pd.DataFrame(rows),
            hide_index=True,
            column_config={
                "Confidence": st.column_config.ProgressColumn(
                    "Confidence", min_value=0.0, max_value=1.0, format="%.2f"
                )
            },
        )
    else:
        st.json(fields)


if has_fields:
    render_fields(extraction_result["fields"])
else:
    st.info(
        "Waiting for the OCR and validation module. It should call "
        "`publish_extraction_result(document_id, validated_fields)` when ready."
    )

st.subheader("👁️ Human review dashboard")
if has_fields:
    st.success("Validated extraction is ready for review.")
else:
    st.caption("The review dashboard opens here after extraction and validation.")

# ---------------------------------------------------------------------------
# Integration contract (for teammates)
# ---------------------------------------------------------------------------
with st.expander("🔗 Integration contract (for teammates)"):
    st.markdown(
        """
**To OCR:** `st.session_state['ocr_payloads']` is a list with one dict per unique
document (`PreparedDocument.ocr_payload()`): `filename`, `content` (bytes),
`mime_type`, `sha256`, `warnings`, `page_count`, `is_pdf`, `preprocessing`.
PDFs arrive as the original bytes (the OCR module renders pages); images arrive
as prepared PNG bytes. The untouched upload is `PreparedDocument.original_bytes`.

**To the review dashboard:** `document_bytes` and `document_type` hold the single
original upload; `document_id`, `review_run_id` and `extraction_result` identify
this run. OCR/validation publishes results with
`publish_extraction_result(document_id, validated_fields)`.
        """
    )