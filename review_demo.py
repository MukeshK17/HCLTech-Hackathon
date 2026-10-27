from pathlib import Path

import streamlit as st

from review import render_review
from sample_data import sample_documents

st.set_page_config(page_title="DocuSense Review", layout="wide")

extraction_result = st.session_state.get("extraction_result")

if extraction_result is not None:
    # Integrated mode: reuse Member 1's original upload and Member 2's result.
    render_review(
        extraction_result,
        document_bytes=st.session_state.get("document_bytes"),
        document_type=st.session_state.get("document_type"),
        validator=st.session_state.get("field_validator"),
        authorized_export=st.session_state.get("authorized_export", False),
        document_id=st.session_state.get("document_id"),
        review_run_id=st.session_state.get("review_run_id"),
        confidential_fields=st.session_state.get("confidential_fields", set()),
    )
else:
    # Demo mode is used only when no integrated extraction result was supplied.
    selected_document_id = st.sidebar.selectbox(
        "Select sample bill",
        options=[document["document_id"] for document in sample_documents],
        format_func=lambda document_id: next(
            document["display_name"] for document in sample_documents if document["document_id"] == document_id
        ),
    )

    demo_document = next(document for document in sample_documents if document["document_id"] == selected_document_id)
    selected_bill_number = next(
        index
        for index, document in enumerate(sample_documents, start=1)
        if document["document_id"] == selected_document_id
    )
    st.sidebar.caption(f"Showing bill {selected_bill_number} of {len(sample_documents)}")

    sample_document_path = Path(__file__).with_name("assets") / "sample_bills" / f"{selected_document_id}.png"
    if not sample_document_path.exists():
        st.error("The selected sample bill image is missing. Run `python generate_sample_bills.py` to recreate it.")
        st.stop()

    render_review(
        demo_document,
        document_bytes=sample_document_path.read_bytes(),
        document_type="image/png",
        authorized_export=False,
    )
