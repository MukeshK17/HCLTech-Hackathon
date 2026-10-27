import base64
import hashlib
import json
from collections.abc import Mapping

import streamlit as st

from database import DATABASE_PATH, save_verified_document
from exporter import export_csv, export_json, prepare_export
from rules_engine import RulesEngineConfig
from rules_integration import (
    AMOUNT_FIELDS as RULES_AMOUNT_FIELDS,
)
from rules_integration import (
    is_rules_result,
    revalidate_review_corrections,
    review_fields_from_rules,
    to_rules_dict,
)
from validation import DEFAULT_AMOUNT_FIELDS, validate_fields

AUTO_ACCEPT_THRESHOLD = RulesEngineConfig().thresholds.auto_accept_threshold
STATE_KEY = "_docsense_review_states"


def _validate_document_structure(document):
    if not isinstance(document, Mapping):
        return "Extraction result must be a document object."
    document_id = document.get("document_id")
    if not isinstance(document_id, str) or not document_id.strip():
        return "Extraction result is missing a valid document_id."
    review_run_id = document.get("review_run_id")
    if not isinstance(review_run_id, str) or not review_run_id.strip():
        return "Extraction result is missing a valid review_run_id."
    fields = document.get("fields")
    if not isinstance(fields, Mapping) or not fields:
        return "Extraction result must contain a non-empty fields object."
    for field_name, details in fields.items():
        if not isinstance(field_name, str) or not isinstance(details, Mapping):
            return "Every extracted field must have a name and field details."
        if "value" not in details:
            return f"Extracted field '{field_name}' is missing its value."
        try:
            confidence = float(details.get("confidence", 0))
        except (TypeError, ValueError):
            return f"Extracted field '{field_name}' has invalid confidence."
        if not 0 <= confidence <= 1:
            return f"Confidence for '{field_name}' must be between 0 and 1."
    return None


def _source_fingerprint(fields):
    serialized = json.dumps(fields, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _widget_prefix(document_id):
    return hashlib.sha256(document_id.encode("utf-8")).hexdigest()[:12]


def _field_widget_key(document_id, field_name):
    return f"review_{_widget_prefix(document_id)}_{field_name}"


def _get_document_state(document_id, fields):
    states = st.session_state.setdefault(STATE_KEY, {})
    fingerprint = _source_fingerprint(fields)
    state = states.get(document_id)
    if state is None or state.get("source_fingerprint") != fingerprint:
        original_values = {name: str(details.get("value", "")) for name, details in fields.items()}
        state = {
            "source_fingerprint": fingerprint,
            "original_values": original_values,
            "corrected_values": original_values.copy(),
            "approved_fields": set(),
            "approval_sources": {},
            "edited_fields": set(),
            "manual_review_fields": set(),
            "revealed_fields": set(),
            "saved_record_id": None,
        }
        states[document_id] = state
        for field_name, value in original_values.items():
            st.session_state[_field_widget_key(document_id, field_name)] = value
    return state


def _on_field_change(document_id, field_name, dependent_amount_fields):
    state = st.session_state[STATE_KEY][document_id]
    state.setdefault("approval_sources", {})
    state.setdefault("edited_fields", set())
    state.setdefault("manual_review_fields", set())
    widget_key = _field_widget_key(document_id, field_name)
    state["corrected_values"][field_name] = str(st.session_state[widget_key])
    state["approved_fields"].discard(field_name)
    state["approval_sources"].pop(field_name, None)
    state["edited_fields"].add(field_name)
    state["manual_review_fields"].add(field_name)
    state["saved_record_id"] = None
    if field_name in dependent_amount_fields:
        state["approved_fields"].difference_update(dependent_amount_fields)
        for dependent_field in dependent_amount_fields:
            state["approval_sources"].pop(dependent_field, None)
        state["manual_review_fields"].update(dependent_amount_fields)


def _start_manual_edit(document_id, field_name, dependent_amount_fields):
    """Revoke AI approval before the reviewer starts typing."""
    state = st.session_state[STATE_KEY][document_id]
    state.setdefault("approval_sources", {})
    state.setdefault("manual_review_fields", set())
    state["approved_fields"].discard(field_name)
    state["approval_sources"].pop(field_name, None)
    state["manual_review_fields"].add(field_name)
    state["saved_record_id"] = None

    if field_name in dependent_amount_fields:
        state["approved_fields"].difference_update(dependent_amount_fields)
        for dependent_field in dependent_amount_fields:
            state["approval_sources"].pop(dependent_field, None)
        state["manual_review_fields"].update(dependent_amount_fields)


def _run_validator(validator, values, fields):
    validation_function = validator or validate_fields
    try:
        result = validation_function(values, fields)
    except TypeError:
        result = validation_function(values)
    if not isinstance(result, Mapping):
        raise ValueError("Validator must return a mapping of field names to errors.")
    normalized = {field_name: [] for field_name in fields}
    for field_name, field_errors in result.items():
        if field_name not in normalized:
            continue
        if isinstance(field_errors, str):
            normalized[field_name] = [field_errors]
        elif field_errors:
            normalized[field_name] = [str(error) for error in field_errors]
    return normalized


def _render_document(
    document_bytes,
    document_type,
    sample_preview=False,
    preview_title=None,
    preview_fields=None,
):
    st.subheader("Original Document")
    if document_bytes is None and sample_preview:
        with st.container(border=True):
            st.markdown(f"### {preview_title or 'Sample Bill'}")
            st.caption("Generated preview for demo mode")
            for field_name, details in (preview_fields or {}).items():
                label = field_name.replace("_", " ").title()
                value = details.get("value", "")
                if details.get("confidential", False):
                    value = "••••••••"
                field_label, field_value = st.columns([2, 3])
                field_label.markdown(f"**{label}**")
                field_value.write(value)
    elif document_bytes is None:
        st.info("The original document is not available in application state.")
    elif document_type == "application/pdf":
        encoded_pdf = base64.b64encode(document_bytes).decode("utf-8")
        st.components.v1.html(
            f'<iframe src="data:application/pdf;base64,{encoded_pdf}" '
            'width="100%" height="650" type="application/pdf"></iframe>',
            height=670,
        )
    elif isinstance(document_type, str) and document_type.startswith("image/"):
        st.image(
            document_bytes,
            caption="Original Uploaded Document",
            width="stretch",
        )
    else:
        st.warning("Preview is unavailable for this document type.")


def render_review(
    document,
    document_bytes=None,
    document_type=None,
    validator=None,
    authorized_export=False,
    database_path=DATABASE_PATH,
    sample_preview=False,
    document_id=None,
    review_run_id=None,
    confidential_fields=None,
    show_header=True,
):
    """Render a reusable review dashboard for one extracted document.

    Masking is only a UI safeguard, not access control. The caller must set
    authorized_export only after applying its own authorization policy.
    """
    rules_payload = None
    if is_rules_result(document):
        rules_payload = to_rules_dict(document)
        metadata = rules_payload.get("metadata", {})
        document_id = document_id or metadata.get("document_id") or st.session_state.get("document_id")
        review_run_id = review_run_id or metadata.get("review_run_id") or st.session_state.get("review_run_id")
        document = {
            "document_id": document_id,
            "review_run_id": review_run_id,
            "document_type": rules_payload.get("document_type", "generic"),
            "fields": review_fields_from_rules(rules_payload, set(confidential_fields or ())),
        }

    structure_error = _validate_document_structure(document)
    if structure_error:
        st.error(structure_error)
        return {"status": "invalid_extraction", "error": structure_error}

    document_id = document["document_id"].strip()
    review_run_id = document["review_run_id"].strip()
    review_state_id = f"{document_id}:{review_run_id}"
    fields = dict(document["fields"])
    if document_bytes is None and not sample_preview:
        document_bytes = st.session_state.get("document_bytes")
    if document_type is None and not sample_preview:
        document_type = st.session_state.get("document_type")

    state = _get_document_state(review_state_id, fields)
    state.setdefault("approval_sources", {})
    state.setdefault("edited_fields", set())
    state.setdefault("manual_review_fields", set())
    dependent_amount_fields = {
        name
        for name, details in fields.items()
        if name in DEFAULT_AMOUNT_FIELDS
        or name in RULES_AMOUNT_FIELDS
        or details.get("type") in {"amount", "currency", "number"}
    }
    for field_name in fields:
        widget_key = _field_widget_key(review_state_id, field_name)
        if widget_key not in st.session_state:
            st.session_state[widget_key] = state["corrected_values"][field_name]
        state["corrected_values"][field_name] = str(st.session_state[widget_key])

    document_errors = []
    try:
        if rules_payload is not None:
            validation_errors, document_errors, revalidated_payload = revalidate_review_corrections(
                rules_payload,
                state["corrected_values"],
                state["edited_fields"],
            )
            state["revalidated_rules_result"] = revalidated_payload
        else:
            validation_errors = _run_validator(validator, state["corrected_values"], fields)
        for field_name, details in fields.items():
            missing_from_rules_result = rules_payload is not None and field_name not in state.get(
                "revalidated_rules_result", {}
            ).get("fields", {})
            validation_errors.setdefault(field_name, [])
            if (
                missing_from_rules_result
                and details.get("validation_status") in {"invalid", "missing"}
                and not validation_errors[field_name]
            ):
                validation_errors[field_name] = [str(issue) for issue in details.get("issues", [])]
    except Exception as error:
        message = f"Validation could not be completed: {error}"
        st.error(message)
        return {"status": "validation_error", "error": message}

    for field_name, errors in validation_errors.items():
        if errors:
            state["approved_fields"].discard(field_name)
            state["approval_sources"].pop(field_name, None)

    for field_name, details in fields.items():
        confidence = float(details.get("confidence", 0))
        issues = details.get("issues", [])
        upstream_auto_accept = details.get("decision") == "auto_accept"
        can_auto_approve = (
            (upstream_auto_accept if rules_payload is not None else confidence >= AUTO_ACCEPT_THRESHOLD)
            and not issues
            and not validation_errors[field_name]
            and field_name not in state["manual_review_fields"]
        )
        if can_auto_approve:
            state["approved_fields"].add(field_name)
            state["approval_sources"][field_name] = "ai"

    if show_header:
        st.title("DocuSense")
        st.subheader("Human Review Dashboard")
        st.divider()
    left, right = st.columns([1, 1], gap="large")

    with left:
        _render_document(
            document_bytes,
            document_type,
            sample_preview=sample_preview,
            preview_title=document.get("display_name"),
            preview_fields=fields,
        )

    with right:
        st.subheader("Extracted Information")
        if document_errors:
            st.error(" | ".join(document_errors))
        for field_name, details in fields.items():
            label = field_name.replace("_", " ").title()
            confidence = float(details.get("confidence", 0))
            issues = [str(issue) for issue in details.get("issues", [])]
            rule_errors = validation_errors[field_name]
            confidential = bool(details.get("confidential", False))
            is_hidden = confidential and field_name not in state["revealed_fields"]
            is_approved = field_name in state["approved_fields"]
            approval_source = state["approval_sources"].get(field_name)
            is_ai_approved = is_approved and approval_source == "ai"
            needs_review = (
                (
                    details.get("decision") != "auto_accept"
                    if rules_payload is not None
                    else confidence < AUTO_ACCEPT_THRESHOLD
                )
                or bool(issues)
                or bool(rule_errors)
            )

            field_column, confidence_column, action_column = st.columns([5, 1, 2], vertical_alignment="bottom")
            with field_column:
                st.text_input(
                    label,
                    key=_field_widget_key(review_state_id, field_name),
                    type="password" if is_hidden else "default",
                    disabled=is_hidden or is_ai_approved,
                    on_change=_on_field_change,
                    args=(review_state_id, field_name, dependent_amount_fields),
                )
            with confidence_column:
                badge_color = "#b54708" if needs_review else "#027a48"
                badge_background = "#fffaeb" if needs_review else "#ecfdf3"
                st.markdown(
                    f"""
                    <div style="margin-bottom:8px;text-align:right">
                        <span style="display:inline-block;padding:4px 8px;
                            border:1px solid {badge_color};border-radius:6px;
                            background:{badge_background};color:{badge_color};
                            font-size:12px;font-weight:600;white-space:nowrap">
                            {confidence * 100:.0f}%
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with action_column:
                if is_hidden:
                    if st.button(
                        "Reveal",
                        key=f"reveal_{_widget_prefix(review_state_id)}_{field_name}",
                        width="stretch",
                    ):
                        state["revealed_fields"].add(field_name)
                        st.rerun()
                elif is_ai_approved:
                    if st.button(
                        "AI Approved · Edit",
                        key=f"edit_{_widget_prefix(review_state_id)}_{field_name}",
                        width="stretch",
                    ):
                        _start_manual_edit(
                            review_state_id,
                            field_name,
                            dependent_amount_fields,
                        )
                        st.rerun()
                elif is_approved:
                    st.success("Approved")
                elif st.button(
                    "Approve",
                    key=f"approve_{_widget_prefix(review_state_id)}_{field_name}",
                    width="stretch",
                    disabled=bool(rule_errors),
                ):
                    state["approved_fields"].add(field_name)
                    state["approval_sources"][field_name] = "human"
                    st.rerun()

            if confidential:
                st.caption("🔒 Confidential field")
            if needs_review and not is_approved:
                st.caption("⚠️ Human review required")
            if issues and not is_approved:
                st.warning(" | ".join(issues))
            if rule_errors:
                st.error(" | ".join(rule_errors))
            st.divider()

        approved_count = len(state["approved_fields"].intersection(fields))
        total_count = len(fields)
        st.markdown("### Review Progress")
        st.progress(approved_count / total_count)
        st.write(f"**{approved_count} of {total_count} fields approved**")

        all_approved = approved_count == total_count
        all_valid = not any(validation_errors.values()) and not document_errors
        can_finalize = all_approved and all_valid
        if all_approved:
            st.success("All fields approved. Ready for final validation.")
        else:
            st.info("Review and approve all fields to continue.")

        st.markdown("### Store and Export Verified Document")
        if can_finalize:
            database_data = prepare_export(state["corrected_values"], fields, include_confidential=True)
            database_data["document_id"] = document_id
            database_data["review_run_id"] = review_run_id
            if st.button(
                "Save to Database",
                type="primary",
                width="stretch",
                disabled=state["saved_record_id"] is not None,
            ):
                try:
                    state["saved_record_id"] = save_verified_document(
                        database_data,
                        document_id=document_id,
                        review_run_id=review_run_id,
                        original_values=state["original_values"],
                        database_path=database_path,
                    )
                    st.rerun()
                except Exception as error:
                    st.error(f"Could not save document: {error}")
            if state["saved_record_id"] is not None:
                st.success(f"Saved to database as record #{state['saved_record_id']}.")

            export_data = prepare_export(
                state["corrected_values"],
                fields,
                include_confidential=authorized_export,
            )
            export_data["document_id"] = document_id
            export_data["review_run_id"] = review_run_id
            if not authorized_export and export_data.get("excluded_confidential_fields"):
                st.caption(
                    "Confidential fields are explicitly excluded from these "
                    "downloads because no authorized-export policy was supplied."
                )
            json_column, csv_column = st.columns(2)
            with json_column:
                st.download_button(
                    "Download JSON",
                    export_json(export_data),
                    file_name=f"{document_id}_verified.json",
                    mime="application/json",
                    width="stretch",
                )
            with csv_column:
                st.download_button(
                    "Download CSV",
                    export_csv(export_data),
                    file_name=f"{document_id}_verified.csv",
                    mime="text/csv",
                    width="stretch",
                )
            with st.expander("Preview Verified JSON"):
                st.json(export_data)
        else:
            st.info("Storage and export are available after all fields are approved and validation passes.")

    return {
        "status": "ready" if can_finalize else "in_review",
        "document_id": document_id,
        "review_run_id": review_run_id,
        "approved_count": approved_count,
        "total_count": total_count,
        "validation_errors": validation_errors,
        "saved_record_id": state["saved_record_id"],
    }
