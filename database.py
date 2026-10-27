import hashlib
import json
import sqlite3
from pathlib import Path

DATABASE_PATH = Path(__file__).with_name("docsense.db")


def _ensure_schema(connection):
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS verified_documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            data_json TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    columns = {row[1] for row in connection.execute("PRAGMA table_info(verified_documents)")}
    additions = {
        "document_id": "TEXT",
        "review_run_id": "TEXT",
        "original_values_json": "TEXT",
        "review_status": "TEXT NOT NULL DEFAULT 'approved'",
        "updated_at": "TEXT",
    }
    for column_name, declaration in additions.items():
        if column_name not in columns:
            connection.execute(f"ALTER TABLE verified_documents ADD COLUMN {column_name} {declaration}")

    connection.execute("DROP INDEX IF EXISTS verified_documents_document_id_idx")
    connection.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS
        verified_documents_review_run_id_idx
        ON verified_documents(review_run_id)
        WHERE review_run_id IS NOT NULL
        """
    )


def save_verified_document(
    data,
    document_id=None,
    review_run_id=None,
    original_values=None,
    database_path=DATABASE_PATH,
):
    """Insert or update one verified record per document ID."""
    payload = json.dumps(data, ensure_ascii=False)
    original_payload = json.dumps(original_values or {}, ensure_ascii=False)
    stable_document_id = document_id or hashlib.sha256(payload.encode("utf-8")).hexdigest()
    stable_review_run_id = review_run_id or stable_document_id

    with sqlite3.connect(database_path) as connection:
        _ensure_schema(connection)
        row = connection.execute(
            "SELECT id FROM verified_documents WHERE review_run_id = ?", (stable_review_run_id,)
        ).fetchone()

        if row is None:
            cursor = connection.execute(
                """
                INSERT INTO verified_documents (
                    document_id,
                    review_run_id,
                    data_json,
                    original_values_json,
                    review_status,
                    updated_at
                )
                VALUES (?, ?, ?, ?, 'approved', CURRENT_TIMESTAMP)
                """,
                (
                    stable_document_id,
                    stable_review_run_id,
                    payload,
                    original_payload,
                ),
            )
            record_id = cursor.lastrowid
        else:
            record_id = row[0]
            connection.execute(
                """
                UPDATE verified_documents
                SET data_json = ?,
                    original_values_json = ?,
                    review_status = 'approved',
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (payload, original_payload, record_id),
            )

    return record_id
