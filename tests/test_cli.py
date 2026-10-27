import json

import extraction.__main__ as cli
from extraction.schemas import failed_result

from .conftest import PNG_BYTES


def test_bulk_folder_outputs_one_line_per_file(tmp_path, monkeypatch, capsys):
    (tmp_path / "sub").mkdir()
    for name in ("a.png", "sub/b.jpg", "c.pdf", "notes.txt"):
        (tmp_path / name).write_bytes(PNG_BYTES)
    monkeypatch.setattr(
        cli, "extract_with_ocr", lambda p: failed_result("ocr", "paddleocr", "x", "y").model_copy(update={"errors": []})
    )
    rc = cli.main([str(tmp_path), "--branch", "ocr"])
    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [r["file"].split("/")[-1] for r in rows] == ["a.png", "c.pdf", "b.jpg"]
    assert rows[1]["ocr"]["errors"][0]["code"] == "invalid_input"  # PDF not sent to OCR
    assert "raw_text" not in rows[0]["ocr"]
    assert rc == 1


def test_single_file_pretty_json(tmp_path, monkeypatch, capsys):
    img = tmp_path / "a.png"
    img.write_bytes(PNG_BYTES)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: None)  # never read a real .env key in tests
    rc = cli.main([str(img), "--branch", "vlm"])
    out = json.loads(capsys.readouterr().out)
    assert out["vlm"]["errors"][0]["code"] == "vlm_not_configured" and rc == 1


def test_no_files(tmp_path, capsys):
    assert cli.main([str(tmp_path)]) == 2
