# HCLTech-Hackathon
System Builders Team (AI Engineer)

## Member 2 — OCR & VLM extraction

`extraction/`: PaddleOCR and a local Qwen2-VL-2B, both producing one shared JSON output format. Everything runs locally.
See [docs/MEMBER2_EXTRACTION.md](docs/MEMBER2_EXTRACTION.md) for the interface, data lifecycle, privacy and setup.

Quick start: `python3 -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]" && pytest`
