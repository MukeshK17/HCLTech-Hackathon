"""Generate deterministic PNG bill fixtures from the demo dataset."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from sample_data import sample_documents

OUTPUT_DIRECTORY = Path(__file__).with_name("assets") / "sample_bills"


def _font(size, bold=False):
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def generate_bill(document):
    fields = document["fields"]
    height = max(900, 310 + len(fields) * 62)
    image = Image.new("RGB", (1000, height), "#f4f1e8")
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle(
        (55, 45, 945, height - 45),
        radius=18,
        fill="#fffef9",
        outline="#cbc5b7",
        width=3,
    )

    title_font = _font(42, bold=True)
    heading_font = _font(23, bold=True)
    value_font = _font(23)
    small_font = _font(18)

    vendor = fields.get("vendor_name", {}).get("value", "SAMPLE BILL")
    draw.text((95, 85), str(vendor), font=title_font, fill="#20242a")
    draw.text((95, 145), document["display_name"], font=small_font, fill="#667085")
    draw.line((95, 195, 905, 195), fill="#d7d1c4", width=2)

    y_position = 225
    for field_name, details in fields.items():
        if field_name == "vendor_name":
            continue
        label = field_name.replace("_", " ").title()
        value = str(details.get("value", ""))
        draw.text((95, y_position), label, font=heading_font, fill="#344054")
        value_width = draw.textbbox((0, 0), value, font=value_font)[2]
        draw.text(
            (905 - value_width, y_position),
            value,
            font=value_font,
            fill="#101828",
        )
        draw.line(
            (95, y_position + 42, 905, y_position + 42),
            fill="#ebe7de",
            width=1,
        )
        y_position += 62

    draw.text(
        (95, height - 92),
        "DOCUSENSE SAMPLE DOCUMENT",
        font=small_font,
        fill="#98a2b3",
    )
    output_path = OUTPUT_DIRECTORY / f"{document['document_id']}.png"
    image.save(output_path, format="PNG", optimize=True)


def main():
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    for document in sample_documents:
        generate_bill(document)


if __name__ == "__main__":
    main()
