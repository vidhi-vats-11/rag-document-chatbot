"""Generate sample documents for trying the app: text PDF, scanned PDF, PNG, DOCX.

Usage:  python scripts/create_sample_documents.py   (writes to ./samples/)
"""

from __future__ import annotations

import io
from pathlib import Path

import pymupdf
from docx import Document
from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parents[1] / "samples"

POLICY_PAGES = [
    "Acme Store - Customer Policy\n\n"
    "1. Refunds\nCustomers may request a full refund within 30 days of purchase. "
    "Refunds are paid to the original payment method within 5 business days after "
    "the returned item is received. Opened software is not refundable.",
    "2. Shipping\nOrders ship from our Berlin warehouse. Standard delivery takes 3 to 7 "
    "working days within the EU. Express delivery (1-2 days) costs 12 EUR.\n\n"
    "3. Warranty\nAll electronics carry a 2-year limited warranty covering manufacturing defects.",
]

SCANNED_TEXT = (
    "INVOICE 2026-0457. Customer: Jordan Lee. Item: Laptop stand x2. "
    "Total due: 89.90 EUR. Payment due within 14 days of the invoice date."
)
IMAGE_TEXT = "Office notice: The meeting room on floor 3 is closed for repairs until 15 November."
DOCX_PARAGRAPHS = [
    "Employee Handbook - Remote Work",
    "Employees may work remotely up to three days per week with manager approval.",
    "Core collaboration hours are 10:00 to 15:00 CET.",
    "Home-office equipment up to 500 EUR per year is reimbursed against receipts.",
]


def _text_image(text: str) -> bytes:
    font = ImageFont.load_default(size=36)
    words, lines, line = text.split(), [], ""
    for w in words:
        if len(line) + len(w) > 45:
            lines.append(line)
            line = w
        else:
            line = f"{line} {w}".strip()
    lines.append(line)
    img = Image.new("RGB", (1700, 100 + 56 * len(lines)), "white")
    draw = ImageDraw.Draw(img)
    for i, ln in enumerate(lines):
        draw.text((50, 50 + 56 * i), ln, fill="black", font=font)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def main() -> None:
    OUT.mkdir(exist_ok=True)

    pdf = pymupdf.open()
    for text in POLICY_PAGES:
        page = pdf.new_page()
        page.insert_textbox(pymupdf.Rect(50, 50, 550, 800), text, fontsize=12)
    pdf.save(OUT / "store_policy.pdf")

    scan = pymupdf.open()
    page = scan.new_page()
    page.insert_image(page.rect, stream=_text_image(SCANNED_TEXT))
    scan.save(OUT / "scanned_invoice.pdf")

    (OUT / "office_notice.png").write_bytes(_text_image(IMAGE_TEXT))

    doc = Document()
    doc.add_heading(DOCX_PARAGRAPHS[0], level=1)
    for p in DOCX_PARAGRAPHS[1:]:
        doc.add_paragraph(p)
    doc.save(OUT / "remote_work_policy.docx")

    for f in sorted(OUT.iterdir()):
        print(f"created {f.relative_to(OUT.parent)}")


if __name__ == "__main__":
    main()
