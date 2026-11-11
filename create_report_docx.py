"""Convert the generated 80-page report PDF into an editable Word document."""

from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.shared import Inches, Pt
from pdf2docx import Converter


ROOT = Path(__file__).resolve().parent
PDF_PATH = ROOT / "reports" / "CMOS_ML_180nm_80_Page_Project_Report.pdf"
DOCX_PATH = ROOT / "reports" / "CMOS_ML_180nm_80_Page_Project_Report.docx"


def paragraphs_in_table(table):
    for row in table.rows:
        for cell in row.cells:
            yield from cell.paragraphs
            for nested_table in cell.tables:
                yield from paragraphs_in_table(nested_table)


def create_docx() -> Path:
    if not PDF_PATH.is_file():
        raise FileNotFoundError(f"Generate the PDF report first: {PDF_PATH}")

    converter = Converter(str(PDF_PATH))
    try:
        converter.convert(str(DOCX_PATH), multi_processing=False)
    finally:
        converter.close()

    document = Document(DOCX_PATH)
    for section in document.sections:
        section.page_width = Inches(8.27)
        section.page_height = Inches(11.69)
        section.top_margin = Inches(1)
        section.bottom_margin = Inches(1)
        section.left_margin = Inches(1)
        section.right_margin = Inches(1)

    normal = document.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(12)
    normal.paragraph_format.line_spacing = 1.0
    normal.paragraph_format.space_before = Pt(0)
    normal.paragraph_format.space_after = Pt(0)
    paragraphs = list(document.paragraphs)
    for table in document.tables:
        paragraphs.extend(paragraphs_in_table(table))
    for paragraph in paragraphs:
        paragraph.paragraph_format.line_spacing = 1.0
        paragraph.paragraph_format.space_before = Pt(0)
        paragraph.paragraph_format.space_after = Pt(0)
        for run in paragraph.runs:
            run.font.name = "Times New Roman"

    document.core_properties.title = (
        "Machine Learning-Assisted Electrical Characterization of 180 nm CMOS Digital Circuits"
    )
    document.core_properties.author = "Ramandeep Nain"
    document.core_properties.subject = (
        "Editable Word conversion of the 80-page single-spaced project report"
    )
    document.save(DOCX_PATH)
    return DOCX_PATH


if __name__ == "__main__":
    print(f"Created {create_docx()}")
