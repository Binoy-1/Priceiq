"""Export utilities — CSV/XLSX/PDF for predictions, dataset, and reports."""
from __future__ import annotations
from io import BytesIO
from datetime import datetime
import pandas as pd


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode()


def df_to_xlsx_bytes(sheets: dict[str, pd.DataFrame]) -> bytes:
    bio = BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as xw:
        for name, df in sheets.items():
            df.to_excel(xw, sheet_name=name[:31] or "Sheet1", index=False)
    return bio.getvalue()


def report_to_pdf_bytes(title: str, sections: list[tuple[str, str]],
                        tables: dict[str, pd.DataFrame] | None = None) -> bytes:
    """Build a polished PDF using reportlab. sections = [(heading, body_text)]."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from reportlab.lib.units import mm
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                    TableStyle, PageBreak)

    bio = BytesIO()
    doc = SimpleDocTemplate(bio, pagesize=A4,
                            leftMargin=18 * mm, rightMargin=18 * mm,
                            topMargin=18 * mm, bottomMargin=18 * mm,
                            title=title)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Brand", fontName="Helvetica-Bold",
                              fontSize=22, textColor=colors.HexColor("#cc785c"),
                              spaceAfter=4))
    styles.add(ParagraphStyle(name="Sub", fontName="Helvetica",
                              fontSize=10, textColor=colors.HexColor("#6b6259"),
                              spaceAfter=14))
    styles.add(ParagraphStyle(name="H2", fontName="Helvetica-Bold",
                              fontSize=13, textColor=colors.HexColor("#3a3631"),
                              spaceBefore=10, spaceAfter=6))
    styles.add(ParagraphStyle(name="Body", fontName="Helvetica",
                              fontSize=10, leading=14,
                              textColor=colors.HexColor("#3a3631")))

    story = []
    story.append(Paragraph("PriceIQ", styles["Brand"]))
    story.append(Paragraph(
        f"{title} · Generated {datetime.now().strftime('%b %d, %Y %H:%M')}",
        styles["Sub"]))
    for heading, body in sections:
        story.append(Paragraph(heading, styles["H2"]))
        story.append(Paragraph(body.replace("\n", "<br/>"), styles["Body"]))
        story.append(Spacer(1, 6))

    if tables:
        for name, df in tables.items():
            story.append(PageBreak())
            story.append(Paragraph(name, styles["H2"]))
            head = list(df.columns.astype(str))
            rows = df.head(40).astype(str).values.tolist()
            tbl = Table([head] + rows, repeatRows=1)
            tbl.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f1d1a")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1),
                 [colors.HexColor("#faf7f2"), colors.white]),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#e6e0d6")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ]))
            story.append(tbl)

    doc.build(story)
    return bio.getvalue()
