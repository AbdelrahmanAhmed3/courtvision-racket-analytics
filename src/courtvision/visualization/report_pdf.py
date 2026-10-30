"""The match report as a PDF: the same wording as report.md, laid out for print."""

from __future__ import annotations

from collections.abc import Sequence
from io import BytesIO
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (
    Image,
    KeepTogether,
    ListFlowable,
    ListItem,
    Paragraph,
    SimpleDocTemplate,
    Table,
    TableStyle,
)

import courtvision
from courtvision.analytics.report import (
    PLAYER_COLUMNS,
    RALLY_COLUMNS,
    highlight_items,
    player_rows,
    rally_rows,
    summary_line,
    trust_notes,
)

MARGIN = 18 * mm
WIDTH = A4[0] - 2 * MARGIN
REPOSITORY = "github.com/AbdelrahmanAhmed3/courtvision-racket-analytics"
ACCENT = colors.HexColor("#1f5f4a")
HEADER_FILL = colors.HexColor("#e4eee9")
STRIPE_FILL = colors.HexColor("#f5f8f6")


def report_pdf(
    report: dict, heatmap_image: Path | None = None, compress: bool = True
) -> bytes:
    """The PDF's bytes. ``compress=False`` leaves the text readable in the file."""
    buffer = BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=MARGIN,
        rightMargin=MARGIN,
        topMargin=MARGIN,
        bottomMargin=MARGIN,
        title=f"Match report: {report['video']}",
        author="CourtVision",
        pageCompression=1 if compress else 0,
    )
    styles = _styles()
    story = [
        Paragraph("Match report", styles["title"]),
        Paragraph(escape(report["video"]), styles["subtitle"]),
        Paragraph(escape(summary_line(report)), styles["body"]),
        Paragraph("Highlights", styles["heading"]),
        _bullets(
            [
                f"<b>{escape(what)}:</b> {escape(text)}"
                for what, text in highlight_items(report)
            ]
            or ["No highlights: nothing was tracked long enough."],
            styles,
        ),
        Paragraph("Players", styles["heading"]),
        _table(PLAYER_COLUMNS, player_rows(report), styles),
    ]
    if heatmap_image is not None and heatmap_image.exists():
        story.append(
            KeepTogether(
                [
                    Paragraph("Where each player stood", styles["heading"]),
                    _image(heatmap_image),
                    Paragraph(
                        "Seconds per square metre, warmer where a player stood longer; "
                        "the far baseline is at the top.",
                        styles["caption"],
                    ),
                ]
            )
        )
    if report["rallies"]:
        story += [
            Paragraph("Rallies", styles["heading"]),
            _table(RALLY_COLUMNS, rally_rows(report), styles),
        ]
    story += [
        Paragraph("How far to trust these numbers", styles["heading"]),
        _bullets([escape(note) for note in trust_notes(report)], styles),
    ]
    document.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return buffer.getvalue()


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=base["BodyText"], fontSize=10, leading=14)
    return {
        "title": ParagraphStyle(
            "title", parent=base["Title"], alignment=0, textColor=ACCENT, spaceAfter=2
        ),
        "subtitle": ParagraphStyle(
            "subtitle", parent=body, textColor=colors.grey, spaceAfter=8
        ),
        "heading": ParagraphStyle(
            "heading",
            parent=base["Heading2"],
            textColor=ACCENT,
            spaceBefore=12,
            spaceAfter=6,
            keepWithNext=1,
        ),
        "body": body,
        "caption": ParagraphStyle(
            "caption", parent=body, fontSize=8, leading=10, textColor=colors.grey
        ),
        "cell": ParagraphStyle("cell", parent=body, fontSize=8, leading=10),
        "header": ParagraphStyle(
            "header", parent=body, fontSize=8, leading=10, fontName="Helvetica-Bold"
        ),
    }


def _bullets(lines: Sequence[str], styles: dict) -> ListFlowable:
    return ListFlowable(
        [ListItem(Paragraph(line, styles["body"]), leftIndent=12) for line in lines],
        bulletType="bullet",
        start="•",
        leftIndent=12,
    )


def _table(columns: Sequence[str], rows: list[list[str]], styles: dict) -> Table:
    data = [[Paragraph(escape(column), styles["header"]) for column in columns]]
    data += [[Paragraph(escape(cell), styles["cell"]) for cell in row] for row in rows]
    table = Table(data, colWidths=[WIDTH / len(columns)] * len(columns), repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), HEADER_FILL),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, STRIPE_FILL]),
                ("LINEBELOW", (0, 0), (-1, 0), 0.6, ACCENT),
                ("LINEBELOW", (0, -1), (-1, -1), 0.3, colors.lightgrey),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    return table


def _image(path: Path, max_width: float = 0.8 * WIDTH) -> Image:
    width, height = ImageReader(str(path)).getSize()
    scale = min(1.0, max_width / width)
    return Image(str(path), width=width * scale, height=height * scale)


def _footer(canvas, document) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(colors.grey)
    canvas.drawString(
        MARGIN, 10 * mm, f"CourtVision {courtvision.__version__} · {REPOSITORY}"
    )
    canvas.drawRightString(A4[0] - MARGIN, 10 * mm, f"Page {document.page}")
    canvas.restoreState()
