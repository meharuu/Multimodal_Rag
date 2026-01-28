from io import BytesIO

from flask import Blueprint, jsonify, request
from werkzeug.utils import secure_filename
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

report_bp = Blueprint("report", __name__)


@report_bp.post("/report")
def report():
    data = request.get_json(silent=True) or {}
    result = data.get("result") or {}
    filename = data.get("filename") or "audio"

    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    title = f"Audio Emotion Report — {filename}"
    story.append(Paragraph(title, styles["Title"]))
    story.append(Spacer(1, 12))

    summary = [
        ["Primary Emotion", result.get("emotion", {}).get("label", "--")],
        ["Confidence", str(result.get("emotion", {}).get("score", "--"))],
        ["Language", result.get("language", "--")],
        ["Duration (s)", str(result.get("duration", "--"))],
    ]
    table = Table(summary, hAlign="LEFT", colWidths=[150, 350])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("BOX", (0, 0), (-1, -1), 0.25, colors.black),
                ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ]
        )
    )
    story.append(table)
    story.append(Spacer(1, 12))

    story.append(Paragraph("Transcript", styles["Heading2"]))
    transcript_text = result.get("transcript") or "No transcript."
    story.append(Paragraph(transcript_text.replace("\n", "<br/>"), styles["BodyText"]))
    story.append(Spacer(1, 12))

    scores = result.get("scores") or []
    if scores:
        story.append(Paragraph("Emotion Probabilities", styles["Heading2"]))
        table_data = [["Label", "Score"]]
        for s in scores:
            table_data.append([s.get("label", "--"), str(s.get("score", ""))])
        s_table = Table(table_data, hAlign="LEFT", colWidths=[200, 100])
        s_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                    ("BOX", (0, 0), (-1, -1), 0.25, colors.black),
                    ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.grey),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ]
            )
        )
        story.append(s_table)
        story.append(Spacer(1, 12))

    segments = result.get("segments") or []
    if segments:
        story.append(Paragraph("Segments", styles["Heading2"]))
        seg_rows = [["Start", "End", "Text"]]
        for seg in segments:
            seg_rows.append(
                [
                    str(seg.get("start", "")),
                    str(seg.get("end", "")),
                    seg.get("text", ""),
                ]
            )
        seg_table = Table(seg_rows, hAlign="LEFT", colWidths=[60, 60, 330])
        seg_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                    ("BOX", (0, 0), (-1, -1), 0.25, colors.black),
                    ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.grey),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ]
            )
        )
        story.append(seg_table)

    doc.build(story)
    pdf = buf.getvalue()
    buf.close()

    return (
        pdf,
        200,
        {
            "Content-Type": "application/pdf",
            "Content-Disposition": f'attachment; filename=\"{secure_filename(filename)}_report.pdf\"',
        },
    )
