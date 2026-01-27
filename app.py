import os
import ssl
import tempfile
from functools import lru_cache
from pathlib import Path
from datetime import datetime

from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

# Lazy-load heavy models so the app can start quickly.


@lru_cache(maxsize=1)
def get_whisper_model():
    import whisper
    model_name = os.getenv("WHISPER_MODEL", "small")
    download_root = os.getenv("WHISPER_DOWNLOAD_ROOT")  # optional local cache dir

    def load(insecure: bool = False):
        if insecure:
            ssl._create_default_https_context = ssl._create_unverified_context
        return whisper.load_model(model_name, download_root=download_root)

    # primary attempt
    try:
        return load(insecure=os.getenv("ALLOW_INSECURE_SSL") == "1")
    except Exception as exc:
        # Retry once with insecure context if cert failure detected
        msg = str(exc).lower()
        if "certificate verify failed" in msg and os.getenv("ALLOW_INSECURE_SSL") != "1":
            return load(insecure=True)
        raise


@lru_cache(maxsize=1)
def get_emotion_pipeline():
    from transformers import pipeline
    if os.getenv("ALLOW_INSECURE_SSL") == "1":
        # HuggingFace hub respects this flag to skip SSL verification.
        os.environ["HF_HUB_DISABLE_SSL_VERIFY"] = "1"

    model_name = os.getenv(
        "EMOTION_MODEL", "j-hartmann/emotion-english-distilroberta-base"
    )
    return pipeline("text-classification", model=model_name, top_k=None)


app = Flask(__name__, static_folder=".", template_folder=".")
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100MB uploads
HISTORY = []


@app.route("/")
def index():
    # Serve the single-page UI
    return app.send_static_file("audio_emotion_analyzer.html")


@app.get("/history")
def history_page():
    return app.send_static_file("query_history.html")


@app.get("/history/data")
def history_data():
    return jsonify({"items": HISTORY})


@app.post("/analyze")
def analyze():
    try:
        if "audio" not in request.files:
            return jsonify({"error": "No audio file uploaded."}), 400

        file = request.files["audio"]
        if not file.filename:
            return jsonify({"error": "Empty filename."}), 400

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / secure_filename(file.filename)
            file.save(path)

            # Transcribe with Whisper
            whisper_model = get_whisper_model()
            result = whisper_model.transcribe(str(path))
            transcript = (result.get("text") or "").strip()
            segments = result.get("segments", [])
            duration = result.get("duration")

            # Classify emotions on the transcript text
            emotion_pipe = get_emotion_pipeline()
            raw_scores = emotion_pipe(transcript) if transcript else []
            # pipeline returns list[list[dict]] when top_k=None
            if raw_scores and isinstance(raw_scores[0], list):
                raw_scores = raw_scores[0]

            scores = [
                {"label": s["label"], "score": round(float(s["score"]), 4)}
                for s in raw_scores
            ]
            top = max(raw_scores, key=lambda x: x["score"]) if scores else None

            response = {
                "transcript": transcript,
                "language": result.get("language"),
                "duration": duration,
                "emotion": {
                    "label": top["label"] if top else None,
                    "score": round(float(top["score"]), 4) if top else None,
                },
                "scores": scores,
                "segments": [
                    {
                        "start": seg.get("start"),
                        "end": seg.get("end"),
                        "text": (seg.get("text") or "").strip(),
                    }
                for seg in segments
            ],
        }
        HISTORY.append(
            {
                "id": len(HISTORY) + 1,
                "filename": file.filename,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "emotion": response["emotion"]["label"],
                "confidence": response["emotion"]["score"],
                "language": response["language"],
                "duration": response["duration"],
                "transcript": response["transcript"],
            }
        )
        # Keep only the last 200 entries
        if len(HISTORY) > 200:
            del HISTORY[:-200]
        return jsonify(response)
    except Exception as exc:  # pragma: no cover
        app.logger.exception("analysis failed")
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=True)


@app.post("/report")
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
            "Content-Disposition": f'attachment; filename="{secure_filename(filename)}_report.pdf"',
        },
    )
