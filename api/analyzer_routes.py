import os
import ssl
import tempfile
from datetime import datetime
from functools import lru_cache
from pathlib import Path

from flask import Blueprint, current_app, jsonify, request
from werkzeug.utils import secure_filename

from api.history_routes import add_history

analyzer_bp = Blueprint("analyzer", __name__)


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


@analyzer_bp.route("/")
def index():
    # Serve the single-page UI
    return current_app.send_static_file("audio_emotion_analyzer.html")


@analyzer_bp.post("/analyze")
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
        add_history(
            {
                "filename": file.filename,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "emotion": response["emotion"]["label"],
                "confidence": response["emotion"]["score"],
                "language": response["language"],
                "duration": response["duration"],
                "transcript": response["transcript"],
            }
        )
        return jsonify(response)
    except Exception as exc:  # pragma: no cover
        current_app.logger.exception("analysis failed")
        return jsonify({"error": str(exc)}), 500
