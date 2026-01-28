import json
import threading
from datetime import datetime
from pathlib import Path

from flask import Blueprint, current_app, jsonify, request

feedback_bp = Blueprint("feedback", __name__)

# Keep data file at project root so moves don't create new copies.
FEEDBACK_FILE = Path(__file__).resolve().parent.parent / "feedback_data.json"
FEEDBACK_LOCK = threading.Lock()


def _load_feedback():
    if not FEEDBACK_FILE.exists():
        return []
    try:
        with FEEDBACK_FILE.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def _write_feedback(entries):
    tmp_path = FEEDBACK_FILE.with_suffix(".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)
    tmp_path.replace(FEEDBACK_FILE)


@feedback_bp.get("/feedback")
def feedback_page():
    return current_app.send_static_file("feedback.html")


@feedback_bp.get("/feedback/list")
def feedback_list_page():
    return current_app.send_static_file("feedback_list.html")


@feedback_bp.post("/api/feedback")
def submit_feedback():
    data = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip()
    rating = str(data.get("rating") or "").strip()

    if not message:
        return jsonify({"error": "Message is required."}), 400
    if rating not in {"1", "2", "3", "4", "5"}:
        return jsonify({"error": "Rating must be 1-5."}), 400

    entry = {
        "id": int(datetime.utcnow().timestamp() * 1000),
        "created_at": datetime.utcnow().isoformat() + "Z",
        "name": (data.get("name") or "").strip() or None,
        "email": (data.get("email") or "").strip() or None,
        "rating": int(rating),
        "message": message,
    }

    with FEEDBACK_LOCK:
        items = _load_feedback()
        items.append(entry)
        # keep file reasonably small
        if len(items) > 500:
            items = items[-500:]
        _write_feedback(items)

    return jsonify({"ok": True, "id": entry["id"]})


@feedback_bp.get("/api/feedback")
def list_feedback():
    with FEEDBACK_LOCK:
        items = _load_feedback()
    # newest first
    items = sorted(items, key=lambda x: x.get("created_at", ""), reverse=True)
    return jsonify({"items": items})
