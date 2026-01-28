from flask import Blueprint, current_app, jsonify

history_bp = Blueprint("history", __name__)

# In-memory query history for the session
HISTORY = []
MAX_HISTORY = 200


def add_history(entry: dict):
    """Append an analysis result to the in-memory history."""
    entry = dict(entry)
    entry.setdefault("id", len(HISTORY) + 1)
    HISTORY.append(entry)
    if len(HISTORY) > MAX_HISTORY:
        del HISTORY[:-MAX_HISTORY]


@history_bp.get("/history")
def history_page():
    return current_app.send_static_file("query_history.html")


@history_bp.get("/history/data")
def history_data():
    return jsonify({"items": HISTORY})
