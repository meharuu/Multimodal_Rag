import os
# Silence noisy semaphore cleanup warnings from loky/resource_tracker globally (applies to child processes too)
os.environ.setdefault("PYTHONWARNINGS", "ignore:resource_tracker:UserWarning")

from flask import Flask, render_template, request, jsonify, redirect
import multiprocessing as mp
import warnings


from config import UPLOAD_DIR, VIDEO_DIR
from services.video_processor import extract_audio
from services.transcriber import transcribe_audio
from services.chunker import chunk_segments
from services.embedder import build_vectorstore
from services.retriever import retrieve_chunks
from services.llm import generate_answer


app = Flask(__name__)

# Use spawn to avoid semaphore leak warnings from forked workers (loky/tokenizers)
try:
    mp.set_start_method("spawn")
except RuntimeError:
    pass

# Suppress noisy semaphore leak warnings from underlying joblib/loky
warnings.filterwarnings(
    "ignore",
    message=r".*resource_tracker: There appear to be .* leaked semaphore objects.*",
    category=UserWarning,
)


STATE = {}


@app.route("/")
def index():
    return render_template("index.html")



@app.route("/upload", methods=["POST"])
def upload_video():
    video = request.files["video"]
    video_path = os.path.join(VIDEO_DIR, video.filename)
    video.save(video_path)

    audio_path = extract_audio(video_path)

    try:
        transcript = transcribe_audio(audio_path)
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 400

    segments = transcript.get("segments", [])
    if not segments:
        return jsonify({"error": "No audio detected or transcription failed"}), 400

    chunks = chunk_segments(segments)
    try:
        index, embedder = build_vectorstore(chunks)  # keep your existing embedding/FAISS code
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    video_id = video.filename
    STATE[video_id] = {
        "chunks": chunks,
        "index": index,
        "embedder": embedder
    }

    return jsonify({"video_id": video_id})




@app.route("/chat")
def chat():
    return render_template("chat.html", video=STATE.get("video"))


TRANSCRIPTS = {}

@app.route("/ask", methods=["POST"])
def ask_question():
    try:
        data = request.get_json(force=True)
        question = data.get("question", "")
        video_id = data.get("video_id")

        if not video_id:
            return jsonify({"error": "Missing video_id"}), 400
        if video_id not in STATE:
            return jsonify({"error": "Video not processed"}), 400

        store = STATE[video_id]

        print(f"[ASK] video_id={video_id} question='{question}'")
        retrieved_chunks = retrieve_chunks(
            question,
            store["index"],
            store["chunks"],
            store["embedder"],
            k=2  # smaller for speed
        )
        print(f"[ASK] retrieved {len(retrieved_chunks)} chunks")

        print("[ASK] generating answer...")
        answer = generate_answer(question, retrieved_chunks)
        print("[ASK] answer ready")

        return jsonify({
            "answer": answer,
            "sources": retrieved_chunks
        })
    except Exception as e:
        return jsonify({"error": f"Ask failed: {e}"}), 500



if __name__ == "__main__":
    # Bind to all interfaces to work with proxies/port-forwarding and avoid localhost blocks
    app.run(host="0.0.0.0", port=5000, debug=True)
