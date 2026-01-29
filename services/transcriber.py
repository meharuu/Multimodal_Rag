# from faster_whisper import WhisperModel
# import os
# from config import AUDIO_DIR

# model = WhisperModel(
#     "base",
#     device="cpu",
#     compute_type="int8"
# )

# def transcribe_audio(audio_filename: str):
#     audio_path = os.path.join(AUDIO_DIR, audio_filename)

#     if not os.path.exists(audio_path):
#         raise FileNotFoundError(f"Audio file not found: {audio_path}")

#     segments, info = model.transcribe(audio_path)

#     return {
#         "text": " ".join(segment.text for segment in segments),
#         "segments": [
#             {
#                 "start": segment.start,
#                 "end": segment.end,
#                 "text": segment.text
#             }
#             for segment in segments
#         ]
#     }

import os
import warnings
import whisper

# Silence the FP16-on-CPU warning emitted by whisper when running on CPU
warnings.filterwarnings(
    "ignore",
    message=r"FP16 is not supported on CPU; using FP32 instead",
    category=UserWarning,
)

# Choose the lightest model by default for faster CPU inference; override via env
MODEL_NAME = os.getenv("WHISPER_MODEL", "tiny")
model = whisper.load_model(MODEL_NAME)

def transcribe_audio(audio_path):
    """
    Transcribes audio and returns a dict with 'text' and 'segments'.
    Raises RuntimeError if transcription fails (e.g., bad audio leading to NaNs).
    """
    try:
        print(f"[TRANSCRIBE] start {audio_path}")
        result = model.transcribe(
            audio_path,
            fp16=False,
            language="en",
            beam_size=1,   # speed: greedy-ish
            best_of=1,
            temperature=0,
        )
        print(f"[TRANSCRIBE] done {audio_path}")
    except Exception as e:
        raise RuntimeError(f"Transcription failed: {e}")

    segments = result.get("segments", []) or []
    text = result.get("text", "") or ""
    return {"text": text, "segments": segments}
