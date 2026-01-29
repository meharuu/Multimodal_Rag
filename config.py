import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
VIDEO_DIR = os.path.join(UPLOAD_DIR, "videos")
AUDIO_DIR = os.path.join(UPLOAD_DIR, "audio")

os.makedirs(VIDEO_DIR, exist_ok=True)
os.makedirs(AUDIO_DIR, exist_ok=True)
