# import subprocess
# import os
# from config import AUDIO_DIR

# def extract_audio(video_path: str) -> str:
#     """
#     Extracts audio from a video and saves it to AUDIO_DIR.
#     Returns the audio filename.
#     """
#     audio_filename = os.path.splitext(os.path.basename(video_path))[0] + ".mp3"
#     audio_path = os.path.join(AUDIO_DIR, audio_filename)

#     command = [
#         "ffmpeg",
#         "-y",
#         "-i", video_path,
#         "-vn",
#         "-acodec", "mp3",
#         audio_path
#     ]

#     subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

#     if not os.path.exists(audio_path):
#         raise RuntimeError(f"FFmpeg failed to extract audio: {audio_path}")

#     return audio_filename

import os
import subprocess

# Make sure audio directory exists
AUDIO_DIR = "audio"
os.makedirs(AUDIO_DIR, exist_ok=True)

def extract_audio(video_path):
    """
    Extracts audio from a video file using ffmpeg.
    Saves as WAV 16kHz mono (required for Whisper)
    Returns the path to the audio file
    """
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    audio_path = os.path.join(AUDIO_DIR, base_name + ".wav")

    command = [
        "ffmpeg",
        "-y",               # overwrite if exists
        "-i", video_path,   # input video
        "-ac", "1",         # mono
        "-ar", "16000",     # 16kHz
        "-vn",              # no video
        "-f", "wav",        # output format
        audio_path
    ]

    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to extract audio: {e.stderr.decode()}")

    return audio_path
