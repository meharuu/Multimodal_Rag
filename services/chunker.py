def chunk_segments(segments, max_words=120):
    """
    Splits transcription segments into manageable chunks.
    segments: list of dicts with 'start', 'end', 'text'
    """
    if not segments:
        return []

    chunks = []
    current_text = ""
    start_time = segments[0]["start"]

    for seg in segments:
        current_text += " " + seg["text"]

        if len(current_text.split()) >= max_words:
            chunks.append({
                "text": current_text.strip(),
                "start": start_time,
                "end": seg["end"]
            })
            current_text = ""
            start_time = seg["end"]

    # Add remaining text
    if current_text.strip():
        chunks.append({
            "text": current_text.strip(),
            "start": start_time,
            "end": segments[-1]["end"]
        })

    return chunks
