import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Keep HF cache inside the repo to avoid re-downloading and to work offline
HF_HOME = os.environ.setdefault("HF_HOME", os.path.join(os.getcwd(), "models", "hf_cache"))
# Avoid HF tokenizer spawning extra worker processes (prevents semaphore leaks)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _resolve_local_model_path():
    """
    Prefer a fully cached local MiniLM model to avoid any network calls.
    Looks inside HF_HOME/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/<hash>
    and returns the newest snapshot path if found; otherwise returns None.
    """
    base = os.path.join(HF_HOME, "hub", "models--sentence-transformers--all-MiniLM-L6-v2", "snapshots")
    if not os.path.isdir(base):
        return None
    # pick the most recently modified snapshot dir
    candidates = [os.path.join(base, d) for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))]
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


local_model_path = _resolve_local_model_path()
model_name_or_path = local_model_path if local_model_path else "all-MiniLM-L6-v2"

# Force local_files_only to avoid HEAD/online checks when cache is present
model = SentenceTransformer(
    model_name_or_path,
    cache_folder=os.path.join(HF_HOME, "hub"),
    local_files_only=bool(local_model_path),
)  # lightweight


def build_vectorstore(chunks):
    print(f"[EMBED] start, chunks={len(chunks)}")
    texts = [c["text"] for c in chunks if c.get("text")]
    if not texts:
        raise ValueError("No text available to embed")

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        batch_size=64,            # larger batch for throughput
        show_progress_bar=False,
    )
    embeddings = np.atleast_2d(embeddings)
    if embeddings.size == 0 or embeddings.shape[0] == 0:
        raise ValueError("Embedding model returned empty embeddings")


    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(np.array(embeddings))
    print(f"[EMBED] built index dim={dim} n={index.ntotal}")


    return index, model
