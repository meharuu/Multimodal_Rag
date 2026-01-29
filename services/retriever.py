import numpy as np


def retrieve_chunks(query, index, chunks, embedder, k=5):
    q_emb = embedder.encode([query], normalize_embeddings=True)
    _, idxs = index.search(np.array(q_emb), k)
    return [chunks[i] for i in idxs[0]]