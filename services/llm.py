from gpt4all import GPT4All
import os
import time

MODEL_DIR = "models"
# Use a smaller, faster local model for quicker responses on CPU
MODEL_FILE = "mistral-7b-instruct-v0.2.Q2_K.gguf"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE)

# Ensure model exists
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found at {MODEL_PATH}. "
        "Download it manually and place it in the models/ directory."
    )

# ✅ CRITICAL FIX: disable downloading
llm = GPT4All(
    model_name=MODEL_FILE,
    model_path=MODEL_DIR,
    allow_download=False,   # 🔒 OFFLINE MODE
    device="cpu",
    n_threads=os.cpu_count() or 4,
)


def generate_answer(question, context_chunks):
    print("[LLM] start generation")
    context = "\n".join(
        f"[{c['start']}s–{c['end']}s] {c['text']}"
        for c in context_chunks
    )

    # Trim context to keep prompt small for speed
    max_words = 800
    words = context.split()
    if len(words) > max_words:
        context = " ".join(words[:max_words])
        print(f"[LLM] context trimmed to {max_words} words")

    prompt = f"""
You are a question-answering assistant.
Answer strictly using the context below.


Context:
{context}

Question:
{question}

Answer:
"""
    start = time.time()
    out = llm.generate(
        prompt,
        max_tokens=256,   # tighter cap for speed
        temp=0.2,
        top_p=0.9,
    )
    print(f"[LLM] generation done in {time.time()-start:.1f}s")
    return out
