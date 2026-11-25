"""Day-one gates. Run from the project folder with the venv active:

    python smoke_test.py                      # gates 1-3 and 5
    python smoke_test.py notes1.pdf notes2.pdf  # plus gate 4 (PDF text check)
"""
import sys
import time

import rag


def ok(msg):
    print(f"  PASS  {msg}")


def bad(msg):
    print(f"  FAIL  {msg}")


print("1. imports")
try:
    import chromadb, openai, pypdf, streamlit  # noqa: E401,F401

    ok(f"streamlit {streamlit.__version__}, chromadb {chromadb.__version__}, "
       f"openai {openai.__version__}, pypdf {pypdf.__version__}")
except Exception as exc:
    bad(repr(exc))
    sys.exit(1)

print("2. embeddings (passage vs query)")
if not rag.has_key():
    bad("API keys missing in .env (NVIDIA_EMBED_API_KEY and NVIDIA_LLM_API_KEY, or NVIDIA_API_KEY)")
    sys.exit(1)
try:
    docs = [
        "Ohm's law states that voltage equals current times resistance (V = I R).",
        "Photosynthesis converts light energy into chemical energy in plants.",
        "A binary search tree keeps smaller keys on the left and larger keys on the right.",
    ]
    vecs = rag.embed_texts(docs, "passage")
    query = rag.embed_texts(["What does Ohm's law say?"], "query")[0]
    sims = [sum(a * b for a, b in zip(query, v)) for v in vecs]
    best = sims.index(max(sims))
    (ok if best == 0 else bad)(f"top match is doc {best} (expected 0); scores {[round(s, 3) for s in sims]}")
    ok(f"vector size {len(query)}; request shape used: passage={rag._variant['passage']}, query={rag._variant['query']}")
except Exception as exc:
    bad(repr(exc)[:400])

print("3. LLM stream (thinking off)")
try:
    t0, first, text = time.time(), None, ""
    for piece in rag.stream_chat([{"role": "user", "content": "Say hello in five words."}], max_tokens=200):
        first = first or time.time() - t0
        text += piece
    if text.strip():
        ok(f"first token in {first:.1f}s, reply: {text.strip()[:80]!r}")
    else:
        bad("empty reply (thinking may be using the token budget)")
    print(f"        thinking_off accepted by endpoint: {rag.LLM_STATE['thinking_off']}")
except Exception as exc:
    bad(repr(exc)[:400])

print("4. PDF text check")
if len(sys.argv) == 1:
    print("  (skipped: pass PDF paths as arguments)")
for path in sys.argv[1:]:
    try:
        with open(path, "rb") as fh:
            pages = rag.extract_pages(fh.read())
        chunks = sum(len(rag.chunk_text(t)) for _, t in pages)
        sample = pages[0][1][:150].replace("\n", " ") if pages else ""
        (ok if pages else bad)(f"{path}: {len(pages)} pages with text, {chunks} chunks")
        print(f"        page 1 starts: {sample!r}")
    except Exception as exc:
        bad(f"{path}: {exc!r}")

print("5. Streamlit st.bottom (pinned mode selector)")
import streamlit as st  # noqa: E402

print(f"  st.bottom: {hasattr(st, 'bottom')}, st._bottom: {hasattr(st, '_bottom')} "
      "(if both False the selector sits above the chat instead; that is fine)")
