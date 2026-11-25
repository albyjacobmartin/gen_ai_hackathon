"""Retrieval and NVIDIA NIM calls. No Streamlit imports here (keeps it testable)."""
import hashlib
import io
import os
import re
from pathlib import Path

import chromadb
from openai import BadRequestError, OpenAI, UnprocessableEntityError
from pypdf import PdfReader


def _load_env():
    """Tiny .env reader so we need no extra dependency."""
    path = Path(__file__).with_name(".env")
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and value and not os.environ.get(key):
            os.environ[key] = value


_load_env()

BASE_URL = "https://integrate.api.nvidia.com/v1"
EMBED_MODEL = os.getenv("EMBED_MODEL", "nvidia/nemotron-3.5-embed-1b")
LLM_MODEL = os.getenv("LLM_MODEL", "nvidia/nemotron-3.5-lightning-30b-a3b")  # one constant: swap here
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "1.0"))  # model card: 1.0 / 0.95
LLM_TOP_P = float(os.getenv("LLM_TOP_P", "0.95"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "4096"))

DB_PATH = str(Path(__file__).with_name("chroma_db"))
COLLECTION = "notes"
CHUNK_SIZE, OVERLAP, EMBED_BATCH = 600, 100, 16

_clients = {}


def _key(kind):
    """kind is 'EMBED' or 'LLM'. NVIDIA_EMBED_API_KEY / NVIDIA_LLM_API_KEY, else the shared NVIDIA_API_KEY."""
    return os.getenv(f"NVIDIA_{kind}_API_KEY") or os.getenv("NVIDIA_API_KEY")


def has_key():
    return bool(_key("EMBED") and _key("LLM"))


def client(kind="LLM"):
    if kind not in _clients:
        key = _key(kind)
        if not key:
            raise RuntimeError(f"NVIDIA_{kind}_API_KEY (or NVIDIA_API_KEY) is missing. Put it in the .env file.")
        # max_retries=1: one automatic retry on 429 / timeout / 5xx, then surface the error.
        _clients[kind] = OpenAI(base_url=BASE_URL, api_key=key, timeout=90, max_retries=1)
    return _clients[kind]


# ---------------------------------------------------------------- PDF -> chunks
def clean(text):
    text = text.replace("\r", "")
    text = re.sub(r"-\n(?=[a-z])", "", text)  # re-join words hyphenated at line end
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pages(data: bytes):
    """Return [(page_number, text)] for pages that have text."""
    reader = PdfReader(io.BytesIO(data))
    pages = []
    for number, page in enumerate(reader.pages, start=1):
        text = clean(page.extract_text() or "")
        if text:
            pages.append((number, text))
    return pages


def chunk_text(text, size=CHUNK_SIZE, overlap=OVERLAP):
    """~size-char chunks, cut at a line/sentence end when possible, with overlap."""
    chunks, start, n = [], 0, len(text)
    while start < n:
        end = min(start + size, n)
        if end < n:
            window = text[start:end]
            cut = max(window.rfind("\n"), window.rfind(". "), window.rfind("; "))
            if cut > size * 0.5:
                end = start + cut + 1
        piece = text[start:end].strip()
        if len(piece) >= 30:
            chunks.append(piece)
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return chunks


# ---------------------------------------------------------------- embeddings
# The hosted embed API details are only partly documented, so try a few request
# shapes and remember the first one that works (per input type).
_EMBED_VARIANTS = [
    lambda t: {"input_type": t, "truncate": "END"},
    lambda t: {"input_type": t, "truncate": "END", "modality": "text"},
    lambda t: {"truncate": "END"},
]
_variant = {"passage": 0, "query": 0}


def _embed_batch(batch, input_type):
    last = None
    for i in range(_variant[input_type], len(_EMBED_VARIANTS)):
        try:
            resp = client("EMBED").embeddings.create(
                input=batch,
                model=EMBED_MODEL,
                encoding_format="float",
                extra_body=_EMBED_VARIANTS[i](input_type),
            )
            _variant[input_type] = i
            return [d.embedding for d in resp.data]
        except (BadRequestError, UnprocessableEntityError) as exc:
            last = exc
    raise last


def embed_texts(texts, input_type):
    """input_type: 'passage' for notes, 'query' for questions."""
    vectors = []
    for i in range(0, len(texts), EMBED_BATCH):
        vectors += _embed_batch(texts[i : i + EMBED_BATCH], input_type)
    return vectors


# ---------------------------------------------------------------- Chroma
_chroma = None


def _db():
    global _chroma
    if _chroma is None:
        _chroma = chromadb.PersistentClient(path=DB_PATH)
    return _chroma


def _col():
    return _db().get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})


def index_pdf(name, data: bytes):
    """Index one PDF. Skips unchanged files (same name + content hash)."""
    digest = hashlib.sha1(data).hexdigest()
    col = _col()
    old = col.get(where={"file": name}, include=["metadatas"])
    if old["ids"]:
        if old["metadatas"][0].get("hash") == digest:
            return {"status": "skipped", "chunks": len(old["ids"])}
        col.delete(where={"file": name})
    items = [(page, chunk) for page, text in extract_pages(data) for chunk in chunk_text(text)]
    if not items:
        return {"status": "empty", "chunks": 0}
    vectors = embed_texts([chunk for _, chunk in items], "passage")
    key = hashlib.sha1(name.encode()).hexdigest()[:8]
    col.add(
        ids=[f"{key}-{i}" for i in range(len(items))],
        embeddings=vectors,
        documents=[chunk for _, chunk in items],
        metadatas=[{"file": name, "page": page, "hash": digest} for page, _ in items],
    )
    return {"status": "indexed", "chunks": len(items)}


def stats():
    """{file name: chunk count} for everything indexed."""
    out = {}
    for meta in _col().get(include=["metadatas"])["metadatas"]:
        out[meta["file"]] = out.get(meta["file"], 0) + 1
    return out


def clear():
    try:
        _db().delete_collection(COLLECTION)  # not a folder delete: avoids Windows file locks
    except Exception:
        pass


def retrieve(query, k=5):
    """Top-k chunks as dicts with similarity score (cosine, higher = closer)."""
    col = _col()
    count = col.count()
    if count == 0:
        return []
    vector = embed_texts([query], "query")[0]
    res = col.query(
        query_embeddings=[vector],
        n_results=min(k, count),
        include=["documents", "metadatas", "distances"],
    )
    return [
        {"text": doc, "file": meta["file"], "page": int(meta["page"]), "score": round(1 - dist, 4)}
        for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
    ]


# ---------------------------------------------------------------- LLM
LLM_STATE = {"thinking_off": None}  # None = untested, True = accepted, False = rejected


def stream_chat(messages, temperature=None, top_p=None, max_tokens=None):
    """Yield answer text as it streams. Reasoning tokens (if any) are ignored."""
    kwargs = dict(
        model=LLM_MODEL,
        messages=messages,
        temperature=LLM_TEMPERATURE if temperature is None else temperature,
        top_p=LLM_TOP_P if top_p is None else top_p,
        max_tokens=max_tokens or LLM_MAX_TOKENS,
        stream=True,
    )
    stream = None
    if LLM_STATE["thinking_off"] is not False:
        try:
            stream = client("LLM").chat.completions.create(
                **kwargs, extra_body={"chat_template_kwargs": {"enable_thinking": False}}
            )
            LLM_STATE["thinking_off"] = True
        except (BadRequestError, UnprocessableEntityError):
            LLM_STATE["thinking_off"] = False  # endpoint rejects it: fall back to default
    if stream is None:
        stream = client("LLM").chat.completions.create(**kwargs)
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if delta.content:
            yield delta.content


def complete_chat(messages, **kw):
    """Non-streamed reply (used by Practice). Retries once if the reply is empty."""
    for _ in range(2):
        text = "".join(stream_chat(messages, **kw))
        if text.strip():
            return text
    raise RuntimeError("The model returned an empty reply.")
