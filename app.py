"""Study Companion: Streamlit UI. Run: streamlit run app.py"""
import os

import streamlit as st

import prompts
import rag

st.set_page_config(page_title="Study Companion", page_icon="📘", layout="wide")
st.session_state.setdefault("messages", [])


# ---------------------------------------------------------------- helpers
def show_banner(in_notes):
    if not in_notes:
        st.warning("**Not from your notes:** no relevant passage was found, so this answer uses general knowledge.")


def show_body(text, mode):
    practice = prompts.parse_practice(text) if mode == prompts.PRACTICE else None
    if practice is None:  # not Practice, or the format was off: show the text as is
        st.markdown(prompts.prettify(text))
        return
    if practice["intro"]:
        st.markdown(prompts.prettify(practice["intro"]))
    for i, item in enumerate(practice["items"], start=1):
        st.markdown(f"**Question {i}.** {item['q']}")
        with st.expander("💡 Hint"):
            st.markdown(item["hint"])
        with st.expander("✅ Answer"):
            st.markdown(prompts.prettify(item["answer"]))
        if item["source"]:
            st.markdown(prompts.prettify(item["source"]))


def show_sources(chunks, in_notes):
    if not chunks:
        return
    with st.expander("Sources"):
        for c in chunks:
            tag = "used" if in_notes else "not used (below threshold)"
            st.markdown(f"**{c['file']}, p.{c['page']}** · score {c['score']:.2f} · {tag}")
            st.caption(c["text"][:240].replace("\n", " "))


def render(message):
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.markdown(message["content"])
            return
        show_banner(message["in_notes"])
        show_body(message["content"], message["mode"])
        show_sources(message["sources"], message["in_notes"])


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("📚 Your notes")
    files = st.file_uploader("Upload PDF notes (one subject)", type="pdf", accept_multiple_files=True)
    if st.button("Index notes", type="primary", disabled=not files):
        with st.status("Indexing...", expanded=True) as status:
            for f in files:
                try:
                    r = rag.index_pdf(f.name, f.getvalue())
                    st.write(
                        {
                            "indexed": f"✅ {f.name}: {r['chunks']} chunks",
                            "skipped": f"↩️ {f.name}: already indexed",
                            "empty": f"⚠️ {f.name}: no readable text (scanned PDF?)",
                        }[r["status"]]
                    )
                except Exception as exc:
                    st.write(f"❌ {f.name}: {str(exc)[:200]}")
            status.update(label="Done", state="complete")
    try:
        indexed = rag.stats()
    except Exception as exc:
        indexed = {}
        st.error(f"Notes store error: {str(exc)[:200]}")
    for name, n in indexed.items():
        st.caption(f"📄 {name} · {n} chunks")
    if not indexed:
        st.caption("No notes indexed yet.")
    if st.button("Clear notes"):
        rag.clear()
        st.session_state.messages = []
        st.rerun()
    with st.expander("Advanced"):
        threshold = st.slider(
            "Relevance threshold", 0.0, 1.0, float(os.getenv("RELEVANCE_THRESHOLD", "0.30")), 0.01,
            help="Top match below this score is treated as 'not in your notes'.",
        )

# ---------------------------------------------------------------- main
st.title("📘 Study Companion")
st.caption("Answers come from your notes, with page tags. Pick a mode, then ask.")

if not rag.has_key():
    st.error(
        "NVIDIA_API_KEY is missing. Add NVIDIA_EMBED_API_KEY and NVIDIA_LLM_API_KEY "
        "(or one shared NVIDIA_API_KEY) to the .env file next to app.py, then restart."
    )
    st.stop()

# Pin the mode selector above the chatbox when this Streamlit version has st.bottom.
dock = getattr(st, "bottom", None) or getattr(st, "_bottom", None)
if dock is not None:
    with dock:
        mode = st.radio("Mode", prompts.MODES, horizontal=True, key="mode", label_visibility="collapsed")
else:
    mode = st.radio("Mode", prompts.MODES, horizontal=True, key="mode")

for message in st.session_state.messages:
    render(message)

prompt = st.chat_input("Ask about your notes...")
if prompt:
    history = list(st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Follow-ups: retrieve with the previous question + this one.
    earlier = [m["content"] for m in history if m["role"] == "user"][-1:]
    retrieval_query = " ".join(earlier + [prompt])

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching your notes..."):
                chunks = rag.retrieve(retrieval_query)
            in_notes = bool(chunks) and chunks[0]["score"] >= threshold
            used = chunks if in_notes else []
            show_banner(in_notes)
            messages = prompts.build_messages(mode, prompt, used, in_notes, history)
            if mode == prompts.PRACTICE:
                with st.spinner("Writing practice questions..."):
                    raw = rag.complete_chat(messages)
                text = prompts.check_tags(raw, used)
                show_body(text, mode)
            else:
                box, acc = st.empty(), ""
                for piece in rag.stream_chat(messages):
                    acc += piece
                    box.markdown(acc + " ▌")
                if not acc.strip():
                    raise RuntimeError("The model returned an empty reply.")
                text = prompts.check_tags(acc, used)
                box.markdown(prompts.prettify(text))
            show_sources(chunks, in_notes)
            st.session_state.messages.append(
                {"role": "assistant", "content": text, "mode": mode, "in_notes": in_notes, "sources": chunks}
            )
        except Exception as exc:
            st.error("The model API did not respond (rate limit or network). Please try again in a few seconds.")
            with st.expander("Details"):
                st.code(str(exc)[:400])
