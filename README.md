# 7CS 1018 - Gen AI Hackathon

PS 06: Personalized AI Learning Companion for Engineering Students (RAG).
Upload your notes (PDF), pick a mode (Explanations, Examples, Practice questions, Revision) and ask.
Answers come from the notes with page tags; anything else is labelled "Not from your notes".

## Setup (Windows, Python 3.13)
```
.venv\Scripts\activate
pip install -r requirements.txt
```
Put your keys in `.env` (one per model, or a single shared `NVIDIA_API_KEY`):
```
NVIDIA_EMBED_API_KEY=nvapi-...
NVIDIA_LLM_API_KEY=nvapi-...
```
(`.env` and `chroma_db/` are git-ignored.)

## Check the setup, then run
```
python smoke_test.py notes1.pdf
streamlit run app.py
```
Use one subject per session: upload the PDF in the sidebar, click **Index notes** (do this before the demo),
and use **Clear notes** to start a new subject.

## Files
- `app.py` - Streamlit UI
- `rag.py` - PDF extraction, chunking, embeddings, ChromaDB, NVIDIA NIM calls
- `prompts.py` - system prompt, mode templates, tag check, Practice parser
- `smoke_test.py` - day-one checks against the real API
