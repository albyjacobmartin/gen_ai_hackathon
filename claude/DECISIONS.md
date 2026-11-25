# DECISIONS (confirmed by the user, 2026-11-25; do not change without asking)

## Product
| Topic | Decision |
|---|---|
| Solution | Grounded Q&A over uploaded notes, 4 modes (Explanations, Examples, Practice questions, Revision) |
| Scope | One subject per session, session reset, no weak-topic tracking or learner profile |
| UI | Collapsible left sidebar (upload, Index notes, Clear notes), main chat, mode selector above the chatbox |
| Prompting | Mode rewrites the prompt; fixed markdown template per mode, streamed (Practice renders after generation) |
| Trust | "Understood need" line; per-section tags [Notes: file, p.N] or [Not from your notes]; citation post-check; Sources expander |
| Off-notes questions | Answer from general knowledge with a "Not from your notes" banner |
| Practice | Questions visible; one hint and the answer hidden in expanders |
| Follow-ups | Retrieval query = last two student messages; last 4 turns in the prompt |
| Chunking | About 600 characters, overlap about 100, file and page metadata |

## Architecture
| Topic | Decision |
|---|---|
| Files | app.py, rag.py, prompts.py |
| Dependencies | streamlit, chromadb, openai (client only), pypdf; any addition needs approval |
| Storage | On-disk ChromaDB; Clear notes uses delete_collection |
| Python | 3.13 venv; fallback to 3.12 if chromadb will not install |

## Models (NVIDIA NIM, base URL https://integrate.api.nvidia.com/v1)
| Task | Model | Status |
|---|---|---|
| Embeddings | nvidia/nemotron-3.5-embed-1b | ID verified. query input_type unverified |
| Generation (answers, practice, revision) | nvidia/nemotron-3.5-lightning-30b-a3b | ID verified. Thinking off (unverified); sampling 1.0 / 0.95 to start |

Out of scope, on purpose:
- nemotron-3.5-parse: PDFs are text-selectable.
- nemotron-3.5-rerank-8b: about 30 chunks, unverified API, extra call. Hook kept: retrieve() returns a plain top-k list.
- nemotron-3.5-content-safety: no safety model in the demo; scope rule in the system prompt only.
- nemotron-3-embed-1b: deprecated (unsupported from 2026-11-23).
- embed-8b, other models: not needed.

## Risk rules
- Retry once on 429 or timeout, then show a friendly error.
- Model ID lives in one constant; swapping is manual.
- Day-one gates before any UI work. Freeze before the demo; record a backup run.
- Key in .env; .env and chroma_db/ in .gitignore before the first commit.
- Free tier is about 40 RPM (forum-reported, unverified); the design uses 2 calls per question.