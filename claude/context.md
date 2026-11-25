# context.md - GenAI Hackathon (PS 06)
Last updated: 2026-11-25, Task 2 (solution design and plan) complete and confirmed by the user.
Status: Design decided. Nothing built, run or tested. Next: the user says "start implementation".
How to use: paste into a new chat or keep in the Project so Claude starts with the facts. Update only at milestones.

## 1. Working rules (user-stated)
- When the user says "update", give the full updated file contents. The user applies files manually.
- Code delivery: give the code and say which file to add or update. The user pastes it (no folder is connected).
- One chat per task; new chats start from this file. Token efficiency matters.
- Technical help only. The user handles judging criteria and the presentation.
- No code, files or slides until the user says "start implementation".
- Ask before adding a dependency or changing a decision. Flag anything that could break the live demo. Plan in 5 bullets before anything bigger than one file. Free tiers only. Concise, code first.
- Ponytail (smallest design that demos well) is on.
- Time is very short: finish as soon as possible.

## 2. The hackathon
- Theme: Gen AI. Category B: Education and Skill Development. SDG 4.
- PS 06: Personalized AI Learning Companion for Engineering Students. A RAG-based learning assistant that generates personalized explanations, examples, practice questions and revision material from the syllabus.
- 6 hours, team of 4, only the user is working.

## 3. Environment
- Windows PC, Python 3.13, a venv. Claude Pro (no API usage), so NVIDIA NIM is the model API.
- NVIDIA API key: created. Demo data: 3 PDFs, 3 different subjects, up to 6 pages each, text-selectable, formulas as plain text. One subject is used per session.
- Empty GitHub repo cloned locally (path unknown). No folder connected to Claude.
- Plugin trimming is undecided and not project-critical.

## 4. NVIDIA NIM facts (checked 2026-11-25)
Verified from NVIDIA docs and samples:
- Base URL https://integrate.api.nvidia.com/v1, OpenAI-compatible; use the openai client.
- Embeddings: POST /v1/embeddings, model nvidia/nemotron-3.5-embed-1b, one 2048-d L2-normalised vector per input, 32,768-token context, truncate "END"/"NONE", free endpoint.
- Do NOT use nemotron-3-embed-1b: deprecated, unsupported from 2026-11-23.
- LLM: nvidia/nemotron-3.5-lightning-30b-a3b via chat.completions with streaming. 30B MoE, 3B active, reasoning model. NVIDIA's sample uses extra_body={"chat_template_kwargs":{"enable_thinking":True},"reasoning_budget":16384} and reads delta.reasoning_content. Model card recommends temperature 1.0, top_p 0.95.
Unverified (test in hour 1):
- input_type: "passage" for documents is documented; "query" for questions is assumed.
- Embedding batch size limit (not stated): use small batches.
- enable_thinking=False accepted by the hosted endpoint.
- Free tier about 40 RPM: forum reports only, not official docs. A forum reply says NVIDIA cannot raise it for free personal accounts.
- Reranker and content-safety hosted request schemas were not found (not used). Parse not checked (not needed).
- Python 3.13: streamlit supports it; chromadb likely (a maintainer reported it installs on 3.13.3), Windows wheel unconfirmed; openai and pypdf unchecked (pure Python).
- Whether st.bottom exists in the installed Streamlit.

## 5. Task 1 findings (full detail in problem_research.md)
Confidence: A = peer-reviewed/official, B = credible reporting or preprint, C = secondary summary.
- Scale: B.Tech seats filled 75.07% of intake in 2024-25 (AICTE via Careers360, B). About 1.5 million engineers graduate a year (TeamLease, C).
- Employability: BE/BTech 70.15% in 2026, CS/IT about 78 to 80% (India Skills Report 2026, vendor test, B). Placement estimates conflict: 47.7% (AICTE) vs about 10% (TeamLease).
- Weak fundamentals: Anna University first-year (2017) physics about 50% failed, maths about 45%, chemistry about 38% (B, dated).
- Faculty: Karnataka government engineering colleges 307 of 646 teaching posts vacant, 47.5% (Dec 2024, B).
- Students use AI but distrust it: Delhi, 6,000+ students: 23% daily use, 6% fully trust accuracy, 45% don't know how to use it well (B).
- For designed tutors: Harvard physics RCT, 194 students, more than 2x active-learning gain, tutor seeded with expert step-by-step solutions (A).
- Against unguarded tutors: Turkish school experiment 17% lower closed-book exam; university trial 0.37 SD lower (C; verify before quoting).
- Existing tools: ChatGPT Study Mode (not notes-limited), NotebookLM (grounded, weak quizzes, no student profile), a RAG tutor preprint (text only, gave full homework answers).
- Not found: rupee cost of failure, coaching spend, VTU/Karnataka pass rates, regional-language data, Indian edtech evaluation.

## 6. Task 2 decisions (all confirmed by the user)
Product (a-c):
- Grounded Q&A over the student's uploaded notes with 4 modes: Explanations, Examples, Practice questions, Revision. One subject per session, session reset. No weak-topic tracking, no learner profile.
- UI: collapsible left sidebar (PDF upload, Index notes, Clear notes); main chat; mode selector pinned above the chatbox.
- The selected mode rewrites the prompt (mode template) before the RAG pipeline; the model must answer in a fixed per-mode template.
- Streaming for Explain, Examples, Revision. Practice renders after generation completes.
- Practice: questions visible; one hint and the answer each hidden in expanders.
- Every answer starts with an "Understood need" line. Each section ends with a tag: [Notes: file, p.N] or [Not from your notes].
- Fixed markdown headings per mode, not JSON.
- Off-notes questions: answered from general knowledge with a "Not from your notes" banner and tags.
- Sources expander under each answer (file, page, snippet, top-1 score for debugging).
- Live upload, done before the demo. No "Load demo notes" button. No search-in filter.
Concepts (d): chunking with file/page metadata; embeddings and semantic search; prompt augmentation; structured generation; hallucination mitigation (grounding rules, tags, post-check, threshold banner); pedagogical hint prompting; zero-shot need inference; conversation memory (last 4 turns; retrieval query = last two student messages joined); streaming.
Design (e): three files: app.py (UI), rag.py (extract, chunk, embed, index, retrieve), prompts.py (system prompt and 4 mode templates). Dependencies approved: streamlit, chromadb, openai (client only), pypdf. On-disk ChromaDB (PersistentClient) so a refresh keeps the notes; "Clear notes" uses delete_collection, not a folder delete.
Models (f): core path only: embed-1b plus lightning-30b. Parse, reranker and safety are out. Thinking off (fallback: keep on, ignore reasoning_content, show a spinner). Sampling starts at 1.0 / 0.95; drop to about 0.6 only if outputs break the template or tags. Model ID in one constant. No safety model for the demo; scope rule in the system prompt only. retrieve() returns a plain top-k list so a reranker can be added later.
Risks (g): see section 8. Citation post-check: yes. One retry on 429 or timeout, then a friendly error.

## 7. Architecture and data flow
```
INGEST (sidebar "Index notes")
 PDF -> pypdf text per page -> normalise -> ~600-char chunks, overlap ~100, keep {file, page}
 -> embed (input_type=passage, small batches) -> Chroma (cosine), skip if file hash unchanged

ASK (chatbox + mode)
 prompt + mode -> retrieval query = last 2 student messages
 -> embed (input_type=query) -> Chroma top-5 + scores
 -> top-1 similarity vs threshold: "in notes" or "not in notes" (banner)
 -> messages: system (role, grounding, tag rules, scope rule) + mode template
 + context chunks labelled [file p.N] + last 4 turns + prompt
 -> stream from lightning-30b (Practice: wait, then parse markers)
 -> tag post-check against the retrieved (file, page) set
 -> render + Sources expander
```
Calls: ingest about 2 to 3 embed calls per subject; ask = 1 embed + 1 LLM.
Practice markers: question, [HINT], [ANSWER]; if a marker is missing, show the raw text.

## 8. Risks and gates
Risks and mitigations:
1. NIM slow, 429 or down: one retry, timeout, friendly error, chat kept; index before the demo; recorded backup run; phone hotspot.
2. Empty or truncated reply (reasoning uses the budget): thinking off; ignore reasoning_content; treat empty as error and retry once; enough max_tokens.
3. Template ignored: Practice raw fallback; tune prompts on about 5 questions per mode.
4. Fake citations: code post-check rewrites invalid tags to [Not from your notes].
5. Wrong in-notes threshold: tune on 3 in-notes and 3 off-notes questions; show the score in Sources.
6. Query/passage mismatch: test 5 known questions in hour 1.
7. PDF text quirks: inspect extracted text; normalise; split at line or sentence boundaries; keep overlap.
8. Windows and Chroma: delete_collection; cache the client with st.cache_resource; one Streamlit process and one tab.
9. Streamlit reruns: history in st.session_state; index only on the button; file hash skip.
10. API key leak: .env or Streamlit secrets; add .env and chroma_db/ to .gitignore before the first commit.
Notes: a sharp topic switch can pull mixed chunks for one turn (accepted). Day-one gates: imports; embed round trip; streamed LLM call with thinking off; PDF text check; st.bottom check. If Chroma will not compile on Python 3.13, recreate the venv on Python 3.12. After the gates pass, pip freeze > requirements.txt.
Working rules: commit hourly; prompts only in prompts.py; print retrieval scores to the console; freeze before the demo.

## 9. Technical defense (draft answers; user justifies in the final presentation)
1. Why RAG when the notes fit in the prompt? At demo size both work; RAG gives page-level provenance and scales to syllabus-length notes with small prompts.
2. Hallucination control? Grounding rules, per-section tags, code check of tags against retrieved pages, threshold banner. Not a guarantee: a valid tag can still misstate a chunk, so snippets are shown.
3. Why 600-char chunks? Focused retrieval but enough for a definition or formula; overlap avoids cutting ideas; tuned on a few questions only.
4. In-notes vs not? Threshold on the top-1 cosine similarity; empirical; edge cases will misfire.
5. Follow-ups? Retrieval joins the last two student messages; a sharp topic switch can mix chunks for one turn.
6. Why this embedder? Free endpoint, normalised 2048-d; NVIDIA reports 58.38 nDCG@10, 95.4% of the 8B teacher. Text side only.
7. Why this LLM? 30B MoE with 3B active, low latency; thinking off for first-token speed.
8. Why no reranker, parse, safety? Parse is unneeded for text PDFs; reranking about 30 chunks adds little; safety is a prompt scope rule. Keeps 2 calls per question.
9. Personalization without a profile? Inferred "understood need" plus mode template; no persistent learner model (first roadmap item).
10. What does the mode change? Same retrieval; different template and output structure; Practice also changes parsing and rendering.
11. Why hide hint and answer? Encourages attempting first; aligns with the guided-step tutor evidence (A). Do not quote the unguarded-tutor figures (C).
12. API failure? One retry, then a friendly error with chat kept; model ID in one constant for manual swap.
13. Privacy? Notes and vectors stay in a local Chroma folder; questions and retrieved chunks go to NVIDIA's hosted endpoint. Retention policy unverified; do not claim one.
14. Evaluation? None yet. Planned metrics: retrieval hit rate on a question set, share of valid tags. The 12-question regression script is the stretch item.
Limitations to volunteer: no learner profile or weak-topic tracking; one subject per session; no OCR for scanned PDFs; no safety model; no formal evaluation; thinking-off and 40 RPM unverified.
Optional after the build: a mock drill where Claude asks these questions and critiques the answers.

## 10. Implementation plan (6 hours; the hour-3 checkpoint is the minimum demo)
- Hour 1: setup, .gitignore, first commit, gates (imports, embed round trip, streamed LLM with thinking off, PDF text check), confirm model IDs with client.models.list().
- Hour 2: rag.py (extract, chunk, embed, Chroma, retrieve, clear). Check 5 known questions; print scores.
- Hour 3: prompts.py plus app.py vertical slice: Explain mode, streaming, tags, Sources, banner, history, follow-up retrieval, retry. Checkpoint: Explain works end to end. Commit.
- Hour 4: Examples and Revision templates, "Understood need", mode selector, citation post-check.
- Hour 5: Practice mode (markers, parser, expanders, fallback); sidebar (uploader, Index notes, Clear notes); tune the threshold.
- Hour 6: full test on all 3 subjects, fixes, then freeze; dry run, backup recording and screenshots, requirements.txt, pre-index the demo subject, final commit.
Cut order if behind: (1) Sources snippets to file and page only; (2) one combined hint-and-answer expander; (3) pinned selector to a plain radio; (4) post-check to banner-only; (5) Clear notes to a manual folder delete. Never cut: Explain end to end, citations, persisted indexing, the backup recording.

## 11. Open items
- The user says "start implementation".
- Day-one tests of the unverified items in section 4.
- Tune the in-notes threshold.
- Optional: mock defense drill; the 12-question regression script.

## 12. Progress log
- 2026-11-25: Task 1 done. Planning reset; context.md created.
- 2026-11-25: Task 2 done: topics (a) to (i) decided one by one with the user; NIM IDs for embed-1b and lightning-30b verified; DECISIONS.md replaced; Project instructions replaced.