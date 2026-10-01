# CLAUDE.md

Academic Research Monitor helps a researcher find new arXiv papers relevant to their written interests.
It is an early, local experiment: one user, Windows + PowerShell, activated `.venv`, Python 3.12.

## Pipeline

Each stage is a CLI command. Stages hand off **only** through CSV files in `data/`; each run overwrites its output.

| Stage | Command | Kind | Reads | Writes |
|---|---|---|---|---|
| 0 | `python scripts/extract_arxiv_taxonomy.py` | deterministic (arxiv.org HTML) | – | `arxiv_taxonomy.csv` (committed) |
| 1 | `find_categories "<interest>"` | LLM | taxonomy | `selected_categories.csv` |
| 2 | `fetch_papers YYYY-MM-DD` | deterministic (arXiv API) | selected categories | `retrieved_papers.csv` |
| 3 | `find_paper_matches "<interest>"` | LLM | retrieved papers | `paper_matches.csv` |
| 3 (alt, experimental) | `rerank_papers "<interest>"` | embeddings + BM25 + LLM | retrieved papers | `paper_matches.csv`, `paper_matches_settings.json` |

Stage 3 alt first has the LLM extract keyword search terms (`extract_keywords`, prompt `keywords-v1`). Its input is the description and details as JSON, and it is told to leave out the exclusions. The checked terms (`clean_keywords`) are the BM25 query only. Embeddings, paper tokenization, fusion, shortlist and screening are unchanged. If no usable terms come back, the run stops with a `SetupError` before any embedding, ranking or screening. There is no fallback to the full description, so results never silently mix query types. Stage 3 alt then ranks all papers by embedding similarity and BM25, fused with RRF. It shortlists the top `SHORTLIST_PERCENT` (10%) of retrieved papers, rounded up with integer maths. The count is shown before confirmation, and the percentage and count are recorded in the settings and report. The LLM accepts or rejects each shortlisted paper with a 0–100 score used only for ordering. Retrieval narrows the pool; relevance is decided only by the LLM. Ranks and scores are never shown to the model, and there is no RRF-score threshold. Accepted papers are saved in score order. The settings file records the models, prompt version, limits, usage and timing. `find_paper_matches.save` deletes that file when the LLM-only path replaces the matches.

`research_monitor` is an interactive layer over stages 1–3. It calls the same functions and writes the same files, only after the user accepts each stage. It takes dates as `dd-mm-yyyy` ranges and passes optional extra details to the matcher as a separate prompt section. Free-text answers (the description, including revisions, and non-empty details) go through `input_check.py`, an LLM classifier (usable / needs_detail / unrelated). It sends the user's text as JSON data, and the program prints its own fixed text for "unrelated". `REDIRECT_RULE` makes any instruction aimed at the program "unrelated", even when mixed with a valid interest. The whole text is rejected, not cleaned. Research about prompt injection, security or harmful topics, and include/exclude criteria, stay usable. Re-run a real evaluation with expectations fixed in advance before changing the checker prompts. It is an input-quality check, not a security boundary. At step 5 it asks for the method: `l` (LLM-only) or `h` (the experimental stage 3 alt). It times the program's own work with `timed()` (`time.perf_counter`), never the user's answering time, and prints a summary at the end. For the LLM-only path, retries happen inside the SDK and aren't counted. For the hybrid path, `rerank_papers` reports new vs cached papers and retry count and wait, and these are also saved in the settings file. After matching, it writes `data/research_report.txt` from data it already holds, and deletes any older report when categories are accepted, so a report never outlives its results. Keep it thin: pipeline logic belongs in the stage modules, not in the flow.

No database, cache, scheduler or UI. Don't add one without a demonstrated need.

`scripts/search_experiment.py` is an experiment outside the pipeline. It ranks the saved papers by embedding search (`text-embedding-3-small`, cosine) and by hybrid search (+ BM25, reciprocal rank fusion), and compares them with the LLM matches, which are a baseline, not ground truth. It reads write-once snapshots in `data/experiments/search/inputs/` and writes only under `data/experiments/search/`. Its `embeddings.json` is the one requested exception to "no cache". It is keyed by model and text hash, so a changed abstract or model is re-embedded rather than mixed. `embed_texts` is its only provider call.

`rerank_papers` reuses `search_experiment`'s ranking and embedding functions, and keeps its own vectors in `data/paper_embeddings.json`. That file is the second exception to "no cache", with the same model-and-text-hash rule. **Known architectural gap:** a pipeline stage now imports an experiment module. If the hybrid path stays, move the shared ranking and embedding functions into a pipeline module.

`scripts/judge_experiment.py` builds on it. `gpt-6-astra` judges a pool: the original LLM matches plus each search method's top 20. It judges one paper per request, and the judge sees only the brief, title and abstract. The rubric is fixed (relevant / partly_relevant / irrelevant), and each verdict comes with an exact abstract quote that is checked in code. Results go to `data/experiments/judge/`. `judgments.jsonl` is appended as each paid judgment arrives and is reused only under the same settings fingerprint (model, effort, prompt, brief). The judge is provisional, not ground truth, and pool-only results say nothing about recall. `judge_paper` is its only provider call.

## Code layout and boundaries

- Modules live in `scripts/` and are registered in `pyproject.toml` `[project.scripts]`. After adding or renaming a command, run `pip install -e .`.
- `find_categories.py` also holds the shared pieces: `ROOT`, `MODEL`, `SetupError`, `load_api_key`, `openai_errors` (a context manager that turns OpenAI SDK errors into one-line `SetupError`s) and `group` (the "field > subfield" label).
- Per module: plain functions for parsing, validation and CSV I/O. Exactly one function makes the provider call (`ask_model`, `match_batch`). Exception: `rerank_papers` has two, `extract_keywords` and `rerank_batch`, and both go through `with_retries`. It imports `openai` lazily and wraps the call in `with openai_errors():` to turn SDK errors into `SetupError`. Everything else must run offline.
- LLM output is never trusted as data. Validate returned codes and labels against what was sent (taxonomy codes, per-batch paper labels), and take names and metadata from our CSVs, not from the model. The matcher sends short per-batch labels (P01, P02, …), not arXiv IDs, and maps them back in code. Saved results keep arXiv IDs.
- `SetupError` means a user-fixable problem: print one line, exit 1, no traceback. Usage errors exit 2.
- Use plain functions and dicts. The only classes are pydantic response schemas and `SetupError`.

## External services

- **OpenAI:** `gpt-6-luna` via `client.responses.parse(..., text_format=<pydantic model>)` (Structured Outputs), `reasoning={"effort": "low"}`. Check the current OpenAI docs before changing the model or call.
- **Paid calls:** a command that sends many requests prints the number of papers and requests and asks `[y/N]` first. **Claude never runs a paid OpenAI call**; give the user the exact command instead.
- **arXiv API rules** ([terms](https://info.arxiv.org/help/api/tou.html)):
  - At most 1 request every 3 s, over one connection.
  - At most 2000 results per page and 30000 per query.
  - Titles and abstracts are CC0. Never download or store PDFs or full text.
- **arXiv transport:** requests go through `curl` because arXiv's CDN answers Python's TLS client with HTTP 406. Don't switch back to `urllib`/`requests` without re-testing. arXiv calls are free, but keep them few and reuse saved responses in tests.

## Reruns, failures, stale state

- Pagination never stops silently. An empty page mid-result is retried, then fails loudly. Deduplicate by version-less arXiv ID.
- Retry only failures that are safe and plausibly transient: network errors, 5xx, rate limits, and an empty arXiv page. Don't retry auth, validation or bad-query errors. The OpenAI SDK already retries (2×, on 408/409/429/5xx); don't wrap another retry loop around it. Exception: `rerank_papers` builds its client with `max_retries=0` and `with_retries` is its only loop. The reason is that the SDK treats every 429 alike, and billing, spend and quota codes (`NO_RETRY_CODES`) must never be retried. The loop retries only rate limits, connection errors and 5xx, at most `RATE_LIMIT_RETRIES` times. It honours Retry-After up to `MAX_WAIT`, and otherwise backs off exponentially.
- On a partial LLM failure, stop starting new requests, keep results already paid for, and label the output incomplete.
- Persisted AI output must say what produced it: model, prompt/schema version, and input identity. When any of these change, recompute rather than mix old and new results. **Current gap:** the CSVs don't record this yet. Until they do, warn on mismatched inputs rather than combining them silently.

## Secrets and config

- `OPENAI_API_KEY` comes from the environment or `.env` (gitignored, never tracked). Never print, log or echo the key. Tests use a fake key.
- Settings (model, batch size, paths, rate limits, prices) are module-level constants at the top of each file. No config framework. Keep environment differences explicit (env vars).

## Dependencies

Only `openai`; everything else is the standard library. A new dependency needs a stated reason and goes into `pyproject.toml`.

## Testing

- Tests go in `tests/`. Prefer the existing test framework if one is established. If introducing the first test framework, choose the simplest appropriate option and explain the choice before adding the dependency.
- Established framework: stdlib `unittest` + `unittest.mock`. Run with `python -m unittest` from the project root. Shared fakes (scripted `input()`, arXiv pages, OpenAI client) are in `tests/fakes.py`. `tests/fixtures/arxiv_page.xml` is a real arXiv response. Patch output paths in tests so they never touch `data/`.
- Deterministic code (parsing, pagination, validation, CSV) gets deterministic tests: saved fixtures such as real arXiv Atom XML, a fake `fetch_page`, and no sleeps or network.
- Fake LLM calls by replacing `openai.OpenAI` with a stub whose `responses.parse` returns `text_format(...)`. Normal tests never spend credits.
- When a real AI output is wrong in an instructive way, save the input and the expected judgement as an eval example. Don't reshape the prompt around one odd output.

## Debugging

Look at real inputs and outputs first: the CSVs and a saved API response. Form a hypothesis, test it with the smallest experiment, then make the smallest fix. For example, the arXiv 406 was isolated by varying the headers, then the client, before switching to curl.

## Workflow

- Before editing: read the relevant code and run `git status`. After editing: run the tests, then check `git status` and `git diff`.
- **Never commit or push unless the user explicitly asks.** Never rewrite git history without asking.
- The README is for humans (usage, outputs); update it when commands or outputs change. Keep it short, for researchers who code a little. Put details in `docs/`: `troubleshooting.md` (errors), `advanced_usage.md` (single-step commands, files, settings) and `experiments.md` (the hybrid option, search and judge experiments). This file holds architecture and conventions; update it when they change.
- Build the smallest thing that works. If you spot an architectural problem, report it rather than refactoring unasked.
- The user is learning. When a non-obvious engineering decision comes up, explain the concept and the reason in a sentence or two.
