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

`research_monitor` is an interactive layer over stages 1–3. It calls the same functions and writes the same files, only after the user accepts each stage. It takes dates as `dd-mm-yyyy` ranges and passes optional extra details to the matcher as a separate prompt section. Keep it thin: pipeline logic belongs in the stage modules, not in the flow.

No database, cache, scheduler or UI. Don't add one without a demonstrated need.

## Code layout and boundaries

- Modules live in `scripts/` and are registered in `pyproject.toml` `[project.scripts]`. After adding or renaming a command, run `pip install -e .`.
- `find_categories.py` also holds the shared pieces: `ROOT`, `MODEL`, `SetupError`, `load_api_key`.
- Per module: plain functions for parsing, validation and CSV I/O. Exactly one function makes the provider call (`ask_model`, `match_batch`). It imports `openai` lazily and turns SDK errors into `SetupError`. Everything else must run offline.
- LLM output is never trusted as data. Validate returned codes and IDs against what was sent (taxonomy codes, batch IDs), and take names and metadata from our CSVs, not from the model.
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
- Retry only failures that are safe and plausibly transient: network errors, 5xx, rate limits, and an empty arXiv page. Don't retry auth, validation or bad-query errors. The OpenAI SDK already retries (2×, on 408/409/429/5xx); don't wrap another retry loop around it.
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
- The README is for humans (usage, outputs); update it when commands or outputs change. This file holds architecture and conventions; update it when they change.
- Build the smallest thing that works. If you spot an architectural problem, report it rather than refactoring unasked.
- The user is learning. When a non-obvious engineering decision comes up, explain the concept and the reason in a sentence or two.
