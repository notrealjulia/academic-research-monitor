# Advanced usage

[Back to the README](../README.md)

## Running single steps

`research_monitor` runs these steps in order. You can also run them one at a time, with the virtual environment active (`(.venv)` in the prompt). Each step reads the previous step's file from `data`.

| Command | What it does | Uses OpenAI? |
|---|---|---|
| `find_categories "<description>"` | Suggests categories and saves them to `data\selected_categories.csv`. There's no accept/revise step. | yes |
| `fetch_papers YYYY-MM-DD` | Downloads the papers in the saved categories for **one** day to `data\retrieved_papers.csv`. Note the year-month-day format. | no |
| `find_paper_matches "<description>"` | Screens every saved paper (option `l`). Saves `data\paper_matches.csv`. | yes |
| `rerank_papers "<description>"` | Search first, then screens the top 10% (option `h`). Saves `data\paper_matches.csv` and `data\paper_matches_settings.json`. | yes |

The two matching commands show a few sample papers, say how many requests they will send, and ask `Proceed? [y/N]` first. They warn you if the description differs from the one used to choose the categories. Unlike `research_monitor`, they don't check the description first, don't take extra details, and **don't write the report**. An older `research_report.txt` may still be in `data`, describing an earlier run.

For example, to try another description on papers you've already downloaded, without contacting arXiv again:

```powershell
find_paper_matches "I study neural machine translation for Basque and other low-resource European languages."
```

## Files in `data`

| File | Contents |
|---|---|
| `research_report.txt` | The readable report (written by `research_monitor` only). |
| `selected_categories.csv` | The accepted categories: code, field, name, the reason each was suggested, and your description. |
| `retrieved_papers.csv` | Every downloaded paper: arXiv ID, title, abstract, authors, first submission date and time, categories, and links to the arXiv page and the PDF. |
| `paper_matches.csv` | The selected papers: the same columns plus `match_reason`. With option `h`, there's also `relevance_score` (0–100) and papers are listed best first. |
| `paper_matches_settings.json` | Option `h` only: the models, keyword search terms, shortlist percentage and size, request counts, token use, retries and timing of the run that wrote `paper_matches.csv`. It's deleted when option `l` replaces the matches. |
| `paper_embeddings.json` | Option `h` only: saved paper embeddings, reused so repeat runs on the same papers make no new embedding requests. You can delete it; it's rebuilt when needed. |
| `arxiv_taxonomy.csv` | arXiv's category list (see below). Part of the project; don't delete it. |

Each run replaces the files it writes.

## The timing summary

At the end of a run, `research_monitor` prints how long each stage took. Time spent waiting for your answers isn't included. For option `h` it also shows how many papers were embedded new or reused from the saved embeddings, and how many retries there were. Option `l` can't count its retries, because they happen inside OpenAI's library.

## Settings

Settings are constants at the top of each file in `scripts`. For example:
- `SHORTLIST_PERCENT` in `rerank_papers.py`: the share of papers option `h` passes to the AI (10, rounded up)
- `MODEL` in `find_categories.py`: the OpenAI model used for checks, categories and matching

After changing anything in `scripts`, run the tests.

## Other project tasks

```powershell
python -m unittest                         # run the automated tests (no OpenAI or arXiv requests)
python scripts/extract_arxiv_taxonomy.py   # rebuild data\arxiv_taxonomy.csv from arXiv's category page
```

## arXiv category list

`data\arxiv_taxonomy.csv` holds arXiv's category list (155 categories in 8 fields). The program checks the AI's category suggestions against it.

| Column | Meaning |
|---|---|
| `field` | Top-level group, e.g. `Computer Science`, `Physics` |
| `subfield` | What arXiv calls an "archive", e.g. `Astrophysics`. Only Physics has these; the column is blank elsewhere. |
| `subject` | Category name, e.g. `Artificial Intelligence` |
| `code` | arXiv category code, e.g. `cs.AI` |
| `description` | arXiv's description of the category |
