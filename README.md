# Academic Research Monitor

The goal is to help researchers find new papers that match their research interests.

## Current state

So far there is one piece: a script that pulls arXiv's category taxonomy from
<https://arxiv.org/category_taxonomy> and saves it to `data/arxiv_taxonomy.csv`.
The current file has **155 categories across 8 fields**.

## Usage

From the project root:

```
python scripts/extract_arxiv_taxonomy.py
```

The script uses only the Python standard library.

## Finding categories for your interests (experiment)

`find_categories` asks an OpenAI model (`gpt-6-luna`) which arXiv categories match a
plain-text description of your interests. It checks every returned code against
`data/arxiv_taxonomy.csv` and saves the matches to `data/selected_categories.csv`,
replacing that file on each run.

One-time setup, in the activated virtual environment, from the project root:

```
pip install -e .
```

This installs the `openai` package (listed in `pyproject.toml`) and adds the `find_categories` command.
Put your key in `.env` as `OPENAI_API_KEY=...`. Git ignores this file.

```
find_categories "I am looking for papers in computational linguistics and AI"
```

## Interactive flow

`research_monitor` runs all the steps below in one guided session:

1. Describe your interests. It suggests arXiv categories; you accept them, revise your description and try again, or exit. Only accepted categories are saved.
2. Enter a start date and an optional end date (`dd-mm-yyyy`; blank end = today). The range covers whole UTC days, inclusive, by each paper's first-version submission date.
3. It fetches every paper in those categories and dates into `data/retrieved_papers.csv` and stops if there are none.
4. Optionally add details about which papers you want. These are sent alongside your original description, not merged into it.
5. It shows the number of papers and OpenAI requests and asks `Run matching? [y/N]`. If you say no, the retrieved papers stay available for `find_paper_matches` later.

```
research_monitor
```

Suggesting categories is a small paid OpenAI call (well under a cent) each time you submit or revise a description.

## Fetching and matching papers (experiment)

This is two steps, so you can fetch once and try different descriptions without contacting arXiv again.

**1. `fetch_papers`** takes the category codes in `data/selected_categories.csv` and fetches every
arXiv paper in those categories (primary or cross-listed) whose first version was submitted on the
given date, 00:00–23:59 UTC, using the [arXiv API](https://info.arxiv.org/help/api/user-manual.html).
It pages through the full result set, at most one request every 3 seconds per arXiv's rules, and
saves every paper to `data/retrieved_papers.csv`. It makes no OpenAI calls.

```
fetch_papers 2026-09-25
```

**2. `find_paper_matches`** reads `data/retrieved_papers.csv` without contacting arXiv. It shows the
paper count, a few samples and the number of OpenAI requests, and asks for confirmation. After you
confirm, `gpt-6-luna` judges each paper's title and abstract against your description (it never
sees the full text). Matches are saved to `data/paper_matches.csv`. It warns you if your description
differs from the one `find_categories` used to choose the categories.

```
find_paper_matches "I study how large language models handle multilingual translation for low-resource languages."
```

Both CSVs are replaced on each run.

arXiv's CDN currently rejects Python's built-in HTTP client (HTTP 406), so requests go through
`curl`, which ships with Windows 10 and later. Papers only show up once arXiv has announced them,
so the last day or so of submissions is usually missing.

## CSV columns

| Column        | Meaning                                                                                                  |
|---------------|----------------------------------------------------------------------------------------------------------|
| `field`       | Top-level group, e.g. `Computer Science`, `Physics`                                                      |
| `subfield`    | What arXiv calls an "archive", e.g. `Astrophysics`. Only Physics has these; the column is blank elsewhere. |
| `subject`     | Category name, e.g. `Artificial Intelligence`                                                            |
| `code`        | arXiv category code, e.g. `cs.AI`                                                                        |
| `description` | arXiv's description of the category                                                                      |
