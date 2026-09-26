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

## CSV columns

| Column        | Meaning                                                                                                  |
|---------------|----------------------------------------------------------------------------------------------------------|
| `field`       | Top-level group, e.g. `Computer Science`, `Physics`                                                      |
| `subfield`    | What arXiv calls an "archive", e.g. `Astrophysics`. Only Physics has these; the column is blank elsewhere. |
| `subject`     | Category name, e.g. `Artificial Intelligence`                                                            |
| `code`        | arXiv category code, e.g. `cs.AI`                                                                        |
| `description` | arXiv's description of the category                                                                      |
