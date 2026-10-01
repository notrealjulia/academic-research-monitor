# Experiments

[Back to the README](../README.md)

The search-first matching option (`h`) and the two comparison scripts below are experiments. Their results are useful, but they haven't been checked against a fixed set of papers whose relevance is known, so neither method has been shown to be more accurate.

## The search-first option in `research_monitor`

Option `h` (also available as the `rerank_papers` command) works in four stages:

1. **Keyword terms.** The AI turns your description and details into keyword search terms, leaving out anything you said you're not interested in. The terms are printed, listed in the report and saved in `paper_matches_settings.json`. If no usable terms come back, the run stops with an error before anything else is sent.
2. **Search.** Every downloaded paper is ranked two ways:
   - by *meaning*: OpenAI's `text-embedding-3-small` compares each title and abstract with your full description and details,
   - by *words*: BM25 keyword ranking with the keyword terms from stage 1.

   The two rankings are combined with reciprocal rank fusion, which favours papers that rank well in both. Paper embeddings are saved in `data\paper_embeddings.json` and reused when the paper text and embedding model are unchanged.
3. **Shortlist.** The top 10% of papers, rounded up, are kept: for example 42 of 411, or 336 of 3,360. The count is shown before you confirm, and it's recorded in the report and the settings file.
4. **Screening.** The AI reads the shortlisted titles and abstracts, 20 per request, and accepts or rejects each one. It never sees the search ranks or scores. Accepted papers get a 0–100 score, which only orders them; it's the AI's judgment, not a probability.

The trade-off: the AI reads far fewer papers, so this option is cheaper and less likely to hit OpenAI rate limits. But a relevant paper outside the top 10% is never read. If OpenAI is briefly busy, the program waits and retries, up to 5 times per request. It never retries billing or quota errors.

## Search experiment: `search_experiment.py`

```powershell
python scripts/search_experiment.py
python scripts/search_experiment.py "another description"
```

This ranks the downloaded papers by embedding search and by hybrid search (embeddings plus BM25 on the full description). It compares both rankings with the AI matches in `paper_matches.csv`: how many of those matches appear in each method's top 10, 20 and 50, and where each one ranks. The AI matches are a comparison point, not a correct answer key.

- **Inputs are copied once.** The first run copies `retrieved_papers.csv`, `paper_matches.csv` and `selected_categories.csv` into `data\experiments\search\inputs\`, and later runs read only those copies. If the main files have changed since, it says so and keeps using the copies. To start a fresh experiment on new data, delete the `inputs` folder first.
- **Cost.** The first run embeds every paper (about $0.01 for 850 papers) and asks `Proceed? [y/N]`. The embeddings are saved, so another description costs only one tiny request.
- **Output.** `embedding_ranking.csv`, `hybrid_ranking.csv` and `run_info.json` (settings and input checksums) go to `data\experiments\search\`, replaced on each run.

## Judge experiment: `judge_experiment.py`

```powershell
python scripts/judge_experiment.py
```

This builds a pool of papers: all the AI matches, plus the top 20 from embedding, BM25 and hybrid search. It runs after the search experiment, using the same inputs and description. A stronger model (`gpt-6-astra`) then rates each paper separately, seeing only the description, title and abstract, as **relevant**, **partly relevant** or **irrelevant**. Each rating comes with a short reason and a quote from the abstract, which the program checks is really there.

- It shows the estimated cost and asks `Proceed? [y/N]` first. Ratings are saved as they arrive, so after an interruption, running it again reuses them.
- **Output:** `judgments.csv`, `judgments.jsonl` and `settings.json` in `data\experiments\judge\`.
- The ratings are a second opinion, not the truth. Only the pool is rated, so the results say nothing about relevant papers outside it.
