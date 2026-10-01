"""Pick the papers in data/retrieved_papers.csv that match a researcher's written interests.

Usage:
  find_paper_matches "I study how large language models handle multilingual translation ..."

Reads the papers saved by fetch_papers (no arXiv requests), shows the number of papers
and OpenAI requests, asks to confirm, then has the model judge each paper from its
title and abstract only. Prints the matches and saves them to data/paper_matches.csv
(overwritten on each run).
"""

import csv
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from fetch_papers import PAPER_FIELDS, RETRIEVED, SELECTED, show_samples
from find_categories import MODEL, ROOT, SetupError, load_api_key, openai_errors

OUT = ROOT / "data" / "paper_matches.csv"
CSV_FIELDS = PAPER_FIELDS + ["match_reason"]

BATCH_SIZE = 40  # papers per OpenAI request
WORKERS = 4  # parallel OpenAI requests

INSTRUCTIONS = """You screen new arXiv papers for a researcher.
You get the researcher's description of their interests and a batch of papers (label, title, abstract).
Return only the papers the researcher would plausibly want to read, judged from the title and abstract alone.
Respect anything the researcher says they are not interested in.
Be selective: a paper that merely shares a broad area (e.g. "uses language models") is not enough.
Returning no papers is fine.
For each match, return the paper's label exactly as given (e.g. P07) and one short sentence saying what in this
specific paper connects to the researcher's interests. The sentence must be supported by that paper's own abstract,
not by any other paper in the batch."""


def load_papers(path=RETRIEVED):
    if not path.exists():
        raise SetupError(f"{path} not found. Run fetch_papers YYYY-MM-DD first.")
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        missing = [c for c in PAPER_FIELDS if c not in (reader.fieldnames or [])]
        if missing:
            raise SetupError(f"{path} is missing columns {missing}. Run fetch_papers again.")
        return list(reader)


def stored_interest(path=SELECTED):
    """The interest find_categories used to choose the categories, or None."""
    if not path.exists():
        return None
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return rows[0].get("interest", "").strip() if rows else None


def label(i):
    """Label of the i-th paper (1-based) within its batch: P01, P02, ..."""
    return f"P{i:02d}"


def batch_labels(batch):
    return {label(i): p for i, p in enumerate(batch, 1)}


def format_batch(batch):
    # Short per-batch labels instead of arXiv IDs: long, similar-looking IDs are easy for the model to
    # miscopy or swap. Labels are mapped back to the paper records in collect().
    return "\n\n".join(f"Paper: {lab}\nTitle: {p['title']}\nAbstract: {p['abstract']}"
                       for lab, p in batch_labels(batch).items())


def format_interest(interest, details=""):
    """The researcher's own words, passed through unchanged. Optional details stay a separate section."""
    text = f"Researcher's interests:\n{interest}"
    if details:
        text += f"\n\nAdditional details about which papers the researcher wants:\n{details}"
    return text


def request_count(papers):
    return (len(papers) + BATCH_SIZE - 1) // BATCH_SIZE


def match_batch(client, interest, batch, details=""):
    from pydantic import BaseModel

    class Match(BaseModel):
        label: str
        reason: str

    class Matches(BaseModel):
        matches: list[Match]

    with openai_errors():
        response = client.responses.parse(
            model=MODEL,
            reasoning={"effort": "low"},
            instructions=INSTRUCTIONS,
            input=f"{format_interest(interest, details)}\n\nPapers:\n{format_batch(batch)}",
            text_format=Matches,
        )
    if response.output_parsed is None:
        raise SetupError("The model returned no usable answer for a batch (it may have refused).")
    return [(m.label, m.reason) for m in response.output_parsed.matches]


def collect(results, batch):
    """Map returned labels to {arxiv_id: reason} for this batch.

    Labels not in this batch are dropped; a repeated label keeps its first reason.
    """
    by_label = batch_labels(batch)
    reasons = {}
    for lab, reason in results:
        paper = by_label.get(lab.strip().upper())
        if paper is not None:
            reasons.setdefault(paper["arxiv_id"], reason.strip())
    return reasons


def match_papers(interest, papers, api_key, details=""):
    """Return (matches, batches checked, error).

    After the first failure, requests not yet started are cancelled; ones already
    in flight are still collected, since they are paid for.
    """
    import openai

    client = openai.OpenAI(api_key=api_key)
    batches = [papers[i:i + BATCH_SIZE] for i in range(0, len(papers), BATCH_SIZE)]
    by_id = {p["arxiv_id"]: p for p in papers}
    reasons, done, error = {}, 0, None
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(match_batch, client, interest, b, details): b for b in batches}
        for future in as_completed(futures):
            if future.cancelled():
                continue
            try:
                reasons.update(collect(future.result(), futures[future]))
            except SetupError as e:
                if error is None:
                    error = e
                    for f in futures:
                        f.cancel()
                continue
            done += 1
            print(f"  checked batch {done}/{len(batches)}")
    matches = [{**by_id[i], "match_reason": r} for i, r in reasons.items()]
    matches.sort(key=lambda p: p["submitted"], reverse=True)
    return matches, done, error


def print_matches(matches):
    for i, p in enumerate(matches, 1):
        print(f"{i}. {p['title']}")
        print(f"   {p['arxiv_id']}  |  {p['submitted'][:10]}  |  {p['categories']}")
        print(f"   {p['url']}")
        print(f"   Why: {p['match_reason']}\n")


def save(matches, path=OUT):
    # Settings written by rerank_papers describe the file they came with; drop them when this file is replaced.
    path.with_name(path.stem + "_settings.json").unlink(missing_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(matches)


def confirm(prompt):
    try:
        return input(prompt).strip().lower() in ("y", "yes")
    except EOFError:
        return False


def report(matches, papers, done, batches, error, path=OUT):
    """Print the matching results and save them."""
    print("\nRelevance was judged from each paper's title and abstract only, not the full text.\n")
    if error:
        print(f"Warning: stopped after {done} of {batches} batches because of an error: {error}")
        print("The results below are incomplete.\n")
    checked = f"{len(papers)} retrieved papers" if not error else f"papers in the {done} batches that finished"
    if matches:
        print_matches(matches)
        print(f"{len(matches)} of the {checked} look relevant.")
    else:
        print(f"None of the {checked} look relevant to your description.")
    save(matches, path)
    print(f"Saved to {path}")


def main():
    interest = " ".join(sys.argv[1:]).strip()
    if not interest:
        print('Usage: find_paper_matches "describe your research interests"', file=sys.stderr)
        sys.exit(2)
    try:
        papers = load_papers()
        earlier = stored_interest()
        if earlier and earlier != interest:
            print("Warning: this description differs from the one find_categories used to choose the categories")
            print(f"these papers were fetched from ({SELECTED.name}):")
            print(f"  \"{earlier}\"")
            print("Papers outside those categories were never retrieved. To re-target, run find_categories "
                  "and fetch_papers again.\n")

        dates = sorted({p["submitted"][:10] for p in papers})
        span = f" submitted {dates[0]}" + (f" to {dates[-1]}" if len(dates) > 1 else "") if dates else ""
        print(f"Loaded {len(papers)} papers{span} from {RETRIEVED.name}.")
        if not papers:
            print("No papers to check, so no OpenAI requests are needed.")
            save([])
            print(f"Saved an empty result to {OUT}")
            return
        show_samples(papers)

        batches = request_count(papers)
        print(f"Matching will send {len(papers)} titles and abstracts to {MODEL} in {batches} requests.")
        api_key = load_api_key()
        if not confirm("Proceed? [y/N] "):
            print("Cancelled. No OpenAI requests were made.")
            return

        matches, done, error = match_papers(interest, papers, api_key)
    except SetupError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    report(matches, papers, done, batches, error)


if __name__ == "__main__":
    main()
