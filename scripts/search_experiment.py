"""Experiment: rank the saved papers by embedding search and by hybrid (embedding + BM25) search,
and compare both rankings with the previous LLM matches.

Usage:
  python scripts/search_experiment.py              # query = the stored research description
  python scripts/search_experiment.py "a query"    # try another query (papers are not re-embedded)

Separate from the pipeline: it reads snapshots of the pipeline CSVs, never writes to them, and
writes only to data/experiments/search/. The previous LLM matches are a comparison baseline,
not ground truth.
"""

import array
import base64
import csv
import hashlib
import json
import math
import re
import shutil
import sys
from datetime import datetime, timezone

from fetch_papers import RETRIEVED, SELECTED
from find_categories import ROOT, SetupError, load_api_key, openai_errors
from find_paper_matches import OUT as MATCHES, confirm, load_papers

EXP_DIR = ROOT / "data" / "experiments" / "search"
INPUTS = EXP_DIR / "inputs"  # write-once copies of the pipeline CSVs
PREVIOUS_IDS_BACKUP = ROOT / "data" / "paper_matches_ids.csv"  # older ID-prompt run, kept for reference
EMBEDDINGS = EXP_DIR / "embeddings.json"
EMBEDDING_CSV = EXP_DIR / "embedding_ranking.csv"
HYBRID_CSV = EXP_DIR / "hybrid_ranking.csv"
RUN_INFO = EXP_DIR / "run_info.json"

EMBED_MODEL = "text-embedding-3-small"  # $0.02 per 1M input tokens (OpenAI pricing page, 2026-09)
EMBED_BATCH = 200  # inputs per request; the API allows 2048 inputs and 300k tokens per request
PRICE_PER_M_TOKENS = 0.02
CHARS_PER_TOKEN = 4  # rough estimate for the cost preview only

BM25_K1 = 1.5  # common defaults (Robertson & Zaragoza; Lucene uses 1.2/0.75)
BM25_B = 0.75
RRF_K = 60  # from Cormack et al. 2009; damps the influence of the very top ranks

CUTOFFS = (10, 20, 50)
SHOW_TOP = 20


# ---------- snapshots and inputs ----------

def snapshot(sources, dest=INPUTS):
    """Copy each existing source into dest once. Never overwrites; warns if the source has changed since."""
    dest.mkdir(parents=True, exist_ok=True)
    for src in sources:
        if not src.exists():
            continue
        dst = dest / src.name
        if not dst.exists():
            shutil.copy2(src, dst)
        elif dst.read_bytes() != src.read_bytes():
            print(f"Note: {src.name} changed since the snapshot; the experiment keeps using {dst}.")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stored_brief(path):
    if not path.exists():
        raise SetupError(f"{path} not found, so the research description cannot be recovered.")
    with path.open(encoding="utf-8", newline="") as f:
        interests = {r.get("interest", "").strip() for r in csv.DictReader(f)} - {""}
    if len(interests) != 1:
        raise SetupError(f"Could not recover a single research description from {path}.")
    return interests.pop()


def paper_text(p):
    return f"{p['title'].strip()}\n\n{' '.join(p['abstract'].split())}"


def text_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------- ranking ----------

def rank(scores):
    """{id: score} -> ids sorted best first. Ties break by ID so reruns give the same order."""
    return sorted(scores, key=lambda i: (-scores[i], i))


def positions(order):
    return {i: n for n, i in enumerate(order, 1)}


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


def tokenize(text):
    """Lowercase alphanumeric words with a crude plural fold (hallucinations -> hallucination).

    Applied identically to query and papers, so an odd fold ("bias" -> "bia") still matches itself.
    """
    words = []
    for w in re.findall(r"[a-z0-9]+", text.lower()):
        if len(w) > 4 and w.endswith("ies"):
            w = w[:-3] + "y"
        elif len(w) > 3 and w.endswith("s") and not w.endswith(("ss", "us", "is")):
            w = w[:-1]
        words.append(w)
    return words


def bm25_scores(query, docs, k1=BM25_K1, b=BM25_B):
    """{id: text} -> {id: BM25 score}. Each distinct query term counts once.

    IDF is the always-positive Lucene form, so very common words ("the", "i") get weight near zero
    instead of a negative one. No stopword list.
    """
    tokens = {i: tokenize(t) for i, t in docs.items()}
    n = len(docs)
    avgdl = sum(len(t) for t in tokens.values()) / n if n else 0.0
    df = {}
    for toks in tokens.values():
        for w in set(toks):
            df[w] = df.get(w, 0) + 1
    terms = [w for w in dict.fromkeys(tokenize(query)) if w in df]
    idf = {w: math.log(1 + (n - df[w] + 0.5) / (df[w] + 0.5)) for w in terms}
    scores = {}
    for i, toks in tokens.items():
        tf = {}
        for w in toks:
            tf[w] = tf.get(w, 0) + 1
        norm = k1 * (1 - b + b * len(toks) / avgdl) if avgdl else k1
        scores[i] = sum(idf[w] * tf[w] * (k1 + 1) / (tf[w] + norm) for w in terms if w in tf)
    return scores


def rrf(*orders, k=RRF_K):
    """Reciprocal rank fusion: score = sum over rankings of 1 / (k + rank)."""
    scores = {}
    for order in orders:
        for n, i in enumerate(order, 1):
            scores[i] = scores.get(i, 0.0) + 1 / (k + n)
    return scores


# ---------- embeddings (the only provider call is embed_texts) ----------

def encode(vector):
    return base64.b64encode(array.array("f", vector).tobytes()).decode("ascii")


def decode(b64):
    return array.array("f", base64.b64decode(b64)).tolist()


def load_cache(path=EMBEDDINGS, model=EMBED_MODEL):
    """Saved embeddings, or an empty cache. Entries from another model are dropped, never mixed."""
    empty = {"model": model, "papers": {}, "queries": {}}
    if not path.exists():
        return empty
    cache = json.loads(path.read_text(encoding="utf-8"))
    if cache.get("model") != model:
        print(f"Note: saved embeddings are from {cache.get('model')}, not {model}; they will be recomputed.")
        return empty
    return cache


def save_cache(cache, path=EMBEDDINGS):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(cache), encoding="utf-8")
    tmp.replace(path)  # never leave a half-written cache behind


def missing_texts(cache, papers, query):
    """(key, kind, text) for every paper or query whose saved embedding is absent or for other text."""
    todo = []
    for p in papers:
        text = paper_text(p)
        entry = cache["papers"].get(p["arxiv_id"])
        if not entry or entry["text_sha256"] != text_hash(text):
            todo.append((p["arxiv_id"], "papers", text))
    if text_hash(query) not in cache["queries"]:
        todo.append((text_hash(query), "queries", query))
    return todo


def embed_texts(client, texts, model=EMBED_MODEL):
    """Embed texts in one request; returns vectors in input order."""
    with openai_errors():
        response = client.embeddings.create(model=model, input=texts, encoding_format="base64")
    data = sorted(response.data, key=lambda d: d.index)
    if [d.index for d in data] != list(range(len(texts))):
        raise SetupError(f"The embedding response did not return exactly one vector per input ({len(texts)}).")
    vectors = [decode(d.embedding) if isinstance(d.embedding, str) else list(d.embedding) for d in data]
    if len({len(v) for v in vectors}) != 1:
        raise SetupError("The embedding response mixed vector sizes.")
    return vectors, response.usage.total_tokens


def fill_cache(cache, todo, api_key, save=save_cache):
    """Embed the missing texts batch by batch, saving after each paid batch. Returns tokens used."""
    import openai

    client = openai.OpenAI(api_key=api_key)
    tokens = 0
    for start in range(0, len(todo), EMBED_BATCH):
        batch = todo[start:start + EMBED_BATCH]
        vectors, used = embed_texts(client, [t for _, _, t in batch])
        for (key, kind, text), v in zip(batch, vectors):
            entry = {"text_sha256": text_hash(text), "embedding_b64": encode(v)}
            if kind == "queries":
                entry["text"] = text
            cache[kind][key] = entry
        save(cache)
        tokens += used
        print(f"  embedded {min(start + EMBED_BATCH, len(todo))}/{len(todo)}")
    return tokens


def embedding_scores(cache, papers, query):
    q = decode(cache["queries"][text_hash(query)]["embedding_b64"])
    return {p["arxiv_id"]: cosine(q, decode(cache["papers"][p["arxiv_id"]]["embedding_b64"])) for p in papers}


# ---------- outputs ----------

def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def embedding_rows(order, scores, titles, previous):
    return [{"rank": n, "arxiv_id": i, "title": titles[i], "similarity": f"{scores[i]:.6f}",
             "previous_match": "yes" if i in previous else "no"} for n, i in enumerate(order, 1)]


def hybrid_rows(order, scores, emb_pos, bm25_pos, bm25, titles, previous):
    return [{"rank": n, "arxiv_id": i, "title": titles[i], "rrf_score": f"{scores[i]:.6f}",
             "embedding_rank": emb_pos[i], "bm25_rank": bm25_pos[i], "bm25_score": f"{bm25[i]:.4f}",
             "previous_match": "yes" if i in previous else "no"} for n, i in enumerate(order, 1)]


def overlap(order, previous, cutoffs=CUTOFFS):
    return {c: len(set(order[:c]) & previous) for c in cutoffs}


def print_top(name, rows, score_field):
    print(f"\n=== {name}: top {SHOW_TOP} ===")
    for r in rows[:SHOW_TOP]:
        mark = "*" if r["previous_match"] == "yes" else " "
        print(f"{r['rank']:>3}. {mark} {r[score_field]}  {r['arxiv_id']}  {r['title'][:80]}")
    print("(* = in the previous LLM matches)")


def print_summary(previous, titles, orders):
    print(f"\n=== Previous LLM matches ({len(previous)}) in each ranking ===")
    for name, order in orders.items():
        counts = overlap(order, previous)
        print(f"{name:>9}: " + ", ".join(f"top {c}: {counts[c]}" for c in CUTOFFS))
    pos = {name: positions(order) for name, order in orders.items()}
    print(f"\n{'embedding':>9} {'bm25':>5} {'hybrid':>6}  arxiv_id    title")
    for i in sorted(previous, key=lambda i: pos["hybrid"][i]):
        print(f"{pos['embedding'][i]:>9} {pos['bm25'][i]:>5} {pos['hybrid'][i]:>6}  {i}  {titles[i][:70]}")


# ---------- main ----------

def main():
    snapshot([RETRIEVED, MATCHES, SELECTED, PREVIOUS_IDS_BACKUP])
    try:
        papers = load_papers(INPUTS / RETRIEVED.name)
        previous_rows = load_papers(INPUTS / MATCHES.name)
        brief = stored_brief(INPUTS / SELECTED.name)
    except SetupError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    query = " ".join(sys.argv[1:]).strip() or brief
    titles = {p["arxiv_id"]: p["title"] for p in papers}
    previous = {r["arxiv_id"] for r in previous_rows}
    unknown = previous - titles.keys()
    if unknown:
        print(f"Warning: {len(unknown)} previous matches are not in the saved papers and are ignored: {sorted(unknown)}")
        previous -= unknown
    print(f"{len(papers)} papers, {len(previous)} previous LLM matches.")
    print(f"Query ({'stored research description' if query == brief else 'from the command line'}):\n  {query}\n")
    if query != brief:
        print("Note: the previous LLM matches were judged against the stored description, not this query.\n")

    # BM25 is offline; the embedding step may cost money and asks first.
    bm25 = bm25_scores(query, {p["arxiv_id"]: paper_text(p) for p in papers})

    try:
        cache = load_cache()
        todo = missing_texts(cache, papers, query)
        tokens = 0
        if todo:
            chars = sum(len(t) for _, _, t in todo)
            requests = (len(todo) + EMBED_BATCH - 1) // EMBED_BATCH
            est = chars / CHARS_PER_TOKEN
            print(f"Embedding will send {len(todo)} texts (~{est:,.0f} tokens, ~${est / 1e6 * PRICE_PER_M_TOKENS:.4f}) "
                  f"to {EMBED_MODEL} in {requests} requests. Saved embeddings are reused next time.")
            api_key = load_api_key()
            if not confirm("Proceed? [y/N] "):
                print("Cancelled. No OpenAI requests were made.")
                return
            tokens = fill_cache(cache, todo, api_key)
            print(f"Used {tokens:,} tokens (~${tokens / 1e6 * PRICE_PER_M_TOKENS:.4f}).")
        else:
            print("All embeddings are already saved; no OpenAI requests needed.")
    except SetupError as e:
        print(f"Error: {e}", file=sys.stderr)
        print("Embeddings finished before the error are saved; rerun to embed the rest. No rankings were written.",
              file=sys.stderr)
        sys.exit(1)

    emb = embedding_scores(cache, papers, query)
    emb_order, bm25_order = rank(emb), rank(bm25)
    hybrid = rrf(emb_order, bm25_order)
    hybrid_order = rank(hybrid)

    emb_rows = embedding_rows(emb_order, emb, titles, previous)
    hyb_rows = hybrid_rows(hybrid_order, hybrid, positions(emb_order), positions(bm25_order), bm25, titles, previous)
    write_csv(EMBEDDING_CSV, emb_rows, list(emb_rows[0]))
    write_csv(HYBRID_CSV, hyb_rows, list(hyb_rows[0]))
    RUN_INFO.write_text(json.dumps({
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "query": query, "query_is_stored_brief": query == brief, "details": None,
        "embedding_model": EMBED_MODEL, "dimensions": len(decode(cache["queries"][text_hash(query)]["embedding_b64"])),
        "paper_text": "title + blank line + abstract (whitespace collapsed)",
        "bm25": {"k1": BM25_K1, "b": BM25_B, "tokenizer": "lowercase [a-z0-9]+, plural fold, no stopwords"},
        "rrf_k": RRF_K, "papers": len(papers), "previous_matches": len(previous),
        "inputs_sha256": {p.name: sha256(p) for p in sorted(INPUTS.iterdir())},
        "overlap": {name: overlap(o, previous) for name, o in
                    {"embedding": emb_order, "bm25": bm25_order, "hybrid": hybrid_order}.items()},
    }, indent=2), encoding="utf-8")

    print_top("Embedding search", emb_rows, "similarity")
    print_top("Hybrid search (embedding + BM25, RRF)", hyb_rows, "rrf_score")
    print_summary(previous, titles, {"embedding": emb_order, "bm25": bm25_order, "hybrid": hybrid_order})
    print("\nThe previous LLM matches are a baseline with known mistakes, not ground truth.")
    print(f"Saved {EMBEDDING_CSV.name}, {HYBRID_CSV.name} and {RUN_INFO.name} to {EXP_DIR}")


if __name__ == "__main__":
    main()
