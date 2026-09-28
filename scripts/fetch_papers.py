"""Fetch arXiv papers submitted on one UTC date in the selected categories.

Usage:
  fetch_papers 2026-09-25

Reads category codes from data/selected_categories.csv (made by find_categories),
queries the arXiv API for every paper in any of those categories (primary or
cross-list) whose first version was submitted on that date, 00:00-23:59 UTC,
pages through the full result set and saves it to data/retrieved_papers.csv
(overwritten on each run). Makes no OpenAI calls; find_paper_matches does the matching.

arXiv API rules (https://info.arxiv.org/help/api/tou.html): at most one request
every 3 seconds, one connection at a time. Titles and abstracts are CC0 metadata.
"""

import csv
import re
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from datetime import date, datetime, time as dtime, timezone
from urllib.parse import urlencode

from find_categories import ROOT, SetupError, load_taxonomy

SELECTED = ROOT / "data" / "selected_categories.csv"
RETRIEVED = ROOT / "data" / "retrieved_papers.csv"

ARXIV_URL = "https://export.arxiv.org/api/query"
USER_AGENT = "academic-research-monitor/0.1 (personal research experiment)"
PAGE_SIZE = 500  # arXiv allows up to 2000 per request; smaller pages are more reliable
ARXIV_MAX_RESULTS = 30000  # arXiv refuses queries that page past this
ARXIV_DELAY = 3.1  # seconds between requests
ARXIV_RETRIES = 3
SAMPLES = 3

NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
    "opensearch": "http://a9.com/-/spec/opensearch/1.1/",
}

PAPER_FIELDS = ["arxiv_id", "title", "abstract", "authors", "submitted", "primary_category",
                "categories", "url", "pdf_url"]


# --- arXiv ---------------------------------------------------------------

def load_codes(path=SELECTED):
    if not path.exists():
        raise SetupError(f"{path} not found. Run find_categories first.")
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    known = load_taxonomy()
    codes = list(dict.fromkeys(r["code"].strip() for r in rows if r["code"].strip() in known))
    if not codes:
        raise SetupError(f"No valid category codes in {path}. Run find_categories again.")
    return codes, rows


def build_query(codes, start, end):
    cats = " OR ".join(f"cat:{c}" for c in codes)
    return f"({cats}) AND submittedDate:[{start:%Y%m%d%H%M} TO {end:%Y%m%d%H%M}]"


def strip_version(arxiv_id):
    return re.sub(r"v\d+$", "", arxiv_id)


def text(el, path):
    found = el.find(path, NS)
    return " ".join(found.text.split()) if found is not None and found.text else ""


def parse_feed(xml):
    """Parse one Atom page into (total_results, papers)."""
    root = ET.fromstring(xml)
    total = int(text(root, "opensearch:totalResults") or 0)
    papers = []
    for entry in root.findall("atom:entry", NS):
        abs_url = text(entry, "atom:id")
        if "/api/errors" in abs_url:
            raise SetupError(f"arXiv rejected the query: {text(entry, 'atom:summary')}")
        arxiv_id = strip_version(abs_url.rsplit("/abs/", 1)[-1])
        primary = entry.find("arxiv:primary_category", NS).get("term")
        cats = [c.get("term") for c in entry.findall("atom:category", NS)]
        cats = [primary] + [c for c in cats if c != primary]
        pdf = next((l.get("href") for l in entry.findall("atom:link", NS) if l.get("title") == "pdf"), "")
        papers.append({
            "arxiv_id": arxiv_id,
            "title": text(entry, "atom:title"),
            "abstract": text(entry, "atom:summary"),
            "authors": "; ".join(text(a, "atom:name") for a in entry.findall("atom:author", NS)),
            "submitted": text(entry, "atom:published"),
            "primary_category": primary,
            "categories": " ".join(cats),
            "url": f"https://arxiv.org/abs/{arxiv_id}",
            "pdf_url": pdf or f"https://arxiv.org/pdf/{arxiv_id}",
        })
    return total, papers


class FetchError(Exception):
    pass


def fetch_page(query, start):
    # arXiv's CDN currently answers Python's built-in HTTP client with 406 Not Acceptable,
    # keyed on the TLS client rather than headers, so requests go through curl (bundled with Windows 10+).
    params = urlencode({"search_query": query, "start": start, "max_results": PAGE_SIZE,
                        "sortBy": "submittedDate", "sortOrder": "ascending"})
    curl = shutil.which("curl")
    if not curl:
        raise SetupError("curl was not found. It ships with Windows 10 and later; install it or add it to PATH.")
    result = subprocess.run(
        [curl, "--silent", "--show-error", "--fail", "--max-time", "120",
         "--user-agent", USER_AGENT, "--header", "Accept: application/atom+xml", f"{ARXIV_URL}?{params}"],
        capture_output=True,
    )
    if result.returncode != 0:
        raise FetchError(result.stderr.decode(errors="replace").strip() or f"curl exit code {result.returncode}")
    return result.stdout


def fetch_papers(codes, start, end):
    """Fetch every paper matching the query, one page at a time, following arXiv's rate limit."""
    query = build_query(codes, start, end)
    papers, offset, total, last_request = {}, 0, None, 0.0
    while total is None or offset < total:
        for attempt in range(1, ARXIV_RETRIES + 1):
            time.sleep(max(0.0, last_request + ARXIV_DELAY * attempt - time.monotonic()))
            last_request = time.monotonic()
            try:
                page_total, page = parse_feed(fetch_page(query, offset))
            except (FetchError, ET.ParseError) as e:
                error = e
                continue
            # arXiv occasionally returns an empty page mid-way; retry rather than stop early.
            if page or page_total <= offset:
                break
            error = "empty page"
        else:
            raise SetupError(f"arXiv request failed at result {offset} after {ARXIV_RETRIES} tries: {error}")

        if total is None:
            total = page_total
            if total > ARXIV_MAX_RESULTS:
                raise SetupError(f"{total} papers match, more than arXiv's {ARXIV_MAX_RESULTS}-result limit. "
                                 "Use fewer categories or a shorter date range.")
            print(f"arXiv reports {total} papers. Fetching {PAGE_SIZE} per request, 3 s apart...")
        if not page:
            break  # the result set shrank while paging; nothing more to fetch
        for p in page:
            papers.setdefault(p["arxiv_id"], p)  # deduplicate
        offset += len(page)
        print(f"  fetched {min(offset, total)}/{total}")

    # submittedDate is the v1 submission time; double-check the window locally (minute precision).
    lo, hi = f"{start:%Y-%m-%dT%H:%M}", f"{end:%Y-%m-%dT%H:%M}"
    kept = [p for p in papers.values() if lo <= p["submitted"][:16] <= hi]
    return kept, total


# --- Output --------------------------------------------------------------

def show_samples(papers, n=SAMPLES):
    print(f"\nSample of retrieved papers ({min(n, len(papers))} of {len(papers)}):")
    step = max(1, len(papers) // n)
    for p in papers[::step][:n]:
        abstract = p["abstract"] if len(p["abstract"]) <= 300 else p["abstract"][:300] + "..."
        print(f"\n  [{p['arxiv_id']}] {p['title']}")
        print(f"  {p['submitted']}  {p['categories']}")
        print(f"  {abstract}")
    print()


def save_papers(papers, path=RETRIEVED):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PAPER_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(papers)


def parse_day(arg):
    try:
        day = date.fromisoformat(arg)
    except ValueError:
        raise SetupError(f"'{arg}' is not a date. Use YYYY-MM-DD, e.g. 2026-09-25.")
    today = datetime.now(timezone.utc).date()
    if day > today:
        raise SetupError(f"{day} is in the future (today is {today} UTC).")
    return day


def main():
    if len(sys.argv) != 2:
        print("Usage: fetch_papers YYYY-MM-DD   (a UTC submission date)", file=sys.stderr)
        sys.exit(2)
    try:
        day = parse_day(sys.argv[1].strip())
        codes, _ = load_codes()
        start = datetime.combine(day, dtime(0, 0), timezone.utc)
        end = datetime.combine(day, dtime(23, 59), timezone.utc)
        print(f"Categories: {', '.join(codes)} (from {SELECTED.name})")
        print(f"Date: papers whose first version was submitted on {day}, "
              f"{start:%H:%M}-{end:%H:%M} UTC, in any of these categories (primary or cross-listed).")
        if (datetime.now(timezone.utc).date() - day).days < 3:
            print("Note: arXiv only returns announced papers, so a very recent date may be incomplete.")

        papers, total = fetch_papers(codes, start, end)
    except SetupError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"\nRetrieved {len(papers)} unique papers submitted on {day} (arXiv reported {total}).")
    if papers:
        show_samples(papers)
    save_papers(papers)
    print(f"Saved to {RETRIEVED}")


if __name__ == "__main__":
    main()
