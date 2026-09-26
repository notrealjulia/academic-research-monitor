"""Extract the arXiv category taxonomy into a CSV.

Source: https://arxiv.org/category_taxonomy

Page structure (inside div#category_taxonomy_list):
  <h2 class="accordion-head">  group, e.g. "Computer Science", "Physics"
  <h3>                         physics-only archive, e.g. "Astrophysics (astro-ph)"
  <h4>code <span>(name)</span> category
  <p>                          category description

Mapping: field = h2 group, subfield = h3 name (blank outside Physics),
subject = h4 name, code = h4 code, description = following <p>.

arXiv calls the h3 level an "archive"; we name the column "subfield" because
that is clearer for people using the dataset.
"""

import csv
import re
import sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.request import Request, urlopen

URL = "https://arxiv.org/category_taxonomy"
OUT = Path(__file__).resolve().parent.parent / "data" / "arxiv_taxonomy.csv"


def clean(text):
    return re.sub(r"\s+", " ", text).strip()


class TaxonomyParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.rows = []
        self.list_depth = 0  # div nesting depth inside the taxonomy list; 0 = outside
        self.field = None
        self.subfield = ""
        self.capture = None  # tag whose text is being captured: "h2", "h3", "h4" or "p"
        self.buf = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "div":
            if self.list_depth:
                self.list_depth += 1
            elif attrs.get("id") == "category_taxonomy_list":
                self.list_depth = 1
            return
        if not self.list_depth:
            return
        if tag in ("h2", "h3", "h4") or (tag == "p" and self.rows and self.rows[-1]["description"] is None):
            self.capture, self.buf = tag, []

    def handle_endtag(self, tag):
        if tag == "div" and self.list_depth:
            self.list_depth -= 1
            return
        if tag != self.capture:
            return
        text = clean("".join(self.buf))
        self.capture = None
        if tag == "h2":
            self.field, self.subfield = text, ""
        elif tag == "h3":
            # "Condensed Matter (cond-mat)" -> "Condensed Matter"
            self.subfield = re.sub(r"\s*\([^)]*\)$", "", text)
        elif tag == "h4":
            m = re.match(r"(\S+)\s*\((.*)\)$", text)
            if not m:
                raise ValueError(f"Unexpected category heading: {text!r}")
            self.rows.append({"field": self.field, "subfield": self.subfield, "subject": m.group(2),
                              "code": m.group(1), "description": None})
        elif tag == "p":
            # arXiv shows "Description coming soon" as a placeholder; treat it as missing.
            self.rows[-1]["description"] = "" if text.lower() == "description coming soon" else text

    def handle_data(self, data):
        if self.capture:
            self.buf.append(data)


def main():
    req = Request(URL, headers={"User-Agent": "academic-research-monitor/0.1"})
    html = urlopen(req, timeout=30).read().decode("utf-8")

    parser = TaxonomyParser()
    parser.feed(html)
    rows = parser.rows
    if not rows:
        sys.exit("No categories found; the page layout may have changed.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["field", "subfield", "subject", "code", "description"])
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "description": row["description"] or ""})

    fields = sorted({r["field"] for r in rows})
    missing = [r["code"] for r in rows if not r["description"]]
    print(f"Wrote {len(rows)} subjects across {len(fields)} fields to {OUT}")
    print(f"Missing descriptions: {len(missing)} {missing if missing else ''}")


if __name__ == "__main__":
    main()
