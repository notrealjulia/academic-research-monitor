"""Suggest arXiv categories to monitor for a researcher's written interests.

Usage:
  find_categories "I am looking for papers in computational linguistics and AI"

Sends the taxonomy from data/arxiv_taxonomy.csv plus the interest text to the
OpenAI API, checks every returned code against the CSV, prints the matches and
saves them to data/selected_categories.csv (overwritten on each run).

Reads OPENAI_API_KEY from the environment, or from .env in the project root.
"""

import csv
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TAXONOMY = ROOT / "data" / "arxiv_taxonomy.csv"
OUT = ROOT / "data" / "selected_categories.csv"
ENV_FILE = ROOT / ".env"

MODEL = "gpt-6-luna"
MAX_CATEGORIES = 5

INSTRUCTIONS = f"""You help a researcher choose arXiv categories to monitor for new papers.
You get the researcher's description of their interests and the full arXiv category list.
Pick the categories most worth monitoring: usually 1 to {MAX_CATEGORIES}, fewer if the interest is narrow.
One interest can match several categories, including categories in different fields.
Only use codes that appear in the list, copied exactly.
For each pick, give one short sentence explaining how it connects to what the researcher wrote."""


class SetupError(Exception):
    """A problem the user can fix, reported without a traceback."""


def load_taxonomy(path=TAXONOMY):
    if not path.exists():
        raise SetupError(f"{path} not found. Run: python scripts/extract_arxiv_taxonomy.py")
    with path.open(encoding="utf-8", newline="") as f:
        return {row["code"]: row for row in csv.DictReader(f)}


def load_api_key(env_file=ENV_FILE):
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key and env_file.exists():
        # utf-8-sig also accepts files saved as "UTF-8 with BOM" (an option in Notepad).
        for line in env_file.read_text(encoding="utf-8-sig").splitlines():
            name, sep, value = line.strip().partition("=")
            if sep and name.strip().removeprefix("export ").strip() == "OPENAI_API_KEY":
                key = value.strip().strip("'\"")
    if not key:
        raise SetupError(f"OPENAI_API_KEY is not set. Add a line OPENAI_API_KEY=... to {env_file}")
    return key


def format_taxonomy(taxonomy):
    lines = []
    for row in taxonomy.values():
        group = row["field"] + (f" > {row['subfield']}" if row["subfield"] else "")
        lines.append(f"{row['code']} | {row['subject']} | {group} | {row['description']}")
    return "code | name | field > subfield | description\n" + "\n".join(lines)


def ask_model(interest, taxonomy, api_key):
    # Imported here so the offline parts (taxonomy, validation, output) work without the SDK.
    import openai
    from pydantic import BaseModel

    class Pick(BaseModel):
        code: str
        reason: str

    class Picks(BaseModel):
        categories: list[Pick]

    client = openai.OpenAI(api_key=api_key)
    try:
        response = client.responses.parse(
            model=MODEL,
            reasoning={"effort": "low"},
            instructions=INSTRUCTIONS,
            input=f"Researcher's interests:\n{interest}\n\narXiv categories:\n{format_taxonomy(taxonomy)}",
            text_format=Picks,
        )
    except openai.AuthenticationError:
        raise SetupError("OpenAI rejected the API key. Check OPENAI_API_KEY in .env.")
    except openai.RateLimitError as e:
        raise SetupError(f"OpenAI rate limit or quota exceeded: {e.message}")
    except openai.APIConnectionError:
        raise SetupError("Could not reach the OpenAI API. Check your internet connection.")
    except openai.APIStatusError as e:
        raise SetupError(f"OpenAI API error {e.status_code}: {e.message}")

    if response.output_parsed is None:
        raise SetupError("The model returned no usable answer (it may have refused). Try rewording.")
    return [(p.code, p.reason) for p in response.output_parsed.categories]


def validate(picks, taxonomy):
    """Keep picks whose code is in the taxonomy, dropping duplicates. Returns (rows, rejected codes)."""
    rows, rejected, seen = [], [], set()
    for code, reason in picks:
        code = code.strip()
        if code not in taxonomy:
            rejected.append(code)
        elif code not in seen:
            seen.add(code)
            row = taxonomy[code]
            rows.append({
                "code": code,
                "field": row["field"],
                "subfield": row["subfield"],
                "subject": row["subject"],
                "reason": reason.strip(),
            })
    return rows, rejected


def print_results(rows, rejected):
    for i, row in enumerate(rows, 1):
        group = row["field"] + (f" > {row['subfield']}" if row["subfield"] else "")
        print(f"{i}. {row['code']}  {row['subject']}  ({group})")
        print(f"   {row['reason']}\n")
    if rejected:
        print(f"Ignored codes not in the taxonomy: {', '.join(rejected)}\n")


def save(rows, interest, path=OUT):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["code", "field", "subfield", "subject", "reason", "interest"])
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "interest": interest})


def main():
    interest = " ".join(sys.argv[1:]).strip()
    if not interest:
        print('Usage: find_categories "describe your research interests"', file=sys.stderr)
        sys.exit(2)
    try:
        taxonomy = load_taxonomy()
        api_key = load_api_key()
        print(f"Asking {MODEL} to match your interests against {len(taxonomy)} arXiv categories...\n")
        rows, rejected = validate(ask_model(interest, taxonomy, api_key), taxonomy)
    except SetupError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if not rows:
        print("No valid categories were suggested. Try describing your interests in more detail.")
        if rejected:
            print(f"Ignored codes not in the taxonomy: {', '.join(rejected)}")
        sys.exit(1)

    print_results(rows, rejected)
    save(rows, interest)
    print(f"Saved {len(rows)} categories to {OUT}")


if __name__ == "__main__":
    main()
