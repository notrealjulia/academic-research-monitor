"""Interactive flow: interest -> categories -> date range -> fetch -> match.

Usage:
  research_monitor

A thin layer over find_categories, fetch_papers and find_paper_matches. Each stage
saves the same CSV as its standalone command, and only after the user has accepted
the previous stage:
  1. Suggest categories for the interest; save data/selected_categories.csv only if accepted.
  2. Ask for an inclusive UTC date range (dd-mm-yyyy).
  3. Fetch every paper first submitted in that range; save data/retrieved_papers.csv.
  4. Ask for optional extra details, kept separate from the original description.
  5. Show the number of OpenAI requests; match and save data/paper_matches.csv only if confirmed.
  6. After matching, write a readable report to data/research_report.txt.

A report from an earlier run is deleted as soon as this run starts replacing result
files (when categories are accepted), so a stale report never sits next to new CSVs.
"""

import sys
import textwrap
from datetime import date, datetime, time as dtime, timezone

import fetch_papers as fp
import find_categories as fc
import find_paper_matches as fm

SELECTED = fp.SELECTED
RETRIEVED = fp.RETRIEVED
MATCHES = fm.OUT
REPORT = fc.ROOT / "data" / "research_report.txt"

DATE_FORMAT = "%d-%m-%Y"
REPORT_WIDTH = 88


def ask(prompt):
    """input() that treats Ctrl+Z/Ctrl+D (end of input) as exit.

    Removes byte-order marks (U+FEFF): PowerShell adds one when piping text into a
    program, and pasted text can carry one. It is invisible but would otherwise be
    saved and sent to the model as part of the answer.
    """
    try:
        return input(prompt).replace("﻿", "").strip()
    except EOFError:
        leave()


def leave(message="Exited."):
    print(f"\n{message}")
    sys.exit(0)


def ask_description(prompt):
    while True:
        text = ask(prompt)
        if text:
            return text
        print("Please enter a description.")


# --- 1. Categories ---------------------------------------------------------

def choose_categories(taxonomy, api_key):
    """Suggest categories until the user accepts. Returns (interest, rows)."""
    interest = ask_description("Describe your research interests:\n> ")
    while True:
        print(f"\nAsking {fc.MODEL} to match your description against {len(taxonomy)} arXiv categories...\n")
        rows, rejected = fc.validate(fc.ask_model(interest, taxonomy, api_key), taxonomy)
        if rows:
            fc.print_results(rows, rejected)
        else:
            print("No valid categories were suggested.")
            if rejected:
                print(f"Ignored codes not in the taxonomy: {', '.join(rejected)}")
        options = "[a] Accept  [r] Revise description  [x] Exit" if rows else "[r] Revise description  [x] Exit"
        while True:
            choice = ask(f"{options}\n> ").lower()
            if choice == "a" and rows:
                return interest, rows
            if choice == "r":
                print(f"\nCurrent description:\n  {interest}")
                interest = ask_description("Revised description:\n> ")
                break
            if choice == "x":
                leave("Exited. No categories were saved.")
            print("Please type " + ("a, r or x." if rows else "r or x."))


# --- 2. Date range ---------------------------------------------------------

def parse_date(text, today):
    """Parse dd-mm-yyyy. Raises ValueError with a message for the user."""
    try:
        day = datetime.strptime(text, DATE_FORMAT).date()
    except ValueError:
        raise ValueError(f"'{text}' is not a valid dd-mm-yyyy date.")
    if day > today:
        raise ValueError(f"{day:%d-%m-%Y} is in the future (today is {today:%d-%m-%Y} UTC).")
    return day


def ask_date(prompt):
    text = ask(prompt)
    if text.lower() == "x":
        leave("Exited. Categories were saved; no papers were fetched.")
    return text


def choose_dates(today):
    """Ask for an inclusive date range. A blank end date means today (UTC)."""
    while True:
        try:
            start = parse_date(ask_date("\nStart date (dd-mm-yyyy, or x to exit): "), today)
            break
        except ValueError as e:
            print(e)
    while True:
        text = ask_date(f"End date (dd-mm-yyyy, Enter for today {today:%d-%m-%Y}): ")
        try:
            end = parse_date(text, today) if text else today
        except ValueError as e:
            print(e)
            continue
        if end < start:
            print(f"The end date must not be before the start date ({start:%d-%m-%Y}).")
            continue
        return start, end


def utc_window(start, end):
    return (datetime.combine(start, dtime(0, 0), timezone.utc),
            datetime.combine(end, dtime(23, 59), timezone.utc))


def describe_range(start, end):
    days = (end - start).days + 1
    return (f"papers whose first arXiv version was submitted from {start:%d-%m-%Y} 00:00 UTC "
            f"through {end:%d-%m-%Y} 23:59 UTC ({days} day{'s' if days != 1 else ''}, inclusive)")


# --- Report ----------------------------------------------------------------

def wrap(text, first_indent, indent):
    # Break only at spaces: never inside hyphenated words ("non-textual") or long words such as
    # URLs. A word longer than the line is left whole, so that line runs past REPORT_WIDTH.
    return textwrap.fill(text, REPORT_WIDTH, initial_indent=first_indent, subsequent_indent=indent,
                         break_on_hyphens=False, break_long_words=False)


def paragraph(text, indent="  "):
    return wrap(text, indent, indent)


def submitted_utc(iso):
    """'2026-09-24T18:13:26Z' -> '24-09-2026 18:13 UTC'."""
    return datetime.strptime(iso, "%Y-%m-%dT%H:%M:%SZ").strftime("%d-%m-%Y %H:%M UTC")


def build_report(interest, details, categories, start, end, papers, matches, done, batches, error, created):
    """Plain-text report of one research_monitor run. Uses only data the run already has."""
    rule = "=" * REPORT_WIDTH
    lines = [
        "RESEARCH MONITOR REPORT",
        rule,
        f"Created: {created:%d-%m-%Y %H:%M} UTC",
        "",
        paragraph(f"Papers were selected by an AI model ({fc.MODEL}) from each paper's title and abstract "
                  "only. The full papers were not read. Check each paper yourself before relying on it.", ""),
    ]
    if error:
        lines += ["", "INCOMPLETE: matching stopped early.",
                  paragraph(f"Only {done} of {batches} request batches finished before this error: {error}. "
                            "Papers in the unfinished batches were never checked, so relevant papers may be "
                            "missing from this report.", "")]
    lines += [
        "", "YOUR SEARCH", rule,
        "Research description:", paragraph(interest), "",
        "Additional details:", paragraph(details) if details else "  (none)", "",
        "arXiv categories (a paper counts if it is listed in any of them):",
        *[f"  {c['code']:<18} {c['subject']} ({c['field']}{' > ' + c['subfield'] if c['subfield'] else ''})"
          for c in categories],
        "",
        "Dates searched:", paragraph(describe_range(start, end)), "",
        "RESULTS", rule,
        f"Papers retrieved from arXiv: {len(papers)}",
        f"Papers selected as matches:  {len(matches)}"
        + (f" (from the {done} of {batches} batches that finished)" if error else ""),
    ]
    if not matches:
        lines += ["", "No papers were selected as matches for your description."]
    for i, p in enumerate(matches, 1):
        lines += [
            "", "-" * REPORT_WIDTH,
            wrap(f"{i}. {p['title']}", "", "   "), "",
            paragraph(f"Authors: {p['authors'].replace('; ', ', ')}"),
            f"  First submitted: {submitted_utc(p['submitted'])}",
            f"  arXiv: {p['url']}", "",
            "  Why it was selected:", paragraph(p["match_reason"], "    "), "",
            "  Abstract:", paragraph(p["abstract"], "    "),
        ]
    return "\n".join(lines) + "\n"


def remove_old_report(path):
    if path.exists():
        path.unlink()
        print(f"Removed the previous report ({path.name}), since this run replaces its results.")


# --- Flow ------------------------------------------------------------------

def run():
    taxonomy = fc.load_taxonomy()
    api_key = fc.load_api_key()

    print("Step 1 of 5: categories")
    interest, rows = choose_categories(taxonomy, api_key)
    fc.save(rows, interest, SELECTED)
    codes = [r["code"] for r in rows]
    print(f"Saved {len(rows)} categories to {SELECTED}")
    remove_old_report(REPORT)

    print("\nStep 2 of 5: date range")
    today = datetime.now(timezone.utc).date()
    start, end = choose_dates(today)
    print(f"\nRange: {describe_range(start, end)}.")
    if (today - end).days < 3:
        print("Note: arXiv only returns announced papers, so the most recent days may be incomplete.")

    print(f"\nStep 3 of 5: fetching papers in {', '.join(codes)} (primary or cross-listed)")
    papers, total = fp.fetch_papers(codes, *utc_window(start, end))
    fp.save_papers(papers, RETRIEVED)
    print(f"\nRetrieved {len(papers):,} unique papers (arXiv reported {total:,}). Saved to {RETRIEVED}")
    if not papers:
        leave("No papers were found for these categories and dates, so there is nothing to match. "
              "Try a wider date range.")

    print("\nStep 4 of 5: what to look for")
    print(f"Your research description:\n  {interest}")
    details = ask("Optional: add details about which papers you want (Enter to skip):\n> ")

    print("\nStep 5 of 5: matching")
    batches = fm.request_count(papers)
    print(f"{len(papers)} papers to screen. Matching will send their titles and abstracts to {fc.MODEL} "
          f"in {batches} requests.")
    if not fm.confirm("Run matching? [y/N] "):
        print(f"\nNo matching was run, so no report was written. The retrieved papers stay in {RETRIEVED}.")
        print(f'To match them later: find_paper_matches "<your description>"')
        return

    matches, done, error = fm.match_papers(interest, papers, api_key, details)
    fm.report(matches, papers, done, batches, error, MATCHES)
    REPORT.write_text(build_report(interest, details, rows, start, end, papers, matches, done, batches, error,
                                   datetime.now(timezone.utc)), encoding="utf-8")
    print(f"{'Incomplete report' if error else 'Report'} saved to {REPORT}")


def main():
    try:
        run()
    except fc.SetupError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()
