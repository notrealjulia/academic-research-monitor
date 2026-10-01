# Academic Research Monitor

Academic Research Monitor finds new [arXiv](https://arxiv.org) papers that match your research interests. You describe your interests in your own words, and the program:

1. suggests arXiv categories to follow, for you to accept or revise,
2. downloads every paper posted in those categories between two dates,
3. asks an AI model which papers fit your description, with a one-sentence reason for each.

You answer its questions in a Windows PowerShell window.

**Before you start:**
- **Cost.** The AI steps use the OpenAI API and are billed to your own OpenAI API account. That is separate from ChatGPT: a ChatGPT subscription does not cover API use. Downloading from arXiv is free. Before matching, the program shows how many requests it will send and waits for you to type `y`.
- **Privacy.** Your research description, and the titles and abstracts of downloaded papers, are sent to OpenAI.
- **Limits.** The AI judges each paper from its **title and abstract only**, never the full text. It can miss relevant papers or include irrelevant ones, so check each result yourself.

---

## Setup (once)

You need:
- **Git:** <https://git-scm.com/download/win> (default options are fine)
- **Python 3.12 or newer:** <https://www.python.org/downloads/>. On the installer's first screen, tick **"Add python.exe to PATH"**.
- **An OpenAI API key with billing set up:** <https://platform.openai.com>

Open **Windows PowerShell** (press the Windows key and type `PowerShell`). Paste each block with right-click or **Ctrl+V**, and press **Enter**.

Check the tools. Each command should print a version; type `curl.exe` exactly, including `.exe`:

```powershell
git --version
py --version
curl.exe --version
```

Download the project and go into its folder:

```powershell
git clone https://github.com/notrealjulia/academic-research-monitor.git
cd academic-research-monitor
```

Create a virtual environment (a private Python setup for this project), switch it on, and install the program:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
```

The prompt should now start with `(.venv)`. If activation fails with *"running scripts is disabled on this system"*, see [Troubleshooting](docs/troubleshooting.md).

Add your API key. This creates a file named `.env` and opens it in Notepad:

```powershell
Copy-Item .env.example .env
notepad .env
```

Paste your key straight after the `=` (no spaces or quotes), so the line reads `OPENAI_API_KEY=sk-...`, then save and close. Keep the key private: anyone who has it can spend from your account. Git never uploads `.env`.

---

## Running it

```powershell
research_monitor
```

Type each answer on a single line and press **Enter**. Press **Ctrl+C** to stop at any time.

1. **Research interests.** Describe what you study and want to read about, including what you're *not* interested in. For example:
   > I study machine translation for low-resource and regional European languages, such as Basque, Welsh or Sámi. I want new methods, datasets and evaluations, especially for dialects. I am not interested in speech recognition without translation.

   The AI checks that the description is specific enough. It then suggests categories: type `a` to accept, `r` to revise your description, or `x` to exit.
2. **Dates.** Enter a start and end date as day-month-year, e.g. `21-09-2026`, or press Enter for today. Both days count. The last day or two may be incomplete, because arXiv lists papers only once it has announced them.
3. **Download.** Papers in your categories are downloaded from arXiv. This is free, but takes a moment for long date ranges.
4. **Optional details.** Add anything that narrows the selection, e.g. *"only papers with new datasets"* or *"skip surveys"*. Press Enter to skip.
5. **Matching.** Choose how the AI screens the papers:
   - **`l`: screen every paper.** The AI reads the title and abstract of every downloaded paper. This is the most thorough option, and it costs the most on large downloads.
   - **`h`: search first, then screen the top 10% (experimental).** A search step ranks all papers by how closely they match your description, and the AI reads only the top 10%. Accepted papers get a 0–100 score that orders them; it's the AI's judgment, not a probability. This option is cheaper and faster, but a relevant paper that search ranks low is never read. [How it works](docs/experiments.md#the-search-first-option-in-research_monitor).

   The program shows how many papers and requests it will send, then asks `Run matching? [y/N]`. Type `y` to start. Anything else stops without sending anything.

At the end it lists the matches and prints how long each stage took.

---

## Your results

Results are saved in the `data` folder. To open it:

```powershell
explorer data
```

- **`research_report.txt`:** start here. It includes your search, the categories and dates, and every selected paper with its reason, link and abstract. It's written only when matching runs. If matching stopped partway, it says **INCOMPLETE** at the top.
- **`paper_matches.csv`:** the selected papers, as a spreadsheet.
- **`retrieved_papers.csv`:** every downloaded paper.
- **`selected_categories.csv`:** the categories you accepted.

**Each run replaces these files.** Copy them elsewhere if you want to keep them.

---

## Next time

Open PowerShell, then:

```powershell
cd academic-research-monitor
.\.venv\Scripts\Activate.ps1
research_monitor
```

---

## More help

- [Troubleshooting](docs/troubleshooting.md): installation problems, error messages, no papers found
- [Advanced usage](docs/advanced_usage.md): running single steps, file contents, settings
- [Experiments](docs/experiments.md): how the search-first option works, and the search and judge experiments
