# Academic Research Monitor

Academic Research Monitor helps you find new papers on [arXiv](https://arxiv.org) that match your research interests. You describe your interests in your own words. The program then:

1. suggests which arXiv subject categories to follow, for you to accept or revise,
2. downloads every paper posted in those categories between two dates,
3. asks an AI model which of those papers fit your description, and gives a short reason for each one.

You use it by typing answers in a terminal window. No programming is needed. This guide assumes Windows.

**What uses your OpenAI account.** Suggesting categories and matching papers both send requests to OpenAI with *your* API key, so they are billed to *your* OpenAI API account. Downloading papers from arXiv is free and does not use OpenAI. Before matching, the program tells you how many papers and requests it will send, and it only starts after you type `y`.

**What the AI sees, and what it doesn't.** The matching step reads each paper's **title and abstract only**, never the full paper. Its reasons can be wrong: it may miss relevant papers or include irrelevant ones. Treat the results as suggestions and check each paper yourself. Your research description and the papers' titles and abstracts are sent to OpenAI.

---

## Start here

Do steps 1–6 once. After that, see [Using it again later](#using-it-again-later).

### What you need beforehand

- **A Windows computer** with an internet connection.
- **Git**, which downloads the project. Install it from <https://git-scm.com/download/win>; the default options are fine.
- **Python 3.12 or newer.** Install it from <https://www.python.org/downloads/>. On the installer's first screen, tick **"Add python.exe to PATH"**.
- **An OpenAI API key, with API billing set up.** Create the key on the OpenAI API platform at <https://platform.openai.com>. API use is billed separately from ChatGPT: **a ChatGPT subscription (Plus, Pro, etc.) does not include API usage.** Your API account needs a payment method or credit, or requests will fail.
- **curl**, a small download tool built into Windows 10 and 11. The program uses it to contact arXiv. You'll check it's there in step 1.

### 1. Open PowerShell and check your tools

PowerShell is the Windows terminal. Press the **Windows key**, type `PowerShell`, and open **Windows PowerShell**. A window with a blinking cursor appears. You type (or paste) a command, then press **Enter**.

To paste in PowerShell, right-click the window or press **Ctrl+V**.

Check that each tool is installed:

```powershell
git --version
py --version
curl.exe --version
```

- **Git** should print something like `git version 2.…`.
- **Python** should print `Python 3.12.…` or a higher number such as 3.13.
- **curl** should print a few lines starting with `curl`. Type `curl.exe` exactly, including `.exe`: in PowerShell, plain `curl` runs a different command.

If any check fails, see [Troubleshooting](#troubleshooting).

### 2. Download your own copy of the project

This downloads ("clones") a copy of the project into a new folder called `academic-research-monitor`, inside the folder PowerShell is currently in (normally `C:\Users\<your name>`). The copy is yours: nothing you do in it changes the original on GitHub.

```powershell
git clone https://github.com/notrealjulia/academic-research-monitor.git
cd academic-research-monitor
```

`cd` means "change directory": it moves PowerShell into the project folder. Run all remaining commands from inside this folder.

### 3. Create and switch on a virtual environment

A virtual environment is a private Python setup for this project, in a folder named `.venv`. It keeps the project's add-ons separate from anything else on your computer.

Create it (only needed once):

```powershell
py -m venv .venv
```

Switch it on ("activate" it):

```powershell
.\.venv\Scripts\Activate.ps1
```

When activation works, the prompt starts with `(.venv)`, for example:

```
(.venv) PS C:\Users\YourName\academic-research-monitor>
```

If you see an error saying *running scripts is disabled on this system*, see [Troubleshooting](#troubleshooting).

### 4. Install the program

With `(.venv)` showing, run:

```powershell
pip install -e .
```

The dot at the end matters: it means "this folder". The command downloads the one add-on the project needs (OpenAI's Python library) and makes the **`research_monitor`** command available in this virtual environment. It should finish with a line starting `Successfully installed`.

The installation points to this folder, so don't move or rename the folder afterwards. If you do, run this step again from the new location.

### 5. Add your OpenAI API key

The program reads your key from a file named `.env` in the project folder. Make it from the template, then open it in Notepad:

```powershell
Copy-Item .env.example .env
notepad .env
```

Notepad shows one line:

```
OPENAI_API_KEY=
```

Paste your key directly after the `=`, with no spaces or quotation marks, so it looks like `OPENAI_API_KEY=sk-...`. Save (**Ctrl+S**) and close Notepad.

Keep this key private. Anyone who has it can use your OpenAI account. Don't share it, email it or paste it anywhere else. The `.env` file is set up so Git never uploads it, so don't try to force it into a commit.

### 6. Run it

```powershell
research_monitor
```

The program asks its questions one at a time, as described in the next section.

---

## What the program asks

Answer each question and press **Enter**. To stop at any time, press **Ctrl+C**. At the category and date questions you can also type `x` to exit.

> **Tip:** type or paste each answer as **a single line** and press Enter once. Don't type anything while the program is working (for example, while it downloads papers). Anything typed early, or a second pasted line, is taken as the answer to the *next* question.

### Step 1 of 5: your research interests

```
Describe your research interests:
>
```

Write a few sentences about what you study and what you want to read about. Saying what you are *not* interested in also helps. For example (this is only an example; write your own):

> *Example:* I study how communities adapt to climate-related flooding, including early-warning systems, risk communication and how households decide to evacuate. I want new work that evaluates these systems with real-world data. I am not interested in purely engineering designs of flood barriers.

The program asks the AI model to suggest arXiv categories (this uses your OpenAI account). It then lists them with a short reason each, in this form:

```
1. <code>  <category name>  (<field>)
   <reason>
```

Then it asks:

```
[a] Accept  [r] Revise description  [x] Exit
```

- **`a` (accept):** keeps these categories and moves on. They are saved to `data\selected_categories.csv`.
- **`r` (revise):** shows your current description and lets you type a new one, then suggests categories again. Each attempt is a new request to OpenAI.
- **`x` (exit):** stops without saving any categories.

### Step 2 of 5: date range

```
Start date (dd-mm-yyyy, or x to exit):
End date (dd-mm-yyyy, Enter for today …):
```

Type dates as **day-month-year with dashes**, e.g. `21-09-2026`. Press Enter without typing an end date to mean "up to today". Both dates are included. Dates count in UTC (universal time), and each paper counts by the date its first version was submitted.

arXiv only lists papers once it has announced them, so the last day or two before today may not have appeared yet.

### Step 3 of 5: downloading papers (no OpenAI use)

The program downloads every paper in your accepted categories and dates from arXiv, including papers mainly filed under another category but cross-listed in yours. A progress bar shows how far it has got:

```
  [##############................]   500/1,069
```

arXiv asks programs to wait 3 seconds between requests, so large date ranges take a little while. When it finishes, it says how many papers it found and saves them to `data\retrieved_papers.csv`. If it finds none, it says so and stops.

### Step 4 of 5: optional extra details

The program shows your research description again and asks:

```
Optional: add details about which papers you want (Enter to skip):
```

You can add anything that narrows the search, such as *"only papers with field studies"* or *"skip review articles"*. The AI receives this alongside your original description; it doesn't replace it. Press Enter to skip.

### Step 5 of 5: matching (uses your OpenAI account)

The program shows how many papers it will screen and how many requests it will send to OpenAI, then asks:

```
Run matching? [y/N]
```

- **`y`:** the AI reads each paper's title and abstract and keeps the ones that fit your interests, each with a one-sentence reason. The matches are printed and saved to `data\paper_matches.csv`.
- **Anything else, or just Enter:** nothing is sent to OpenAI. The downloaded papers stay in `data\retrieved_papers.csv`.

---

## Your results

Results are saved as CSV files (simple spreadsheets) in the `data` folder inside the project folder. You can open them with Excel.

| File | What it contains |
|---|---|
| `selected_categories.csv` | The categories you accepted. Each row has the category code, its field and name, the reason it was suggested, and your research description. |
| `retrieved_papers.csv` | Every paper downloaded for your categories and dates. Each row has the arXiv ID, title, abstract, authors, submission date and time, categories, and links to the arXiv page and the PDF. |
| `paper_matches.csv` | The papers the AI judged relevant: the same columns as above plus `match_reason`, the AI's one-sentence reason. |

To open the folder in File Explorer from PowerShell (inside the project folder):

```powershell
explorer data
```

**Each new run replaces these files.** To keep results, copy the files somewhere else first.

If accented or non-English characters look garbled after double-clicking a file in Excel, open it through Excel's **Data → From Text/CSV** instead and choose UTF-8.

---

## Using it again later

You don't need to download, install or add your key again. Each time:

1. Open PowerShell.
2. Go to the project folder. If you cloned it in the default place:

   ```powershell
   cd academic-research-monitor
   ```

3. Switch on the virtual environment:

   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

4. Run the program:

   ```powershell
   research_monitor
   ```

---

## Troubleshooting

**"py" or "python" is not recognized, or typing `python` opens the Microsoft Store.**
Python isn't installed, or Windows can't find it. Install Python from <https://www.python.org/downloads/> and tick **"Add python.exe to PATH"** on the first screen. Then close PowerShell, open a new window, and check `py --version`. If `py` still isn't found but `python --version` works, use `python` wherever this guide says `py`.

**"running scripts is disabled on this system" when activating.**
Windows blocks activation scripts by default. Allow them for your user account once, answer `Y` if asked, then activate again:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\.venv\Scripts\Activate.ps1
```

**The virtual environment isn't activated (no `(.venv)` at the start of the prompt).**
Every new PowerShell window starts without it. Go to the project folder and run `.\.venv\Scripts\Activate.ps1`. If PowerShell says the path doesn't exist, you're either in the wrong folder (use `cd academic-research-monitor`) or haven't done [step 3](#3-create-and-switch-on-a-virtual-environment) yet.

**"research_monitor is not recognized as the name of a cmdlet…"**
Either the virtual environment isn't activated (check for `(.venv)`, see above), or the program isn't installed yet. With `(.venv)` showing, run `pip install -e .` from the project folder, as in [step 4](#4-install-the-program).

**"Error: OPENAI_API_KEY is not set. Add a line OPENAI_API_KEY=... to …\.env"**
The program found no key. Check that:
- the file is named exactly `.env` (not `.env.txt`) and sits in the project folder. Run `notepad .env` from the project folder to open it;
- the line reads `OPENAI_API_KEY=` followed directly by your key, on one line.

If the file looks right but the error remains, Notepad may have saved it in a format with a hidden marker at the start. Open it with `notepad .env`, choose **File → Save as**, set **Encoding** to **UTF-8** (not "UTF-8 with BOM"), and save.

The program uses a key set in your Windows environment variables named `OPENAI_API_KEY` *instead of* the one in `.env`. If you set one there in the past, it takes priority.

**"Error: OpenAI rejected the API key. Check OPENAI_API_KEY in .env."**
The key is wrong, incomplete or has been revoked. Create a new key on <https://platform.openai.com>, paste it into `.env` again, and save.

**"Error: OpenAI rate limit or quota exceeded: …"**
Usually your OpenAI API account has no billing set up or has run out of credit. Remember that a ChatGPT subscription doesn't cover API use. Check billing on <https://platform.openai.com>. If it's a short-term rate limit, wait a minute and try again.

**"Error: curl was not found…" or `curl.exe --version` fails.**
curl is included with Windows 10 (since 2018) and Windows 11. If it's missing, install it from <https://curl.se/windows/>, then open a new PowerShell window and check `curl.exe --version`.

**"No papers were found for these categories and dates…"**
- Try a wider date range.
- Leave out the last day or two: arXiv hasn't announced the newest papers yet.
- Check the dates are day-month-year (`21-09-2026`, not `09-21-2026`).
- Consider revising your description so that different categories are suggested.

**Something else went wrong mid-run.**
Press **Ctrl+C** to stop, then start again with `research_monitor`. Files from finished steps stay in the `data` folder until the next run replaces them.

---

## Advanced: running the steps separately

`research_monitor` runs three smaller commands in sequence. You can also run them one at a time, with the virtual environment activated. Each reads the previous one's file from `data`.

| Command | What it does | Uses OpenAI? |
|---|---|---|
| `find_categories "<your description>"` | Suggests categories and saves them to `data\selected_categories.csv` (no accept/revise step). | yes |
| `fetch_papers YYYY-MM-DD` | Downloads papers in the saved categories for **one** day (note the year-month-day format here) to `data\retrieved_papers.csv`. | no |
| `find_paper_matches "<your description>"` | Matches the saved papers against a description, after showing a few sample papers and asking `Proceed? [y/N]`. Saves `data\paper_matches.csv`. It warns you if the description differs from the one used to choose the categories. | yes |

For example, to try a different description on papers you've already downloaded, without contacting arXiv again:

```powershell
find_paper_matches "I study early-warning systems for floods and how people respond to them."
```

Other project tasks:

```powershell
python scripts/extract_arxiv_taxonomy.py   # rebuild data/arxiv_taxonomy.csv from arXiv's category page
python -m unittest                         # run the automated tests (no OpenAI or arXiv requests)
```

---

## Reference: arXiv category list

`data/arxiv_taxonomy.csv` holds arXiv's category list (155 categories across 8 fields). The program uses it to check the AI's category suggestions.

| Column        | Meaning |
|---------------|---------|
| `field`       | Top-level group, e.g. `Computer Science`, `Physics` |
| `subfield`    | What arXiv calls an "archive", e.g. `Astrophysics`. Only Physics has these; the column is blank elsewhere. |
| `subject`     | Category name, e.g. `Artificial Intelligence` |
| `code`        | arXiv category code, e.g. `cs.AI` |
| `description` | arXiv's description of the category |
