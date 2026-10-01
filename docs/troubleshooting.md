# Troubleshooting

[Back to the README](../README.md)

## Installation and starting the program

**"py" or "python" is not recognized, or typing `python` opens the Microsoft Store.**
Python isn't installed, or Windows can't find it. Install it from <https://www.python.org/downloads/> and tick **"Add python.exe to PATH"** on the installer's first screen. Then close PowerShell, open a new window, and run `py --version`. If `py` still isn't found but `python --version` works, use `python` wherever the README says `py`.

**`curl.exe --version` fails, or "Error: curl was not found…"**
curl is built into Windows 10 (since 2018) and Windows 11, and the program uses it to contact arXiv. If it's missing, install it from <https://curl.se/windows/>, open a new PowerShell window and check `curl.exe --version` again. Type `curl.exe` with `.exe`: in PowerShell, plain `curl` runs a different command.

**"running scripts is disabled on this system" when activating the virtual environment.**
Windows blocks activation scripts by default. Allow them for your user account once (answer `Y` if asked), then activate again:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\.venv\Scripts\Activate.ps1
```

**No `(.venv)` at the start of the prompt.**
Every new PowerShell window starts without it. Go to the project folder and run `.\.venv\Scripts\Activate.ps1`. If PowerShell says the path doesn't exist, you're in the wrong folder (`cd academic-research-monitor`), or the virtual environment hasn't been created yet (`py -m venv .venv`).

**"research_monitor is not recognized as the name of a cmdlet…"**
Either the virtual environment isn't active (look for `(.venv)`), or the program isn't installed. With `(.venv)` showing, run `pip install -e .` from the project folder. Run it again too if you moved or renamed the project folder.

## API key and billing

**"Error: OPENAI_API_KEY is not set. Add a line OPENAI_API_KEY=... to …\.env"**
Check that the file is named exactly `.env` (not `.env.txt`), sits in the project folder (`notepad .env` opens it), and contains one line: `OPENAI_API_KEY=` followed directly by your key. A Windows environment variable named `OPENAI_API_KEY` takes priority over `.env`, so if you set one in the past, that key is used instead.

**"Error: OpenAI rejected the API key. Check OPENAI_API_KEY in .env."**
The key is wrong, incomplete or revoked. Create a new key at <https://platform.openai.com>, paste it into `.env` and save.

**"Error: OpenAI rate limit or quota exceeded: …"**
Usually the API account has no billing set up or has run out of credit; a ChatGPT subscription doesn't cover API use. Check billing at <https://platform.openai.com>. If the message mentions a rate limit (requests or tokens per minute), wait a minute and try again. Very large downloads screened with option `l` are the most likely to hit this. The search-first option `h` sends fewer requests.

**With the search-first option (`h`):**
- *"OpenAI quota, spend or billing limit reached (…)"*: billing or a spending limit. Waiting won't help, so the program stops without retrying. Check billing and limits.
- *"OpenAI rate limit still reached after 5 retries"* or *"OpenAI asked to wait …s before retrying"*: the program already waited and retried. Try again later.
- *"The model returned no usable keyword search terms for your description…"*: the run stopped before anything else was sent, and your previous results are unchanged. Reword your description and run again.

**"Error: Could not reach the OpenAI API. Check your internet connection."**
Check the connection, then run the program again.

## Searching

**"No papers were found for these categories and dates…"**
- Use a wider date range.
- Leave out the last day or two: arXiv hasn't announced the newest papers yet.
- Check the date format: day-month-year (`21-09-2026`, not `09-21-2026`).
- Revise your description so that different categories are suggested.

**"… papers match, more than arXiv's 30000-result limit."**
Use fewer categories or a shorter date range.

**Your answer went to the wrong question.**
Type or paste each answer as a single line and press Enter once. Don't type while the program is working, for example during the download. Anything typed early, or a second pasted line, becomes the answer to the next question.

**Matching stopped partway.**
The results so far are saved, and `research_report.txt` says **INCOMPLETE** at the top. Papers in the unfinished part were never checked. Run again to check everything.

**Something else went wrong mid-run.**
Press **Ctrl+C** to stop, then run `research_monitor` again. Files from finished steps stay in `data` until the next run replaces them.

## Results files

**Accented or non-English characters look garbled in Excel.**
Open the CSV through Excel's **Data → From Text/CSV** instead of double-clicking it, and choose UTF-8.
