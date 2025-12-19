COB Traffic Project — Setup and Run Guide
=========================================

This guide walks you through installing Python, setting up your environment, training the model, running analyses, and generating the final timing report. It is written for non‑programmers and uses simple, copy‑paste commands.

Requirements
------------

- Windows 10 or 11
- Internet access
- Command Prompt or PowerShell

------------------
*1) Install Python*
------------------

First, check if Python 3 is already installed:

```powershell
py -V
# or
python --version
```

If you see a version (e.g., Python 3.11.x), continue. Otherwise install Python:
- From Microsoft Store: search “Python 3.11” and install
- Or download the Windows installer: https://www.python.org/downloads/

Make sure “Add Python to PATH” is checked during installation.

--------------------------------------------
*2) Create and Activate a Virtual Environment*
--------------------------------------------

Open Command Prompt or PowerShell in the project folder (the one containing this README), then run:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\activate
# If the "py" launcher isn’t available, try:
# python -m venv .venv
# .\.venv\Scripts\activate
```

When activated, your prompt will start with `(.venv)`.

--------------------------------------------
*3) Install Project Dependencies*
-------------------------------

Install all required Python packages listed in requirements:

```powershell
pip install -r requirements.txt
```

--------------------------------------------
*4) Train and Optimize PM Timing Plans*
-------------------------------------

This step trains the model(s) and generates improved timing plans for the PM periods (Plan 61 and Plan 64). It reads data from the `data` folder and writes optimized plans into `data\improved_timings`.

```powershell
python ml\models\train_pm_plans.py
```

You should see:
- Progress messages for loading/preprocessing data
- Optimization steps for each intersection
- A final summary and the path where improved timings are saved

--------------------------------------------
*5) Run Volume Analysis (Notebook)*
---------------------------------

Option A — Open in VS Code (recommended):
1. Open this project in VS Code.
2. Install the “Python” and “Jupyter” extensions if prompted.
3. Open the notebook: volume_analysis.ipynb.
4. Select the Python interpreter from `.venv` when prompted.
5. Run the cells from top to bottom.

Option B — Run in your browser using Jupyter:

```powershell
pip install jupyter
jupyter notebook
```

Then open `volume_analysis.ipynb` in the browser tab that appears and run the cells top to bottom.

--------------------------------------------
*6) Run LOS Analysis (Notebook)*
------------------------------

This notebook evaluates Level of Service (LOS) before and after optimization.

Use the same approach as Step 5, but open and run: `LOS_analysis.ipynb`.

Notes:
- The LOS engine lives in LOS.py and is used by the training/optimization pipeline.
- If you re‑run optimization, re‑run this notebook to reflect updates.

--------------------------------------------
*7) Generate the Visual Timing Report*
------------------------------------

This compiles all improved timing plans into a single, easy‑to‑read HTML report grouped by intersection and plan, with color‑coded LOS and split changes.

```powershell
python generate_timing_report.py
```

The report is written to `timing_improvement_report.html`. Double‑click it in File Explorer or open it in your browser.

Typical Workflow Summary
------------------------

1. Setup once: Steps 1–3
2. Whenever data changes:
	 - Re‑run optimization: `python ml\models\train_pm_plans.py`
	 - (Optional) Re‑run notebooks: `volume_analysis.ipynb`, `LOS_analysis.ipynb`
	 - Regenerate report: `python generate_timing_report.py`

Project Structure (key items)
-----------------------------

- `data/` — Input data and outputs
	- `times/` and `volume/` — Source CSVs for timing and volumes
	- `improved_timings/` — JSON outputs from optimization (by intersection/plan)
- `ml/models/train_pm_plans.py` — Main training/optimization entry point for PM plans
- `LOS.py` — LOS calculator used by the pipeline and analysis
- `volume_analysis.ipynb` — Explore and visualize traffic volumes
- `LOS_analysis.ipynb` — Evaluate LOS before/after improvements
- `generate_timing_report.py` — Builds the final HTML timing report

Troubleshooting
---------------

- "'py' is not recognized" — Use `python` instead of `py` in the commands.
- "python is not recognized" — Reinstall Python and ensure “Add Python to PATH” is checked, then restart the terminal.
- "pip is not recognized" — Try `python -m pip install -r requirements.txt`.
- Execution policy blocks activation (PowerShell): run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, then activate again.
- Jupyter/Notebooks won’t open — Ensure `pip install jupyter` ran in the active `.venv` and that VS Code uses the `.venv` interpreter.

Need Help?
----------

If you get stuck, share the console output and we can assist with next steps. Feel free to reach me at: brettberry07@gmail.com
