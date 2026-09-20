# Junior AI Investigator

A Streamlit workbench for reviewing 50 synthetic long-term-care claim referrals.

The app separates the queue into three triage lanes, shows the evidence for each case, recommends the first check, answers case questions, and records the investigator's decision. The data has no fraud outcomes, so the app does not estimate fraud probability or claim model accuracy.

## Run locally

Python 3.10 or newer is required.

### Windows PowerShell

```powershell
py -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m streamlit run app.py
```

### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open **http://localhost:8501**.

No API key is required. The repository includes checked briefs in `briefs_cache.json`. When no key is present, follow-up questions use fixed rules and computed case facts.

For live model calls, copy `.env.example` to `.env` and set one supported key. `.env` and `.streamlit/secrets.toml` are ignored by Git.

| Variable | Use |
|---|---|
| `NVIDIA_API_KEY`, `OPENAI_API_KEY`, or `LLM_API_KEY` | Set one key to enable the model |
| `NVIDIA_MODEL`, `OPENAI_MODEL`, or `LLM_MODEL` | Optional model override |
| `LLM_BASE_URL` | Optional OpenAI-compatible endpoint |
| `LLM_TIMEOUT_S` | Optional timeout in seconds |

## Checks

```bash
pytest -q
python evaluate_briefs.py
```

The test suite covers data validation, triage rules, stress tests, generated-text checks, cache binding, the audit gate, and a headless run of every Streamlit view.

`evaluate_briefs.py` is a frozen 19-case regression comparison from an earlier prompt that let the model select evidence. It compares those selections with the current engine-selected evidence. It is not an independent validation set, and it does not show that model-written prose improves investigator decisions.

To regenerate the brief cache with a configured key:

```bash
python ai.py --prepare
```

## Method

The ten signals are converted to percentiles within this referred queue and averaged with equal weights. Three natural groups are used only when both cuts fall in clear gaps. A case cannot enter the likely-false-positive lane when any signal is outside the observed benign-profile range. A missing or shared claim number also sends the case to a person.

The result for this sample is:

- 8 priority investigations
- 13 cases needing judgment
- 29 likely false positives

Equal weights are a design choice because there are no outcomes from which to estimate supervised weights. The first principal component is close to the index, but individual cases can move when weights or the reference queue change. The app reports profile-group retention from random-weight, bootstrap, and leave-one-signal-out stress tests. It is a sensitivity measure, not a correctness probability.

"Likely false positive" is a workflow lane, not an estimated outcome probability. Its audit gate reads two low-stability cases deliberately and samples 14 cases at random from the other 27. If reviewers would detect a problem in a sampled case and all 14 random reads are clean, the 90% hypergeometric upper bound is at most 2 hidden problems in that 27-case lot. Any adverse decision blocks bulk closing. A production team should set this tolerance as policy and validate it with real outcomes.

## Language-model boundary

`triage.py` decides the lane, numbers, evidence list, and first check. Before the shift, `ai.py` asks the model to write short prose for all 50 cases around those fixed facts. The committed cache contains 44 first-pass drafts and 6 repaired drafts. `checks.py` checks every draft before display. A failed draft is repaired once, then replaced with a rule-built brief. Each cached brief is tied to a fingerprint of the facts used to create it.

The model does not decide claims, change lanes, select evidence, or receive claim number, state, or dates. The investigator must record a decision and can override a recommendation with a reason.

## Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit interface and decision log |
| `triage.py` | data checks, index, groups, guards, stress tests, and audit sample |
| `ai.py` | fact sheet, cached briefs, questions, and model client |
| `checks.py` | grounding and safety checks for generated text |
| `analysis.ipynb` | analysis with saved outputs |
| `evaluate_briefs.py` | frozen evidence-selection comparison |
| `test_triage.py` | automated tests |
| `data/` | synthetic referrals, signal definitions, and evaluation baseline |
| `WRITEUP.md` | short written explanation |
| `Write-up.pdf` | PDF version of the write-up |
| `Write-up-slides.pptx` | three-slide presentation requested in the brief |

## AI tool use

An AI coding assistant helped draft and revise code and text. I set the approach, checked the calculations, ran the tests, and made the product decisions.
