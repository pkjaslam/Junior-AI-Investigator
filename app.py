"""
app.py - the investigator's screen.      streamlit run app.py

Three views: Morning queue (a work queue), Case review (a workbench for one case) and Methods & audit.
The layout follows what case-management tools in banking and insurance do: numbers and tables first,
one card per job, detail behind tabs and tooltips, colour only for lane and severity.
This file only draws. Every number comes from triage.py and every sentence from ai.py.
"""
from __future__ import annotations

import csv
import html
import io
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ai
import triage as T

ROOT = Path(__file__).parent                               # the decision log lives here: decisions_<queue fingerprint>.csv

# blue / amber / red (not red-green, for colour-blind readers); shape and label always repeat what colour says
LANES = ("priority", "judge", "clear")                     # the order the work should be done in: highest risk first
LANE_COLOR = {"clear": "#2a78d6", "judge": "#eda100", "priority": "#d03b3b"}
LANE_TINT = {"clear": "#e6f0fc", "judge": "#fdf1d4", "priority": "#fbe3e3"}
LANE_SYMBOL = {"clear": "circle", "judge": "diamond", "priority": "square"}
LANE_SHAPE = {"clear": "&#9679;", "judge": "&#9670;", "priority": "&#9632;"}
LANE_HINT = {"judge": "Moderate, mixed signals. One first check settles most of each case. Start here.",
             "priority": "Every signal far out of range. Confirm and open a full investigation.",
             "clear": "Every signal inside the benign-profile range. Read the audit sample, then close the rest together."}
FLAG_SHORT = {"duplicate_service_billed": "Duplicate billing", "shared_contact_with_provider": "Shared contact",
              "recent_policy_change_flag": "Policy change", "service_overlap_other_provider": "Provider overlap"}
SIGNAL_SHORT = {"weekly_visit_frequency": "Visits / wk", "member_provider_distance_miles": "Distance (mi)",
                "prior_claims_last_12mo": "Prior claims", "weekend_billing_ratio": "Weekend share",
                "amount_vs_peer_avg_pct": "vs peer (%)", "round_dollar_billing_ratio": "Round-$ share", **FLAG_SHORT}
BAND_TINT = {"reference": "", "elevated": "background-color: #fdf1d4", "severe": "background-color: #fbe3e3"}     # as on the case page
# what the work queue can be ordered by: label -> (column, ascending). The index is deliberately not on this list.
SORTS = {"Signals out of range, most first": ("Out of range", False), "Hard flags, most first": ("Flags", False),
         "Claim amount, largest first": ("Amount ($)", False), "Claim amount, smallest first": ("Amount ($)", True),
         "Amount vs peer, highest first": ("vs peer (%)", False), "Profile group held least often, first": ("Profile group held (%)", True),
         "Claim date, newest first": ("Claim date", False), "Care type, A to Z": ("Care type", True), "Case id": ("Case", True)}
INK, INK2, MUTED, GRID, SURFACE, NAVY = "#111827", "#4b5563", "#6b7280", "#e5e7eb", "#ffffff", "#1e3a5f"
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'
ROW = 34                                                   # pixel height of one grid row (and of the grid header)

CLOSE, DESK, INVESTIGATE, HOLD, RETURN = ("Close case: no indicator supported", "Desk check requested",
                                          "Open full investigation", "Hold: data question",
                                          "Return to claims: billing or coverage issue")
DISPOSITIONS = [CLOSE, DESK, INVESTIGATE, HOLD, RETURN]
ACTION_TO_DISPOSITION = {"close_alert": CLOSE, "open_investigation": INVESTIGATE, "reconcile_claim_number": HOLD}
OVERRIDE_REASONS = ["I know context the data does not show", "A signal here is a known false alarm",
                    "The evidence is stronger than the tool says", "The evidence is weaker than the tool says",
                    "The data looks wrong", "Other"]
DECISION_FIELDS = ["timestamp_utc", "investigator", "case_id", "lane", "risk_level", "recommended", "decision",
                   "disposition", "override_reason", "finding_feedback", "note", "review_type", "in_audit_sample",
                   "brief_mode", "model", "prompt_version", "brief_sha", "facts_sha"]

CSS = """
<style>
[data-testid="stAppViewContainer"] {background:#f4f5f7;} [data-testid="stHeader"] {background:transparent;} [data-testid="stSidebar"] {background:#fff;}
[data-testid="stMainBlockContainer"] {padding:4.2rem 2.5rem 2rem; max-width:1480px; container-type:inline-size; container-name:page;}
[data-testid="stMainBlockContainer"] [data-testid="stVerticalBlock"] {gap:.7rem;}
div[class*="st-key-card"] {background:#fff; border-radius:10px; container-type:inline-size;}
.brand {display:flex; align-items:center; gap:10px;}
.brand .logo {width:32px; height:32px; border-radius:8px; background:#1e3a5f; color:#fff; font-weight:700; font-size:.8rem;
              display:flex; align-items:center; justify-content:center; letter-spacing:.03em; flex:none;}
.brand .name {font-weight:700; font-size:1.05rem; color:#111827; line-height:1.15;}
.brand .tag {color:#6b7280; font-size:.76rem;}
.st-key-nav button {min-height:2.1rem; padding:.15rem .75rem;} .st-key-nav button p {font-size:.85rem; white-space:nowrap;}
.st-key-ai_status button {min-height:1.7rem; padding:0 .75rem; border-radius:999px; background:#ece9fb; border-color:rgba(17,24,39,.1);}
.st-key-ai_status button p {font-size:.76rem; white-space:nowrap;}
.st-key-toolbar button p, .st-key-next_case button p {white-space:nowrap;}
.sec {font-size:.7rem; letter-spacing:.08em; text-transform:uppercase; color:#6b7280; font-weight:650; margin:0 0 6px 0;}
.page-title {font-size:1.12rem; font-weight:700; color:#111827;} .page-sub {color:#6b7280; font-size:.84rem;}
.chip {display:inline-block; padding:1px 9px; border-radius:999px; font-size:.74rem; margin:2px 4px 2px 0; font-weight:500;
       background:#f1f2f4; color:#111827; border:1px solid rgba(17,24,39,.08); white-space:nowrap;}
.chip.warn {background:#fdf1d4;} .chip.bad {background:#fbe3e3;} .chip.ok {background:#e6f0fc;} .chip.ai {background:#ece9fb;}
.lane {display:inline-block; padding:2px 10px; border-radius:6px; font-weight:600; font-size:.78rem; color:#111827; white-space:nowrap;}
.lane .shape {margin-right:6px;}
.key {display:flex; flex-wrap:wrap; gap:2px 16px; color:#4b5563; font-size:.78rem; margin-bottom:2px;} .key span span {margin-right:5px;}
.kpi {background:#fff; border:1px solid #e4e6ea; border-top:3px solid var(--c, #1e3a5f); border-radius:10px; padding:10px 14px; height:100%;}
.kpi .l {font-size:.7rem; letter-spacing:.07em; text-transform:uppercase; color:#6b7280; font-weight:650;}
.kpi .n {font-size:1.6rem; font-weight:700; line-height:1.15; color:#111827; font-variant-numeric:tabular-nums;}
.kpi .s {color:#4b5563; font-size:.8rem;}
.bar {height:6px; border-radius:3px; background:#e5e7eb; overflow:hidden; margin-top:6px;} .bar div {height:100%; background:#1e3a5f;}
.attn {background:#fff; border:1px solid #e4e6ea; border-radius:10px; padding:9px 14px; font-size:.82rem; color:#4b5563; line-height:1.4;}
.attn .row {display:flex; gap:6px; align-items:baseline; padding:2px 0;} .attn .row .chip {flex:none;}
.hero {background:#fff; border:1px solid #e4e6ea; border-left:4px solid var(--c, #1e3a5f); border-radius:10px; padding:12px 18px;}
.hero .meta {color:#6b7280; font-size:.8rem;} .hero .meta b {color:#111827;}
.hero h2 {margin:6px 0 10px 0; padding:0; font-size:1.12rem; font-weight:700; color:#111827; line-height:1.3;}
.facts {display:grid; grid-template-columns:repeat(auto-fit, minmax(150px, 1fr)); gap:8px 18px; border-top:1px solid #eef0f3; padding-top:9px;}
.fact .k {font-size:.66rem; text-transform:uppercase; letter-spacing:.07em; color:#6b7280; font-weight:650;}
.fact .v {font-weight:650; color:#111827; font-size:.92rem; font-variant-numeric:tabular-nums;}
.body {font-size:.9rem; line-height:1.5; color:#111827;}
.finding {font-size:.84rem; line-height:1.4; color:#4b5563;} .finding b {color:#111827; font-weight:650;}
.next {font-weight:650; color:#111827; font-size:.92rem; line-height:1.4;}
.result {border-radius:8px; padding:7px 10px; font-size:.8rem; line-height:1.4; height:100%; color:#111827;}
.result.good {background:#e6f0fc;} .result.bad {background:#fbe3e3;} .result b {display:block; font-size:.68rem; letter-spacing:.06em; text-transform:uppercase; color:#4b5563;}
.sig {display:grid; grid-template-columns:minmax(150px, 1.2fr) minmax(76px, .6fr) minmax(90px, 1.5fr) 100px; gap:0 12px;
      align-items:center; padding:6px 0; border-bottom:1px solid #eef0f3; font-size:.84rem;}
.sig .name {color:#111827;} .sig .val {font-weight:650; font-variant-numeric:tabular-nums; color:#111827;}
.sig .band {text-align:right;} .sig .band .chip {margin-right:0;}
.track {height:10px; border-radius:5px; overflow:hidden; display:flex;}
.track .z1 {background:#dbe9fb;} .track .z2 {background:#fbe9bd;} .track .z3 {background:#f6cccc;}
.trackwrap {position:relative;}
.marker {position:absolute; top:-4px; width:10px; height:18px; margin-left:-5px; border-radius:3px;
         background:#111827; border:2px solid #fff; box-sizing:border-box;}
.legend {color:#6b7280; font-size:.74rem; margin-top:8px;}
.note {padding:7px 10px; border-radius:8px; margin:5px 0; font-size:.83rem; line-height:1.4; color:#111827;}
.note.clash {background:#fbe3e3;} .note.info {background:#e6f0fc;} .note.context {background:#f1f2f4;}
.note b {font-weight:650;}
.small {color:#6b7280; font-size:.8rem;}
.ground {color:#6b7280; font-size:.76rem; margin-top:4px;}
div[class*="st-key-card"] [data-testid="stMarkdownContainer"] li, div[class*="st-key-card"] [data-testid="stMarkdownContainer"] p {font-size:.86rem; line-height:1.5;}
div[data-testid="stRadio"] label p {font-size:.82rem;}
/* a narrow card (small window, or the sidebar open): a signal takes two lines, a finding sits above its buttons */
@container (max-width: 470px) {
  .sig {grid-template-columns:84px 1fr max-content; grid-template-areas:"name name band" "val track track"; row-gap:5px; padding:8px 0;}
  .sig .name {grid-area:name;} .sig .val {grid-area:val;} .sig .trackwrap {grid-area:track;} .sig .band {grid-area:band;}
}
@container (max-width: 660px) {
  .st-key-card_findings [data-testid="stColumn"] {flex:1 1 100% !important; min-width:100% !important;}
}
/* a narrow page: the two main columns stack, toolbars and tile rows wrap */
@container page (max-width: 1040px) {
  div[class*="st-key-wrap"] [data-testid="stColumn"] {flex:1 1 30% !important; min-width:30% !important;}
}
@container page (max-width: 880px) {
  div[class*="st-key-split"] > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {flex:1 1 100% !important; min-width:100% !important;}
}
/* the work queue, rendered as a wrapping table so nothing is ever cut off */
.qscroll {max-height:560px; overflow:auto; border:1px solid #e4e6ea; border-radius:10px; background:#fff;}
.qt {width:100%; border-collapse:separate; border-spacing:0; font-size:.85rem; color:#111827; table-layout:fixed;}
.qt th {text-align:left; font-size:.68rem; letter-spacing:.05em; text-transform:uppercase; color:#6b7280; font-weight:650;
        padding:9px 12px; border-bottom:1px solid #e4e6ea; position:sticky; top:0; background:#f7f8fa; z-index:1;}
.qt td {padding:9px 12px; border-bottom:1px solid #eef0f3; vertical-align:top; line-height:1.35; overflow-wrap:anywhere;}
.qt tbody tr:last-child td {border-bottom:none;}
.qt tbody tr:hover td {background:#f6f9fd;}
.qt .c-case {font-weight:650; color:#1e3a5f; white-space:nowrap;}
.qt .c-find {color:#1f2733;}
.qt .c-amt {text-align:right; font-variant-numeric:tabular-nums; font-weight:600; white-space:nowrap;}
.qt .c-oor {white-space:nowrap; color:#4b5563; font-variant-numeric:tabular-nums;}
.qt .flagwrap {display:flex; flex-wrap:wrap; gap:4px;}
.oorbar {display:inline-block; width:52px; height:7px; border-radius:4px; background:#e6e8ec; overflow:hidden; vertical-align:middle; margin-right:7px;}
.oorbar i {display:block; height:100%; background:#1e3a5f;}
/* clickable, wrapping work-queue table (one bordered container; header aligns with the rows) */
div[class*="st-key-qtbl_"] {border:1px solid #e4e6ea; border-radius:10px; overflow:hidden; background:#fff;}
div[class*="st-key-qtbl_"] [data-testid="stVerticalBlock"] {gap:0 !important;}
div[class*="st-key-qtbl_"] [data-testid="stHorizontalBlock"] {padding:7px 12px; align-items:center; border-bottom:1px solid #eef0f3;}
div[class*="st-key-qtbl_"] [data-testid="stHorizontalBlock"]:first-child {background:#f7f8fa; border-bottom:1px solid #dfe3e8;}
div[class*="st-key-qtbl_"] [data-testid="stHorizontalBlock"]:last-child {border-bottom:none;}
div[class*="st-key-qtbl_"] [data-testid="stHorizontalBlock"]:not(:first-child):hover {background:#f6f9fd;}
div[class*="st-key-qtbl_"] .stButton button {border:none; background:transparent; color:#1e3a5f; font-weight:650;
   padding:0; min-height:0; box-shadow:none; justify-content:flex-start;}
div[class*="st-key-qtbl_"] .stButton button:hover {color:#12283f; text-decoration:underline;}
div[class*="st-key-qtbl_"] .stButton button:active, div[class*="st-key-qtbl_"] .stButton button:focus {color:#12283f; box-shadow:none;}
.qh {font-size:.68rem; letter-spacing:.05em; text-transform:uppercase; color:#6b7280; font-weight:650;}
.qf {font-size:.85rem; line-height:1.34; color:#1f2733; overflow-wrap:anywhere;}
.qo {font-size:.85rem; color:#4b5563; white-space:nowrap;}
.qflags {display:flex; flex-wrap:wrap; gap:4px;}
.qamt {font-size:.85rem; font-weight:600; text-align:right; font-variant-numeric:tabular-nums;}
.qcell {font-size:.85rem; color:#111827;}
</style>
"""


# ------------------------------------------------------------------ loading (cached) and small helpers
ENGINE_BUILD = "priority-order-v11"     # bump this whenever triage logic changes, so the cached engine rebuilds


@st.cache_resource(show_spinner="Triaging the queue ...")
def load_engine(uploaded: bytes | None = None, build: str = ENGINE_BUILD):
    """Fit once per queue: the brief's CSV by default, or one loaded from the sidebar.
    `build` is part of the cache key so a triage change rebuilds instead of serving a stale result."""
    model = T.fit_triage(T.load_cases(io.BytesIO(uploaded) if uploaded else T.default_csv()))
    packages = {c: T.assess_case(model, c) for c in model.case_ids}
    return model, packages, T.queue_table(model), T.morning_briefing(model)


@st.cache_data(show_spinner=False)
def load_briefs(cache_stamp: float, uploaded: bytes | None = None):
    return ai.briefs_for_queue(load_engine(uploaded, ENGINE_BUILD)[1])


@st.cache_resource(show_spinner=False)
def load_diagnostics(uploaded: bytes | None = None):
    return T.diagnostics(load_engine(uploaded, ENGINE_BUILD)[0])


def decisions_file() -> Path:
    return ROOT / f"decisions_{st.session_state.get('fingerprint', 'queue')}.csv"


def esc(text) -> str:
    return html.escape(str(text))


def md(text) -> str:
    """Streamlit markdown reads $...$ as LaTeX, and these texts are full of dollar amounts."""
    return str(text).replace("$", "\\$")


def money(x: float) -> str:
    return f"${x:,.0f}"


def lane_badge(lane_key: str) -> str:
    return (f'<span class="lane" style="background:{LANE_TINT[lane_key]}" title="{esc(LANE_HINT[lane_key])}"><span class="shape" '
            f'style="color:{LANE_COLOR[lane_key]}">{LANE_SHAPE[lane_key]}</span>{esc(T.LANES[lane_key]["label"])}</span>')


def flags_of(pkg: dict) -> list[str]:
    return [FLAG_SHORT.get(s["key"], s["label"]) for s in pkg["signals"] if s["kind"] == "flag" and s["band"] != "reference"]


FIND_NAME = {"weekly_visit_frequency": "Weekly visits", "member_provider_distance_miles": "Distance",
             "prior_claims_last_12mo": "Prior claims", "weekend_billing_ratio": "Weekend share",
             "amount_vs_peer_avg_pct": "Amount vs peer", "round_dollar_billing_ratio": "Round-dollar share"}


def short_finding(pkg: dict) -> str:
    """A compact, scannable lead for the work-queue table: the most notable signal and its value. The full
    headline is on the case card, so this stays short enough to read in the cell without truncation."""
    clashes = {n["signal"] for n in pkg["care_notes"] if n["effect"] == "clash"}
    measures = [s for s in pkg["signals"] if s["kind"] == "measure" and s["band"] != "reference"]
    measures.sort(key=lambda s: (s["key"] not in clashes, -s["queue_percentile"]))
    if measures:
        return f"{FIND_NAME.get(measures[0]['key'], measures[0]['label'])} {measures[0]['display']}"
    if pkg["counts"]["flags"]:
        n = pkg["counts"]["flags"]
        return f"{n} hard flag{'s' if n != 1 else ''} triggered"
    if pkg["lane"]["reason"] in ("data_hold", "missing_claim"):
        return "Claim number to reconcile"
    return "All signals in the benign range"


def queue_table_html(shown: pd.DataFrame, key: str) -> str:
    """The work queue as a wrapping table: the full finding and every hard flag show in full — the browser wraps
    long cells instead of clipping them, which the data grid cannot do. Rows are read here; a case opens from the map."""
    n_sig = len(T.SIGNALS)
    last = "Lane" if key == "all" else "Care type"
    cols = [("Case", "7%"), ("Finding", "30%"), ("Out of range", "11%"), ("Hard flags", "27%"),
            ("Amount", "9%"), (last, "9%"), ("Status", "7%")]
    colgroup = "".join(f'<col style="width:{w}">' for _, w in cols)
    thead = "".join(f"<th>{esc(h)}</th>" for h, _ in cols)
    body = []
    for _, r in shown.iterrows():
        oor = int(r["Out of range"])
        oor_cell = f'<span class="oorbar"><i style="width:{round(100 * oor / n_sig)}%"></i></span>{oor} of {n_sig}'
        flags = "".join(f'<span class="chip">{esc(f)}</span>' for f in r["Hard flags"]) or '<span style="color:#9aa1ab">—</span>'
        last_cell = (f'<span class="lane" style="background:{LANE_TINT[r["lane_key"]]}">{esc(r["Lane"])}</span>'
                     if key == "all" else esc(r["Care type"]))
        body.append(f'<tr><td class="c-case">{esc(r["Case"])}</td><td class="c-find">{esc(r["headline"])}</td>'
                    f'<td class="c-oor">{oor_cell}</td><td><div class="flagwrap">{flags}</div></td>'
                    f'<td class="c-amt">{money(r["Amount ($)"])}</td><td>{last_cell}</td><td>{esc(r["Status"])}</td></tr>')
    return (f'<div class="qscroll"><table class="qt"><colgroup>{colgroup}</colgroup>'
            f'<thead><tr>{thead}</tr></thead><tbody>{"".join(body)}</tbody></table></div>')


def chip(text: str, kind: str = "", tip: str = "") -> str:
    return f'<span class="chip {kind}" title="{esc(tip)}">{esc(text)}</span>'


def section(label: str) -> None:
    st.html(f'<div class="sec">{esc(label)}</div>')


def kpi(label: str, number: str, sub: str, color: str = NAVY, tip: str = "", extra: str = "") -> str:
    return (f'<div class="kpi" style="--c:{color}" title="{esc(tip)}"><div class="l">{esc(label)}</div>'
            f'<div class="n">{esc(number)}</div><div class="s">{esc(sub)}</div>{extra}</div>')


def grid(frame, key: str, config: dict | None = None, max_rows: int = 16) -> str | None:
    """A data grid (a DataFrame or a styled one) whose first column is the case id. A click on any cell opens that case."""
    data = getattr(frame, "data", frame)
    event = st.dataframe(frame, hide_index=True, width="stretch", row_height=ROW, height=ROW * (min(len(data), max_rows) + 1) + 3,
                         key=key, on_select="rerun", selection_mode="single-cell", column_config=config or {})
    try:
        return str(data.iloc[event["selection"]["cells"][0][0], 0])
    except (KeyError, IndexError, TypeError):
        return None


# ---- decisions: an append-only CSV is the audit trail

def load_decisions() -> pd.DataFrame:
    if not decisions_file().exists():
        return pd.DataFrame(columns=DECISION_FIELDS)
    return pd.read_csv(decisions_file(), dtype=str).fillna("")


def latest_decisions() -> dict[str, dict]:
    d = load_decisions()
    return {} if d.empty else d.groupby("case_id").tail(1).set_index("case_id").to_dict("index")


def audit_gate_state(audit: dict, clear_cases: list[str], decided: dict[str, dict]) -> dict:
    """One source of truth for the queue banner and the bulk-close screen."""
    status = {case_id: ("open" if case_id not in decided else
                        "clean" if decided[case_id]["disposition"] == CLOSE else "problem")
              for case_id in audit["cases"]}
    problems = [case_id for case_id, result in status.items() if result == "problem"]
    strays = [case_id for case_id in clear_cases
              if case_id not in audit["cases"] and case_id in decided and decided[case_id]["disposition"] != CLOSE]
    remaining = [case_id for case_id in clear_cases if case_id not in decided and case_id not in audit["cases"]]
    complete = bool(audit["cases"]) and all(result != "open" for result in status.values())
    clean = complete and not problems and not strays
    return dict(status=status, problems=problems, strays=strays, remaining=remaining,
                n_read=sum(result != "open" for result in status.values()), complete=complete,
                clean=clean, unlocked=clean and bool(remaining))


def save_decision(pkg: dict, brief: dict, decision: str, disposition: str, note: str, reason: str = "",
                  feedback: dict | None = None, review_type: str = "individual", in_audit: bool = False) -> None:
    """Append one row. brief_sha / facts_sha are fingerprints of the exact text and facts the investigator saw."""
    row = dict(timestamp_utc=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
               investigator=(st.session_state.get("who") or "").strip() or "unnamed",
               case_id=pkg["case"]["case_id"], lane=pkg["lane"]["label"], risk_level=pkg["lane"]["risk_display"],
               recommended=pkg["recommended_action"]["action_id"], decision=decision, disposition=disposition,
               override_reason=reason, finding_feedback=json.dumps(feedback or {}), note=note.strip(),
               review_type=review_type, in_audit_sample=str(in_audit), brief_mode=brief["mode"],
               model=brief.get("model", ""), prompt_version=brief.get("prompt_version", ""),
               brief_sha=T.package_hash({k: brief[k] for k in ai.BRIEF_KEYS}), facts_sha=T.package_hash(pkg))
    new_file = not decisions_file().exists()
    with open(decisions_file(), "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=DECISION_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def name_input(where) -> None:
    """Streamlit forgets a widget's value when the widget is not on screen, so the name is copied to plain state."""
    st.session_state["who"] = where.text_input("Investigator", value=st.session_state.get("who", ""), placeholder="Your name",
                                               key="investigator", help="Stored with every decision, for the audit trail.")


def scroll_to_top() -> None:
    """Streamlit keeps the scroll position between reruns, so a case opened from far down the queue would open mid-page."""
    nonce = st.session_state.get("nonce", 0)
    if st.session_state.get("scrolled_for") != nonce:
        st.session_state["scrolled_for"] = nonce
        st.html(f"<script>/* view change {nonce} */ document.querySelector('[data-testid=\"stMain\"]')?.scrollTo(0, 0);</script>",
                unsafe_allow_javascript=True)


def go_to(view: str, case_id: str | None = None, worklist: list | None = None):
    """worklist: the case ids of the table a case was opened from, in the order shown, so Previous / Next walk that list."""
    st.session_state["view"] = view
    if case_id:
        st.session_state["case_id"] = case_id
    if worklist is not None:
        st.session_state["worklist"] = list(worklist)
    st.session_state["nonce"] = st.session_state.get("nonce", 0) + 1     # new widget keys, which clears chart and grid selections


# ------------------------------------------------------------------ charts

def base_layout(fig: go.Figure, height: int, **kwargs) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=56, r=12, t=30, b=44), paper_bgcolor=SURFACE,
                      plot_bgcolor=SURFACE, font=dict(family=FONT, size=12, color=INK2),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(color=INK)),
                      hoverlabel=dict(bgcolor="#ffffff", font=dict(family=FONT, color=INK)), **kwargs)
    fig.update_xaxes(gridcolor=GRID, linecolor="#c9ccd1", zeroline=False)
    fig.update_yaxes(gridcolor=GRID, linecolor="#c9ccd1", zeroline=False)
    return fig


def dodge(x_px: np.ndarray, gap: float) -> np.ndarray:
    """Dot-plot stacking for one row of markers: a marker that would touch one already placed moves up or down a step."""
    steps, placed = np.zeros(len(x_px)), []
    for i in np.argsort(x_px):
        for step in (0, 1, -1, 2, -2, 3, -3):
            if all((x_px[i] - px) ** 2 + ((step - ps) * gap) ** 2 >= gap ** 2 for px, ps in placed):
                break
        steps[i] = step
        placed.append((x_px[i], step))
    return steps


def fig_queue_map(table, packages, briefs, decided, labelled):
    """One marker per case: signals out of range (y, a count) against claim amount (x, log), the two things the queue is
    ordered by. Markers never overlap, so each one can be clicked; a decided case is drawn hollow."""
    n_signals = len(T.SIGNALS)
    lo, hi = np.log10(table["exposure"].min()) - 0.08, np.log10(table["exposure"].max()) + 0.08
    x_px = (np.log10(table["exposure"]) - lo) / (hi - lo) * 560       # a narrow plot is assumed; a wider one only adds room
    y, lift = table["elevated"].astype(float), {}
    for level, rows in table.groupby("elevated").groups.items():
        steps = dodge(x_px[rows].to_numpy(), gap=15)
        y[rows] = level + 0.75 * steps                                # 0.75 of a row is about 15 px at this height
        lift.update({c: 15 * (steps.max() - step) for c, step in zip(rows, steps)})     # pixels to clear the markers above
    fig = go.Figure()
    for key in LANES:
        part = table[table["lane"] == key]
        done = [c in decided for c in part.index]
        custom = []
        for c in part.index:                                           # customdata[0] = case id
            nf = len(flags_of(packages[c]))
            custom.append([c, T.LANES[key]["label"], short_finding(packages[c]), f"{int(part.loc[c, 'elevated'])} of {n_signals} signals out of range",
                           f"{nf} hard flag{'s' if nf != 1 else ''}" if nf else "no hard flag", money(part.loc[c, "exposure"]), part.loc[c, "care_type"],
                           decided[c]["disposition"] if c in decided else "Open"])
        fig.add_trace(go.Scatter(
            x=part["exposure"], y=y[part.index], mode="markers", name=T.LANES[key]["label"],
            marker=dict(size=[10 if d else 12 for d in done], color=LANE_COLOR[key],
                        symbol=[LANE_SYMBOL[key] + ("-open" if d else "") for d in done],
                        line=dict(width=2, color=[LANE_COLOR[key] if d else SURFACE for d in done])),
            customdata=custom,
            hovertemplate="<b>%{customdata[0]}</b> · %{customdata[1]}<br>%{customdata[2]}<br>%{customdata[3]} · %{customdata[4]}"
                          "<br>%{customdata[5]} · %{customdata[6]} · %{customdata[7]}<extra></extra>"))
    for c in labelled:                                                 # the cases named under "Needs attention"
        lean = [np.sign(x_px[c] - x_px[o]) for o in labelled if o != c and abs(x_px[c] - x_px[o]) < 70 and abs(y[c] - y[o]) < 3]
        fig.add_annotation(x=float(np.log10(table.loc[c, "exposure"])), y=float(y[c]), text=c, showarrow=True, arrowhead=0,
                           arrowwidth=1, arrowcolor=MUTED, ax=26 * (lean[0] if lean else 0), ay=-(20 + lift[c]), standoff=7,
                           font=dict(size=11, color=INK2))
    ticks = [v for v in (500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000) if lo <= np.log10(v) <= hi]
    base_layout(fig, 310, showlegend=False, hovermode="closest", hoverdistance=28)
    fig.update_layout(margin=dict(l=52, r=12, t=26, b=44))
    fig.update_xaxes(title_text="Claim amount (log scale)", type="log", range=[lo, hi], tickvals=ticks,
                     ticktext=[f"${v / 1000:g}k" for v in ticks])
    fig.update_yaxes(title_text="Signals out of range", range=[-1.7, n_signals + 1.3], tickvals=list(range(n_signals + 1)), title_standoff=6)
    return fig


def fig_rank_ranges(table: pd.DataFrame) -> go.Figure:
    part = table[(table["lane"] != "clear") & (table["elevated"] > 0)].copy()     # the data-hold case has nothing to rank
    part["mid"] = (part["rank_lo"] + part["rank_hi"]) / 2
    part = part.sort_values("mid", ascending=False)
    fig = go.Figure()
    for key in ("priority", "judge"):
        p = part[part["lane"] == key]
        fig.add_trace(go.Scatter(
            x=p["mid"], y=p["case_id"], mode="markers", name=T.LANES[key]["label"],
            marker=dict(size=9, color=LANE_COLOR[key], symbol=LANE_SYMBOL[key], line=dict(width=2, color=SURFACE)),
            error_x=dict(type="data", symmetric=False, array=p["rank_hi"] - p["mid"],
                         arrayminus=p["mid"] - p["rank_lo"], thickness=2, width=0, color=LANE_COLOR[key]),
            customdata=np.stack([p["rank_lo"], p["rank_hi"]], axis=-1),
            hovertemplate=f"<b>%{{y}}</b><br>position %{{customdata[0]}} to %{{customdata[1]}} of {len(table)}<extra></extra>"))
    base_layout(fig, 470)
    fig.update_xaxes(title_text="Queue position (1 = most signals out of range)", autorange="reversed")
    fig.update_yaxes(categoryorder="array", categoryarray=list(part["case_id"]), showgrid=False)
    return fig


def clicked_case(event) -> str | None:
    """Case id behind a click on the chart (the first item of the point's customdata)."""
    try:
        return str(event["selection"]["points"][0]["customdata"][0])
    except (KeyError, IndexError, TypeError):
        return None


# ------------------------------------------------------------------ HTML pieces

def signal_panel(pkg: dict, model) -> str:
    """'Lab report' for the ten signals, one line each. The track is drawn in queue percentiles so all measures share
    one picture: blue = inside the benign-profile range, amber = elevated, red = where the top group starts."""
    rows = []
    for s in pkg["signals"]:
        if s["kind"] == "measure":
            column = model.df[s["key"]]
            z1 = 100 * float((column <= s["ref_max"]).mean())
            z2 = max(100 * float((column < s["severe_min"]).mean()) - z1, 0)
            band = {"reference": "in range", "elevated": "elevated", "severe": "severe"}[s["band"]]
            kind = {"reference": "ok", "elevated": "warn", "severe": "bad"}[s["band"]]
            tip = f"Benign-profile range up to {s['ref_display']}. Percentile {100 * s['queue_percentile']:.0f} of this queue."
            rows.append(
                f'<div class="sig" title="{esc(tip)}"><div class="name">{esc(s["label"])}</div><div class="val">{esc(s["display"])}</div>'
                f'<div class="trackwrap"><div class="track"><div class="z1" style="width:{z1:.1f}%"></div>'
                f'<div class="z2" style="width:{z2:.1f}%"></div><div class="z3" style="flex:1"></div></div>'
                f'<div class="marker" style="left:{100 * s["queue_percentile"]:.1f}%"></div></div>'
                f'<div class="band">{chip(band, kind)}</div></div>')
    for s in pkg["signals"]:
        if s["kind"] == "flag":
            on = s["band"] != "reference"
            tip = f"Triggered on {s['queue_prevalence']:.0%} of this queue."
            rows.append(f'<div class="sig" title="{esc(tip)}"><div class="name">{esc(s["label"])}</div>'
                        f'<div class="val">{"Triggered" if on else "No"}</div><div></div>'
                        f'<div class="band">{chip("hard flag", "bad") if on else chip("not triggered", "ok")}</div></div>')
    return "".join(rows)


def handover_note(case_id: str, d: dict) -> str:
    return (f"{case_id} | {d['lane']} | {d['disposition']} ({d['decision']}"
            + (f": {d['override_reason']}" if d.get("override_reason") else "") + f") | {d['note']} | "
            f"{d['investigator']}, {d['timestamp_utc']} UTC")


def provenance(brief: dict) -> tuple[str, str, str]:
    """Who wrote this brief and what was checked: a chip, one short line, and the full sentence for the tooltip."""
    v = brief["validation"]
    rejected = "; ".join(v.get("rejected_draft_issues", [])[:2])
    built = "This text is built directly from the case data."
    return {
        "llm": ("AI-written · checked", f"{brief['model']} · {v['numbers_checked']} numbers checked against the case data",
                f"Written by {brief['model']}; {v['numbers_checked']} numbers, {v['signals_checked']} signals and all counts were "
                "machine-checked against the case data. Free text is not: judge it against the signals."),
        "llm-repaired": ("AI-written · corrected once", f"{brief['model']} · first draft failed a check and was corrected",
                         f"Written by {brief['model']} and corrected once after its first draft failed a check ({rejected}). "
                         "Numbers, signals and counts are machine-checked, free text is not."),
        "fallback": ("AI draft discarded", "the model's draft did not pass every check; this text is built from the case data",
                     f"The model's draft did not pass every check and was discarded ({rejected}). {built}"),
        "unreachable": ("Model unreachable", "the model could not be reached when briefs were written",
                        f"The language model could not be reached when briefs were written. {built}"),
        "offline": ("Rule-built", "built from the case data by fixed rules", "No language model wrote this brief. " + built),
    }[brief["mode"]]


# ------------------------------------------------------------------ view 1: the morning queue

def queue_frame(packages, table, briefs, decided) -> pd.DataFrame:
    """One row per case, in work order, with everything the work queue can show, filter or sort on."""
    rows = []
    for c in table.index:
        p = packages[c]
        row = {"Case": c, "Lane": p["lane"]["label"], "Finding": short_finding(p), "headline": briefs[c]["headline"],
               "Out of range": p["counts"]["elevated"],
               "Hard flags": flags_of(p), "Flags": p["counts"]["flags"], "Amount ($)": p["case"]["claim_amount_usd"],
               "Care type": p["case"]["care_type"], "State": p["case"]["state"], "Claim date": str(p["case"]["claim_date"]),
               "Claim no.": p["case"]["claim_number"], "Profile group held (%)": float(table.loc[c, "stability"]),
               "Status": decided[c]["disposition"] if c in decided else "Open", "lane_key": p["lane"]["key"]}
        for s in p["signals"]:                                      # the raw values, for the signal-by-signal view
            name = SIGNAL_SHORT.get(s["key"], s["label"])
            row[name] = s["value"] if s["kind"] == "measure" else ("Yes" if s["band"] != "reference" else "")
        rows.append(row)
    return pd.DataFrame(rows)


def apply_view(frame: pd.DataFrame, search: str = "", care=(), flags=(), status: str = "All", sort=()) -> pd.DataFrame:
    """Filter the queue, then sort it on several columns at once, in the order they were picked.
    With no sort picked the rows keep the work order they arrived in."""
    keep = pd.Series(True, index=frame.index)
    if search.strip():
        text = (frame[["Case", "Claim no.", "Care type", "State", "Finding", "headline", "Status"]].astype(str).agg(" ".join, axis=1)
                + " " + frame["Hard flags"].str.join(" "))
        keep &= text.str.contains(search.strip(), case=False, regex=False)
    if care:
        keep &= frame["Care type"].isin(care)
    for flag in flags:                                               # every picked flag must be present
        keep &= frame["Hard flags"].map(lambda have: flag in have)
    if status != "All":
        keep &= (frame["Status"] == "Open") == (status == "Open")
    out = frame[keep]
    if sort:
        columns, ascending = zip(*(SORTS[k] for k in sort))
        out = out.sort_values(list(columns), ascending=list(ascending), kind="stable")
    return out


def work_table(frame: pd.DataFrame, key: str, model, packages, order_note: str) -> None:
    """Toolbar (search, filters, multi-column sort, view) and the grid. A click on a row opens the case, and
    Previous / Next in the case view then walk the rows exactly as they are shown here."""
    nonce = st.session_state.get("nonce", 0)
    with st.container(key=f"wrap_bar_{key}"):
        bar = st.columns([2.2, 1.7, 1.8, 1.5, 3.0, 2.0], vertical_alignment="center")
    search = bar[0].text_input("Search", key=f"q_{key}", placeholder="Search cases", label_visibility="collapsed")
    care = bar[1].multiselect("Care type", sorted(frame["Care type"].unique()), key=f"care_{key}", placeholder="Care type",
                              label_visibility="collapsed")
    flags = bar[2].multiselect("Hard flags", list(FLAG_SHORT.values()), key=f"flags_{key}", placeholder="Has hard flag",
                               label_visibility="collapsed", help="Cases that have every flag you pick.")
    status = bar[3].selectbox("Status", ["All", "Open", "Decided"], key=f"status_{key}", label_visibility="collapsed",
                              format_func=lambda v: "All statuses" if v == "All" else v)
    sort = bar[4].multiselect("Sort by", list(SORTS), key=f"sort_{key}", max_selections=3, label_visibility="collapsed",
                              placeholder="Sort by (up to 3, in order)")
    view = bar[5].radio("View", ["Summary", "Signals"], key=f"view_{key}", horizontal=True, label_visibility="collapsed",
                        help="Summary: one line per case. Signals: every signal value side by side, tinted where it is out of range.")
    shown = apply_view(frame, search, care, flags, status, sort)
    order = "; then ".join(k.lower() for k in sort) if sort else order_note
    hint = "Click a row to open it." if view == "Signals" else "Click a case to open it."
    st.caption(f"{len(shown)} of {len(frame)} cases · {money(shown['Amount ($)'].sum())} in claims · order: {order}. {hint}")
    if shown.empty:
        st.info("No case matches these filters.")
        return
    if view == "Signals":
        config = {"Case": st.column_config.TextColumn(width=78),
                  "Amount ($)": st.column_config.NumberColumn(format="localized", width=92)}
        names = {s["key"]: SIGNAL_SHORT.get(s["key"], s["label"]) for s in packages[shown["Case"].iloc[0]]["signals"]}
        data = shown[["Case", *names.values(), "Amount ($)"]]
        tint = pd.DataFrame("", index=data.index, columns=data.columns)
        for key_, name in names.items():
            tint[name] = [BAND_TINT["severe" if key_ in T.FLAGS and b != "reference" else b] for b in model.bands.loc[shown["Case"], key_]]
        for s in packages[shown["Case"].iloc[0]]["signals"]:
            if s["kind"] == "measure":
                spec = T.SIGNAL_INFO[s["key"]]["fmt"]                  # "{:+.0f}%" in signals.json -> "%+.0f" for the grid
                config[names[s["key"]]] = st.column_config.NumberColumn(format="%" + spec[spec.index(":") + 1:spec.index("}")],
                                                                        help=f"Benign-profile range: up to {s['ref_display']}")
        opened = grid(data.style.apply(lambda _: tint, axis=None), f"grid_{key}_{view}_{nonce}", config)
        if opened:
            go_to("case", opened, worklist=shown["Case"])
            st.rerun()
    else:                                       # Summary: clickable, wrapping rows — the full finding and every flag show, uncut
        n_sig = len(T.SIGNALS)
        ratios = [0.8, 3.1, 1.3, 2.7, 1.05, 1.35, 0.8]
        heads = ["Case", "Finding", "Out of range", "Hard flags", "Amount", "Lane" if key == "all" else "Care type", "Status"]
        order_list = list(shown["Case"])
        with st.container(key=f"qtbl_{key}_{nonce}"):
            for col, h in zip(st.columns(ratios, vertical_alignment="center"), heads):
                col.markdown(f"<div class='qh'>{esc(h)}</div>", unsafe_allow_html=True)
            for _, r in shown.iterrows():
                c = st.columns(ratios, vertical_alignment="center")
                c[0].button(r["Case"], key=f"open_{key}_{r['Case']}_{nonce}", on_click=go_to,
                            args=("case", r["Case"], order_list), help="Open this case")
                c[1].markdown(f"<div class='qf'>{esc(r['headline'])}</div>", unsafe_allow_html=True)
                oor = int(r["Out of range"])
                c[2].markdown(f"<div class='qo'><span class='oorbar'><i style='width:{round(100 * oor / n_sig)}%'></i>"
                              f"</span>{oor} of {n_sig}</div>", unsafe_allow_html=True)
                flags = "".join(f"<span class='chip'>{esc(f)}</span>" for f in r["Hard flags"]) or "<span style='color:#9aa1ab'>—</span>"
                c[3].markdown(f"<div class='qflags'>{flags}</div>", unsafe_allow_html=True)
                c[4].markdown(f"<div class='qamt'>{money(r['Amount ($)'])}</div>", unsafe_allow_html=True)
                last_cell = (f"<span class='lane' style='background:{LANE_TINT[r['lane_key']]}'>{esc(r['Lane'])}</span>"
                             if key == "all" else esc(r["Care type"]))
                c[5].markdown(f"<div class='qcell'>{last_cell}</div>", unsafe_allow_html=True)
                c[6].markdown(f"<div class='qcell'>{esc(r['Status'])}</div>", unsafe_allow_html=True)


def view_queue(model, packages, table, briefing, briefs):
    decided = latest_decisions()
    lanes = briefing["lanes"]
    nonce = st.session_state.get("nonce", 0)
    frame = queue_frame(packages, table, briefs, decided)

    st.html(f'<div class="page-title">Morning queue</div><div class="page-sub">Triaged before you arrived · '
            f'{briefing["n_cases"]} cases · {money(briefing["total_exposure"])} in claims</div>')
    done = len(decided) / max(briefing["n_cases"], 1)
    tiles = [kpi(T.LANES[k]["label"], str(lanes[k]["n"]), f"{money(lanes[k]['exposure'])} in claims · "
                 f"{sum(c in decided for c in lanes[k]['cases'])} decided", LANE_COLOR[k], LANE_HINT[k]) for k in LANES]
    tiles.append(kpi("Progress", f"{len(decided)} of {briefing['n_cases']}", "cases decided", NAVY,
                     extra=f'<div class="bar"><div style="width:{100 * done:.0f}%"></div></div>'))

    attention, audit = [], briefing["audit"]["cases"]                # (label, colour, what to do)
    if not briefing["structure_ok"]:
        attention.append(("Lanes switched off", "bad", T.LANE_REASON_TEXT["no_structure"]))
    if briefing["borderline"]:
        attention.append(("Borderline", "warn", ", ".join(briefing["borderline"]) + " sit right on a cut between two lanes, "
                          "so the lane is the least certain here — worth confirming before you rely on it."))
    if briefing["repeated_claim_cases"]:
        attention.append(("Shared claim number", "warn", " and ".join(briefing["repeated_claim_cases"])
                          + ": neither is closed in bulk until someone reconciles them."))
    if briefing.get("missing_claim_cases"):
        attention.append(("Missing claim number", "warn", ", ".join(briefing["missing_claim_cases"])
                          + (" has" if len(briefing["missing_claim_cases"]) == 1 else " have")
                          + " no claim number, so the shared-number check cannot run. Held for a person until it is supplied."))
    if audit:
        gate = audit_gate_state(briefing["audit"], lanes["clear"]["cases"], decided)
        if not gate["remaining"] and gate["complete"]:
            message = "Every case in this lane has a recorded decision."
        elif gate["problems"] or gate["strays"]:
            message = "An adverse decision keeps bulk closing locked. Review the remaining cases one by one."
        elif gate["unlocked"]:
            message = "The full sample is clean, so bulk closing is available."
        else:
            message = "Bulk closing stays locked until every sample case has a clean decision."
        attention.append(("Audit sample", "ok" if not gate["problems"] and not gate["strays"] else "warn",
                          f"{gate['n_read']} of {len(audit)} read. A case counts only after a decision is recorded. {message}"))
    todo = [c for c in table.index if c not in decided and (table.loc[c, "lane"] != "clear" or c in briefing["audit"]["cases"])]
    with st.container(key="split_queue"):
        left, right = st.columns([5, 7], gap="medium")
    with left:
        for row in (tiles[:2], tiles[2:]):
            for col, tile in zip(st.columns(2), row):
                col.html(tile)
        if attention:
            st.html('<div class="attn"><div class="sec">Needs attention</div>' + "".join(
                f'<div class="row">{chip(label, kind)}<span>{esc(text)}</span></div>' for label, kind, text in attention) + "</div>")
        if todo:
            st.button(f"Open the next case: {todo[0]}", key="next_case", type="primary", on_click=go_to, args=("case", todo[0], list(todo)),
                      help="The first case still needing a decision, in the priority-first work order; Next walks the rest in that order.")
    with right, st.container(border=True, key="card_map"):
        section("Queue map · one marker per case · click a marker to open it")
        st.html('<div class="key">' + "".join(f'<span><span style="color:{LANE_COLOR[k]}">{LANE_SHAPE[k]}</span>{esc(T.LANES[k]["label"])}</span>'
                                              for k in LANES) + "<span><span>&#9675;</span>hollow: decided</span></div>")
        labelled = set(briefing["borderline"]) | set(briefing["repeated_claim_cases"])
        event = st.plotly_chart(fig_queue_map(table, packages, briefs, decided, labelled), key=f"map_{nonce}", theme=None,
                                config=dict(displayModeBar=False), width="stretch", on_select="rerun", selection_mode="points")
        if clicked_case(event):
            go_to("case", clicked_case(event), worklist=[])
            st.rerun()
    with st.container(border=True, key="card_queue"):
        labels = {"priority": "Priority investigation", "judge": "Needs judgment", "clear": "Likely false positives", "all": "All cases"}
        counts = {**{k: lanes[k]["n"] for k in LANES}, "all": briefing["n_cases"]}
        tab_names = [f"{labels[k]} · {counts[k]}" for k in labels]
        last_lane = packages.get(st.session_state.get("case_id"), {}).get("lane", {}).get("key")
        tabs = st.tabs(tab_names, default=tab_names[LANES.index(last_lane)] if last_lane in LANES else None)
        notes = {"judge": "signals out of range, then claim amount", "priority": "claim amount (the signals cannot tell these cases apart)",
                 "all": "lane, then each lane's own order"}
        for tab, key in zip(tabs, labels):
            with tab:
                if key == "clear":
                    clear_lane(packages, table, briefing, briefs, decided)
                else:
                    work_table(frame if key == "all" else frame[frame["lane_key"] == key], key, model, packages, notes[key])


def clear_lane(packages, table, briefing, briefs, decided):
    """Close the likely false positives together - but only after the audit sample has been read clean."""
    lane, audit = briefing["lanes"]["clear"], briefing["audit"]
    nonce = st.session_state.get("nonce", 0)
    gate = audit_gate_state(audit, lane["cases"], decided)
    status, problems, strays = gate["status"], gate["problems"], gate["strays"]
    n_read, remaining = gate["n_read"], gate["remaining"]
    random_complete = all(status[c] != "open" for c in audit["random"])
    random_clean = random_complete and all(status[c] == "clean" for c in audit["random"])
    bound = T.acceptance_bound(audit["lot"], len(audit["random"]), audit["confidence"]) if random_clean else None

    with st.container(key="wrap_audit"):
        a, b, c = st.columns(3)
    a.html(kpi("Step 1 · audit sample read", f"{n_read} of {len(audit['cases'])}",
               f"{len(audit['purposive'])} least stable + {len(audit['random'])} random", LANE_COLOR["clear"],
               tip="A case counts as read only when you record a clean or not-clean decision on it — opening it is not enough."))
    b.html(kpi("Could still hide a problem", f"at most {bound}" if bound is not None else "pending",
               (f"of {audit['lot']} · {audit['confidence']:.0%} confidence, hypergeometric" if bound is not None
                else f"shown after all {len(audit['random'])} random reads are clean"), LANE_COLOR["clear"],
               "The purposive reads do not enter this bound. An adverse random read makes the clean-sample bound inapplicable."))
    c.html(kpi("Step 2 · close together", str(len(remaining)), "nothing left to close" if not remaining else "switched off by a finding"
               if problems or strays else "unlocks when every audit case reads clean" if not gate["complete"] else "ready to close",
               LANE_COLOR["clear"]))
    with st.popover("How this works"):
        st.markdown(f"None of these {lane['n']} cases has a hard flag, every measure sits inside the benign-profile range, and "
                    "their index lies below a clear gap in the queue.\n\n"
                    f"Read the {len(audit['purposive'])} targeted and {len(audit['random'])} random cases in full. The random sample "
                    f"was sized so a clean result limits the {audit['confidence']:.0%} upper bound to "
                    f"{audit['max_hidden_share']:.0%} of its lot. When every audit case reads clean, the rest can be closed in "
                    "one step. If any does not, bulk closing is switched off. Bulk-closed rows are marked in the log and are "
                    "never used as labels.")
    if problems:
        st.error(f"The audit found a problem in {', '.join(problems)}. Bulk closing is off: review the remaining "
                 f"{len(remaining)} cases one by one.")
    if strays:
        verb = "it was" if len(strays) == 1 else "they were"
        st.error(f"{', '.join(strays)} in this lane {'was' if len(strays) == 1 else 'were'} opened and not closed, so "
                 f"{verb} not a clean likely false positive. Bulk closing is off: an adverse finding here means the lane "
                 "assumption is not holding for this queue, so review the rest one by one.")

    section("Audit sample · click a row to open it")
    word = {"open": "To read", "clean": "Read: clean", "problem": "Read: not clean"}
    frame = pd.DataFrame([{"Case": x, "Why in the sample": "Profile group least stable" if x in audit["purposive"] else "Random draw",
                           "Profile group held (%)": float(table.loc[x, "stability"]), "Care type": packages[x]["case"]["care_type"],
                           "Amount ($)": packages[x]["case"]["claim_amount_usd"], "Status": word[status[x]]} for x in audit["cases"]])
    money_col = {"Case": st.column_config.TextColumn(width=80), "Amount ($)": st.column_config.NumberColumn(format="localized", width=95),
                 "Why in the sample": st.column_config.TextColumn(width=170), "Care type": st.column_config.TextColumn(width=150),
                 "Profile group held (%)": st.column_config.NumberColumn(format="percent", width=150)}
    opened = grid(frame, f"grid_audit_{nonce}", money_col) if len(frame) else None
    worklist = audit["cases"]

    n1, n2 = st.columns([2, 6], vertical_alignment="bottom")
    name_input(n1)
    note = n2.text_input("Note for the audit trail (required)", key=f"bulk_note_{nonce}", placeholder="What you read, and what you found.")
    blocked = n_read < len(audit["cases"]) or bool(problems) or bool(strays) or not remaining
    with st.container(horizontal=True, vertical_alignment="center", key="toolbar_bulk"):
        close = st.button(f"Close the other {len(remaining)} cases", type="primary", disabled=blocked, key="bulk_close")
        st.caption("Nothing left to close." if not remaining else "Off: a case in this lane was opened and not closed." if strays
                   else "Off: the audit found a problem." if problems else
                   f"Locked until all {len(audit['cases'])} audit cases are read ({len(audit['cases']) - n_read} to go)." if n_read < len(audit["cases"]) else
                   "The audit read clean. Closing is recorded as one bulk decision per case.")
    if close:
        if not note.strip():
            st.warning("Write the audit note first; nothing was closed.")
        else:
            for case_id in remaining:
                save_decision(packages[case_id], briefs[case_id], "accept", CLOSE, note, review_type="bulk")
            st.session_state["flash"] = (f"Closed {len(remaining)} cases together after a clean audit. They are marked "
                                         "'bulk' in the log and should never be used as training labels.")
            go_to("queue")
            st.rerun()

    with st.expander(f"The other {lane['n'] - len(audit['cases'])} cases in this lane"):
        rest = [x for x in lane["cases"] if x not in audit["cases"]]
        frame = pd.DataFrame({"Case": rest, "Care type": [packages[x]["case"]["care_type"] for x in rest],
                              "Amount ($)": [packages[x]["case"]["claim_amount_usd"] for x in rest],
                              "Profile group held (%)": [float(table.loc[x, "stability"]) for x in rest],
                              "Status": [decided[x]["disposition"] if x in decided else "Open" for x in rest]})
        opened_rest = grid(frame, f"grid_rest_{nonce}", money_col) if len(frame) else None
        if opened_rest:
            opened, worklist = opened_rest, rest
    if opened:
        go_to("case", opened, worklist=worklist)
        st.rerun()


# ------------------------------------------------------------------ view 2: one case

def find_cases(query: str, packages: dict) -> list[str]:
    """Case ids whose id or claim number equals the text, else contains it. Exact on purpose: a search box that
    ranks fuzzy matches can put the wrong claim first."""
    q = query.strip().lower()
    names = {c: (c.lower(), str(p["case"]["claim_number"]).lower()) for c, p in packages.items()}
    exact = [c for c, pair in names.items() if q in pair]
    return exact or [c for c, pair in names.items() if q and any(q in name for name in pair)]


def worklist_for(case_id: str, briefing: dict, lane_key: str) -> list:
    """The list Previous / Next walk: the table the case was opened from, else the audit sample, else the lane's work order."""
    work = st.session_state.get("worklist") or []
    if case_id in work:
        return work
    audit = briefing["audit"]["cases"]
    return audit if case_id in audit else briefing["lanes"][lane_key]["cases"]


def view_case(model, packages, table, briefing, briefs, client):
    case_id = st.session_state.get("case_id") or table.index[0]
    pkg, brief = packages[case_id], briefs[case_id]
    case, lane, counts = pkg["case"], pkg["lane"], pkg["counts"]
    decided = latest_decisions()

    work = worklist_for(case_id, briefing, lane["key"])
    pos = work.index(case_id)
    with st.container(key="split_tools"):
        tools, find = st.columns([6, 5], gap="medium", vertical_alignment="center")
    with tools.container(horizontal=True, vertical_alignment="center", key="toolbar_case"):
        st.button("Back to queue", on_click=go_to, args=("queue",), key="back")
        st.button("Previous", disabled=pos == 0, on_click=go_to, args=("case", work[max(pos - 1, 0)]), key="prev")
        st.button("Next", disabled=pos == len(work) - 1, on_click=go_to, args=("case", work[min(pos + 1, len(work) - 1)]), key="next")
        st.caption(f"{pos + 1} of {len(work)} in " + ("your list" if st.session_state.get("worklist") == work else "this lane's work order"))
    query = find.text_input("Find a case", key=f"find_{st.session_state.get('nonce', 0)}", label_visibility="collapsed",
                            placeholder="Find a case: type a case id or claim number, then Enter")
    hits = find_cases(query, packages)
    if len(hits) == 1:                                           # open it (this also empties the box, whose key changes)
        go_to("case", hits[0], worklist=None if hits[0] == case_id else [])
        st.rerun()
    elif len(hits) > 1:                                          # a claim number can sit on two cases (C1001 and C1031)
        with find.container(horizontal=True, vertical_alignment="center"):
            st.caption(f"{len(hits)} cases match:")
            for hit in hits[:6]:
                st.button(hit, key=f"hit_{hit}", on_click=go_to, args=("case", hit, []))
    elif query.strip() and not hits:
        find.caption("No case id or claim number matches.")

    facts = [("Risk level", lane["risk_display"]), ("Claim amount", money(case["claim_amount_usd"])),
             ("Signals out of range", f"{counts['elevated']} of {counts['signals']}"),
             ("Hard flags", f"{counts['flags']} of {len(T.FLAGS)}"),
             ("Profile group held in stress tests", "lane set by a safety rule" if lane["reason"] != "index" else
              f"{lane['stability']:.0%} of runs" + (" · borderline" if lane["borderline"] else ""))]
    st.html(f'<div class="hero" style="--c:{LANE_COLOR[lane["key"]]}">{lane_badge(lane["key"])} <span class="meta">&nbsp;<b>{esc(case_id)}</b> · '
            f'{esc(case["claim_number"])} · {esc(case["care_type"])} · {esc(case["state"])} · {esc(case["claim_date"])}</span>'
            f'<h2>{esc(brief["headline"])}</h2><div class="facts">'
            + "".join(f'<div class="fact"><div class="k">{esc(k)}</div><div class="v">{esc(v)}</div></div>' for k, v in facts)
            + "</div></div>")
    for caveat in pkg["data_caveats"]:
        st.warning(md(caveat))
    if case_id in decided:
        st.success(md("Decided. " + handover_note(case_id, decided[case_id])))

    with st.container(key="split_case"):
        left, right = st.columns([6, 5], gap="medium")
    with left:
        feedback = panel_assessment(pkg, brief)
        panel_decision(pkg, brief, briefing, feedback)
    with right, st.container(border=True, key="card_evidence"):
        tabs = st.tabs(["Signals", "Context and gaps", "Similar cases", "Ask AI"])
        with tabs[0]:
            st.html(signal_panel(pkg, model))
            st.html(f'<div class="legend">Marker: this case. Blue: range of the {pkg["queue"]["benign_reference_n"]} benign-profile '
                    'cases (descriptive, not a tolerance limit). Amber: elevated. Red: where the top group starts. '
                    'Hover a row for the numbers.</div>')
        with tabs[1]:
            panel_context(pkg, brief)
        with tabs[2]:
            panel_similar(pkg, model, decided)
        with tabs[3]:
            panel_chat(pkg, model, client)


def panel_assessment(pkg: dict, brief: dict) -> dict:
    """The brief, the findings to confirm or reject, and the first check. Returns the investigator's view on each finding."""
    action, case_id = pkg["recommended_action"], pkg["case"]["case_id"]
    short, line, full = provenance(brief)
    with st.container(border=True, key="card_summary"):
        section("AI assessment")
        st.html(f'<div class="body">{esc(brief["summary"])}</div><div class="ground">'
                f'{chip(short, "ai" if brief["mode"].startswith("llm") else "", full)}'
                f'{esc(line)}</div>')

    feedback = {}
    if brief["key_indicators"]:
        with st.container(border=True, key="card_findings"):
            section("Key indicators · confirm or reject each finding")
            labels = {s["key"]: s["label"] for s in pkg["signals"]}
            for i, item in enumerate(brief["key_indicators"]):
                name, finding = labels.get(item["signal"], ""), esc(item["finding"])
                if name and not item["finding"].lower().startswith(name.lower()):       # a model finding may not name its signal
                    finding = f"<b>{esc(name)}.</b> {finding}"
                text, mark = st.columns([3, 2.5], vertical_alignment="center")
                text.html(f'<div class="finding">{finding}</div>')
                view = mark.radio(f"Your view on finding {i + 1}", ["Agree", "Disagree", "Not sure"], index=None, horizontal=True,
                                  label_visibility="collapsed", key=f"find_{case_id}_{i}")
                if view:
                    feedback[item["signal"]] = view

    with st.container(border=True, key="card_next"):
        section("Recommended next step" + (f" · {action['question']}" if action.get("question") else ""))
        st.html(f'<div class="next">{esc(action["text"])}</div>')
        if action.get("if_clean"):
            r1, r2 = st.columns(2)
            r1.html(f'<div class="result good"><b>If it comes back clean</b>{esc(action["if_clean"])}</div>')
            r2.html(f'<div class="result bad"><b>If it does not</b>{esc(action["if_not"])}</div>')
        if action.get("steps"):
            with st.expander(f"Steps ({len(action['steps'])})"):
                st.markdown("\n".join(f"{i}. {md(s)}" for i, s in enumerate(action["steps"], 1)))
    return feedback


def panel_context(pkg: dict, brief: dict):
    """What changes the reading of the signals, and what the signals cannot show."""
    if pkg["care_notes"]:
        section(f"Care setting · {pkg['case']['care_type']}")
        word = {"clash": "Hard to reconcile", "context": "Context"}
        st.html("".join(f'<div class="note {n["effect"]}"><b>{word[n["effect"]]}.</b> {esc(n["text"])}</div>'
                        for n in sorted(pkg["care_notes"], key=lambda n: n["effect"] != "clash")))
    c1, c2 = st.columns(2)
    with c1:
        section("Most plausible ordinary reading")
        st.html(f'<div class="finding">{esc(brief["innocent_reading"])}</div>')
    with c2:
        section("What is missing")
        st.html(f'<div class="finding">{esc(brief["missing_information"])}</div>')
    section("How firmly it sits in this lane")
    st.html(f'<div class="finding">{esc(brief["stability_note"])}</div>')
    section("Not tested by these signals")
    st.html('<div class="finding">' + "".join(f"&bull; {esc(item)}<br>" for item in pkg["not_tested"]) + "</div>")


def panel_decision(pkg: dict, brief: dict, briefing: dict, feedback: dict):
    case_id = pkg["case"]["case_id"]
    suggested = ACTION_TO_DISPOSITION.get(pkg["recommended_action"]["action_id"], DESK)
    with st.container(border=True, key="card_decision"):
        section("Your decision")
        choice = st.radio(f"The tool recommends: {suggested}", ["Accept the recommendation", "Override"], index=None, horizontal=True,
                          key=f"choice_{case_id}",
                          help="Nothing is pre-selected and nothing is decided until you record it. Both paths need a note.")
        disposition, reason = suggested, ""
        if choice == "Override":
            o1, o2 = st.columns(2)
            disposition = o1.selectbox("Your disposition", [d for d in DISPOSITIONS if d != suggested], index=None,
                                       placeholder="Choose a disposition", key=f"disp_{case_id}")
            reason = o2.selectbox("Why", OVERRIDE_REASONS, index=None, placeholder="Choose a reason", key=f"why_{case_id}")
        d1, d2 = st.columns([1, 2.4])
        name_input(d1)
        note = d2.text_area("What you checked and what you found (required)", height=68, key=f"note_{case_id}")
        if st.button("Record decision", type="primary", key=f"save_{case_id}"):          # checked on click, so one click is enough
            if choice is None or not note.strip() or (choice == "Override" and not (disposition and reason)):
                st.warning("Nothing was recorded. Choose accept or override (an override needs a disposition and a reason) "
                           "and write the note.")
            else:
                save_decision(pkg, brief, "accept" if choice.startswith("Accept") else "override", disposition, note,
                              reason, feedback, in_audit=case_id in briefing["audit"]["cases"])
                decided = latest_decisions()
                audit = briefing["audit"]["cases"]
                remaining = [c for c in worklist_for(case_id, briefing, pkg["lane"]["key"]) if c not in decided]
                st.session_state["flash"] = f"Recorded {case_id}: {disposition}." + (
                    " The audit sample is read; the queue shows what it allows." if case_id in audit and all(c in decided for c in audit) else "")
                go_to("case", remaining[0]) if remaining else go_to("queue")
                st.rerun()


def panel_similar(pkg: dict, model, decided: dict):
    section("Closest signal profiles · how were they decided? · click a row to open it")
    near = T.similar_cases(model, pkg["case"]["case_id"], k=3)
    n_signals = len(T.SIGNALS)
    frame = pd.DataFrame([{"Case": r["case_id"], "Lane": r["lane"], "Out of range": r["elevated"], "Amount ($)": r["claim_amount_usd"],
                           "Decision": decided[r["case_id"]]["disposition"] if r["case_id"] in decided else "Open"} for r in near])
    opened = grid(frame, f"grid_similar_{pkg['case']['case_id']}_{st.session_state.get('nonce', 0)}",
                  {"Case": st.column_config.TextColumn(width=75), "Lane": st.column_config.TextColumn(width=150),
                   "Amount ($)": st.column_config.NumberColumn(format="localized", width=85),
                   "Out of range": st.column_config.ProgressColumn(min_value=0, max_value=n_signals, format=f"%d of {n_signals}", width=125)})
    if opened:
        go_to("case", opened, worklist=[])
        st.rerun()
    st.html('<div class="legend">Nearest cases by distance between signal profiles (percentiles). A precedent, not a verdict.</div>')


def panel_chat(pkg: dict, model, client):
    case_id = pkg["case"]["case_id"]
    history = st.session_state.setdefault("chat", {}).setdefault(case_id, [])
    lead = [f"What if the {s['label'].lower()} is explained?" for s in ai.rank_indicators(pkg, 1)]     # none on a benign case
    suggestions = ["Why is it in this lane?", "What should I check first?", "Could this be ordinary?", *lead]
    question = None
    with st.container(horizontal=True):        # a wrapping row: buttons keep their full text at any width
        for text in suggestions:
            if st.button(text, key=f"sugg_{case_id}_{text}"):
                question = text
    with st.container():                       # inline, so the page does not open scrolled to the bottom
        question = st.chat_input("Ask about this case in your own words", key=f"ask_{case_id}") or question
    if question:
        with st.spinner("Checking the case data ..."):
            result = ai.answer_question(question, pkg, model, client, history)
        history += [dict(role="user", content=question),
                    dict(role="assistant", content=result["answer"], mode=result["mode"], tools=result["tools"],
                         checked=result["validation"].get("numbers_checked", 0), model=getattr(client, "model", ""),
                         error=getattr(client, "last_error", "") if result["mode"] == "unreachable" else "")]
    for turn in history:
        with st.chat_message(turn["role"]):
            st.write(md(turn["content"]))
            if turn["role"] == "assistant":
                n = turn.get("checked", 0)
                who = f"Answered by {turn.get('model') or 'the model'}"
                how = {"llm": (f"{who}; {n} numbers plus wording, names, lane and counts checked against the case data" if n else
                               f"{who}; it quotes no numbers; wording, names, lane and counts were checked against the case data"),
                       "fallback": "The model's answer failed a check; answered from the case data by fixed rules instead",
                       "unreachable": "The model could not be reached" + (f" ({turn['error']})" if turn.get("error") else "")
                                      + "; answered from the case data by fixed rules",
                       "offline": "Answered from the case data by fixed rules"}[turn["mode"]]
                st.caption(how + (f" · computed: {', '.join(turn['tools'])}" if turn.get("tools") else ""))
    if not history:
        st.html('<div class="legend">Answers use only this case\'s data plus computed what-ifs, similar cases and queue totals. '
                'Questions about paying a claim are never answered.</div>')


# ------------------------------------------------------------------ view 3: methods and audit

def view_methods(model, table, briefing):
    d = load_diagnostics(st.session_state.get("uploaded"))
    dom = d["dominance"]
    stable = int((table["stability"] >= T.BORDERLINE).sum())
    top_n = int((model.groups == 2).sum())
    st.html('<div class="page-title">Methods and audit</div><div class="page-sub">How the lanes are made, what they cannot '
            'tell you, and how the tool is monitored. The full working is in analysis.ipynb.</div>')
    tabs = st.tabs(["How the lanes are made", "Closing safely", "Decision log", "Assumptions and data notes"])

    with tabs[0]:
        tiles = [kpi("Correlation between signals", f"{d['min_corr']:.2f} to {d['max_corr']:.2f}",
                     "pooled; inside a group on average " + ", ".join(f"{d['within_group_corr'][g]:+.2f}" for g in range(3))),
                 kpi("First principal component", f"{d['pc1_share']:.1%}", f"plain average correlates {d['index_vs_pc1']:.3f} with it"),
                 kpi("Gaps under the two cuts", " / ".join(f"{g:.0%}" for g in model.gap_share),
                     f"of the index range; the rule asks for {T.MIN_GAP_SHARE:.0%}" + ("" if model.structure_ok else " · lanes off")),
                 kpi("Stable profile groups", f"{stable} of {len(table)}",
                     f"{model.n_weights:,} weightings, {model.n_boot} resampled queues, {len(T.SIGNALS)} leave-one-out")]
        with st.container(key="wrap_method_tiles"):
            for col, tile in zip(st.columns(4), tiles):
                col.html(tile)
        with st.container(key="split_method"):
            a, b = st.columns([4, 6], gap="medium")
        with a, st.container(border=True, key="card_method"):
            section("In short")
            st.markdown(md(
                "- **One index.** Each signal becomes a percentile within this queue; the index is their plain average. No outcomes exist to learn weights from.\n"
                f"- **Three groups, not a continuum.** {model.n_reference} benign-profile, {int((model.groups == 1).sum())} middle, {top_n} extreme. "
                "Inside a group the index cannot rank cases, so it never sets the order inside a lane.\n"
                f"- **What holds without weights.** Each of the {top_n} extreme cases is at least as high as every case outside that group on all "
                f"{len(T.SIGNALS)} signals. None of the {dom['pairs_inside_top_group']} pairs among them can be ordered.\n"
                "- **Two guards.** Any signal out of range blocks bulk closing; a shared claim number holds the case for a person.\n"
                f"- **Limits.** Everything is relative to these {len(table)} referred cases. Without the three-group structure the lanes switch "
                "themselves off. In production the reference would be frozen from all claims, by care type."))
            st.caption("This is the share of stress-test runs in which a case keeps its statistical profile group. It measures sensitivity to modelling choices, not correctness probability.")
        with b, st.container(border=True, key="card_ranks"):
            section("The lanes are identifiable; the order inside a lane is not")
            st.plotly_chart(fig_rank_ranges(table), key="ranks", theme=None, config=dict(displayModeBar=False), width="stretch")
            st.caption(f"Each bar: the 5th to 95th percentile of the case's position across {model.n_weights:,} random weightings of the signals.")

    with tabs[1]:
        st.caption("Read a random sample before closing the rest. If every read is clean, the hypergeometric distribution bounds how "
                   "many problems can remain. The random reads are also the only unbiased labels from the low end "
                   "(AML teams call this below-the-line testing).")
        s1, s2, s3 = st.columns(3)
        n_lot = int(s1.number_input("Cases eligible for bulk closing", 1, 100000, max(1, int(briefing["audit"]["lot"])), key="acc_n"))
        target = float(s2.selectbox("Largest share allowed to hide a problem", [0.20, 0.10, 0.05, 0.02, 0.01], index=1, key="acc_t",
                                    format_func=lambda v: f"{v:.0%}"))
        conf = float(s3.selectbox("Confidence", [0.90, 0.95, 0.99], key="acc_c", format_func=lambda v: f"{v:.0%}"))
        need = T.audit_size_for(n_lot, target, conf)
        k1, k2, k3 = st.columns(3)
        k1.html(kpi("Read at random", f"{need} of {n_lot:,}", f"{need / n_lot:.0%} of the lot"))
        k2.html(kpi("If all are clean, at most", f"{T.acceptance_bound(n_lot, need, conf)} cases", f"({target:.0%}) could still hide a problem"))
        k3.html(kpi("Confidence", f"{conf:.0%}", "a statement about the sampling procedure; assumes a reader would spot a problem"))

    with tabs[2]:
        log = load_decisions()
        if log.empty:
            st.caption("No decisions recorded yet. Override rate by lane is the first drift signal; disagreement with individual "
                       "findings is the second. Bulk-closed rows are marked and must never be used as training labels.")
        else:
            latest = log.groupby("case_id").tail(1)
            read = latest[latest["review_type"] != "bulk"]          # a bulk close is not a judgment, so it cannot be an override
            summary = pd.DataFrame({
                "read by a person": read.groupby("lane").size(),
                "overridden": read[read["decision"] == "override"].groupby("lane").size(),
                "closed in bulk": latest[latest["review_type"] == "bulk"].groupby("lane").size()}).fillna(0).astype(int)
            summary["override rate"] = [f"{o / n:.0%}" if n else "" for o, n in zip(summary["overridden"], summary["read by a person"])]
            section("Override rate by lane · the first drift signal")
            st.dataframe(summary, width="stretch")
            section("Every recorded decision, newest first")
            st.dataframe(log.iloc[::-1], hide_index=True, width="stretch")
            st.download_button("Download the decision log (CSV)", log.to_csv(index=False), "decisions.csv", "text/csv")

    with tabs[3]:
        with st.container(key="split_notes"):
            a, b = st.columns(2, gap="medium")
        with a, st.container(border=True, key="card_assume"):
            section("Assumptions")
            st.markdown("\n".join(f"- {md(x)}" for x in T.ASSUMPTIONS))
            st.caption(f"Workload estimate on those assumed minutes: about {briefing['minutes_tool'] / 60:.1f} h with the lanes and the "
                       f"audit sample, against {briefing['minutes_manual'] / 60:.1f} h reading the queue top to bottom. A hypothesis until measured.")
        with b, st.container(border=True, key="card_data"):
            section("Not tested by these signals")
            st.markdown("\n".join(f"- {x}" for x in T.NOT_TESTED))
            section("Data notes")
            for f in model.audit:
                (st.warning if f["level"] == "warn" else st.info)(md(f["text"]))


# ------------------------------------------------------------------ page

def ai_status(packages: dict, briefs: dict, client) -> None:
    """Header button: how many briefs the language model wrote, and what happened to the rest."""
    need = [c for c, p in packages.items() if p["lane"]["key"] != "clear"]
    mode = {c: briefs[c]["mode"] for c in need}
    written = [c for c in need if mode[c].startswith("llm")]
    repaired = [c for c in need if mode[c] == "llm-repaired"]
    dropped = [c for c in need if mode[c] == "fallback"]
    other = [c for c in need if mode[c] in ("offline", "unreachable")]
    with st.popover(f"AI briefs {len(written)} of {len(need)}" if written else "Rule-built briefs"):
        lines = [f"**{len(need)} cases need a person**, so {len(need)} briefs are drafted by the language model before the shift. "
                 f"The other {len(packages) - len(need)} are likely false positives: every signal is in range, so a fixed rule "
                 "writes their one line and no model is called.",
                 "The statistics choose the exact evidence shown on each case; the model only writes the prose around it, "
                 "so a brief can never omit a triggered flag or a top indicator."]
        if written:
            lines.append(f"**{len(written)} drafts passed every check** against the case data"
                         + (f" ({len(repaired)} after one correction: {', '.join(repaired)})." if repaired else "."))
        if dropped:
            lines.append(f"**{len(dropped)} drafts did not pass every check** and were replaced by rule-built text: "
                         f"{', '.join(dropped)}. The case page says why.")
        if other:
            lines.append(f"**{len(other)} have no model draft** (no model was configured or reachable when briefs were written); "
                         "they show rule-built text.")
        lines.append((f"Live questions in a case are answered by **{client.model}** and checked the same way." if client else
                      "No language model is configured now, so follow-up questions are answered from the case data by fixed rules."))
        st.markdown(md("\n\n".join(lines)))


def main():
    st.set_page_config(page_title="Junior AI Investigator", layout="wide", initial_sidebar_state="collapsed")
    st.html(CSS)
    with st.sidebar:
        st.subheader("Queue")
        upload = st.file_uploader("Load a different queue (CSV with the same columns)", type="csv", key="upload")
    uploaded = upload.getvalue() if upload else None
    try:
        model, packages, table, briefing = load_engine(uploaded, ENGINE_BUILD)
    except (ValueError, FileNotFoundError) as problem:          # a bad file is reported, never guessed at
        st.error(f"That file was not loaded: {problem}")
        uploaded = None
        model, packages, table, briefing = load_engine(None)
    if st.session_state.get("fingerprint") not in (None, model.fingerprint):      # a new queue: forget the old case
        st.session_state.pop("case_id", None)
        go_to("queue")
    st.session_state["fingerprint"], st.session_state["uploaded"] = model.fingerprint, uploaded
    client = ai.get_client()
    briefs = load_briefs(ai.CACHE_PATH.stat().st_mtime if ai.CACHE_PATH.exists() else 0.0, uploaded)
    st.session_state.setdefault("view", "queue")
    nonce = st.session_state.get("nonce", 0)

    with st.container(key="split_head"):
        head, nav, status = st.columns([3, 4.4, 2.6], vertical_alignment="center")
    head.html('<div class="brand"><div class="logo">JAI</div><div><div class="name">Junior AI Investigator</div>'
              '<div class="tag">Long-term care · case triage</div></div></div>')
    views = {"Morning queue": "queue", "Case review": "case", "Methods & audit": "methods"}
    with nav.container(horizontal=True, horizontal_alignment="center", key="nav"):
        for label, view in views.items():
            st.button(label, key=f"nav_{view}", type="primary" if view == st.session_state["view"] else "secondary",
                      on_click=go_to, args=(view,))
    with status.container(horizontal=True, horizontal_alignment="right", key="ai_status"):
        ai_status(packages, briefs, client)
    slot = st.container()                                   # always present, so a message never shifts (and remounts) what is below it
    if st.session_state.get("flash"):
        slot.success(md(st.session_state.pop("flash")))

    if st.session_state["view"] == "case":
        view_case(model, packages, table, briefing, briefs, client)
    elif st.session_state["view"] == "methods":
        view_methods(model, table, briefing)
    else:
        view_queue(model, packages, table, briefing, briefs)
    scroll_to_top()

    with st.sidebar:
        st.subheader("Demo controls")
        if st.button("Reset recorded decisions", key="reset") and decisions_file().exists():
            decisions_file().unlink()
            st.rerun()
        todo = [c for c, p in packages.items() if p["lane"]["key"] != "clear"]
        if client and st.button(f"Write AI briefs ({len(todo)})", key="prepare",
                                help="The pre-shift batch: one checked model call per case that needs a person. Replaces briefs_cache.json."):
            bar = st.progress(0.0, text="Starting ...")
            meta = ai.prepare_briefs(packages, client, progress=lambda i, n, c, mode: bar.progress(i / n, text=f"{c}: {mode}"))
            st.session_state["flash"] = f"Briefs written: {meta['outcomes']}."
            st.rerun()


if __name__ == "__main__":
    main()
