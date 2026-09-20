"""
triage.py - the statistics behind the Junior AI Investigator.

Everything the app shows as a number, a lane or a finding is computed here with pandas / numpy.
ai.py only turns these facts into text; it never changes them.

The file follows the order I worked through the problem:
  1 data   2 evidence index   3 lanes   4 benign-profile ranges   5 stress tests
  6 LTC domain notes and the first check   7 one case as a dict   8 queue and audit sample   9 diagnostics

There are no fraud labels, so nothing here is a probability of fraud. The output organises the work.
Percentiles, cuts and ranges are learned from this one queue of referred cases. In production they
would be frozen from all claims (by care type) and versioned; assess_case() already scores one case
against a fixed model, which is also how the what-if works.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# ------------------------------------------------------------------ 0. vocabulary
MEASURES = ["weekly_visit_frequency", "member_provider_distance_miles", "prior_claims_last_12mo",
            "weekend_billing_ratio", "amount_vs_peer_avg_pct", "round_dollar_billing_ratio"]
FLAGS = ["duplicate_service_billed", "shared_contact_with_provider", "recent_policy_change_flag",
         "service_overlap_other_provider"]
SIGNALS = MEASURES + FLAGS
ATTRIBUTES = ["case_id", "claim_number", "claim_date", "care_type", "claim_amount_usd", "state"]

# data/signals.json holds the wording for each signal: label, number format, why it matters, the ordinary
# explanation, the step that tests it, the record that settles it, and what a clean / not clean result means.
# "by_setting" replaces wording where the care setting changes the meaning (a facility bills days, not visits).
# It is content for an SIU expert to edit, so it lives outside the code.
SIGNAL_INFO = json.loads((Path(__file__).parent / "data" / "signals.json").read_text(encoding="utf-8"))


def signal_text(signal: str, field: str, setting: str = "unknown") -> str:
    """Wording for one signal; the care setting may replace the default."""
    info = SIGNAL_INFO[signal]
    return info.get("by_setting", {}).get(setting, {}).get(field, info[field])


# Every signal belongs to exactly one question an investigator can test.
QUESTIONS = {
    "care_delivered": dict(
        label="Was care delivered as billed?",
        signals=["weekly_visit_frequency", "prior_claims_last_12mo",
                 "member_provider_distance_miles", "service_overlap_other_provider"],
        request={   # the record to ask for depends on the care setting
            "home": "Request the plan of care and the agency schedule or caregiver daily notes for the billed period.",
            "day_centre": "Request the centre's attendance (sign-in) sheets and opening days for the billed period.",
            "residential": "Request the facility's itemised statement and residency record (admission date, hospital or leave days).",
            "unknown": "Request the plan of care and the provider's visit record for the billed period."},
        effort="medium", action_id="request_care_records"),
    "invoices": dict(
        label="Do the invoices hold up?",
        signals=["duplicate_service_billed", "round_dollar_billing_ratio",
                 "amount_vs_peer_avg_pct", "weekend_billing_ratio"],
        request="Desk-review the itemised invoices and claim-line history already on file.",
        effort="low", action_id="review_invoices"),
    "control": dict(
        label="Who controls the claim and its payments?",
        signals=["shared_contact_with_provider", "recent_policy_change_flag"],
        request="Check the insurer's own records: the matched contact field and the policy change log.",
        effort="low", action_id="verify_relationship"),
}
QUESTION_OF = {s: q for q, spec in QUESTIONS.items() for s in spec["signals"]}

# Care type -> care setting. Domain knowledge, not learned from this file.
CARE_SETTING = {"Home Health Aide": "home", "In-Home Care": "home", "Adult Day Care": "day_centre",
                "Assisted Living": "residential", "Skilled Nursing": "residential"}
SETTING_LABEL = {"home": "home-care", "day_centre": "day-centre", "residential": "facility", "unknown": "these"}

# Thresholds used only for the care-setting notes, never for the lane.
WEEKEND_OVERWEIGHT = 0.35       # clearly above 2/7 = 0.29, what care on all 7 days puts on weekends
DAY_CENTRE_MAX_VISITS = 5       # weekday service
PER_DIEM_MAX_UNITS = 7          # days in a week
LONG_TRIP_MILES = 50            # each way, for in-home care
FREQUENT_VISITS = 7             # daily or more

LANES = {
    "clear": dict(label="Likely false positive", risk="Low", order=2,
                  meaning="Every signal sits inside the range of the benign-profile group and no hard flag is triggered."),
    "judge": dict(label="Needs judgment", risk="Medium", order=1,
                  meaning="Several signals are moderately elevated. A targeted check should settle it."),
    "priority": dict(label="Priority investigation", risk="High", order=0,
                     meaning="Every signal is far outside the benign-profile range and all three questions are open."),
}
LANE_KEYS = ["clear", "judge", "priority"]
LANE_REASON_TEXT = {          # shown when a guard, not the index, decided the lane
    "signal_guard": "The average looks benign, but at least one signal is outside the benign-profile range, so it cannot be closed with the rest.",
    "data_hold": "Every signal is inside the benign-profile range, but the claim number also appears on another case. Held for a person to reconcile.",
    "missing_claim": "Every signal is inside the benign-profile range, but this referral has no claim number, so the check for a shared or re-used number cannot run. Held for a person until the identifier is supplied.",
    "no_structure": "This queue does not split into the three groups the lanes rely on, so every case goes to a person, listed by how many signals are out of range.",
}
ACTIONS = {
    "close_alert": "Close the case: no indicator is supported by the available data.",
    "review_invoices": "Desk-review the itemised invoices and claim-line history.",
    "reconcile_claim_number": "Reconcile the shared claim number before anything else.",
    "supply_claim_number": "Get the claim number for this referral before anything else.",
    "open_investigation": "Open a full investigation with the evidence pack.",
}
# What these ten signals cannot test. Shown on every case so nobody reads silence as assurance.
NOT_TESTED = [
    "whether the insured meets the benefit trigger (help with 2 of 6 activities of daily living, or cognitive impairment)",
    "a hospital stay or death on billed dates",
    "provider licensure, and whether the caregiver is agency staff, independent or a relative",
    "why the upstream engine referred the claim",
]
MINUTES = dict(manual_per_case=20, clear_skim=1.5, clear_audit=10, judge=15, priority=10)   # assumed handling times
MIN_GAP_SHARE = 0.10      # a cut counts as real only if it sits in a gap of at least 10% of the index range
BORDERLINE = 0.90         # a case that keeps its profile group in fewer than 90% of stress-test runs is flagged
AUDIT_CONFIDENCE = 0.90
AUDIT_MAX_HIDDEN_SHARE = 0.10   # a clean random sample must cap hidden problems at 10% of the sampled lot
AUDIT_PURPOSIVE_MAX = 2         # also read the least stable clear cases; these do not enter the random-sample bound

ASSUMPTIONS = [
    "Each CSV row is one referral from an upstream rules + ML engine; flags are referral signals, not verified findings.",
    "The users are the fraud investigation team itself, so the strongest outcome is 'open a full investigation', not a hand-off.",
    "All ten non-attribute columns are treated as signals (the brief mentions nine; the file has ten).",
    "The brief says 0 means normal. That works as a rule for the four flags only: the measures are almost never 0 (four of the "
    "six never are) and one is negative for 16 cases, so normal is estimated (the benign-profile range) and every signal is "
    "read one-sided: higher is more unusual.",
    "Claim amount is exposure. It orders work inside a lane and is deliberately left out of the evidence index.",
    "State is displayed (it sets prompt-pay and fraud-bureau reporting clocks) but never scored and never sent to the language model.",
    "Percentiles, cut points and reference ranges describe this one queue of already-referred cases. Production would freeze them from all claims, by care type.",
    "Skilled Nursing and Assisted Living are treated as residential facilities; Adult Day Care as a weekday day centre.",
    "Care-setting notes (2/7 weekend share, 5 day-centre days, 7 per-diem days, 50-mile trips) are domain assumptions for an SIU expert to confirm. They shape the narrative, never the lane.",
    "The tool moves a review forward. It never pends or decides a payment, and record requests go out as routine verification.",
    "Minutes-per-case figures behind the workload estimate are placeholders for measured handling times.",
]


# ------------------------------------------------------------------ 1. data
TEXT_PATTERNS = {"case_id": r"[A-Za-z0-9_-]{1,20}", "claim_number": r"[A-Za-z0-9_-]{1,30}",
                 "claim_date": r"\d{4}-\d{2}-\d{2}", "care_type": r"[A-Za-z][A-Za-z /&-]{1,39}", "state": r"[A-Za-z]{2}"}


def load_cases(path) -> pd.DataFrame:
    """Read and check the CSV. Missing signal values are rejected, not imputed: missing is not the same as normal."""
    df = pd.read_csv(path, encoding="utf-8-sig")            # the file starts with a BOM
    df.columns = [c.strip() for c in df.columns]
    missing = [c for c in ATTRIBUTES + SIGNALS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing expected columns: {missing}")
    if df["case_id"].duplicated().any():
        raise ValueError("case_id must be unique")
    numeric = SIGNALS + ["claim_amount_usd"]
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="raise")
    incomplete = df.loc[df[numeric].isna().any(axis=1), "case_id"].tolist()
    if incomplete:
        raise ValueError(f"Missing signal values for {incomplete}. Fix the extract; this tool does not impute.")
    if not df[FLAGS].isin([0, 1]).all().all():
        raise ValueError("flag columns must be 0/1")
    for col, pattern in TEXT_PATTERNS.items():               # text columns end up inside a model prompt
        if col == "claim_number":                            # a missing claim number is a caution, not a rejection (see audit_data)
            df[col] = df[col].fillna("").astype(str).str.strip().replace({"nan": "", "None": "", "<NA>": "", "NaN": ""})
            check = df.loc[df[col] != "", col]               # the empty (missing) ones are held later, not rejected here
        else:
            df[col] = df[col].astype(str).str.strip()
            check = df[col]
        bad = check[~check.str.fullmatch(pattern)].tolist()
        if bad:
            raise ValueError(f"Unexpected text in column {col}: {bad[:3]}")
    pd.to_datetime(df["claim_date"], format="%Y-%m-%d", errors="raise")
    return df.reset_index(drop=True)


def default_csv(root=Path(__file__).parent) -> Path:
    """The brief calls the file sample_cases.csv, the e-mail sample_cases_synthetic.csv: accept either, in data/ or here."""
    for name in ("data/sample_cases.csv", "data/sample_cases_synthetic.csv", "sample_cases.csv", "sample_cases_synthetic.csv"):
        if (Path(root) / name).exists():
            return Path(root) / name
    raise FileNotFoundError("Put sample_cases.csv (or sample_cases_synthetic.csv) in the data/ folder.")


def audit_data(df: pd.DataFrame) -> list[dict]:
    """Data-quality findings that change what the tool is allowed to say."""
    numbers = df["claim_number"].astype(str).str.strip()
    missing = df.loc[numbers.isin(["", "nan", "None", "<NA>", "NaN"]), "case_id"].tolist()
    out = [dict(code="completeness", level="ok",
                text=f"{len(df)} cases, {len(SIGNALS)} signals, no missing signal values"
                     + (f"; {len(missing)} with no claim number." if missing else "; every case has a claim number."))]
    for case_id in missing:                                  # a lost identifier is itself a reason for caution
        out.append(dict(code="missing_claim_number", level="warn", cases=[case_id], text=(
            f"{case_id} has no claim number on file. Without it the check for a claim number shared or re-used across "
            "referrals cannot run, so a duplicate would be missed. A missing identifier is treated as a caution, not a "
            "pass: the case is held for a person until the number is supplied.")))
    has_number = df[~numbers.isin(["", "nan", "None", "<NA>", "NaN"])]
    repeated = has_number[has_number.duplicated("claim_number", keep=False)]
    for number, grp in repeated.groupby("claim_number"):
        out.append(dict(code="repeated_claim_number", level="warn", cases=list(grp["case_id"]), text=(
            f"Claim number {number} appears on {' and '.join(grp['case_id'])} with different dates, care types, amounts "
            "and states. In LTC one claim can cover several providers, so this may be one insured who moved (which would "
            "itself explain a policy change and a long distance), or it may be a keying error. Reconcile before treating "
            "the two as independent.")))
    future = df.loc[pd.to_datetime(df["claim_date"]) > pd.Timestamp.today(), "case_id"].tolist()
    if future:
        out.append(dict(code="future_dates", level="warn", cases=future,
                        text=f"Claim dates in the future: {', '.join(future)}. Check the extract."))
    out.append(dict(code="signal_count", level="info", text=(
        "The brief describes 5 attributes and 9 signals; the file has 6 attribute columns (case id and claim number "
        "are separate) and 10 signal columns. All 10 signals are used.")))
    out.append(dict(code="undefined_units", level="info", text=(
        "Ratios come without numerators or billing periods, the peer group behind 'amount vs peer average' is undefined, "
        "and the distance basis (office, place of service, residence) is not stated. Each brief names the record that would settle it.")))
    out.append(dict(code="selected_sample", level="info", text=(
        f"All {len(df)} cases were already referred as suspicious. Anything relative (percentiles, reference ranges, "
        "lanes) describes this queue, not the full claims population.")))
    return out


# ------------------------------------------------------------------ 2. evidence index
def ecdf_mid(values, reference) -> np.ndarray:
    """Mid-rank percentile of each value inside a reference sample, in [0, 1]:
    (number below + half the number equal) / n.  Rank based, so one 247-mile distance cannot dominate;
    unit free, so visits, miles and ratios are comparable; and a new or what-if value can be scored
    against the same reference."""
    ref = np.sort(np.asarray(reference, dtype=float))
    v = np.asarray(values, dtype=float)
    return (np.searchsorted(ref, v, side="left") + np.searchsorted(ref, v, side="right")) / (2.0 * len(ref))


def percentile_table(values: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({s: ecdf_mid(values[s], reference[s]) for s in SIGNALS}, index=values.index)


def evidence_index(pct: pd.DataFrame) -> pd.Series:
    """Plain average of the ten signal percentiles. No outcomes exist to estimate supervised weights.
    The first principal component has similar loadings and is strongly associated with this index, while the
    stress tests below show which individual group assignments change when the weights change."""
    return pct[SIGNALS].mean(axis=1).rename("evidence_index")


# ------------------------------------------------------------------ 3. lanes
def natural_breaks_3(x) -> tuple[np.ndarray, tuple[float, float], tuple[float, float]]:
    """Best split of a 1-D sample into 3 groups (Jenks natural breaks = exact 1-D k-means with k = 3).
    Sort, try every pair of cut positions, keep the pair with the smallest within-group sum of squares.
    Returns the group of each point (0 = lowest), the two cut values (middle of each gap) and each gap
    as a share of the sample range, which the structure check uses."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 3:
        raise ValueError("need at least 3 cases to form 3 groups")
    order = np.argsort(x, kind="stable")
    s = x[order]
    c1 = [0.0] + np.cumsum(s).tolist()              # running sums make each group's sum of squares O(1)
    c2 = [0.0] + np.cumsum(s * s).tolist()

    def sse(i, j):                                  # sum of squares of s[i:j] around its own mean
        d = c1[j] - c1[i]
        return (c2[j] - c2[i]) - d * d / (j - i)

    best = None
    for i in range(1, n - 1):
        left = sse(0, i)
        for j in range(i + 1, n):
            total = left + sse(i, j) + sse(j, n)
            if best is None or total < best[0]:
                best = (total, i, j)
    _, i, j = best
    groups = np.empty(n, dtype=int)
    groups[order[:i]], groups[order[i:j]], groups[order[j:]] = 0, 1, 2
    spread = max(s[-1] - s[0], 1e-12)
    cuts = (float((s[i - 1] + s[i]) / 2), float((s[j - 1] + s[j]) / 2))
    gap_share = (float((s[i] - s[i - 1]) / spread), float((s[j] - s[j - 1]) / spread))
    return groups, cuts, gap_share


def assign_lane(index_value, cuts, n_elevated, hold=None, structure_ok=True) -> tuple[str, str]:
    """Lane from the index, plus a structure check and two guards. Returns (lane, reason).
    structure: 3-means cuts any sample into three, so if the cuts are not in real gaps the lanes are off.
    guard 1:   any signal outside the benign-profile range blocks the 'clear' lane, whatever the average says.
    guard 2:   an open data question about the claim number holds the case for a person. hold is
               'shared' (the number appears on another referral) or 'missing' (no number on file); either holds it."""
    if not structure_ok:
        return "judge", "no_structure"
    lane = "clear" if index_value < cuts[0] else "judge" if index_value < cuts[1] else "priority"
    if lane == "clear" and n_elevated > 0:
        return "judge", "signal_guard"
    if lane == "clear" and hold:
        return "judge", "missing_claim" if hold == "missing" else "data_hold"
    return lane, "index"


# ------------------------------------------------------------------ 4. benign-profile ranges ("lab report" logic)
def reference_ranges(df: pd.DataFrame, groups) -> pd.DataFrame:
    """Observed range of the lowest group for each measure, and where the top group starts.
    Descriptive only: the group was picked by the same signals and nobody verified those cases,
    so the maximum is not a tolerance limit."""
    benign, top = df[groups == 0], df[groups == 2]
    return pd.DataFrame({s: dict(ref_min=float(benign[s].min()), ref_median=float(benign[s].median()),
                                 ref_max=float(benign[s].max()), severe_min=float(top[s].min()))
                         for s in MEASURES}).T


def band_of(signal: str, value: float, ref: pd.DataFrame) -> str:
    """'reference', 'elevated' or 'severe'. A triggered flag counts as elevated."""
    if signal in FLAGS:
        return "elevated" if value >= 1 else "reference"
    if value <= ref.loc[signal, "ref_max"]:
        return "reference"
    return "severe" if value >= ref.loc[signal, "severe_min"] else "elevated"


# ------------------------------------------------------------------ 5. stress tests
def robustness(df, pct, base_groups, n_weights=2000, n_boot=300, alpha=1.0, seed=7) -> pd.DataFrame:
    """How much of the result survives if my choices had been different?
    (a) random weights from Dirichlet(alpha); alpha = 1 means every weighting is equally likely
    (b) the reference queue is bootstrapped   (c) one signal is left out.
    rank_lo / rank_hi = 5th and 95th percentile of the rank under (a).
    stability = the worst of the three shares of runs in which the case keeps its group.
    It measures sensitivity to my choices. It is not a confidence level."""
    rng = np.random.default_rng(seed)
    P = pct[SIGNALS].to_numpy()
    X = df[SIGNALS].to_numpy(dtype=float)
    n, m = P.shape
    rows = np.arange(n)
    landed = np.zeros((n, 3))                    # how often each case landed in each group

    ranks, keep_w = np.empty((n_weights, n)), np.zeros(n)
    for b, w in enumerate(rng.dirichlet(np.full(m, alpha), n_weights)):
        score = P @ w
        ranks[b] = (-score).argsort(kind="stable").argsort(kind="stable") + 1
        g = natural_breaks_3(score)[0]
        keep_w += g == base_groups
        landed[rows, g] += 1

    keep_b = np.zeros(n)
    for _ in range(n_boot):
        ref = X[rng.integers(0, n, n)]
        score = np.column_stack([ecdf_mid(X[:, j], ref[:, j]) for j in range(m)]).mean(axis=1)
        g = natural_breaks_3(score)[0]
        keep_b += g == base_groups
        landed[rows, g] += 1

    keep_l = np.zeros(n)
    for j in range(m):
        g = natural_breaks_3(np.delete(P, j, axis=1).mean(axis=1))[0]
        keep_l += g == base_groups
        landed[rows, g] += 1

    landed[rows, base_groups] = -1               # what is left: where the case goes when it moves
    out = pd.DataFrame({"rank_lo": np.percentile(ranks, 5, axis=0).round().astype(int),
                        "rank_hi": np.percentile(ranks, 95, axis=0).round().astype(int),
                        "stab_weights": keep_w / n_weights, "stab_boot": keep_b / max(n_boot, 1),
                        "stab_loo": keep_l / m, "alt_group": landed.argmax(axis=1)}, index=pct.index)
    out["stability"] = out[["stab_weights", "stab_boot", "stab_loo"]].min(axis=1)
    return out


def gower_distance(pct: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Mean absolute difference over the ten signals (measures as percentiles, flags as 0/1)."""
    Z = pct[SIGNALS].copy()
    Z[FLAGS] = df[FLAGS].to_numpy(dtype=float)
    A = Z.to_numpy()
    return pd.DataFrame(np.abs(A[:, None, :] - A[None, :, :]).mean(axis=2), index=pct.index, columns=pct.index)


# ------------------------------------------------------------------ the fitted model: everything learned from one queue
@dataclass
class TriageModel:
    df: pd.DataFrame                 # indexed by case_id
    pct: pd.DataFrame                # signal percentiles within the queue
    index: pd.Series                 # evidence index
    groups: pd.Series                # statistical group 0 / 1 / 2, before any guard
    cuts: tuple                      # the two cut points on the index
    gap_share: tuple                 # gap under each cut, as a share of the index range
    structure_ok: bool               # do the cuts sit in real gaps?
    ref: pd.DataFrame                # benign-profile range per measure
    setting_ref: dict                # the same, per care setting
    bands: pd.DataFrame              # reference / elevated / severe per signal
    lane: pd.Series
    robust: pd.DataFrame             # rank ranges and stability
    distance: pd.DataFrame           # case-to-case Gower distance
    audit: list = field(default_factory=list)
    holds: dict = field(default_factory=dict)      # case_id -> other cases sharing its claim number
    missing_claim: set = field(default_factory=set)  # case ids with no claim number on file
    fingerprint: str = ""
    n_weights: int = 0
    n_boot: int = 0

    @property
    def case_ids(self) -> list[str]:
        return list(self.df.index)

    @property
    def n_reference(self) -> int:
        return int((self.groups == 0).sum())


def fit_triage(df: pd.DataFrame, n_weights=2000, n_boot=300, alpha=1.0) -> TriageModel:
    """Learn the index, groups, ranges and stability from one queue."""
    audit = audit_data(df)
    d = df.set_index("case_id", drop=False)
    pct = percentile_table(d, d)
    index = evidence_index(pct)
    groups, cuts, gap_share = natural_breaks_3(index.to_numpy())
    structure_ok = min(gap_share) >= MIN_GAP_SHARE
    ref = reference_ranges(d, groups)
    bands = pd.DataFrame({s: [band_of(s, v, ref) for v in d[s]] for s in SIGNALS}, index=d.index)
    n_elevated = (bands != "reference").sum(axis=1)
    holds = {c: [o for o in f["cases"] if o != c]
             for f in audit if f["code"] == "repeated_claim_number" for c in f["cases"]}
    missing_claim = {c for f in audit if f["code"] == "missing_claim_number" for c in f["cases"]}
    hold_of = lambda c: "shared" if c in holds else "missing" if c in missing_claim else None
    lane = pd.Series([assign_lane(index[c], cuts, int(n_elevated[c]), hold_of(c), structure_ok)[0] for c in d.index],
                     index=d.index, name="lane")
    setting = d["care_type"].map(lambda t: CARE_SETTING.get(t, "unknown"))
    setting_ref = {k: {s: (float(g[s].min()), float(g[s].max())) for s in MEASURES}
                   for k, g in d[groups == 0].groupby(setting[groups == 0])}
    fingerprint = hashlib.sha1(pd.util.hash_pandas_object(df[ATTRIBUTES + SIGNALS], index=False)
                               .to_numpy().tobytes()).hexdigest()[:12]
    return TriageModel(df=d, pct=pct, index=index, groups=pd.Series(groups, index=d.index), cuts=cuts,
                       gap_share=gap_share, structure_ok=structure_ok, ref=ref, setting_ref=setting_ref, bands=bands,
                       lane=lane, robust=robustness(d, pct, groups, n_weights, n_boot, alpha),
                       distance=gower_distance(pct, d), audit=audit, holds=holds, missing_claim=missing_claim,
                       fingerprint=fingerprint, n_weights=n_weights, n_boot=n_boot)


# ------------------------------------------------------------------ 6. LTC domain: care-setting notes, three questions
def care_context(row: pd.Series, model: TriageModel) -> list[dict]:
    """How the care setting changes the reading of a signal.
    effect 'clash'   = the value is hard to reconcile with this care setting
    effect 'context' = worth knowing before judging the signal
    A note never removes a signal and never moves a lane: without outcomes, domain context is advice."""
    ref = model.ref
    setting = CARE_SETTING.get(row["care_type"], "unknown")
    visits, miles = row["weekly_visit_frequency"], row["member_provider_distance_miles"]
    weekend, prior = row["weekend_billing_ratio"], row["prior_claims_last_12mo"]
    up = lambda s: row[s] > ref.loc[s, "ref_max"]                       # outside the benign-profile range?
    seen = lambda s: "{:.2f} to {:.2f}".format(                          # what benign-profile cases of this setting show
        *model.setting_ref.get(setting, {}).get(s, (ref.loc[s, "ref_min"], ref.loc[s, "ref_max"])))
    notes = []

    def add(signal, effect, text, short="", also=()):                    # also = a second signal the note is about
        notes.append(dict(signal=signal, effect=effect, text=text, short=short, also=list(also)))

    if setting == "residential":
        if visits > PER_DIEM_MAX_UNITS:
            add("weekly_visit_frequency", "clash",
                f"A residential stay is billed per day, so a weekly visit count is an unusual unit here; benign-profile "
                f"{SETTING_LABEL['residential']} claims in this queue sit at {seen('weekly_visit_frequency')}, and this one "
                f"bills {visits:.0f}. Ask what a billed unit counts and read it against the itemised statement.",
                short=f"{visits:.0f} billed units a week, against a benign residential pattern of {seen('weekly_visit_frequency')}")
        if up("member_provider_distance_miles"):
            add("member_provider_distance_miles", "context",
                f"A resident lives at the facility, so {miles:.0f} miles suggests the address on file is elsewhere. Confirm the admission date.")
        if row["shared_contact_with_provider"] == 1:
            add("shared_contact_with_provider", "context",
                (f"The address on file is {miles:.0f} miles from the facility, so the shared contact is probably a "
                 "phone, e-mail or bank detail rather than the address.") if up("member_provider_distance_miles") else
                "A resident's address on file is often the facility itself, so an address match can be innocent. Confirm which field matched.")
        if up("round_dollar_billing_ratio"):
            add("round_dollar_billing_ratio", "context",
                "Facilities usually charge flat daily or monthly amounts, which are round by nature; yet benign-profile "
                f"facility claims in this queue sit at {seen('round_dollar_billing_ratio')}, so ask what this ratio counts.")
    if setting == "day_centre":
        if visits > DAY_CENTRE_MAX_VISITS:
            add("weekly_visit_frequency", "clash",
                f"Day centres typically open on weekdays; {visits:.0f} attendances a week needs the centre's schedule to explain.",
                short=f"{visits:.0f} day-centre attendances a week")
        if up("weekend_billing_ratio"):
            add("weekend_billing_ratio", "clash",
                f"A weekday day-centre service should bill little on weekends; {weekend:.2f} is unusual. Opening days are usually public.",
                short=f"{weekend:.2f} of a day centre's billing falls on weekends")
    if setting == "home" and miles > LONG_TRIP_MILES and visits >= FREQUENT_VISITS:
        add("member_provider_distance_miles", "clash",
            f"{visits:.0f} visits a week over {miles:.0f} miles each way is hard to deliver in person.",
            short=f"{visits:.0f} home visits a week over {miles:.0f} miles", also=["weekly_visit_frequency"])
    if setting != "day_centre" and up("weekend_billing_ratio"):
        if weekend > WEEKEND_OVERWEIGHT:
            add("weekend_billing_ratio", "clash",
                f"Care on every day of the week would put 0.29 of billing on weekends; {weekend:.2f} means weekends are over-represented.",
                short=f"{weekend:.2f} of billing falls on weekends")
        else:                                                           # my expectation and the file disagree: say so
            add("weekend_billing_ratio", "context",
                f"For reference, care on every day of the week puts 2/7 = 0.29 of billing on weekends; this case shows {weekend:.2f}. "
                f"Benign-profile {SETTING_LABEL[setting]} claims in this queue sit at {seen('weekend_billing_ratio')}, so ask how the ratio is defined.")
    if up("prior_claims_last_12mo") and prior <= 12:
        add("prior_claims_last_12mo", "context",
            f"Ongoing LTC claims are commonly invoiced monthly, so {prior:.0f} claims in 12 months can be routine.")
    return notes


def open_questions(bands: pd.Series, notes: list[dict], setting: str) -> list[dict]:
    """For each of the three questions: its elevated signals and the check that tests them.
    Order of work is a plain sort, not a weighted score: a question with a triggered hard flag first
    (cheapest thing to confirm or kill), then the largest share of elevated signals, then clashes
    with the care setting, then the cheaper check."""
    clashes = {n["signal"] for n in notes if n["effect"] == "clash"}
    out = []
    for key, spec in QUESTIONS.items():
        elevated = [s for s in spec["signals"] if bands[s] != "reference"]
        if not elevated:
            continue
        request = spec["request"] if isinstance(spec["request"], str) else spec["request"][setting]
        out.append(dict(key=key, label=spec["label"], n_signals=len(spec["signals"]), open=elevated,
                        flags=[s for s in elevated if s in FLAGS], clashes=[s for s in elevated if s in clashes],
                        request=request, steps=[signal_text(s, "step", setting) for s in elevated],
                        if_clean=" ".join(signal_text(s, "if_clean", setting) for s in elevated[:2]),
                        if_not=" ".join(signal_text(s, "if_not", setting) for s in elevated[:2]),
                        effort=spec["effort"], action_id=spec["action_id"]))
    effort_rank = {"low": 0, "medium": 1, "high": 2}
    out.sort(key=lambda q: (-min(len(q["flags"]), 1), -len(q["open"]) / q["n_signals"],
                            -len(q["clashes"]), effort_rank[q["effort"]]))
    return out


# ------------------------------------------------------------------ 7. one case as a plain dict (+ what-if, similar cases)
def _fmt(signal: str, value: float) -> str:
    if signal in FLAGS:
        return "triggered" if value >= 1 else "not triggered"
    return SIGNAL_INFO[signal]["fmt"].format(value)


def assess_case(model: TriageModel, case_id: str, overrides: dict | None = None) -> dict:
    """Everything known about one case (JSON-ready). `overrides` replaces signal values for a what-if:
    the reference queue, cuts and ranges stay fixed, so it is a clean counterfactual."""
    if case_id not in model.df.index:
        raise KeyError(f"unknown case_id {case_id}")
    row = model.df.loc[case_id].copy()
    for key, value in (overrides or {}).items():
        if key not in SIGNALS:
            raise KeyError(f"cannot override {key}; not a signal")
        row[key] = float(value)

    pct = {s: float(ecdf_mid([row[s]], model.df[s])[0]) for s in SIGNALS}
    index = float(np.mean([pct[s] for s in SIGNALS]))
    bands = pd.Series({s: band_of(s, float(row[s]), model.ref) for s in SIGNALS})
    elevated = [s for s in SIGNALS if bands[s] != "reference"]
    twins = model.holds.get(case_id, [])
    missing_number = case_id in model.missing_claim
    lane, reason = assign_lane(index, model.cuts, len(elevated),
                               "shared" if twins else "missing" if missing_number else None, model.structure_ok)
    setting = CARE_SETTING.get(row["care_type"], "unknown")
    notes = care_context(row, model)
    plan = open_questions(bands, notes, setting)
    rb = model.robust.loc[case_id]

    signals = []
    for s in SIGNALS:
        item = dict(key=s, label=SIGNAL_INFO[s]["label"], kind="flag" if s in FLAGS else "measure",
                    value=float(row[s]), display=_fmt(s, float(row[s])), band=bands[s],
                    queue_percentile=round(pct[s], 2), question=QUESTION_OF[s], concern=SIGNAL_INFO[s]["concern"],
                    benign=signal_text(s, "benign", setting), records=signal_text(s, "records", setting))
        if s in MEASURES:
            item.update(ref_max=float(model.ref.loc[s, "ref_max"]), ref_display=_fmt(s, float(model.ref.loc[s, "ref_max"])),
                        severe_min=float(model.ref.loc[s, "severe_min"]))
        else:
            item.update(queue_prevalence=round(float(model.df[s].mean()), 2))
        signals.append(item)

    # the recommended first step
    reconcile = [f"Ask the source system whether {case_id} and {', '.join(twins)} are one insured's claim served by "
                 "two providers, a re-used number or a keying error."] if twins else []
    if lane == "clear":
        action = dict(action_id="close_alert", text=ACTIONS["close_alert"], steps=[], effort="low")
    elif lane == "priority":
        action = dict(action_id="open_investigation", text=ACTIONS["open_investigation"],
                      steps=reconcile + [q["request"] for q in plan], effort="high")
    elif reason == "data_hold":
        twin_lanes = ", ".join(f"{t} is in {LANES[model.lane[t]]['label']}" for t in twins)
        action = dict(action_id="reconcile_claim_number", text=ACTIONS["reconcile_claim_number"], effort="low",
                      steps=reconcile + [f"Do not close this case on its own until then: {twin_lanes}."],
                      if_clean="The rows are separate claims; this one has no signal outside the benign-profile range and can be closed.",
                      if_not="The rows belong together; review this case as part of the other referral.")
    elif reason == "missing_claim":
        action = dict(action_id="supply_claim_number", text=ACTIONS["supply_claim_number"], effort="low",
                      steps=["Ask the source system for this referral's claim number, then re-run the shared-number check.",
                             "Do not close this case on its own until the number is on file and checked."],
                      if_clean="With the number supplied and unique, this case has no signal outside the benign-profile range and can be closed.",
                      if_not="If the number is shared with another referral, reconcile the two before deciding either.")
    elif plan:
        lead = plan[0]
        action = dict(action_id=lead["action_id"], text=lead["request"], steps=reconcile + lead["steps"],
                      effort=lead["effort"], question=lead["label"], if_clean=lead["if_clean"], if_not=lead["if_not"])
    else:
        action = dict(action_id="review_invoices", text=ACTIONS["review_invoices"], steps=[], effort="low")

    borderline = bool(rb["stability"] < BORDERLINE) and reason == "index" and not overrides
    alt = LANE_KEYS[int(rb["alt_group"])]
    risk = LANES[lane]["risk"]
    risk_display = ("Low, held for a data question" if reason in ("data_hold", "missing_claim") else
                    f"{risk}, borderline {LANES[alt]['risk']}" if borderline else risk)
    return dict(
        case=dict(case_id=case_id, claim_number=str(row["claim_number"]), claim_date=str(row["claim_date"]),
                  care_type=str(row["care_type"]), care_setting=setting, state=str(row["state"]),
                  claim_amount_usd=float(row["claim_amount_usd"])),
        lane=dict(key=lane, label=LANES[lane]["label"], reason=reason, risk_level=risk, risk_display=risk_display,
                  meaning=LANE_REASON_TEXT.get(reason, LANES[lane]["meaning"]), evidence_index=round(index, 3),
                  rank_range=[int(rb["rank_lo"]), int(rb["rank_hi"])], stability=round(float(rb["stability"]), 2),
                  moves_lane_share=round(1 - float(rb["stability"]), 2), borderline=borderline,
                  alt_lane=LANES[alt]["label"] if borderline else None),
        counts=dict(elevated=len(elevated), severe=int((bands == "severe").sum()),
                    flags=int(sum(row[s] for s in FLAGS)), signals=len(SIGNALS)),
        signals=signals, elevated_signals=elevated, care_notes=notes, plan=plan, recommended_action=action,
        not_tested=NOT_TESTED,
        data_caveats=[f["text"] for f in model.audit
                      if f["code"] in ("repeated_claim_number", "missing_claim_number") and case_id in f["cases"]],
        shares_claim_number_with=twins,
        claim_number_missing=missing_number,
        queue=dict(n_cases=len(model.df), benign_reference_n=model.n_reference),
    )


def what_if(model: TriageModel, case_id: str, explained: list[str]) -> dict:
    """Suppose these signals have an ordinary explanation: measures go to the benign-profile median and the case
    is re-scored against the unchanged queue. A hard flag is a factual finding from the referral, not a value to
    relax, so the what-if never zeroes one; a requested flag is reported as still standing. Because of guard 1 a
    case leaves the judgment lane only when every elevated signal is explained, so the useful output is what would
    still be open - and a case with a hard flag cannot be reasoned clean while the flag stands."""
    measures = [s for s in explained if s not in FLAGS]
    flags_kept = [s for s in explained if s in FLAGS]
    overrides = {s: float(model.ref.loc[s, "ref_median"]) for s in measures}
    before = assess_case(model, case_id)
    after = assess_case(model, case_id, overrides) if measures else before
    return dict(explained=[SIGNAL_INFO[s]["label"] for s in measures],
                flags_kept=[SIGNAL_INFO[s]["label"] for s in flags_kept], relaxed_any=bool(measures),
                lane_before=before["lane"]["label"], lane_after=after["lane"]["label"],
                lane_changed=before["lane"]["key"] != after["lane"]["key"],
                elevated_before=before["counts"]["elevated"], elevated_after=after["counts"]["elevated"],
                still_elevated=[SIGNAL_INFO[s]["label"] for s in after["elevated_signals"]])


def similar_cases(model: TriageModel, case_id: str, k: int = 3) -> list[dict]:
    """Nearest neighbours by signal profile: the precedent an investigator would ask for."""
    nearest = model.distance[case_id].drop(case_id).sort_values().head(k)
    return [dict(case_id=c, distance=round(float(v), 3), lane=LANES[model.lane[c]]["label"], lane_key=model.lane[c],
                 care_type=str(model.df.loc[c, "care_type"]), claim_amount_usd=float(model.df.loc[c, "claim_amount_usd"]),
                 elevated=int((model.bands.loc[c] != "reference").sum())) for c, v in nearest.items()]


# ------------------------------------------------------------------ 8. queue: audit sample, acceptance bound, briefing
def acceptance_bound(n_lot: int, n_read: int, confidence: float = 0.90) -> int:
    """If n_read cases drawn at random from a lot of n_lot all read clean, how many problems could remain?
    Hypergeometric with zero defects found: the largest D for which 'no problem in the sample' still has
    probability above 1 - confidence. It is a statement about the sampling procedure and assumes a reader
    would spot a problem. (Acceptance sampling; AML teams call it below-the-line testing.)"""
    n_read = min(max(n_read, 0), n_lot)
    d = 0
    while d < n_lot - n_read:
        p_zero = math.comb(n_lot - (d + 1), n_read) / math.comb(n_lot, n_read)
        if p_zero <= 1 - confidence + 1e-12:
            break
        d += 1
    return d


def audit_size_for(n_lot: int, max_hidden_share: float, confidence: float = 0.90) -> int:
    """Smallest random sample whose clean result bounds hidden problems at max_hidden_share of the lot."""
    for n in range(n_lot + 1):
        if acceptance_bound(n_lot, n, confidence) <= max_hidden_share * n_lot:
            return n
    return n_lot


def audit_plan(model: TriageModel) -> dict:
    """Which likely-false-positive cases to read in full before the rest can be closed together.
    purposive = the least stable ones (always read, never counted in the bound)
    random    = a seeded simple random sample of the others; only this part supports the bound,
                and only this part gives unbiased labels."""
    clear = list(model.lane[model.lane == "clear"].index)
    shaky = model.robust.loc[clear].sort_values("stability")
    purposive = list(shaky.index[shaky["stability"] < BORDERLINE][:AUDIT_PURPOSIVE_MAX])
    rest = [c for c in clear if c not in purposive]
    n_random = audit_size_for(len(rest), AUDIT_MAX_HIDDEN_SHARE, AUDIT_CONFIDENCE)
    rng = np.random.default_rng(int(model.fingerprint[:8], 16))           # same draw every time for this queue
    random_part = sorted(str(c) for c in rng.choice(rest, size=n_random, replace=False)) if n_random else []
    return dict(purposive=[str(c) for c in purposive], random=random_part, lot=len(rest),
                confidence=AUDIT_CONFIDENCE, max_hidden_share=AUDIT_MAX_HIDDEN_SHARE,
                cases=[str(c) for c in purposive] + random_part)


def queue_table(model: TriageModel) -> pd.DataFrame:
    """One row per case, in the order the work should be done."""
    d = model.df
    t = pd.DataFrame({
        "case_id": d.index, "care_type": d["care_type"], "state": d["state"], "claim_date": d["claim_date"],
        "exposure": d["claim_amount_usd"].astype(float), "lane": model.lane,
        "evidence_index": model.index.round(3), "elevated": (model.bands != "reference").sum(axis=1),
        "flags": d[FLAGS].sum(axis=1).astype(int), "rank_lo": model.robust["rank_lo"],
        "rank_hi": model.robust["rank_hi"], "stability": model.robust["stability"].round(2),
    }).set_index("case_id", drop=False)
    # The index never orders cases inside a lane. Judgment lane: most signals out of range first (a plain count),
    # then the larger claim. The other two lanes: the larger claim first.
    t["lane_order"] = t["lane"].map(lambda k: LANES[k]["order"])
    t["count_key"] = np.where(t["lane"] == "judge", -t["elevated"], 0)
    return (t.sort_values(["lane_order", "count_key", "exposure"], ascending=[True, True, False])
             .drop(columns=["count_key", "lane_order"]))


def morning_briefing(model: TriageModel) -> dict:
    """The numbers behind the first screen."""
    t = queue_table(model)
    lanes = {k: dict(label=LANES[k]["label"], n=int((t["lane"] == k).sum()),
                     exposure=float(t.loc[t["lane"] == k, "exposure"].sum()), cases=list(t.index[t["lane"] == k]))
             for k in LANES}
    plan = audit_plan(model)
    minutes_tool = (lanes["clear"]["n"] * MINUTES["clear_skim"] + len(plan["cases"]) * MINUTES["clear_audit"]
                    + lanes["judge"]["n"] * MINUTES["judge"] + lanes["priority"]["n"] * MINUTES["priority"])
    return dict(n_cases=int(len(t)), lanes=lanes, audit=plan, structure_ok=model.structure_ok,
                minutes_manual=len(t) * MINUTES["manual_per_case"], minutes_tool=round(minutes_tool),
                borderline=list(t.index[(t["stability"] < BORDERLINE) & (t["lane"] != "clear")])
                if model.structure_ok else [],                  # no lanes, so nothing sits between two of them
                total_exposure=float(t["exposure"].sum()),
                repeated_claim_cases=[c for f in model.audit if f["code"] == "repeated_claim_number" for c in f["cases"]],
                missing_claim_cases=sorted(model.missing_claim))


# ------------------------------------------------------------------ 9. diagnostics (Methods view and notebook)
def dominance(model: TriageModel) -> dict:
    """Weight-free ordering: A dominates B if A >= B on every signal and > on at least one."""
    V = model.df[SIGNALS].to_numpy(dtype=float)
    ge = (V[:, None, :] >= V[None, :, :]).all(axis=2)                    # ge[a, b]: a >= b on every signal
    dom = ge & (V[:, None, :] > V[None, :, :]).any(axis=2)
    n, top = len(V), (model.groups == 2).to_numpy()
    return dict(comparable_share=float((dom | dom.T).sum() / (n * (n - 1))),
                top_group_dominates_all_others=bool(ge[np.ix_(top, ~top)].all()),
                ordered_pairs_inside_top_group=int((dom | dom.T)[np.ix_(top, top)].sum() // 2),
                pairs_inside_top_group=int(top.sum() * (top.sum() - 1) // 2))


def diagnostics(model: TriageModel) -> dict:
    """How the signals move together: pooled, inside each group, and against the first principal component.
    (The notebook shows the same analysis step by step.)"""
    d = model.df
    corr = d[SIGNALS].corr(method="spearman").to_numpy()
    pooled = corr[np.triu_indices(len(SIGNALS), 1)]
    within = {}
    for g in range(3):                                                   # measures only: flags barely vary inside a group
        c = d.loc[model.groups == g, MEASURES].corr(method="spearman").to_numpy()
        within[g] = float(np.nanmean(c[np.triu_indices(len(MEASURES), 1)]))
    Z = model.pct[SIGNALS]
    Z = ((Z - Z.mean()) / Z.std(ddof=1).replace(0, np.nan)).dropna(axis=1)    # a constant signal carries no structure
    eigval, eigvec = np.linalg.eigh(np.corrcoef(Z.to_numpy().T))              # eigenvalues come smallest first
    pc1 = pd.Series(Z.to_numpy() @ eigvec[:, -1], index=Z.index)
    return dict(min_corr=float(np.nanmin(pooled)), max_corr=float(np.nanmax(pooled)), within_group_corr=within,
                pc1_share=float(eigval[-1] / eigval.sum()), index_vs_pc1=float(abs(pc1.corr(model.index))),
                amount_vs_index=float(d["claim_amount_usd"].corr(model.index, method="spearman")),
                dominance=dominance(model))


def package_hash(obj) -> str:
    """Short fingerprint of any JSON-ready object; the decision log stores it for the text and facts a person saw."""
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:12]


def build_queue(path) -> tuple[TriageModel, dict[str, dict]]:
    """CSV path -> fitted model and one package per case."""
    model = fit_triage(load_cases(path))
    return model, {c: assess_case(model, c) for c in model.case_ids}


if __name__ == "__main__":                                    # quick look from the command line
    m, _ = build_queue(default_csv())
    print(queue_table(m)[["lane", "elevated", "flags", "exposure", "rank_lo", "rank_hi", "stability"]].to_string())
