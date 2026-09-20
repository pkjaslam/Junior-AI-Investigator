"""
checks.py - what stands between the model and the screen.

Every model-written brief and chat answer goes through these checks before an investigator sees it.
A text that fails is repaired once or thrown away (ai.py), never shown. Each check exists because of a
failure I expected, or one I saw when a real model wrote the briefs.

  check_numbers           every number must be on the fact sheet
  check_wording           no accusation, no claim decision, nothing about people or history
  check_names             a capitalised word the facts do not contain is probably an invented name
  check_lanes_and_counts  only this case's lane; "5 of 10 signals" must match the engine
  check_findings          a finding is about an elevated signal and quotes only that signal's numbers
  check_shape             empty fields, prompt echo, a headline that only repeats the lane

What they cannot catch: a right number used wrongly, or an invented fact that avoids every pattern.
"""
from __future__ import annotations

import json
import re

import triage as T

BRIEF_KEYS = ["headline", "summary", "key_indicators", "innocent_reading", "missing_information", "stability_note"]

IDENTIFIER = re.compile(r"\b[A-Za-z]{1,5}-?\d{3,}\b|\b\d{4}-\d{2}-\d{2}\b")          # C1023, LTC-2034786, dates
NUMBER = re.compile(r"(?<![\w.])[-+]?\$?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?[k%]?|(?<![\w.])[-+]?\.\d+%?")
PERCENT_AFTER = re.compile(r"(?:st|nd|rd|th)?\s*(?:percent|%)", re.I)
NUMBER_WORD = re.compile(r"\b(?:eleven|twelve|\w+teen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|"
                         r"hundred|thousand|million|double|triple|twice|thrice|half|third|quarter)\b", re.I)
SMALL_NUMBER = {word: value for value, word in enumerate(
    ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"))}
SMALL_NUMBER_CONTEXT = re.compile(
    r"\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten)\b"
    r"(?=\s+(?:of\b|signals?\b|measures?\b|flags?\b|questions?\b|days?\b|weeks?\b|months?\b|years?\b|"
    r"visits?\b|miles?\b|claims?\b|percent\b))", re.I)
WORDING = re.compile(          # accusations and claim decisions: never, not even negated
    r"\bfraud\w*|\bscam\w*|\bscheme\w*|\bcollu\w+|\bkickback\w*|\bphantom\b|\bfake\w*|\bfabricat\w+|\bguilty\b|"
    r"\bcriminal\w*|\bprosecut\w+|law enforcement|\bden(?:y|ies|ied|ial)\b|\bwithh[oe]ld\w*|\bsuspend\w*|\brescind\w*|"
    r"stop payment|do not pay|pay (?:the|this) claim|\b(?:low|medium|high)[ -]risk\b", re.I)
STATUS_CLAIM = re.compile(     # facts about people and history that ten signals can never contain
    r"\bsanction\w*|\blicen[cs]\w*|\bconvict\w*|\bindict\w*|\blawsuit\w*|\bsubstantiated\b|\bproven\b|"
    r"\b(?:was|were|been|already) (?:confirmed|verified|investigated|found)\b|\bprior (?:investigation|referral)s?\b|"
    r"\bhistory of\b|\bdaughter\b|\bson\b|\bspouse\b|\bhospitali[sz]\w+|\bdeceased\b|\bdied\b", re.I)
UNSUPPORTED_DURATION = re.compile(
    r"\b(?:member|patient|claimant|provider|caregiver)\b[^.]{0,30}\b(?:stayed|resided|lived|worked)\b"
    r"[^.]{0,12}\b(?:\d+|zero|one|two|three|four|five|six|seven|eight|nine|ten)\s+"
    r"(?:days?|weeks?|months?|years?)\b", re.I)
PROMPT_ECHO = re.compile(r"CASE_FACTS|TOOL_RESULTS|\bJSON\b|fact sheet|concrete fact|the prompt|lane\.why")
BROKEN_TEXT = re.compile(r"\ufffd|[\x00-\x08\x0b\x0c\x0e-\x1f]")
RESIDENCY_FRAME = [        # billed units are not residency days: a resident can be billed several services per day
    re.compile(r"\bexceed\w*\b[^.]{0,35}\b\d+[- ]?days?\b", re.I),                 # "exceeds 7-day / 7 days"
    re.compile(r"\bmore\b[^.]{0,35}\bthan\b[^.]{0,25}\bdays?\b(?![^.]{0,15}\brecord)", re.I),  # "more units billed than days available"
    re.compile(r"\b\d+[- ]?day (?:limit|residency|maximum|cap)\b", re.I),          # "7-day limit", "7-day residency"
    re.compile(r"\bonly\b[^.]{0,25}\b\d+ days?\b[^.]{0,25}\b(?:possible|available|residency)\b", re.I),
    re.compile(r"\bdays?\b[^.]{0,15}\b(?:are|is)\b[^.]{0,10}\b(?:possible|available)\b", re.I),  # "7 days of residency are possible"
    re.compile(r"\binconsistent with residency\b", re.I),
]
CARE_CONTEXT_FRAME = [
    # 0.13 is an observed upper bound in this queue, not a general daily-care expectation.
    re.compile(r"\b0\.13\b[^.]{0,45}\b(?:expected|benchmark|pattern)\b[^.]{0,35}\bdaily care\b", re.I),
    re.compile(r"\bdaily care\b[^.]{0,35}\b(?:expected|benchmark|pattern)\b[^.]{0,45}\b0\.13\b", re.I),
    # 2/7 = 0.29 is a calendar benchmark for care on all seven days, not an empirical benign profile.
    re.compile(r"\b(?:0\.29|2/7)\b[^.]{0,45}\bbenign (?:pattern|profile|range|expectation)\b", re.I),
    re.compile(r"\bbenign (?:pattern|profile|range|expectation)\b[^.]{0,45}\b(?:0\.29|2/7)\b", re.I),
    re.compile(r"\b(?:0\.29|2/7)\b[^.]{0,45}\bexpected\b[^.]{0,30}\bdaily care\b", re.I),
    # The data do not establish how commonly home-health-aide providers use flat rates.
    re.compile(r"\bhome health aide\b[^.]{0,60}\b(?:often|usually|typically)\b[^.]{0,35}\bflat\b", re.I),
    re.compile(r"\bround-dollar (?:charges|billing)\b[^.]{0,35}\b(?:often|usually|typically)\b[^.]{0,35}\bflat\b", re.I),
]
DAYS = {d + s for d in ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday") for s in ("", "s")}
NOT_NAMES = {"LTC", "SIU", "AI", "I", "EVV", "ADL", "ADLs"} | DAYS          # capitalised words that are not invented names
MIN_LENGTH = {"headline": 15, "summary": 60, "innocent_reading": 30, "missing_information": 20}


def numbers_in(text: str) -> list[tuple[str, float, bool]]:
    """(token, value, is_percent) for every number in the text. Identifiers and dates are not quantities;
    a spelled-out number gets the value NaN because it cannot be checked."""
    text = IDENTIFIER.sub(" ", text)
    out = [(w, float("nan"), False) for w in NUMBER_WORD.findall(text)]
    out += [(m.group(0), float(SMALL_NUMBER[m.group(0).lower()]), False)
            for m in SMALL_NUMBER_CONTEXT.finditer(text)]
    for m in NUMBER.finditer(text):
        token = m.group(0)
        raw = token.lstrip("+-").lstrip("$").replace(",", "")
        value = float(raw.rstrip("k%")) * (1000.0 if raw.endswith("k") else 1.0)
        out.append((token, value, token.endswith("%") or bool(PERCENT_AFTER.match(text, m.end()))))
    return out


def allowed_numbers(*sources) -> set[float]:
    text = " ".join(s if isinstance(s, str) else json.dumps(s, default=str) for s in sources)
    return {abs(v) for _, v, _ in numbers_in(text) if v == v}


def is_grounded(token: str, value: float, is_pct: bool, allowed: set[float], counting_ok: bool = True) -> bool:
    if value != value:                                            # NaN: a spelled-out number
        return False
    for a in allowed:
        if abs(value - a) <= 0.005 * max(1.0, a):                 # the number itself, up to display rounding
            return True
        if is_pct and 0 < a <= 1 and abs(value - 100 * a) <= 0.5:     # 0.97 written as 97%
            return True
        if token.endswith("k") and a >= 1000 and abs(value - a) <= 0.02 * a:     # "$32k" for 32,373
            return True
    return False


def _strings(obj) -> list[str]:
    """Every string inside a nested dict / list, in order."""
    if isinstance(obj, dict):
        return [t for v in obj.values() for t in _strings(v)]
    if isinstance(obj, list):
        return [t for v in obj for t in _strings(v)]
    return [str(obj)]


def check_numbers(text: str, facts: dict) -> list[str]:
    """Every number must be on the fact sheet."""
    allowed = allowed_numbers(facts)
    return [f"number '{t}' is not in the case facts" for t, v, p in numbers_in(text) if not is_grounded(t, v, p, allowed)]


def check_wording(text: str, facts_text: str) -> list[str]:
    """No accusation, no claim decision, nothing about people or history the data cannot contain."""
    issues = []
    hit = WORDING.search(text)
    if hit:
        issues.append(f"wording the tool must never use: '{hit.group(0)}'")
    hit = STATUS_CLAIM.search(text)
    if hit and hit.group(0).lower() not in facts_text:
        issues.append(f"a claim the data cannot support: '{hit.group(0)}'")
    hit = UNSUPPORTED_DURATION.search(text)
    if hit:
        issues.append(f"a duration the data cannot support: '{hit.group(0)}'")
    return issues


def check_names(fragments: list[str], facts_text: str) -> list[str]:
    """A capitalised word inside a sentence that the fact sheet does not contain is probably an invented name."""
    known = set(re.findall(r"[a-z][\w'-]*", facts_text))
    issues = []
    for fragment in fragments:
        for sentence in re.split(r"(?<=[.;:!?])\s+", IDENTIFIER.sub(" ", fragment)):
            for word in re.findall(r"[A-Za-z][\w'-]*", sentence)[1:]:              # skip the first word of the sentence
                if word[0].isupper() and word not in NOT_NAMES and word.lower() not in known:
                    issues.append(f"name or term '{word}' is not in the case facts")
    return issues


def check_lanes_and_counts(text: str, pkg: dict, facts_text: str) -> list[str]:
    """Only this case's lane may be named, and '5 of 10 signals' must match the engine's counts."""
    issues = []
    own = pkg["lane"]["label"].lower()
    alt = (pkg["lane"]["alt_lane"] or "").lower()
    lower = text.lower()
    for lane in T.LANES.values():
        label = lane["label"].lower()
        if label not in lower or label == own:
            continue
        if label == alt:
            context = r"(?:stress|sensitivity|borderline|change|move|land)"
            qualified = (re.search(rf"{context}[^.]{{0,80}}{re.escape(label)}", lower)
                         or re.search(rf"{re.escape(label)}[^.]{{0,80}}{context}", lower))
            if qualified:
                continue
        if label != own:
            issues.append(f"names a lane this case is not in: '{lane['label']}'")
    c = pkg["counts"]
    legal = {"signal": ({c["elevated"], c["signals"] - c["elevated"], c["signals"]}, c["signals"]),
             "measure": ({c["severe"], c["elevated"] - c["flags"], len(T.MEASURES)}, len(T.MEASURES)),
             "flag": ({c["flags"], len(T.FLAGS)}, len(T.FLAGS)),
             "question": ({len(pkg["plan"]), len(T.QUESTIONS)}, len(T.QUESTIONS))}
    count_token = r"\d+|zero|one|two|three|four|five|six|seven|eight|nine|ten"
    pattern = (rf"\b({count_token}) of (?:the )?({count_token}) (?:referral |hard |open )?"
               r"(signal|measure|flag|question)")
    to_int = lambda value: int(value) if value.isdigit() else SMALL_NUMBER[value.lower()]
    for n, denominator, noun in re.findall(pattern, text, re.I):
        allowed_counts, expected_total = legal[noun.lower()]
        if to_int(n) not in allowed_counts or to_int(denominator) != expected_total:
            issues.append(f"count '{n} of {denominator} {noun}s' does not match the engine")
    return issues


def check_residency_frame(text: str) -> list[str]:
    """Reject text that treats a billed count as a number of residency days. A resident is there every day and can
    be billed several services per day, so 'N visits exceed 7 days' proves nothing; compare against the benign pattern."""
    return (["frames a billed count as a number of residency days; compare it against the benign residential pattern instead"]
            if any(p.search(text) for p in RESIDENCY_FRAME) else [])


def check_care_context_frame(text: str) -> list[str]:
    """Reject correct values given a stronger meaning than the supplied data and assumptions support."""
    return (["misstates a care-setting benchmark or queue reference range"]
            if any(p.search(text) for p in CARE_CONTEXT_FRAME) else [])


def check_text(fragments: list[str], facts: dict, pkg: dict | None = None) -> list[str]:
    """The checks shared by briefs and chat answers."""
    text, facts_text = " ".join(fragments), " ".join(_strings(facts)).lower()
    issues = check_numbers(text, facts) + check_wording(text, facts_text) + check_names(fragments, facts_text)
    issues += check_residency_frame(text) + check_care_context_frame(text)
    return issues + (check_lanes_and_counts(text, pkg, facts_text) if pkg else [])


def check_findings(brief: dict, pkg: dict) -> list[str]:
    """A finding must be about an elevated signal, state that signal's own value, and quote no other signal's numbers."""
    issues, signals = [], {s["key"]: s for s in pkg["signals"]}
    if len(brief["key_indicators"]) > len(T.SIGNALS):        # the engine owns this list; guard only against a runaway
        issues.append("too many key indicators")
    for item in brief["key_indicators"]:
        key = item.get("signal") if isinstance(item, dict) else None
        finding = str(item.get("finding", "")) if isinstance(item, dict) else ""
        if key not in pkg["elevated_signals"]:
            issues.append(f"indicator '{key}' is not an elevated signal for this case")
            continue
        if not finding.strip():
            issues.append(f"indicator '{key}' has no finding")
        s = signals[key]
        own = allowed_numbers(s["display"], s.get("ref_display", ""), str(s["queue_percentile"]) if s["kind"] == "measure" else "",
                              str(s.get("queue_prevalence", "")), s["concern"], s["benign"],
                              *[n["text"] for n in pkg["care_notes"] if key in (n["signal"], *n["also"])])
        quoted = numbers_in(finding)
        for token, value, is_pct in quoted:
            if not is_grounded(token, value, is_pct, own, counting_ok=False):
                issues.append(f"finding for '{key}' quotes '{token}', which is not a number of that signal")
        shown = allowed_numbers(s["display"])                          # the value as displayed, e.g. "9 visits/wk"
        if s["kind"] == "measure" and not any(is_grounded(t, v, p, shown, counting_ok=False) for t, v, p in quoted):
            issues.append(f"finding for '{key}' must state the signal's own value ({s['display']})")
    return issues


def check_shape(brief: dict, pkg: dict, text: str) -> list[str]:
    """Problems that are not about facts: empty fields, prompt echo, a headline that only repeats the lane."""
    issues = [f"'{f}' is empty or too short to be useful" for f, least in MIN_LENGTH.items() if len(str(brief[f]).strip()) < least]
    if PROMPT_ECHO.search(text):
        issues.append("the text talks about its own instructions instead of the case")
    if BROKEN_TEXT.search(text):
        issues.append("the text contains a broken or unsupported character")
    words = lambda t: set(re.findall(r"[a-z]{4,}", str(t).lower()))
    if len(words(brief["headline"]) & words(pkg["lane"]["meaning"])) >= 5:
        issues.append("the headline repeats the lane description; name this case's most unusual concrete fact instead")
    if re.search(r"claim numbers?\s+C\d{3,}", text):
        issues.append("C-numbers are case ids, not claim numbers")
    if len(str(brief["headline"])) > 140 or len(str(brief["summary"])) > 900:
        issues.append("headline or summary too long for a scannable brief")
    return issues


def validate_brief(brief: dict, pkg: dict, facts: dict) -> dict:
    """Run every check on a brief against the fact sheet the model was given. Any issue discards it."""
    missing = [k for k in BRIEF_KEYS if k not in brief]
    if missing or not isinstance(brief.get("key_indicators"), list):
        return dict(passed=False, issues=[f"missing or malformed fields: {missing or ['key_indicators']}"],
                    numbers_checked=0, signals_checked=0)
    fragments = _strings({k: brief[k] for k in BRIEF_KEYS})
    text = " ".join(fragments)
    issues = check_text(fragments, facts, pkg) + check_findings(brief, pkg) + check_shape(brief, pkg, text)
    return dict(passed=not issues, issues=issues, numbers_checked=len(numbers_in(text)),
                signals_checked=len(brief["key_indicators"]))


def validate_answer(answer: str, facts: dict, pkg: dict | None = None) -> dict:
    """The same text checks as a brief, lane and count claims included. The investigator's own question is
    NOT a source of facts. Passing pkg turns on check_lanes_and_counts (the case's own lane and the engine's
    counts); a chat answer that names a lane this case is not in, or an unsupported 'k of n', is rejected."""
    issues = check_text([answer], facts, pkg)
    return dict(passed=not issues, issues=issues, numbers_checked=len(numbers_in(answer)))
