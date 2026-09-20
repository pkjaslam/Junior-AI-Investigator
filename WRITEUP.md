# Junior AI Investigator

## Product and user

The user is a fraud investigator starting a morning queue of long-term-care referrals. The immediate job is to decide which cases need a full investigation, which need one focused check, and which may be false positives.

The prototype opens with all 50 cases already organized into three triage lanes. Each case shows the signals, a short brief, the first check to run, and what a clean or adverse result would mean. The investigator records agree, disagree, or not sure. An override requires a reason, and every decision is written to a log.

I focused on this review path. I did not build authentication, a database, multi-user workflow, or a claims-system integration. The CSV is the input. Cached briefs allow the demo to run without an API key.

## What the data supports

The file contains 50 synthetic referrals, ten signals, and claim amount. It contains no fraud outcomes. I therefore did not train a classifier and do not report a fraud probability or model accuracy.

Across the full queue, the signals are strongly associated. Pairwise Spearman correlations range from 0.62 to 0.89. The first principal component explains 74.5% of the variance, and the equal-weight percentile index correlates 0.999 with that component. The principal-component loadings are similar across signals.

I use equal weights because there are no outcomes from which to estimate supervised weights. The high correlation with the first component supports this as a simple summary, but it does not make the case order invariant to weights. The app therefore reports sensitivity to alternative weights, bootstrap samples of the queue, and leaving out one signal.

The index has two large gaps, about 15% and 21% of its range. A three-component Gaussian mixture also has the lowest BIC among one to five components. I use three groups only when both cuts fall in gaps of at least 10% of the index range. If that structure is absent, the app switches the lanes off and sends every case to a person.

The three statistical groups contain 30, 12, and 8 cases. Two safety rules then act on the display lanes. Any signal outside the benign-profile range blocks bulk closing. A missing or repeated claim number also holds the case for a person. The final lane counts are 29 likely false positives, 13 needing judgment, and 8 priority investigations. "Likely false positive" is a workflow lane, not an estimated outcome probability.

Claim amount does not enter the index. It is financial exposure and is associated with the index in this sample. It is used only to order work within a lane. For the eight priority cases, a Pareto check shows that each is at least as high as every non-priority case on all ten signals. The signals do not provide a defensible order among those eight, so claim amount is the tie-breaker.

## System design

`triage.py` validates the data, builds the percentile index, finds the groups, applies the two guards, runs the stress tests, and chooses the first check. These steps are deterministic.

`ai.py` creates a case fact sheet and asks the language model for a short headline, summary, ordinary explanation, and missing records for every case before the shift. The committed cache contains 44 first-pass drafts and 6 drafts that passed after one repair. The model does not choose the lane, evidence list, numbers, or next action. It does not receive the claim number, state, or dates.

`checks.py` validates model text before it reaches the screen. It checks numbers, lane and signal counts, unsupported names and history, prohibited claim decisions, evidence-to-signal matches, and the mistake of treating billed service units as residency days. A failed draft gets one repair attempt. If it still fails, the app uses the rule-built brief. Cached briefs are tied to a hash of the case facts and are checked again when loaded.

I use a frozen 19-case regression comparison from an earlier prompt that allowed the model to select indicators. The model omitted a triggered flag or care-setting clash in 11 of 19 cases. Mean coverage of required evidence was 62%, and mean coverage of triggered flags was 39%. The engine-selected list covers both by construction. This is not an independent validation set; it supports the current division of work and guards against reintroducing model-selected evidence.

That comparison does not show that model-written prose is better than rule-built prose. A proper usefulness test would blind investigators to the source, rate clarity and actionability, and compare time to decision and audited misses. The rule brief remains the fallback until such a test shows a benefit.

## Uncertainty and audit control

The sensitivity value is the lowest profile-group retention across 2,000 random signal weightings, 300 bootstrap samples of the queue, and ten leave-one-signal-out runs. Forty-six of the 50 cases retain their statistical group in at least 90% of every stress test. This is a sensitivity measure. It is not the probability that the lane is correct.

The likely-false-positive lane is not closed without review. Two low-stability cases are selected deliberately. Fourteen cases are then sampled at random from the remaining 27. Assuming reviewers would detect a problem in a sampled case, 14 clean random reads give a 90% hypergeometric upper bound of at most 2 hidden problems in that 27-case lot. The two deliberate reads do not enter this bound. Any adverse decision in the lane blocks bulk closing.

The 10% tolerance and 90% confidence level are prototype policy choices. A production team should set them with compliance and operations, then monitor audited misses.

## Human control, scale, and risks

The tool recommends work; it does not decide a claim. It refuses questions about paying or denying a claim. Every finding requires a human response, and every decision records the displayed brief, model, prompt version, and fact fingerprint.

For a larger deployment, the percentile references, cut points, and benign ranges should be estimated from a fixed calibration window, separated by care type, versioned, and monitored for drift. Investigator decisions and random audit reads can provide the first outcome labels. Real claim notes and policy documents would justify retrieval; this prototype has only the structured CSV.

The main risks are false confidence from an unlabeled sample, generated text that sounds more certain than the data, privacy, and automation bias. The current controls are explicit limits, deterministic evidence, text checks, a rule fallback, human decisions, and an audit log. In production, claim data and model calls must remain inside an approved environment.

## Assumptions

- The assignment mentions nine signals, while the CSV contains ten signal columns. I use all ten.
- A zero value means normal for the four binary flags. The six continuous measures use the observed range of the lowest profile group as a descriptive reference, not as a clinical or policy threshold.
- Higher values are treated as more unusual for all six continuous measures.
- Care-setting notes are domain assumptions for an investigator to confirm. They affect the explanation, not the lane.
- All statistics describe this referred queue and should not be generalized to the full claims population.
