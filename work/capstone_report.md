# Capstone Report — Lane 2: Refresh / Content Opportunity Scoring

- **Author:** Fawad Wazir
- **Lane:** Lane 2 — Refresh / Content Opportunity Scoring
- **Repo:** https://github.com/fawadwazir/flyrank-Internship
- **Date:** September 2026

## 0. Abstract

Which of a client's existing pages should an editor refresh first, out of thousands that
could plausibly qualify? This paper builds a page-level scoring model on FlyRank's content
performance dataset and evaluates it against a transparent three-gate baseline rule (stale +
strikable + visible) using a held-out set of clients the model never trained on. On that
honest split, a logistic regression scorer roughly doubles the baseline's precision at the
top of the queue (Precision@50: 0.34 → 0.74) while a more complex Random Forest does not
improve on the simple model — and the Random Forest's apparent 0.92 Precision@50 disappears
to 0.56 the moment the split stops leaking client identity, which is itself the paper's
central finding. The output is a ranked, reason-coded refresh queue meant to sit in front of
a human editor, not to replace one.

## 1. Problem framing

**Decision supported:** which pages a content team should refresh next, this sprint, given
limited editor time.

**Unit of analysis:** one page (`content_id`) belonging to one client (`client_id`), scored
on its most recent 90-day performance window.

**Output:** a ranked queue with three tiers (`refresh_priority`, `refresh_backlog`,
`no_action`) plus a short reason code per page (e.g. `stale+strikable_position+visible`).

**Action a human takes:** an editor works down the `refresh_priority` tier first — updating,
re-optimizing, or consolidating the page — then the backlog tier as time allows.

**Cost of a wrong call:** a false positive wastes an editor's time refreshing a page that
wasn't actually declining. A false negative lets a real opportunity keep losing traffic
unnoticed. Neither is catastrophic on its own, which is why this is framed as a
*prioritization aid*, not an automated action — a ranked list, reviewed by a person, is the
right shape for this cost profile.

**Why ML helps at all:** a content team can eyeball a handful of pages, but not thousands
per client across dozens of clients. A rule can encode one or two obvious signals (stale +
losing rank); a model can weigh many weak, correlated signals (visibility consistency,
content age, word count, intent, position) at once and rank the whole set consistently.

## 2. Data safety

**Data used:** `data/raw/content_refresh_anonymized.csv` — the 30,000-row, 32-client starter
slice provided with the internship (52 columns: identifiers, content metadata, 90-day
performance metrics, and pre-computed trend/tier fields). All identifiers
(`client_id`, `content_id`) are already anonymized hashes.

**Note on scope:** the internship's data contract (`work/notebooks/w03_data_contract.ipynb`)
scoped a label definition against the full 79M-row `FlyRank/internship-warehouse` on
Hugging Face. This capstone iteration was built and evaluated on the 30,000-row starter
slice — the same one the baseline (Section 3) was built and validated on — so baseline and
model are compared on identical data. Scaling the same pipeline to the full warehouse is
listed as future work (Section 8).

**Columns deliberately excluded from features:**
- `trend_direction`, `trend_pct` — these ARE the label (or derived directly from it); using
  them as features would leak the answer into the input.
- `impressions_last_30d` / `impressions_prev_30d` and the equivalent `clicks_*` / `sessions_*`
  30-day pairs — these are the raw components `trend_direction` is computed from. Excluded
  for the same reason, even though they aren't the label column itself.
- `client_id`, `content_id` — used only for grouping (train/test split, queue identity),
  never as model features.

**Confirmed:** no client name, URL, or query string appears anywhere in `work/` or in this
report — only anonymized hash IDs and aggregate metrics.

## 3. Baseline

The baseline (built in `work/notebooks/w04_baseline_score.ipynb`) is a transparent three-gate
rule, not a model:

```
score = impressions_90d   if  stale AND strikable AND visible   else 0
  stale      = freshness_tier == "91-180"
  strikable  = position_tier in {page_1, striking}
  visible    = impression_tier in {moderate, good, excellent}
```

It flags a page only when all three conditions hold, then ranks flagged pages by raw
impression volume. It is a fair comparison because it uses only public-safe, pre-existing
tier columns (no fitting, no leakage) and produces the same kind of ranked output the model
does.

**Baseline on the honest test split** (Section 5 — same 7,115-page, 8-client holdout used to
score the model): flags 5.3% of pages, **ROC AUC 0.503, Precision@20 0.20, Precision@50
0.34**. An AUC of 0.503 is barely better than random — the gate gets the *flagging* roughly
right (see feature overlap below) but does a poor job of *ranking* within the flagged set,
since it only sorts by raw impressions.

## 4. Model / analysis

**Target:** `is_declining_label = 1` if `trend_direction == "down"`, else `0` — a proxy for
"this page's traffic is trending down," not a guarantee of a future outcome.

**Method:** Logistic Regression and Random Forest, both compared on identical features and
splits. Logistic regression fits this lane well because the tiered, largely-monotonic signals
here (freshness, position, visibility) are close to linearly separable once scaled, and a
linear model is easy to hand an editor as "why" — the coefficients in Section 6 are the whole
explanation. Random Forest was included as the standard non-linear comparison.

**Features used (18 numeric + 8 categorical, 26 total):**
`search_volume, competition, cpc, word_count, char_count, log_impressions_90d,
log_clicks_90d, log_sessions_90d, log_ai_sessions_90d, days_with_impressions,
days_with_sessions, content_age_days, days_since_last_update, ctr, avg_position,
engagement_rate, scroll_rate, ai_traffic_pct` (numeric, standardized) and
`competition_level, content_type, main_intent, age_tier, freshness_tier, word_count_tier,
impression_tier, position_tier` (categorical, one-hot encoded).

**Deliberately left out:** the label-derived and leakage-adjacent columns listed in Section 2,
and both ID columns (used for grouping only).

## 5. Evaluation

**Split:** grouped by `client_id` (`GroupShuffleSplit`, 75/25, seed=42) — **24 training
clients (22,885 pages) / 8 held-out test clients (7,115 pages)**. A page-level random split
was deliberately rejected as the primary evaluation: pages from the same client share a
CMS, an industry, and an SEO history, so a random split lets the model partly memorize
per-client patterns rather than learn signals that generalize to a *new* client. Base rates:
55.0% (train) vs 51.7% (test) — close enough that the split isn't skewed.

**Results — model vs. baseline, same client-holdout test set, same metrics:**

| Method | ROC AUC | Precision@20 | Precision@50 |
|---|---|---|---|
| Baseline rule (3-gate) | 0.503 | 0.20 | 0.34 |
| Random Forest | 0.603 | 0.50 | 0.56 |
| **Logistic Regression** | **0.611** | **0.80** | **0.74** |

Logistic Regression wins on every metric and roughly doubles the baseline's Precision@50.
Random Forest, despite being the more flexible model, does *not* beat the simpler linear
model here — with only 24 training clients, the extra capacity has little real signal left
to fit once client identity is held out, and likely picks up noise instead.

**The split-honesty check** (`random_split_check` in `final_results.json`): the same Random
Forest, using the *same features*, evaluated on a random 75/25 row split instead of a
client-holdout split, scores **ROC AUC 0.753, Precision@50 0.92** — numbers that look like a
clear win. That gap (0.603 → 0.753 AUC; 0.56 → 0.92 P@50) is not a better model; it's the
model partly recognizing clients it already saw during training. This is the paper's central
methodological finding: **evaluating this kind of data on a random split overstates
performance by a wide margin**, and any number quoted without stating the split type should
be treated with suspicion.

**Error analysis (Logistic Regression, top 50 of the test queue):** 37/50 (74%) are true
declining pages, matching Precision@50 exactly. Two clients account for 35 of the 50 flagged
pages (22 + 13) — the same per-client concentration weakness the baseline had. Of the 13
false positives, most share `freshness_tier="0-30"` (recently updated) with a `page_1` or
`striking` position and a low-to-moderate impression tier — pages that look "at risk" on
position and freshness alone but haven't actually started declining yet.

## 6. Interpretation

**What the model found, in plain words** — top standardized Logistic Regression weights:

| Feature | Weight | Reading |
|---|---|---|
| `log_impressions_90d` | +1.51 | Higher-traffic pages are *more* likely to be flagged as declining — likely because they have more room, and more data, to show a real drop |
| `impression_tier = good` | +0.82 | Being in the "good" visibility tier (vs. low/excellent) raises risk |
| `position_tier = striking` | −0.82 | Pages already in the striking-distance band (positions 4–10) are *less* likely to be flagged — there's more room to climb than fall |
| `freshness_tier = 91–180` | +0.63 | Confirms the baseline's core intuition: staleness in the 3–6 month range is a real risk signal |
| `log_clicks_90d` | −0.57 | More clicks (holding impressions constant, i.e. a better CTR) is protective |

Random Forest's top features (`days_with_impressions`, `log_impressions_90d`,
`avg_position`, `content_age_days`) tell a consistent story: **visibility consistency**
matters as much as raw traffic — a page that shows up in search results every day is a
different risk profile than one with the same total impressions bunched into a few spikes.

**Surprise / negative result:** the more complex model did not win. That's a legitimate
finding for this dataset size (24 training clients) and this feature set, not a failure of
execution — it's reported as-is rather than tuned until Random Forest wins.

## 7. Recommendation

Ranked, in order an editor should act on them:

1. **Adopt the Logistic Regression queue as the refresh worklist, reviewed by a human before
   any page is touched.** It roughly doubles the baseline's hit rate at the top of the queue
   on a fair, client-holdout evaluation.
2. **Cap how many pages any single client contributes to the top of the queue.** Both the
   baseline and the model concentrate top picks in 2–4 clients; without a per-client quota,
   editors serving many clients will keep re-refreshing the same few accounts.
3. **Treat `refresh_priority` (top decile) as this sprint's list, `refresh_backlog` (next 30%)
   as next sprint's, and `no_action` as skip** — don't try to action the full ranked file at
   once.
4. **Re-validate quarterly.** Search behavior and competitive tiers shift; a model trained on
   one quarter's tier definitions should be refit, not assumed stable indefinitely.
5. **Do not present Precision@50 numbers without stating the split.** Given the gap in
   Section 5, any future comparison on this dataset must state client-holdout vs. random
   split explicitly, or the numbers aren't comparable.

**Confidence:** directional, not causal. This model predicts a labeled proxy
(`trend_direction`) associated with likely-declining pages; it does not establish that
refreshing a flagged page *causes* a traffic recovery — that would need a pre/post or
holdout experiment, which is future work, not this paper.

## 8. Reproducibility

**Environment:** Python 3.12, `scikit-learn==1.8.0`, `pandas==3.0.2`, `numpy==2.4.4` (see
`requirements.txt`). Random seed `42` used everywhere a split or a stochastic model is fit.

**To re-run from a fresh clone:**
```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute work/notebooks/w04_baseline_score.ipynb
jupyter nbconvert --to notebook --execute work/notebooks/w05_model.ipynb
jupyter nbconvert --to notebook --execute work/notebooks/w06_validation_audit.ipynb
jupyter nbconvert --to notebook --execute work/notebooks/w07_action_playbook.ipynb
```
Each notebook writes its own metrics file (`work/artifacts/*.json`) and chart SVGs
(`work/artifacts/charts/*.svg`), which this paper embeds directly — the numbers in this
report are a checkable re-run, not a one-off.

**Future work — scaling to the full warehouse:** the data contract
(`work/notebooks/w03_data_contract.ipynb`) already scoped a label definition against the full
79M-row `FlyRank/internship-warehouse`. The next iteration should re-run this exact pipeline
(same features, same grouped-split logic, same two models) against a full monthly panel
pulled via DuckDB/HF, to confirm the client-holdout result holds at scale and isn't an
artifact of the 32-client starter sample.

## 9. Acknowledgments & data credit

Built on the [FlyRank ML Internship dataset](https://flyrank.ai).
