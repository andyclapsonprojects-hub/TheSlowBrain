# How TheSlowBrain Works

This document explains the system as it stands: what each stage does, how the pieces fit
together, the safety model, and how to extend it. For a visual, level-by-level tour see
[index.html](index.html) (open it locally in a browser).

---

## The idea in one paragraph

TheSlowBrain treats a trading decision the way a careful analyst would: gather evidence, score
it against an explicit rubric, *try* to disprove any apparent edge with out-of-sample tests and
statistical guards, have it reviewed, and only then act — and even then, only on paper/shadow by
default. The goal is not speed; it is **not fooling yourself**. Weak or missing evidence is
labelled honestly rather than rounded up to a confident answer.

---

## The pipeline, end to end

Every run flows through the same chain (orchestrated by `slowbrain.workflow.run_first_cycle`).
Any new feature should slot into this chain rather than bypass the guards.

1. **Import raw evidence only** — `data_import` copies raw data and ledgers and records a
   checksum manifest (file, size, SHA-256, source path). Previously generated reports/verdicts
   are deliberately *not* imported, so the system never treats stale output as fresh truth.

2. **Load features and outcomes** — `features` joins research ideas to their historical forward
   returns across horizons of 1, 5, 10, and 20 days. Bounded runs read the latest slice for
   speed; evidence/training paths read the full eligible set. The reference universe used during
   development spans **~297,000 research ideas** and **~1.18 million forward-return observations**
   across **3,809 tickers** from **2019 to 2026** — large enough to test rubrics and train the
   learning gate out-of-sample, with the 30 human-verified labels held strictly out. (That source
   dataset is private; the public repo ships a synthetic stand-in via `scripts/run_demo_cycle.py`.)

3. **Attach market and technical context** — `technical_context` derives trend, momentum, gaps,
   candle geometry, and volume confirmation. `market_data` / `market_data_vendors` optionally
   add read-only benchmark and liquidity evidence via an Alpha Vantage → Finnhub → Yahoo
   fallback chain, cached per symbol by `market_data_cache`.

4. **Quarantine bad evidence** — `data_quality` flags malformed rows and implausible outcomes
   (for example, sub-$1 names showing phantom triple-digit returns). Error-severity rows are
   excluded from BUY decisions, backtest evidence, and model training.

5. **Score with the active rubric** — `rubrics.decide_feature` applies the current versioned
   rubric and emits one of five labels (below). The active rubric is loaded from persisted state
   when present, falling back to the seed rubric otherwise.

6. **Generate candidate improvements** — `grader_council` can propose rubric variations. They
   are only ever *candidates*; nothing is adopted without surviving the next stage.

7. **Backtest with statistical guards** — `backtest` evaluates candidates using validation
   selection plus a *separate* confirmation holdout, purged/embargoed splits (to prevent
   leakage), walk-forward windows, after-cost returns (`costs`), drawdown, a **deflated Sharpe
   ratio** with p-value, **probability of backtest overfitting** (CSCV-style), skew/kurtosis,
   and multiple-testing correction. `optimizer.select_rubric` adopts a candidate only when the
   corrected, after-cost improvement is real.

8. **Review and calibrate** — `eval_council` scores quality dimensions (profit evidence, risk
   control, data quality, execution safety, report honesty, overfitting robustness, economic
   rationale), with an optional cached OpenAI judge that degrades to "unknown" when unavailable.
   Results are calibrated against a held-out set of **30 human-verified labels**
   (`human_anchor`), reported with an honest low-confidence, non-binding caveat at this sample
   size.

9. **Run the learning gate in shadow** — `gating_model` trains a small logistic gate (built on
   the from-scratch autograd engine in `microgix`) over the five labels using historical
   forward-return outcomes. It reports accuracy, Brier score, expected calibration error, anchor
   agreement, a drift-guard result, and its weights — then **stays behind a hard baseline
   fallback**. The 30 human labels are never in a training fold.

10. **Apply portfolio and broker safety** — `cio` enforces cash reserve, position sizing, sector
    caps, and concentration limits, producing **blocked** order intents. `trading212` /
    `live_execution` provide a manual, heavily gated broker path that stays disabled by default.

11. **Write reports and append evidence** — `reporting` emits the structured report and the
    concise brief; `decision_capture` writes the latest decisions plus an append-only outcome
    stream; `learning_state` persists the active rubric and gate state; and a daily track record
    is appended so live performance can accrue over time.

---

## The five labels

| Label | Meaning |
|---|---|
| **BUY** | Strong score that passes every quality, risk, and data gate. |
| **SELL** | Exit / negative signal for a held name. |
| **HOLD** | No action — neutral or insufficient edge. |
| **WATCHLIST** | A near-miss just below the buy threshold; watch, don't buy. |
| **AVOID** | A high score *actively blocked* by a failed quality / risk / data gate. |

The distinction between WATCHLIST and AVOID matters: WATCHLIST is "almost good enough", while
AVOID is "looks good but something is wrong" — an active negative screen, not a shrug.

---

## The learning loop (and why it's slow)

State persists between runs: the adopted rubric and the gate's weights are saved and reloaded,
and each run appends to an outcome stream and a daily track record, so evidence accumulates. But
accumulation is not authority. The learned gate can only ever be *promoted* out of shadow by
explicit, earned criteria — beating the baseline on accuracy **and** calibration **and**
held-out anchor agreement, out of sample. The cron/daily runner exists to gather evidence, not
to grant the model control. Until those bars are cleared, decisions follow the baseline rubric.

---

## Safety model

- **Live execution is off by default.** A real broker order requires, together: a fresh ready
  preview, a process-scoped enable flag, an explicit `--execute`, a matching approval token,
  valid broker health, and clearance against a duplicate-prevention ledger. Submissions are
  reconciled against positions/orders/history afterwards.
- **Secrets never enter the repo.** Configuration is read from the environment (`.env` is
  gitignored); reports are scrubbed of credentials.
- **No false proof.** Proxy, missing, low-sample, or stale evidence is labelled as such.
- **Leakage control is built in.** Purged/embargoed splits and a separate confirmation holdout
  guard against optimistic in-sample results; the human anchor is strictly held out.

---

## Running it

```bash
uv sync                                       # install locked dependencies
uv run python scripts/run_demo_cycle.py       # one full cycle on synthetic data, offline
uv run pytest                                 # unit + integration suites
uv run ruff check && uv run mypy              # lint + strict type-check
```

Other operator entrypoints in `scripts/` include repeated shadow runs, the Telegram brief
(dry-run by default), the daily runner, and the read-only broker health / preview tools. To run
on your own dataset, set `SLOWBRAIN_LEGACY_STOCK_PROJECT` and use
`scripts/run_first_slowbrain_cycle.py`.

---

## Extending it safely

- Add new evidence or features upstream of scoring, and let them flow through the existing
  guards — do not add a path that reaches a decision without backtest + eval + safety checks.
- Keep modules single-responsibility and under ~500 lines (enforced by an integration test).
- Anything that touches money, persisted state, or branching decision logic needs tests at both
  the unit and integration layers before it is trusted.
- The learned gate may gain influence only through the documented earn-promotion criteria —
  never by default and never because a schedule ran.
