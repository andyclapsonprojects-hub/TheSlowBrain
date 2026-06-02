# TheSlowBrain

![License: MIT](https://img.shields.io/badge/License-MIT-22c55e.svg)
![Python 3.13+](https://img.shields.io/badge/python-3.13%2B-3776AB.svg)
![Typed: mypy strict](https://img.shields.io/badge/typed-mypy%20strict-1d4ed8.svg)
![Lint: ruff](https://img.shields.io/badge/lint-ruff-7c3aed.svg)

**A cautious stock-trading research "brain" that studies the market and only ever _suggests_ — it never trades your money by default.**

TheSlowBrain turns raw market evidence into honest BUY / SELL / HOLD / WATCHLIST / AVOID
decisions. It is deliberately *slow*: every improvement must survive data-quality checks,
cost and risk checks, out-of-sample backtests, an evaluation council, and calibration against
human-verified examples before it is trusted. Weak, missing, or low-confidence evidence is
labelled as such instead of being dressed up as proof.

> 📊 **Scale.** Developed and back-tested against a historical universe of **~297,000 research
> ideas** and **~1.18 million forward-return observations** across **3,809 tickers**, spanning
> **2019–2026** at 1/5/10/20-day horizons. The learning gate trains on this evidence in shadow
> mode; 30 human-verified labels are held strictly out. (That source dataset is private and not
> redistributed — see [the demo](#quickstart) for a runnable synthetic stand-in.)

> ⚠️ **Research only — not financial advice.** This project does not place live orders by
> default and makes no promise of profit. See [Safety](#safety) and the [LICENSE](LICENSE).

---

## What it does

- **Imports raw evidence** (research ideas joined to historical forward returns) and validates
  its provenance with a checksum manifest.
- **Builds decision features** — price/volume/trend/momentum context plus optional read-only
  market-data from Alpha Vantage → Finnhub → Yahoo.
- **Scores each idea** with an active, versioned rubric and emits a **5-label vocabulary**.
- **Tests candidate improvements** with purged/embargoed cross-validation, a deflated Sharpe
  ratio, probability-of-backtest-overfitting, and multiple-testing correction — adopting a new
  rubric only when guarded, after-cost evidence genuinely beats the current one.
- **Reviews quality** with an evaluation council (deterministic dimensions + an optional OpenAI
  judge) and calibrates against a held-out set of human-verified labels.
- **Learns slowly in shadow mode**: a tiny from-scratch neural gate trains on historical
  outcomes and reports its metrics, but stays behind a hard baseline fallback — it cannot take
  over trading.
- **Reports** a concise brief and append-only track record. Broker execution stays gated.

### The five labels

| Label | Meaning |
|---|---|
| **BUY** | Strong score that passes every quality, risk, and data gate. |
| **SELL** | Exit / negative signal for a held name. |
| **HOLD** | No action — neutral or insufficient edge. |
| **WATCHLIST** | A near-miss just below the buy threshold; worth watching, not buying. |
| **AVOID** | A high score actively blocked by a failed quality / risk / data gate. |

---

## Quickstart

Requirements: **Python ≥ 3.13** and [**uv**](https://docs.astral.sh/uv/).

```bash
# 1. Install dependencies (creates a local .venv from the locked versions)
uv sync

# 2. (optional) configure integrations — everything works with an empty .env
cp .env.example .env

# 3a. Prove it works end-to-end with zero private data (synthetic dataset):
uv run python scripts/run_demo_cycle.py

# 3b. ...or run the quality gates:
uv run pytest
uv run ruff check
uv run mypy
```

`run_demo_cycle.py` builds a small synthetic dataset, runs one full research cycle offline, and
prints the resulting brief plus the path to the generated report — no API keys or external data
required. To run on your **own** dataset instead, point `SLOWBRAIN_LEGACY_STOCK_PROJECT` at it
and use `scripts/run_first_slowbrain_cycle.py`.

---

## Documentation

- 📘 **[Live visual explainer →](https://andyclapsonprojects-hub.github.io/TheSlowBrain/)** — a
  single, self-contained page with **Beginner / Intermediate / Expert** views of what the project
  is and how to use it. (Source: [docs/index.html](docs/index.html) — also opens locally in any browser.)
- 📗 **[docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md)** — the architecture, the pipeline, the
  evaluation layers, the safety model, and how to extend the system, in prose.

---

## Dependencies

The core is intentionally **standard-library only** — there is no numpy, pandas, or PyTorch.
The learning gate is a ~150-line scalar autograd engine written from scratch.

| Scope | Packages |
|---|---|
| Runtime | `openai` (the only third-party runtime dependency; used for the optional LLM judge) |
| Dev / tooling | `ruff`, `mypy` (strict), `pytest`, `pytest-cov` (coverage gate ≥ 93%) |

Exact, reproducible versions are pinned in [`pyproject.toml`](pyproject.toml) and `uv.lock`.

---

## Project layout

```
src/slowbrain/      Core library (features, rubrics, backtest, eval council,
                    learning gate, market data, CIO/risk, reporting, broker boundary)
scripts/            Operator entrypoints (run a cycle, shadow runs, briefs, broker preview)
tests/              Unit + integration test suites
docs/               index.html explainer and HOW_IT_WORKS.md
```

---

## Safety

- **Live trading is blocked by default.** A manual broker path exists, but a real order requires
  *all* of: a fresh ready preview, a process-scoped enable flag, an `--execute` flag, a matching
  approval token, valid broker health, and duplicate-ledger clearance.
- **Secrets stay in the environment.** `.env` is gitignored and never committed; reports never
  persist tokens or credentials.
- **Honesty over optimism.** Proxy, missing, low-sample, or stale evidence is explicitly labelled
  — never treated as production-grade proof.
- The optional OpenAI judge is research/evaluation only and degrades gracefully when unavailable.

---

## License

[MIT](LICENSE) © 2026 Andy Clapson.
