"""HTML rendering for human-labeling packs."""

from __future__ import annotations

from ..technical_context import RecentBar, TechnicalContext
from .models import HumanLabelingCase, HumanLabelingPack
from .utils import _e, _fmt


def _html(pack: HumanLabelingPack) -> str:
    cards = "\n".join(_case_card(case) for case in pack.cases)
    allowed_labels = _e(", ".join(pack.label_values_allowed))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>TheSlowBrain Human Label Pack</title>
  <style>
    :root {{
      --ink: #15151f;
      --muted: #5a6175;
      --bg: #fff8e7;
      --cyan: #00c2ff;
      --pink: #ff4fa3;
      --lime: #a6ff4d;
      --yellow: #ffd23f;
      --green: #17b26a;
      --red: #f04438;
      --panel: #ffffff;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Inter, Segoe UI, Arial, sans-serif;
      color: var(--ink);
      background:
        linear-gradient(135deg, rgba(0,194,255,.18), transparent 32%),
        linear-gradient(225deg, rgba(255,79,163,.18), transparent 32%),
        var(--bg);
    }}
    header {{
      padding: 28px clamp(18px, 4vw, 54px);
      background: linear-gradient(90deg, var(--cyan), var(--lime), var(--yellow), var(--pink));
      border-bottom: 4px solid #15151f;
    }}
    h1 {{ margin: 0 0 8px; font-size: clamp(2rem, 5vw, 4.3rem); line-height: .95; }}
    h2 {{ margin: 0 0 14px; font-size: 1.35rem; }}
    p {{ line-height: 1.55; }}
    main {{ padding: 24px clamp(14px, 3vw, 42px) 48px; }}
    .summary {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
      gap: 12px;
      margin-bottom: 18px;
    }}
    .pill {{
      background: var(--panel);
      border: 2px solid #15151f;
      border-radius: 8px;
      padding: 12px;
      box-shadow: 4px 4px 0 #15151f;
      overflow-wrap: anywhere;
    }}
    .case {{
      background: var(--panel);
      border: 2px solid #15151f;
      border-radius: 8px;
      margin: 18px 0;
      box-shadow: 5px 5px 0 #15151f;
      overflow: hidden;
    }}
    .case-head {{
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      padding: 14px 16px;
      background: #15151f;
      color: white;
    }}
    .ticker {{ font-size: 1.55rem; font-weight: 850; letter-spacing: 0; }}
    .body {{
      display: grid;
      grid-template-columns: minmax(260px, 1.1fr) minmax(280px, 1fr);
      gap: 16px;
      padding: 16px;
    }}
    @media (max-width: 820px) {{ .body {{ grid-template-columns: 1fr; }} }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
      gap: 8px;
    }}
    .metric {{
      border: 1px solid #d9dce8;
      border-radius: 6px;
      padding: 9px;
      min-width: 0;
    }}
    .label {{ color: var(--muted); font-size: .78rem; font-weight: 750; text-transform: uppercase; }}
    .value {{ font-weight: 800; overflow-wrap: anywhere; }}
    .chart {{
      width: 100%;
      min-height: 190px;
      border: 1px solid #d9dce8;
      border-radius: 8px;
      background: #f7fbff;
      padding: 8px;
    }}
    .notes {{
      margin-top: 12px;
      padding: 12px;
      border-left: 6px solid var(--pink);
      background: #fff1f7;
      border-radius: 6px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 10px;
      font-size: .9rem;
    }}
    th, td {{ border-bottom: 1px solid #e5e7f0; padding: 7px; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    .todo {{
      background: #15151f;
      color: white;
      border-radius: 8px;
      padding: 16px;
      margin-bottom: 18px;
    }}
    code {{ background: rgba(255,255,255,.2); padding: 2px 5px; border-radius: 4px; }}
  </style>
</head>
<body>
  <header>
    <h1>TheSlowBrain Human Label Pack</h1>
    <p>
      Use this instead of the raw JSONL. It gives decision-time price, volume,
      candle, trend, Slow Brain, and outcome context for each Andy label.
    </p>
  </header>
  <main>
    <section class="summary">
      <div class="pill"><div class="label">Generated</div><div class="value">{_e(pack.generated_at)}</div></div>
      <div class="pill"><div class="label">Cases</div><div class="value">{pack.case_count}</div></div>
      <div class="pill"><div class="label">Allowed Labels</div><div class="value">{allowed_labels}</div></div>
      <div class="pill"><div class="label">Human Labels?</div><div class="value">Not yet. You fill them in.</div></div>
    </section>
    <section class="todo">
      <h2>Your job</h2>
      <p>
        For each case, decide whether a world-class trader would label it
        <code>BUY</code>, <code>SELL</code>, <code>HOLD</code>, or <code>UNKNOWN</code>,
        then write a short rationale in the CSV file. Do not treat the 10-day
        return as information that would have been known at trade time.
      </p>
    </section>
    {cards}
  </main>
</body>
</html>
"""


def _case_card(case: HumanLabelingCase) -> str:
    context = case.technical_context
    head = (
        f'<div class="ticker">{_e(case.ticker)} '
        f'<span style="font-size:.9rem;font-weight:600;">{_e(case.signal_date)}</span></div>'
    )
    label_status = (
        f"SlowBrain: <strong>{_e(case.slowbrain_action)}</strong> | "
        f"Human label: <strong>{_e(case.human_label or 'blank')}</strong>"
    )
    source_line = (
        f"<strong>Source:</strong> {_e(case.source)} | "
        f"Price source: {_e(context.price_source)} | Status: {_e(context.status)}"
    )
    fill_line = (
        "Fill in CSV columns <strong>human_label</strong> and "
        f"<strong>human_rationale</strong> for example <code>{_e(case.example_id)}</code>."
    )
    return f"""<article class="case">
  <div class="case-head">
    {head}
    <div>{label_status}</div>
  </div>
  <div class="body">
    <div>
      <h2>Decision Evidence</h2>
      <div class="metrics">
        {_metric('Close', context.close)}
        {_metric('Open / High / Low', _ohl(context))}
        {_metric('Volume', context.volume)}
        {_metric('Volume vs 20d', context.volume_ratio_20d)}
        {_metric('Day change %', context.day_change_pct)}
        {_metric('Gap %', context.gap_pct)}
        {_metric('Trend', context.trend)}
        {_metric('Pattern', ', '.join(context.pattern_names) or 'none')}
        {_metric('Outcome 10d net %', case.outcome_10d_net_return_pct)}
        {_metric('SlowBrain score', case.slowbrain_score)}
      </div>
      <div class="notes">
        <strong>Candles:</strong> {_e(context.pattern_summary)}<br>
        <strong>SlowBrain reason:</strong> {_e(case.slowbrain_reason)}<br>
        {source_line}
      </div>
      {_recent_table(context.recent_bars)}
    </div>
    <div>
      <h2>Last Available Candles</h2>
      <div class="chart">{_candle_svg(context.recent_bars)}</div>
      <div class="notes">
        {fill_line}
      </div>
    </div>
  </div>
</article>"""


def _metric(label: str, value: object) -> str:
    return f'<div class="metric"><div class="label">{_e(label)}</div><div class="value">{_e(_fmt(value))}</div></div>'


def _ohl(context: TechnicalContext) -> str:
    return f"{_fmt(context.open)} / {_fmt(context.high)} / {_fmt(context.low)}"


def _recent_table(bars: tuple[RecentBar, ...]) -> str:
    if not bars:
        return "<p>No recent OHLCV bars were available.</p>"
    rows = "\n".join(
        f"<tr><td>{_e(bar.date)}</td><td>{_fmt(bar.open)}</td><td>{_fmt(bar.high)}</td>"
        f"<td>{_fmt(bar.low)}</td><td>{_fmt(bar.close)}</td><td>{_fmt(bar.volume)}</td></tr>"
        for bar in bars[-5:]
    )
    return f"""<table>
  <thead><tr><th>Date</th><th>Open</th><th>High</th><th>Low</th><th>Close</th><th>Volume</th></tr></thead>
  <tbody>{rows}</tbody>
</table>"""


def _candle_svg(bars: tuple[RecentBar, ...]) -> str:
    if not bars:
        return "<p>No chart available.</p>"
    width = 640
    height = 220
    pad = 18
    lows = [bar.low for bar in bars]
    highs = [bar.high for bar in bars]
    low = min(lows)
    high = max(highs)
    span = max(high - low, 0.0001)
    step = (width - pad * 2) / max(len(bars), 1)
    candle_width = max(5.0, step * 0.48)

    def y(value: float) -> float:
        return pad + (high - value) / span * (height - pad * 2)

    pieces = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="OHLC candlestick chart">',
        f'<line x1="{pad}" y1="{y(high):.2f}" x2="{width - pad}" y2="{y(high):.2f}" stroke="#d9dce8"/>',
        f'<line x1="{pad}" y1="{y(low):.2f}" x2="{width - pad}" y2="{y(low):.2f}" stroke="#d9dce8"/>',
    ]
    for index, bar in enumerate(bars):
        center = pad + step * index + step / 2
        color = "#17b26a" if bar.close >= bar.open else "#f04438"
        y_open = y(bar.open)
        y_close = y(bar.close)
        top = min(y_open, y_close)
        body_height = max(abs(y_close - y_open), 2.0)
        pieces.append(
            f'<line x1="{center:.2f}" y1="{y(bar.high):.2f}" x2="{center:.2f}" '
            f'y2="{y(bar.low):.2f}" stroke="{color}" stroke-width="2"/>'
        )
        pieces.append(
            f'<rect x="{center - candle_width / 2:.2f}" y="{top:.2f}" width="{candle_width:.2f}" '
            f'height="{body_height:.2f}" fill="{color}" rx="1"/>'
        )
    pieces.append("</svg>")
    return "".join(pieces)


