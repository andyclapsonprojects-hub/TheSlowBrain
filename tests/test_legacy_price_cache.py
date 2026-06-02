from __future__ import annotations

import json
from pathlib import Path

from slowbrain.legacy_price_cache import LegacyN8nPriceCacheProvider


def test_legacy_n8n_price_cache_provider_parses_daily_ohlcv(tmp_path: Path) -> None:
    cache_dir = tmp_path / "reports" / "price-cache" / "daily"
    cache_dir.mkdir(parents=True)
    (cache_dir / "AAPL.json").write_text(
        json.dumps(
            {
                "fetched_at": "2026-06-01T00:00:00Z",
                "prices": {
                    "2026-05-30": {
                        "1. open": "100",
                        "2. high": "103",
                        "3. low": "99",
                        "4. close": "102",
                        "5. volume": "123456",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    provider = LegacyN8nPriceCacheProvider.from_legacy_project_root(tmp_path)

    prices = provider.daily_prices("aapl")

    assert len(prices) == 1
    assert prices[0].symbol == "AAPL"
    assert prices[0].close == 102.0
    assert prices[0].volume == 123456.0
    assert prices[0].source == "legacy_n8n_price_cache"


def test_legacy_n8n_price_cache_provider_returns_empty_for_missing_or_bad_files(tmp_path: Path) -> None:
    provider = LegacyN8nPriceCacheProvider(tmp_path)

    assert provider.daily_prices("MISSING") == ()

    (tmp_path / "BAD.json").write_text("{ bad json", encoding="utf-8")
    assert provider.daily_prices("BAD") == ()

    (tmp_path / "EMPTY.json").write_text(json.dumps({"prices": {}}), encoding="utf-8")
    assert provider.daily_prices("EMPTY") == ()


def test_legacy_n8n_price_cache_provider_skips_malformed_rows_and_sanitizes_symbol(tmp_path: Path) -> None:
    (tmp_path / "EVIL.json").write_text(
        json.dumps(
            {
                "prices": {
                    "2026-05-30": {
                        "1. open": True,
                        "2. high": "bad",
                        "3. low": "99",
                        "4. close": "102",
                        "5. volume": "123456",
                    },
                    "2026-05-31": {
                        "1. open": "100",
                        "2. high": "103",
                        "3. low": "99",
                        "4. close": "102",
                        "5. volume": "123456",
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    provider = LegacyN8nPriceCacheProvider(tmp_path)

    prices = provider.daily_prices("E/VIL")

    assert len(prices) == 1
    assert prices[0].date == "2026-05-31"
