"""Read-only market-data vendor adapters for research and backtest evidence."""

from .providers import (
    AlphaVantageProvider,
    FinnhubProvider,
    YahooProvider,
    YahooUsdGbpRateProvider,
    build_market_data_provider,
    build_price_history_provider,
)
from .types import (
    FinnhubQuote,
    FxRate,
    FxRateSource,
    MarketDataTransport,
    StdlibHttpTransport,
    VendorProvider,
)

__all__ = [
    "AlphaVantageProvider",
    "FinnhubProvider",
    "FinnhubQuote",
    "FxRate",
    "FxRateSource",
    "MarketDataTransport",
    "StdlibHttpTransport",
    "VendorProvider",
    "YahooProvider",
    "YahooUsdGbpRateProvider",
    "build_market_data_provider",
    "build_price_history_provider",
]
