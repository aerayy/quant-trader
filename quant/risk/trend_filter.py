from __future__ import annotations

import pandas as pd

from .base import RiskOverlay


class TrendFilterOverlay(RiskOverlay):
    """Mask long positions when price is below long-term moving average.

    Hipotez: trend rejiminde pozisyon al, downtrend'de flat. Klasik long-only
    risk filtresi (Faber 2007 "A Quantitative Approach to Tactical Asset
    Allocation"). Mean-reversion stratejilerine uygulanmamalı (hipotezle
    ters), trend-following'e uygulanmalı.

    Paramlar:
        ma_window: uzun dönem MA pencere uzunluğu (default 200 gün)
        side: hangi tarafa filtre? "long" (default) sadece longları zorlar;
              "both" hem long hem short tarafına uygular (long ↔ above MA,
              short ↔ below MA).
    """

    def transform(self, weights: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
        window = int(self.params.get("ma_window", 200))
        side = str(self.params.get("side", "long")).lower()

        ma = prices.rolling(window, min_periods=max(window // 4, 20)).mean()
        above = prices > ma

        result = weights.copy()
        if side == "long":
            # Zero out longs when below MA; leave shorts/flat alone
            result = result.where(above | (result <= 0), 0.0)
        elif side == "both":
            # Long allowed only above MA; short only below
            long_mask = (result > 0) & above
            short_mask = (result < 0) & ~above
            keep = long_mask | short_mask | (result == 0)
            result = result.where(keep, 0.0)
        else:
            raise ValueError(f"side must be 'long' or 'both', got {side!r}")

        return result
