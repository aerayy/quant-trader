from __future__ import annotations

import pandas as pd

from .base import Strategy


class MomentumStrategy(Strategy):
    """Absolute time-series momentum.

    Hipotez: bir varlığın son N-gün getirisi pozitifse, eğilim devam eder.
    Long pozisyon al; aksi halde flat. Her N-gün'de bir rebalance.

    Akademik referans: Moskowitz, Ooi & Pedersen (2012) "Time Series Momentum".
    Cross-sectional momentum'dan (Jegadeesh-Titman 1993) farkı: tek varlık
    bazlı, asset evrenine bağlı değil. Az sembolle de anlamlı.

    Paramlar:
        lookback_days: getiri penceresi (default 60)
        rebalance_days: kararı kaç günde bir yenile (default 5)
    """

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        lookback = int(self.params.get("lookback_days", 60))
        rebalance = int(self.params.get("rebalance_days", 5))

        returns = prices.pct_change(lookback)
        raw_weights = (returns > 0).astype(float)

        if rebalance > 1:
            sampled = raw_weights.iloc[::rebalance]
            weights = sampled.reindex(prices.index).ffill()
        else:
            weights = raw_weights

        return weights.fillna(0.0)
