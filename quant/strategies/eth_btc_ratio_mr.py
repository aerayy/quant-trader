from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Strategy, StrategyConfig


class EthBtcRatioMeanReversionStrategy(Strategy):
    """ETH/BTC oranı için z-score-tabanlı mean-reversion (pair trade).

    Hipotez: BTC ve ETH iki yüksek-korelasyonlu (0.85+) blue-chip varlık.
    Mutlak fiyatları trend olabilir ama göreli fiyatları (ETH/BTC oranı)
    uzun-vadeli ortalamaya çekilir. Sentiment/rotasyon ile uçlara giden
    oran kısa vadede mean-reverting.

    Mekanizma: göreli pricing arbitrajı. Tek varlık trend'lerinden bağımsız
    bir alfa kaynağı — bu yüzden directional stratejilerle (momentum)
    rejim-ortogonal olabilir.

    Akademik: Gatev, Goetzmann, Rouwenhorst (2006) "Pairs Trading"; Avellaneda
    & Lee (2010) "Statistical Arbitrage in the US Equities Market". Crypto
    pair trading literatürü daha az ama BTC/ETH için relative-value rationale
    sağlam.

    Sinyal (hysteresis):
        z = (ratio - rolling_mean) / rolling_std
        z < -entry  → long long_symbol  / short short_symbol
        z > +entry  → short long_symbol / long short_symbol
        |z| < exit  → flat (her iki bacak 0)
        else        → mevcut pozisyonu koru (ffill)

    Pozisyon boyutu: aktif iken her bacak ±0.5 ağırlık → toplam |w|=1.0,
    backtest engine'inin normalizasyonu (|w_sum|>1 ise scale-down) tetiklenmez.

    Paramlar:
        z_window: rolling z-score penceresi, gün (default 60)
        entry_threshold: |z| bunun üzerine çıkınca aç (default 2.0)
        exit_threshold: |z| bunun altına inince kapat (default 0.5)
        long_symbol: oran payına denk gelen sembol (default ETHUSDT)
        short_symbol: oran paydasına denk gelen sembol (default BTCUSDT)
    """

    def __init__(self, config: StrategyConfig):
        super().__init__(config)

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        z_window = int(self.params.get("z_window", 60))
        entry = float(self.params.get("entry_threshold", 2.0))
        exit_thr = float(self.params.get("exit_threshold", 0.5))
        long_sym = str(self.params.get("long_symbol", "ETHUSDT"))
        short_sym = str(self.params.get("short_symbol", "BTCUSDT"))

        if entry <= exit_thr:
            raise ValueError(
                f"entry_threshold ({entry}) must be > exit_threshold ({exit_thr})"
            )
        if long_sym not in prices.columns or short_sym not in prices.columns:
            raise ValueError(
                f"Need both {long_sym} and {short_sym} in price columns; got {list(prices.columns)}"
            )

        ratio = prices[long_sym] / prices[short_sym]
        min_p = max(z_window // 2, 10)
        roll_mean = ratio.rolling(z_window, min_periods=min_p).mean()
        roll_std = ratio.rolling(z_window, min_periods=min_p).std()
        z = (ratio - roll_mean) / roll_std.replace(0.0, np.nan)

        signal = pd.Series(np.nan, index=prices.index)
        signal[z < -entry] = 1.0
        signal[z > entry] = -1.0
        signal[z.abs() < exit_thr] = 0.0
        state = signal.ffill().fillna(0.0)

        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        weights[long_sym] = state * 0.5
        weights[short_sym] = -state * 0.5

        return weights
