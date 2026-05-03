from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Strategy, StrategyConfig


class FundingMeanReversionStrategy(Strategy):
    """Long when 8h perpetual funding rate goes sufficiently negative
    (shorts crowded → premium accrues to longs); flat when it normalizes.

    Hipotez: aşırı negatif funding = short tarafı kalabalık. Long pozisyon
    iki edge'i birleştirir: (a) reversion premium (pozisyonlanma uçtan
    döndüğünde fiyat hareketi), (b) carry (live perp'te funding payment).

    NOT: bu backtest spot fiyatları kullanır — funding payment carry'si
    P&L'e DAHIL EDİLMEZ. Live perp implementasyonu ek alfa görür; backtest
    sayıları konservatif tarafta.

    Akademik: Burnside et al. (2011) carry trade; Avellaneda (2020) crypto
    basis; broader: Koijen, Moskowitz, Pedersen, Vrugt (2018) "Carry".

    Sinyal: rolling z-score üzerinden tetikleme. Mutlak eşik (örn. < -%0.03)
    çok seyrek tetikleniyor — kripto funding %95+ zaman pozitif bias'lı.
    z-score = (mevcut - son N gün ortalaması) / son N gün std. Funding
    normal rejimine göre uç değer aldığında trigger.

    Paramlar:
        z_long_threshold: funding z-score'u bu değerin ALTINDA → long aç.
            Default -1.5 (1.5σ aşağı sapma).
        z_flat_threshold: z-score bu değerin ÜSTÜNE çıktığında → flat ol.
            Default 0.0 (mean reversion tamamlandı).
        z_window: rolling z-score için pencere (gün). Default 30.
        smoothing_periods: günlük funding'in son N-gün ortalaması (anti-noise).
            Default 3.
    """

    def __init__(self, config: StrategyConfig, funding_data: dict[str, pd.Series] | None = None):
        super().__init__(config)
        self.funding_data = funding_data or {}

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        z_long = float(self.params.get("z_long_threshold", -1.5))
        z_flat = float(self.params.get("z_flat_threshold", 0.0))
        z_window = int(self.params.get("z_window", 30))
        smoothing = int(self.params.get("smoothing_periods", 3))

        if z_long >= z_flat:
            raise ValueError(
                f"z_long_threshold ({z_long}) must be < z_flat_threshold ({z_flat})"
            )

        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        for sym in prices.columns:
            funding = self.funding_data.get(sym)
            if funding is None or funding.empty:
                continue

            daily = funding.resample("1D").mean()
            smooth = daily.rolling(smoothing, min_periods=1).mean()

            rolling_mean = smooth.rolling(z_window, min_periods=max(z_window // 2, 5)).mean()
            rolling_std = smooth.rolling(z_window, min_periods=max(z_window // 2, 5)).std()
            z = (smooth - rolling_mean) / rolling_std.replace(0.0, np.nan)
            z = z.reindex(prices.index, method="ffill")

            signal = pd.Series(np.nan, index=prices.index)
            signal[z < z_long] = 1.0
            signal[z > z_flat] = 0.0
            state = signal.ffill().fillna(0.0)

            weights[sym] = state.astype(float)

        return weights
