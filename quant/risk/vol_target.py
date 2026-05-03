from __future__ import annotations

import numpy as np
import pandas as pd

from .base import RiskOverlay


class VolTargetOverlay(RiskOverlay):
    """Scale per-asset weights so that realized vol approaches a target.

    weight_t = base_t * (target_vol / realized_vol_t)

    Default annualization assumes 365 (crypto trades 7d). Capped at
    max_leverage (default 1.0 = no upsize beyond the original weight).

    Paramlar:
        target_vol: hedef yıllıklandırılmış volatilite (default 0.30 = %30)
        vol_window: realized vol için rolling pencere (gün, default 30)
        max_leverage: ölçek katsayısının üst sınırı (default 1.0)
        annualization: yıllık period sayısı (default 365 — kripto)
    """

    def transform(self, weights: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
        target_vol = float(self.params.get("target_vol", 0.30))
        vol_window = int(self.params.get("vol_window", 30))
        max_lev = float(self.params.get("max_leverage", 1.0))
        annualization = int(self.params.get("annualization", 365))

        if target_vol <= 0:
            raise ValueError(f"target_vol must be > 0, got {target_vol}")

        returns = prices.pct_change()
        realized = (
            returns.rolling(vol_window, min_periods=max(vol_window // 2, 10)).std()
            * np.sqrt(annualization)
        )
        scale = (target_vol / realized.replace(0.0, np.nan)).clip(upper=max_lev).fillna(0.0)
        return (weights * scale).reindex(weights.index).fillna(0.0)
