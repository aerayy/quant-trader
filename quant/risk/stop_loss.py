from __future__ import annotations

import numpy as np
import pandas as pd

from .base import RiskOverlay


class StopLossOverlay(RiskOverlay):
    """Force-flat any active position whose drawdown from running peak
    (since entry) exceeds the threshold. Stays flat until upstream signal
    flips back to 0 then non-zero (= a fresh position run starts).

    Paramlar:
        max_loss_pct: tetik eşiği, pozitif sayı (default 0.15 = %15 DD)
    """

    def transform(self, weights: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
        threshold = float(self.params.get("max_loss_pct", 0.15))
        if threshold <= 0:
            raise ValueError(f"max_loss_pct must be > 0, got {threshold}")

        returns = prices.pct_change().fillna(0.0)
        result = weights.copy().astype(float)

        for col in weights.columns:
            w_target = weights[col].values.astype(float)
            r = returns[col].values
            out = w_target.copy()

            in_run = False
            stopped = False
            eq = 1.0
            peak = 1.0
            position = 0.0  # weight currently in effect (set last iteration)

            for t in range(len(w_target)):
                # 1. Apply return from current (prior-set) position
                if position != 0.0:
                    eq *= 1.0 + position * r[t]
                    if eq > peak:
                        peak = eq
                    dd = eq / peak - 1.0 if peak > 0 else 0.0
                    if dd < -threshold:
                        stopped = True

                # 2. Decide new position from target
                target = w_target[t]
                if target == 0.0:
                    in_run = False
                    stopped = False
                    eq = 1.0
                    peak = 1.0
                    position = 0.0
                    out[t] = 0.0
                elif stopped:
                    position = 0.0
                    out[t] = 0.0
                else:
                    if not in_run:
                        in_run = True
                        eq = 1.0
                        peak = 1.0
                    position = target
                    out[t] = target

            result[col] = out

        return result
