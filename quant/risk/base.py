from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class RiskOverlay(ABC):
    """Composable risk-shaping layer. Takes target weights from the upstream
    strategy (or earlier overlay), returns modified weights.

    Order matters: trend_filter masks before sizing; vol_target scales raw
    weights; stop_loss overrides per-position drawdowns. Recommended order
    is trend_filter → vol_target → stop_loss.
    """

    def __init__(self, params: dict[str, Any] | None = None):
        self.params = params or {}

    @abstractmethod
    def transform(self, weights: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
        ...

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.params})"
