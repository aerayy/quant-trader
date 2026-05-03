from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class StrategyConfig:
    name: str
    params: dict[str, Any] = field(default_factory=dict)


class Strategy(ABC):
    """Strategy contract: take historical close prices, return desired position weights.

    Weight convention: 1.0 = full long, 0.0 = flat, -1.0 = full short.
    Weights are interpreted at the bar where they're set; the backtest engine
    applies a 1-bar lag (executes at next bar's open) for realistic execution.
    """

    def __init__(self, config: StrategyConfig):
        self.config = config
        self.params = config.params

    @abstractmethod
    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Args:
            prices: DataFrame with DatetimeIndex, columns=symbols, values=close price.

        Returns:
            DataFrame with same index/columns, values = target weight per symbol.
        """
        ...
