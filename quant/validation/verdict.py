from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .oos import OOSResult
from .walk_forward import WalkForwardResult


@dataclass
class Verdict:
    decision: str  # "GO" | "MARGINAL" | "REJECT"
    score: int     # 0-3
    notes: list[str] = field(default_factory=list)

    def report_text(self) -> str:
        line = "─" * 50
        return "\n".join([
            line,
            f"  VERDICT: {self.decision}  (score {self.score}/3)",
            line,
            *self.notes,
            line,
        ])


# Conservative threshold rules — designed to fail naive strategies that
# only look good in calm/trending periods.
OOS_SHARPE_MIN = 0.5               # OOS Sharpe must clear 0.5 in absolute terms
OOS_RETENTION_MIN = 0.5            # AND be ≥50% of IS Sharpe (catches overfit)
WF_POSITIVE_FOLDS_FRAC = 0.6       # ≥60% of folds must have positive Sharpe
WF_WORST_FOLD_MIN = -1.0           # Worst fold Sharpe must be > -1.0 (no catastrophes)
SENS_CV_MAX = 0.5                  # Sharpe coefficient of variation < 0.5
SENS_MEAN_MIN = 0.3                # Mean Sharpe across grid must clear 0.3
SENS_DD_MAX = -0.40                # Median Max DD across grid must be > -40%


def assess(
    oos: OOSResult,
    wf: WalkForwardResult,
    sensitivity: pd.DataFrame,
) -> Verdict:
    notes: list[str] = []
    score = 0

    # 1. OOS check: absolute Sharpe AND retention
    is_s = float(oos.is_metrics.get("sharpe", 0))
    oos_s = float(oos.oos_metrics.get("sharpe", 0))
    abs_ok = oos_s >= OOS_SHARPE_MIN
    if is_s <= 0:
        retention_ok = False
        retention_str = "n/a (IS Sharpe ≤ 0)"
    else:
        retention = oos_s / is_s
        retention_ok = retention >= OOS_RETENTION_MIN
        retention_str = f"{retention:.0%}"
    if abs_ok and retention_ok:
        score += 1
        notes.append(f"  ✓ OOS: Sharpe {oos_s:+.2f} (≥{OOS_SHARPE_MIN}); retention {retention_str} of IS {is_s:+.2f}")
    else:
        why = []
        if not abs_ok:
            why.append(f"OOS Sharpe {oos_s:+.2f} < {OOS_SHARPE_MIN}")
        if not retention_ok:
            why.append(f"retention {retention_str} weak vs IS {is_s:+.2f}")
        notes.append(f"  ✗ OOS: " + "; ".join(why))

    # 2. Walk-forward: fold positivity AND no catastrophic fold
    pos = wf.summary["positive_folds"]
    total = max(wf.summary["total_folds"], 1)
    frac = pos / total
    worst = wf.summary["sharpe_min"]
    frac_ok = frac >= WF_POSITIVE_FOLDS_FRAC
    worst_ok = worst > WF_WORST_FOLD_MIN
    if frac_ok and worst_ok:
        score += 1
        notes.append(f"  ✓ Walk-forward: {pos}/{total} positive ({frac:.0%}); "
                     f"worst fold Sharpe {worst:+.2f}; "
                     f"μ={wf.summary['sharpe_mean']:+.2f} σ={wf.summary['sharpe_std']:.2f}")
    else:
        why = []
        if not frac_ok:
            why.append(f"only {pos}/{total} folds positive ({frac:.0%})")
        if not worst_ok:
            why.append(f"catastrophic fold (Sharpe {worst:+.2f} < {WF_WORST_FOLD_MIN})")
        notes.append(f"  ✗ Walk-forward: " + "; ".join(why) +
                     f"; μ={wf.summary['sharpe_mean']:+.2f} σ={wf.summary['sharpe_std']:.2f}")

    # 3. Parameter sensitivity: mean clears bar, low CV, tolerable DD
    sens_sharpes = sensitivity["sharpe"]
    sens_mean = float(sens_sharpes.mean())
    sens_std = float(sens_sharpes.std())
    cv = sens_std / abs(sens_mean) if abs(sens_mean) > 1e-9 else float("inf")
    median_dd = float(sensitivity["max_drawdown"].median()) if "max_drawdown" in sensitivity.columns else 0.0
    mean_ok = sens_mean > SENS_MEAN_MIN
    cv_ok = cv < SENS_CV_MAX
    dd_ok = median_dd > SENS_DD_MAX
    if mean_ok and cv_ok and dd_ok:
        score += 1
        notes.append(f"  ✓ Sensitivity: mean Sharpe {sens_mean:+.2f}, CV {cv:.2f}, "
                     f"median MaxDD {median_dd:.0%} (across {len(sensitivity)} combos)")
    else:
        why = []
        if not mean_ok:
            why.append(f"mean Sharpe {sens_mean:+.2f} < {SENS_MEAN_MIN}")
        if not cv_ok:
            why.append(f"CV {cv:.2f} ≥ {SENS_CV_MAX} (unstable params)")
        if not dd_ok:
            why.append(f"median MaxDD {median_dd:.0%} ≤ {SENS_DD_MAX:.0%} (intolerable)")
        notes.append(f"  ✗ Sensitivity: " + "; ".join(why))

    if score >= 3:
        decision = "GO"
    elif score >= 2:
        decision = "MARGINAL"
    else:
        decision = "REJECT"

    return Verdict(decision=decision, score=score, notes=notes)
