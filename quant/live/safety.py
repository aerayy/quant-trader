from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PreflightResult:
    ok: bool
    issues: list[str] = field(default_factory=list)
    info: dict[str, Any] = field(default_factory=dict)


def preflight(client: Any, configured_symbols: list[str]) -> PreflightResult:
    """Verify the API key is safe to use for automated trading.

    Hard refuses if:
      - WITHDRAWAL permission is enabled (kritik — hack riski)
      - SPOT trading not enabled
      - Account is locked or in maintenance
      - Configured symbols are not actually tradable on this venue

    Returns a result with issues list. Caller is expected to abort if not ok.
    """
    issues: list[str] = []
    info: dict[str, Any] = {}

    # 1. Account info
    try:
        account = client.get_account()
    except Exception as e:
        issues.append(f"Cannot fetch account info: {e!r}")
        return PreflightResult(ok=False, issues=issues)

    if not account.get("canTrade", False):
        issues.append("Account `canTrade` is FALSE — spot trading disabled.")

    permissions = account.get("permissions", []) or [account.get("accountType", "")]
    if not any(p == "SPOT" or p == "MARGIN" for p in permissions):
        issues.append(f"Account permissions {permissions} do not include SPOT.")

    info["accountType"] = account.get("accountType", "?")
    info["canTrade"] = account.get("canTrade")
    info["canDeposit"] = account.get("canDeposit")
    info["canWithdraw"] = account.get("canWithdraw")

    # 2. API key permissions — the critical check
    try:
        api_perm = client.get_account_api_permissions()
    except Exception as e:
        issues.append(f"Cannot fetch API key permissions ({e!r}); refusing to proceed.")
        return PreflightResult(ok=False, issues=issues, info=info)

    info["apiPermissions"] = api_perm

    if api_perm.get("enableWithdrawals", True):
        issues.append(
            "🚨 API KEY HAS WITHDRAWAL ENABLED. Disable in Binance "
            "→ API Management → Edit Restrictions → uncheck 'Enable Withdrawals'. "
            "Refusing to start until this is disabled."
        )

    if not api_perm.get("enableSpotAndMarginTrading", False):
        issues.append(
            "API key 'Enable Spot & Margin Trading' is OFF. "
            "Enable it in Binance → API Management."
        )

    if api_perm.get("ipRestrict") is False:
        issues.append(
            "⚠️ API key is NOT IP-restricted. Strongly recommended: edit the "
            "key in Binance and add this machine's public IP to the allow list."
        )
        # Warning, not a hard block.

    # 3. Symbols tradable
    try:
        ex_info = client.get_exchange_info()
        tradable = {
            s["symbol"]
            for s in ex_info["symbols"]
            if s["status"] == "TRADING" and "SPOT" in s.get("permissions", [])
        }
        info["exchangeInfo_count"] = len(tradable)
    except Exception as e:
        issues.append(f"Cannot fetch exchange info: {e!r}")
        return PreflightResult(ok=False, issues=issues, info=info)

    missing = [s for s in configured_symbols if s not in tradable]
    if missing:
        issues.append(f"Configured symbols not currently SPOT-tradable: {missing}")

    # Hard fail criteria — only the truly critical issues
    hard_failures = [
        i for i in issues
        if "WITHDRAWAL ENABLED" in i
        or "canTrade` is FALSE" in i
        or "do not include SPOT" in i
        or "not currently SPOT-tradable" in i
        or "Spot & Margin Trading' is OFF" in i
    ]

    return PreflightResult(ok=not hard_failures, issues=issues, info=info)


def fetch_symbol_filters(client: Any, symbols: list[str]) -> dict[str, dict[str, float]]:
    """Return per-symbol stepSize, tickSize, minNotional, minQty for order rounding."""
    ex_info = client.get_exchange_info()
    out: dict[str, dict[str, float]] = {}
    by_symbol = {s["symbol"]: s for s in ex_info["symbols"]}
    for sym in symbols:
        if sym not in by_symbol:
            continue
        filters = {f["filterType"]: f for f in by_symbol[sym]["filters"]}
        lot = filters.get("LOT_SIZE", {})
        price_f = filters.get("PRICE_FILTER", {})
        notional = filters.get("NOTIONAL") or filters.get("MIN_NOTIONAL", {})
        out[sym] = {
            "stepSize": float(lot.get("stepSize", "0.0001")),
            "minQty": float(lot.get("minQty", "0")),
            "tickSize": float(price_f.get("tickSize", "0.01")),
            "minNotional": float(notional.get("minNotional", "10")),
        }
    return out
