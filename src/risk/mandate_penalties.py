"""Pure mandate breach penalty components.

These functions are intentionally not wired into the active reward function.
They provide isolated, testable building blocks for future mandate-aware reward
experiments.
"""

from src.risk.mandate_profiles import MandateLimits


BREACH_KEYS = {
    "drawdown_breach",
    "volatility_breach",
    "max_weight_breach",
    "effective_assets_breach",
    "turnover_breach",
}
DEFAULT_PENALTY_WEIGHTS = {
    "drawdown_breach": 1.0,
    "volatility_breach": 1.0,
    "max_weight_breach": 1.0,
    "effective_assets_breach": 1.0,
    "turnover_breach": 1.0,
}


def positive_part(value: float) -> float:
    """Return max(value, 0.0)."""
    _validate_number(value, "value")

    return max(float(value), 0.0)


def drawdown_breach(
    current_drawdown: float,
    max_drawdown_limit: float,
) -> float:
    """Return absolute drawdown excess beyond a negative drawdown limit."""
    _validate_drawdown(current_drawdown, "current_drawdown")
    _validate_drawdown_limit(max_drawdown_limit)

    return positive_part(max_drawdown_limit - current_drawdown)


def volatility_breach(
    current_volatility: float,
    max_volatility_limit: float,
) -> float:
    """Return volatility excess over the mandate limit."""
    _validate_number(current_volatility, "current_volatility")
    if current_volatility < 0.0:
        raise ValueError("current_volatility must be >= 0.")
    _validate_number(max_volatility_limit, "max_volatility_limit")
    if max_volatility_limit <= 0.0:
        raise ValueError("max_volatility_limit must be > 0.")

    return positive_part(current_volatility - max_volatility_limit)


def max_weight_breach(
    max_weight: float,
    max_weight_limit: float,
) -> float:
    """Return concentration excess over the max-weight mandate limit."""
    _validate_number(max_weight, "max_weight")
    if max_weight < 0.0 or max_weight > 1.0:
        raise ValueError("max_weight must be >= 0 and <= 1.")
    _validate_number(max_weight_limit, "max_weight_limit")
    if max_weight_limit <= 0.0 or max_weight_limit > 1.0:
        raise ValueError("max_weight_limit must be > 0 and <= 1.")

    return positive_part(max_weight - max_weight_limit)


def effective_assets_breach(
    effective_assets: float,
    min_effective_assets: float,
) -> float:
    """Return diversification shortfall versus the mandate minimum."""
    _validate_number(effective_assets, "effective_assets")
    if effective_assets < 1.0:
        raise ValueError("effective_assets must be >= 1.")
    _validate_number(min_effective_assets, "min_effective_assets")
    if min_effective_assets < 1.0:
        raise ValueError("min_effective_assets must be >= 1.")

    return positive_part(min_effective_assets - effective_assets)


def turnover_breach(
    turnover: float,
    max_turnover_limit: float,
) -> float:
    """Return turnover excess over the mandate limit."""
    _validate_number(turnover, "turnover")
    if turnover < 0.0:
        raise ValueError("turnover must be >= 0.")
    _validate_number(max_turnover_limit, "max_turnover_limit")
    if max_turnover_limit < 0.0:
        raise ValueError("max_turnover_limit must be >= 0.")

    return positive_part(turnover - max_turnover_limit)


def compute_cash_breach(
    cash_weight: float,
    normal_cash_max: float = 0.10,
    risk_off_state: bool = False,
) -> float:
    """Return CASH exposure above the normal band unless risk-off is active."""
    _validate_number(cash_weight, "cash_weight")
    if cash_weight < 0.0 or cash_weight > 1.0:
        raise ValueError("cash_weight must be >= 0 and <= 1.")
    _validate_number(normal_cash_max, "normal_cash_max")
    if normal_cash_max < 0.0 or normal_cash_max > 1.0:
        raise ValueError("normal_cash_max must be >= 0 and <= 1.")
    if not isinstance(risk_off_state, bool):
        raise ValueError("risk_off_state must be bool.")
    if risk_off_state:
        return 0.0

    return positive_part(cash_weight - normal_cash_max)


def compute_mandate_breaches(
    current_drawdown: float,
    current_volatility: float,
    max_weight: float,
    effective_assets: float,
    turnover: float,
    mandate_limits: MandateLimits,
) -> dict:
    """Compute all mandate breach magnitudes."""
    if not isinstance(mandate_limits, MandateLimits):
        raise ValueError("mandate_limits must be a MandateLimits instance.")

    return {
        "drawdown_breach": drawdown_breach(
            current_drawdown,
            mandate_limits.max_drawdown_limit,
        ),
        "volatility_breach": volatility_breach(
            current_volatility,
            mandate_limits.max_volatility_limit,
        ),
        "max_weight_breach": max_weight_breach(
            max_weight,
            mandate_limits.max_weight_limit,
        ),
        "effective_assets_breach": effective_assets_breach(
            effective_assets,
            mandate_limits.min_effective_assets,
        ),
        "turnover_breach": turnover_breach(
            turnover,
            mandate_limits.max_turnover_limit,
        ),
    }


def compute_weighted_mandate_penalty(
    breaches: dict,
    penalty_weights: dict | None = None,
) -> float:
    """Compute weighted sum of mandate breaches."""
    unknown_breach_keys = set(breaches) - BREACH_KEYS
    if unknown_breach_keys:
        raise ValueError(f"Unknown mandate breach keys: {sorted(unknown_breach_keys)}.")

    weights = DEFAULT_PENALTY_WEIGHTS.copy()
    if penalty_weights is not None:
        unknown_weight_keys = set(penalty_weights) - BREACH_KEYS
        if unknown_weight_keys:
            raise ValueError(f"Unknown mandate penalty weight keys: {sorted(unknown_weight_keys)}.")
        for key, value in penalty_weights.items():
            _validate_number(value, f"penalty_weights.{key}")
            if value < 0.0:
                raise ValueError(f"penalty_weights.{key} must be >= 0.")
            weights[key] = float(value)

    penalty = 0.0
    for key, breach_value in breaches.items():
        _validate_number(breach_value, key)
        if breach_value < 0.0:
            raise ValueError(f"{key} must be >= 0.")
        penalty += float(breach_value) * weights[key]

    return float(penalty)


def compute_mandate_penalty(
    current_drawdown: float,
    current_volatility: float,
    max_weight: float,
    effective_assets: float,
    turnover: float,
    mandate_limits: MandateLimits,
    penalty_weights: dict | None = None,
) -> dict:
    """Compute mandate breaches and their weighted penalty."""
    breaches = compute_mandate_breaches(
        current_drawdown=current_drawdown,
        current_volatility=current_volatility,
        max_weight=max_weight,
        effective_assets=effective_assets,
        turnover=turnover,
        mandate_limits=mandate_limits,
    )

    return {
        "penalty": compute_weighted_mandate_penalty(
            breaches,
            penalty_weights=penalty_weights,
        ),
        "breaches": breaches,
    }


def _validate_drawdown(value: float, field_name: str) -> None:
    _validate_number(value, field_name)
    if value > 0.0 or value <= -1.0:
        raise ValueError(f"{field_name} must be <= 0 and > -1.")


def _validate_drawdown_limit(value: float) -> None:
    _validate_number(value, "max_drawdown_limit")
    if value > 0.0 or value <= -1.0:
        raise ValueError("max_drawdown_limit must be <= 0 and > -1.")


def _validate_number(value, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be numeric and not bool.")
