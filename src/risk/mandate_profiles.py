"""Mandate profile limits for diagnostics and future reward design.

This module maps simple profile names to quantitative risk limits. It is not
connected to training or reward logic; it only provides reusable configuration
objects for mandate diagnostics and future experiments.
"""

from dataclasses import asdict, dataclass


ALLOWED_MANDATE_CONFIG_KEYS = {
    "profile",
    "max_drawdown_limit",
    "max_volatility_limit",
    "max_weight_limit",
    "min_effective_assets",
    "max_turnover_limit",
}
LIMIT_FIELDS = ALLOWED_MANDATE_CONFIG_KEYS - {"profile"}


@dataclass
class MandateLimits:
    max_drawdown_limit: float
    max_volatility_limit: float
    max_weight_limit: float
    min_effective_assets: float
    max_turnover_limit: float

    def to_dict(self) -> dict:
        """Return mandate limits as a plain dictionary."""
        return asdict(self)


def get_default_mandate_profiles() -> dict[str, MandateLimits]:
    """Return fresh default mandate profile objects."""
    return {
        "conservative": MandateLimits(
            max_drawdown_limit=-0.10,
            max_volatility_limit=0.15,
            max_weight_limit=0.50,
            min_effective_assets=2.00,
            max_turnover_limit=0.50,
        ),
        "moderate": MandateLimits(
            max_drawdown_limit=-0.20,
            max_volatility_limit=0.25,
            max_weight_limit=0.80,
            min_effective_assets=1.25,
            max_turnover_limit=0.75,
        ),
        "aggressive": MandateLimits(
            max_drawdown_limit=-0.35,
            max_volatility_limit=0.45,
            max_weight_limit=1.00,
            min_effective_assets=1.00,
            max_turnover_limit=1.50,
        ),
    }


def get_mandate_limits(
    profile: str | None = None,
    overrides: dict | None = None,
) -> MandateLimits:
    """Return validated mandate limits for a profile with optional overrides."""
    selected_profile = "moderate" if profile is None else profile
    profiles = get_default_mandate_profiles()
    if selected_profile not in profiles:
        valid_profiles = ", ".join(sorted(profiles))
        raise ValueError(f"Unsupported mandate profile: {selected_profile}. Use one of: {valid_profiles}.")

    values = profiles[selected_profile].to_dict()
    if overrides is not None:
        unknown_keys = set(overrides) - LIMIT_FIELDS
        if unknown_keys:
            raise ValueError(f"Unknown mandate override keys: {sorted(unknown_keys)}.")
        values.update(overrides)

    limits = MandateLimits(**values)
    _validate_mandate_limits(limits)

    return limits


def load_mandate_limits_from_config(config: dict) -> MandateLimits:
    """Load mandate limits from an optional config mapping."""
    mandate_config = config.get("mandate")
    if mandate_config is None:
        return get_mandate_limits()
    if not isinstance(mandate_config, dict):
        raise ValueError("Config field mandate must be a mapping.")

    unknown_keys = set(mandate_config) - ALLOWED_MANDATE_CONFIG_KEYS
    if unknown_keys:
        raise ValueError(f"Unknown mandate config keys: {sorted(unknown_keys)}.")

    profile = mandate_config.get("profile")
    overrides = {
        key: value
        for key, value in mandate_config.items()
        if key != "profile"
    }

    return get_mandate_limits(profile=profile, overrides=overrides or None)


def _validate_mandate_limits(limits: MandateLimits) -> None:
    _validate_numeric(limits.max_drawdown_limit, "max_drawdown_limit")
    if limits.max_drawdown_limit > 0.0 or limits.max_drawdown_limit <= -1.0:
        raise ValueError("max_drawdown_limit must be <= 0 and > -1.")

    _validate_numeric(limits.max_volatility_limit, "max_volatility_limit")
    if limits.max_volatility_limit <= 0.0:
        raise ValueError("max_volatility_limit must be > 0.")

    _validate_numeric(limits.max_weight_limit, "max_weight_limit")
    if limits.max_weight_limit <= 0.0 or limits.max_weight_limit > 1.0:
        raise ValueError("max_weight_limit must be > 0 and <= 1.")

    _validate_numeric(limits.min_effective_assets, "min_effective_assets")
    if limits.min_effective_assets < 1.0:
        raise ValueError("min_effective_assets must be >= 1.")

    _validate_numeric(limits.max_turnover_limit, "max_turnover_limit")
    if limits.max_turnover_limit < 0.0:
        raise ValueError("max_turnover_limit must be >= 0.")


def _validate_numeric(value, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be numeric and not bool.")
