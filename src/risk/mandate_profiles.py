"""Canonical mandate profile limits for diagnostics and reporting.

Max-weight caps are structural TD3 experiment constraints, not official
investor mandate constraints. Concentration discipline is represented by the
effective-assets minimum.
"""

from dataclasses import asdict, dataclass


ALLOWED_MANDATE_CONFIG_KEYS = {
    "profile",
    "max_drawdown",
    "max_annualized_volatility",
    "min_effective_assets",
    "max_average_turnover",
    "max_drawdown_limit",
    "max_volatility_limit",
    "max_turnover_limit",
    "description",
}
LIMIT_FIELDS = ALLOWED_MANDATE_CONFIG_KEYS - {"profile"}


@dataclass
class MandateLimits:
    profile_name: str
    max_drawdown: float
    max_annualized_volatility: float
    min_effective_assets: float
    max_average_turnover: float
    description: str

    def to_dict(self) -> dict:
        """Return canonical mandate limits as a plain dictionary."""
        return asdict(self)

    @property
    def max_drawdown_limit(self) -> float:
        """Backward-compatible alias for the canonical drawdown field."""
        return self.max_drawdown

    @property
    def max_volatility_limit(self) -> float:
        """Backward-compatible alias for the canonical volatility field."""
        return self.max_annualized_volatility

    @property
    def max_turnover_limit(self) -> float:
        """Backward-compatible alias for the canonical turnover field."""
        return self.max_average_turnover

    @property
    def max_weight_limit(self) -> float:
        """Legacy non-binding alias.

        Max weight is no longer part of the canonical mandate. Returning 1.0
        preserves old diagnostic components without introducing an official
        max-weight mandate constraint.
        """
        return 1.0


def get_default_mandate_profiles() -> dict[str, MandateLimits]:
    """Return fresh default mandate profile objects."""
    return {
        "conservative": MandateLimits(
            profile_name="conservative",
            max_drawdown=-0.10,
            max_annualized_volatility=0.10,
            min_effective_assets=3.00,
            max_average_turnover=0.05,
            description=(
                "Low-risk mandate with tight drawdown, volatility, "
                "diversification, and implementation-discipline limits."
            ),
        ),
        "moderate": MandateLimits(
            profile_name="moderate",
            max_drawdown=-0.15,
            max_annualized_volatility=0.15,
            min_effective_assets=2.30,
            max_average_turnover=0.10,
            description=(
                "Balanced mandate with moderate drawdown, volatility, "
                "diversification, and turnover constraints."
            ),
        ),
        "aggressive": MandateLimits(
            profile_name="aggressive",
            max_drawdown=-0.25,
            max_annualized_volatility=0.25,
            min_effective_assets=1.50,
            max_average_turnover=0.20,
            description=(
                "Higher-risk mandate with looser drawdown, volatility, "
                "diversification, and turnover constraints."
            ),
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
        values.update(_canonicalize_overrides(overrides))

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
    if not isinstance(limits.profile_name, str) or not limits.profile_name.strip():
        raise ValueError("profile_name must be a non-empty string.")

    _validate_numeric(limits.max_drawdown, "max_drawdown")
    if limits.max_drawdown > 0.0 or limits.max_drawdown <= -1.0:
        raise ValueError("max_drawdown must be <= 0 and > -1.")

    _validate_numeric(limits.max_annualized_volatility, "max_annualized_volatility")
    if limits.max_annualized_volatility <= 0.0:
        raise ValueError("max_annualized_volatility must be > 0.")

    _validate_numeric(limits.min_effective_assets, "min_effective_assets")
    if limits.min_effective_assets < 1.0:
        raise ValueError("min_effective_assets must be >= 1.")

    _validate_numeric(limits.max_average_turnover, "max_average_turnover")
    if limits.max_average_turnover < 0.0:
        raise ValueError("max_average_turnover must be >= 0.")

    if not isinstance(limits.description, str) or not limits.description.strip():
        raise ValueError("description must be a non-empty string.")


def _canonicalize_overrides(overrides: dict) -> dict:
    aliases = {
        "max_drawdown_limit": "max_drawdown",
        "max_volatility_limit": "max_annualized_volatility",
        "max_turnover_limit": "max_average_turnover",
    }
    return {aliases.get(key, key): value for key, value in overrides.items()}


def _validate_numeric(value, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be numeric and not bool.")
