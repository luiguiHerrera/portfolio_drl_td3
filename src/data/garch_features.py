"""GARCH-style volatility features.

This module provides a simple GARCH(1,1)-style conditional volatility filter
that can be used as an opt-in feature source later. It is not a fitted
maximum-likelihood GARCH model.

It also provides an explicit rolling-fitted GARCH(1,1) feature path. The fitted
path estimates parameters using only returns available through t-1 and produces
one-step-ahead volatility forecasts for t. The fitted path uses the standard
``arch`` package when available, with a zero-mean, normal GARCH(1,1) model, and
returns weekly volatility by default.
"""

import math
from numbers import Real
from typing import Any

import numpy as np
import pandas as pd

try:
    from scipy.optimize import minimize
except ImportError:  # pragma: no cover - exercised only without scipy installed.
    minimize = None

try:
    from arch import arch_model as _arch_model
except ImportError:  # pragma: no cover - arch is optional.
    _arch_model = None


GARCH_MODE_DETERMINISTIC = "deterministic_filter"
GARCH_MODE_ROLLING_FITTED = "rolling_fitted"
GARCH_FALLBACK_ROLLING_REALIZED = "rolling_realized_vol"


def validate_garch_parameters(
    omega: float,
    alpha: float,
    beta: float,
    periods_per_year: int,
) -> None:
    """Validate deterministic GARCH filter parameters."""
    _validate_numeric(omega, "omega")
    _validate_numeric(alpha, "alpha")
    _validate_numeric(beta, "beta")
    if not isinstance(periods_per_year, int) or isinstance(periods_per_year, bool):
        raise ValueError("periods_per_year must be a positive integer.")

    if omega <= 0.0:
        raise ValueError("omega must be greater than 0.")
    if alpha < 0.0:
        raise ValueError("alpha must be greater than or equal to 0.")
    if beta < 0.0:
        raise ValueError("beta must be greater than or equal to 0.")
    if alpha + beta >= 1.0:
        raise ValueError("alpha + beta must be less than 1.")
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be a positive integer.")


def compute_garch_volatility_series(
    returns: pd.Series,
    omega: float = 1e-6,
    alpha: float = 0.05,
    beta: float = 0.90,
    periods_per_year: int = 52,
    annualize: bool = True,
) -> pd.Series:
    """Compute a deterministic GARCH(1,1)-style volatility series.

    The recursion uses lagged returns only:
    sigma2_t = omega + alpha * r_{t-1}^2 + beta * sigma2_{t-1}.
    """
    validate_garch_parameters(omega, alpha, beta, periods_per_year)
    if not isinstance(returns, pd.Series):
        raise TypeError("returns must be a pandas Series.")

    numeric_returns = pd.to_numeric(returns, errors="coerce")
    if numeric_returns.isna().any():
        raise ValueError("returns must not contain missing or non-numeric values.")

    sigma2 = np.empty(len(numeric_returns), dtype=float)
    initial_sigma2 = omega / (1.0 - alpha - beta)
    if len(numeric_returns) > 0:
        sigma2[0] = initial_sigma2
    for index in range(1, len(numeric_returns)):
        lagged_return = float(numeric_returns.iloc[index - 1])
        sigma2[index] = omega + alpha * lagged_return**2 + beta * sigma2[index - 1]

    volatility = np.sqrt(sigma2)
    if annualize:
        volatility = volatility * math.sqrt(periods_per_year)

    name = f"{returns.name}_garch_vol" if returns.name is not None else "garch_vol"

    return pd.Series(volatility, index=returns.index, name=name)


def build_garch_volatility_features(
    returns: pd.DataFrame,
    assets: list[str] | None = None,
    omega: float = 1e-6,
    alpha: float = 0.05,
    beta: float = 0.90,
    periods_per_year: int = 52,
    annualize: bool = True,
    prefix: str = "garch_vol",
) -> pd.DataFrame:
    """Build absolute deterministic GARCH volatility features for assets."""
    _validate_returns_dataframe(returns)
    if not isinstance(prefix, str) or not prefix.strip():
        raise ValueError("prefix must be a non-empty string.")

    selected_assets = list(returns.columns) if assets is None else list(assets)
    missing_assets = [asset for asset in selected_assets if asset not in returns.columns]
    if missing_assets:
        raise ValueError(f"Requested assets are missing from returns: {missing_assets}.")

    features = pd.DataFrame(index=returns.index)
    for asset in selected_assets:
        volatility = compute_garch_volatility_series(
            returns[asset],
            omega=omega,
            alpha=alpha,
            beta=beta,
            periods_per_year=periods_per_year,
            annualize=annualize,
        )
        features[f"{prefix}_{asset}"] = volatility

    return features


def build_garch_relative_features(
    garch_vol_features: pd.DataFrame,
    market_asset: str = "SPY",
    prefix: str = "garch",
) -> pd.DataFrame:
    """Build cross-sectional GARCH volatility ratios and ranks."""
    if not isinstance(garch_vol_features, pd.DataFrame):
        raise TypeError("garch_vol_features must be a pandas DataFrame.")
    if garch_vol_features.empty:
        raise ValueError("garch_vol_features must be a non-empty DataFrame.")
    if not isinstance(market_asset, str) or not market_asset.strip():
        raise ValueError("market_asset must be a non-empty string.")
    if not isinstance(prefix, str) or not prefix.strip():
        raise ValueError("prefix must be a non-empty string.")

    absolute_prefix = f"{prefix}_vol_"
    volatility_columns = [
        column
        for column in garch_vol_features.columns
        if isinstance(column, str) and column.startswith(absolute_prefix)
    ]
    if not volatility_columns:
        raise ValueError(f"No columns found with prefix '{absolute_prefix}'.")

    market_column = f"{absolute_prefix}{market_asset}"
    if market_column not in garch_vol_features.columns:
        raise ValueError(f"Market volatility column '{market_column}' is missing.")

    volatility = garch_vol_features[volatility_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )
    if volatility.isna().any().any():
        raise ValueError("garch_vol_features must contain only numeric values.")
    if (volatility[market_column] <= 0.0).any():
        raise ValueError("market volatility must be strictly positive.")

    relative = pd.DataFrame(index=garch_vol_features.index)
    market_volatility = volatility[market_column]
    ranks = volatility.rank(axis=1, method="average", ascending=True)
    for column in volatility_columns:
        asset = column.removeprefix(absolute_prefix)
        relative[f"{prefix}_vol_ratio_{asset}_vs_{market_asset}"] = (
            volatility[column] / market_volatility
        )
        relative[f"{prefix}_vol_rank_{asset}"] = ranks[column]

    return relative


def build_garch_feature_set(
    returns: pd.DataFrame,
    assets: list[str] | None = None,
    market_asset: str = "SPY",
    include_relative: bool = True,
    omega: float = 1e-6,
    alpha: float = 0.05,
    beta: float = 0.90,
    periods_per_year: int = 52,
) -> pd.DataFrame:
    """Build absolute and optional relative GARCH-style volatility features."""
    absolute_features = build_garch_volatility_features(
        returns=returns,
        assets=assets,
        omega=omega,
        alpha=alpha,
        beta=beta,
        periods_per_year=periods_per_year,
        annualize=True,
        prefix="garch_vol",
    )
    if not include_relative:
        return absolute_features

    relative_features = build_garch_relative_features(
        absolute_features,
        market_asset=market_asset,
        prefix="garch",
    )

    return pd.concat([absolute_features, relative_features], axis=1)


def build_rolling_fitted_garch_volatility_features(
    returns: pd.DataFrame,
    assets: list[str] | None = None,
    min_history: int = 104,
    window: int | None = 156,
    periods_per_year: int = 52,
    annualize: bool = False,
    exclude_cash: bool = True,
    fallback: str = GARCH_FALLBACK_ROLLING_REALIZED,
    min_vol: float = 1e-8,
    prefix: str = "garch_vol",
    return_diagnostics: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Build rolling-fitted one-step-ahead GARCH volatility features.

    For each date t, this function fits GARCH(1,1) using only returns through
    t-1 and assigns the one-step-ahead forecast to t. CASH is excluded by
    default because zero-return cash has degenerate variance.
    """
    _validate_returns_dataframe(returns)
    _validate_rolling_garch_options(min_history, window, periods_per_year, fallback)
    if not isinstance(prefix, str) or not prefix.strip():
        raise ValueError("prefix must be a non-empty string.")

    selected_assets = _select_garch_assets(returns, assets=assets, exclude_cash=exclude_cash)
    features = pd.DataFrame(index=returns.index)
    diagnostics = []
    for asset in selected_assets:
        series, asset_diagnostics = compute_rolling_fitted_garch_volatility_series(
            returns[asset],
            min_history=min_history,
            window=window,
            periods_per_year=periods_per_year,
            annualize=annualize,
            fallback=fallback,
            min_vol=min_vol,
            asset_name=asset,
        )
        features[f"{prefix}_{asset}"] = series
        diagnostics.extend(asset_diagnostics)

    diagnostics_frame = pd.DataFrame(diagnostics)
    if return_diagnostics:
        return features, diagnostics_frame
    return features


def compute_rolling_fitted_garch_volatility_series(
    returns: pd.Series,
    min_history: int = 104,
    window: int | None = 156,
    periods_per_year: int = 52,
    annualize: bool = False,
    fallback: str = GARCH_FALLBACK_ROLLING_REALIZED,
    min_vol: float = 1e-8,
    asset_name: str | None = None,
) -> tuple[pd.Series, list[dict[str, Any]]]:
    """Compute rolling-fitted one-step-ahead GARCH volatility for one asset."""
    if not isinstance(returns, pd.Series):
        raise TypeError("returns must be a pandas Series.")
    _validate_rolling_garch_options(min_history, window, periods_per_year, fallback)
    _validate_numeric(min_vol, "min_vol")
    if min_vol <= 0.0:
        raise ValueError("min_vol must be greater than 0.")

    numeric_returns = pd.to_numeric(returns, errors="coerce")
    if numeric_returns.isna().any():
        raise ValueError("returns must not contain missing or non-numeric values.")

    name = str(asset_name or returns.name or "asset")
    values = []
    diagnostics = []
    for position, date in enumerate(numeric_returns.index):
        history = numeric_returns.iloc[:position]
        if window is not None:
            fit_history = history.tail(window)
        else:
            fit_history = history
        status = "fitted"
        fallback_reason = ""
        params = {"omega": np.nan, "alpha": np.nan, "beta": np.nan}
        variance = np.nan
        backend = _primary_fitted_garch_backend()
        if len(history) < 2:
            status = "fallback"
            fallback_reason = "insufficient_history"
        elif len(history) < min_history:
            status = "fallback"
            fallback_reason = "insufficient_history"
        else:
            fit, backend, fallback_reason = _fit_one_step_garch_forecast(fit_history)
            if fit is None:
                status = "fallback"
            else:
                params = {
                    "omega": fit["omega"],
                    "alpha": fit["alpha"],
                    "beta": fit["beta"],
                }
                variance = float(fit["forecast_variance"])
                if not np.isfinite(variance) or variance <= 0.0:
                    status = "fallback"
                    fallback_reason = _append_reason(
                        fallback_reason,
                        "non_positive_forecast",
                    )

        if status == "fallback":
            variance = _rolling_realized_variance(history, window)

        volatility = np.sqrt(variance) if pd.notna(variance) else np.nan
        if pd.notna(volatility):
            volatility = max(float(volatility), min_vol)
            if annualize:
                volatility *= math.sqrt(periods_per_year)
        values.append(volatility)
        diagnostics.append(
            {
                "date": date,
                "asset": name,
                "status": status,
                "fallback_reason": fallback_reason,
                "n_history": len(history),
                "fit_window": len(fit_history),
                "forecast_variance": variance,
                "forecast_volatility": volatility,
                "omega": params["omega"],
                "alpha": params["alpha"],
                "beta": params["beta"],
                "backend": backend,
                "arch_available": _arch_model is not None,
                "scipy_available": minimize is not None,
                "volatility_unit": "annualized" if annualize else "weekly",
            }
        )

    return (
        pd.Series(values, index=numeric_returns.index, name=f"{returns.name}_garch_vol"),
        diagnostics,
    )


def build_real_garch_feature_set(
    returns: pd.DataFrame,
    assets: list[str] | None = None,
    market_asset: str = "SPY",
    include_relative: bool = True,
    min_history: int = 104,
    window: int | None = 156,
    periods_per_year: int = 52,
    annualize: bool = False,
    exclude_cash: bool = True,
    fallback: str = GARCH_FALLBACK_ROLLING_REALIZED,
    return_diagnostics: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Build absolute and optional rolling-fitted GARCH features."""
    absolute_result = build_rolling_fitted_garch_volatility_features(
        returns=returns,
        assets=assets,
        min_history=min_history,
        window=window,
        periods_per_year=periods_per_year,
        annualize=annualize,
        exclude_cash=exclude_cash,
        fallback=fallback,
        prefix="garch_vol",
        return_diagnostics=True,
    )
    absolute_features, diagnostics = absolute_result
    if not include_relative:
        if return_diagnostics:
            return absolute_features, diagnostics
        return absolute_features

    relative_features = build_garch_relative_features(
        absolute_features.dropna(),
        market_asset=market_asset,
        prefix="garch",
    )
    features = pd.concat([absolute_features, relative_features], axis=1, sort=False)
    if return_diagnostics:
        return features, diagnostics
    return features


def build_garch_feature_set_by_mode(
    returns: pd.DataFrame,
    assets: list[str] | None = None,
    market_asset: str = "SPY",
    include_relative: bool = True,
    mode: str = GARCH_MODE_DETERMINISTIC,
    omega: float = 1e-6,
    alpha: float = 0.05,
    beta: float = 0.90,
    periods_per_year: int = 52,
    min_history: int = 104,
    window: int | None = 156,
    annualize: bool = False,
    exclude_cash: bool = False,
    fallback: str = GARCH_FALLBACK_ROLLING_REALIZED,
    return_diagnostics: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Build GARCH features using either deterministic or rolling-fitted mode."""
    if mode == GARCH_MODE_DETERMINISTIC:
        features = build_garch_feature_set(
            returns=returns,
            assets=assets,
            market_asset=market_asset,
            include_relative=include_relative,
            omega=omega,
            alpha=alpha,
            beta=beta,
            periods_per_year=periods_per_year,
        )
        if return_diagnostics:
            diagnostics = pd.DataFrame(
                {
                    "mode": [GARCH_MODE_DETERMINISTIC],
                    "status": ["deterministic_filter"],
                    "backend": ["fixed_parameter_filter"],
                    "arch_available": [_arch_model is not None],
                    "scipy_available": [minimize is not None],
                }
            )
            return features, diagnostics
        return features
    if mode == GARCH_MODE_ROLLING_FITTED:
        return build_real_garch_feature_set(
            returns=returns,
            assets=assets,
            market_asset=market_asset,
            include_relative=include_relative,
            min_history=min_history,
            window=window,
            periods_per_year=periods_per_year,
            annualize=annualize,
            exclude_cash=exclude_cash,
            fallback=fallback,
            return_diagnostics=return_diagnostics,
        )
    raise ValueError(
        "garch mode must be 'deterministic_filter' or 'rolling_fitted'."
    )


def _validate_returns_dataframe(returns: pd.DataFrame) -> None:
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns must be a pandas DataFrame.")
    if returns.empty:
        raise ValueError("returns must be a non-empty DataFrame.")
    if not isinstance(returns.index, pd.DatetimeIndex):
        raise TypeError("returns index must be a pandas DatetimeIndex.")


def _validate_rolling_garch_options(
    min_history: int,
    window: int | None,
    periods_per_year: int,
    fallback: str,
) -> None:
    _validate_positive_integer(min_history, "min_history")
    if window is not None:
        _validate_positive_integer(window, "window")
    _validate_positive_integer(periods_per_year, "periods_per_year")
    if fallback != GARCH_FALLBACK_ROLLING_REALIZED:
        raise ValueError("fallback must be 'rolling_realized_vol'.")


def _validate_positive_integer(value: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")


def _select_garch_assets(
    returns: pd.DataFrame,
    assets: list[str] | None,
    exclude_cash: bool,
) -> list[str]:
    selected_assets = list(returns.columns) if assets is None else list(assets)
    missing_assets = [asset for asset in selected_assets if asset not in returns.columns]
    if missing_assets:
        raise ValueError(f"Requested assets are missing from returns: {missing_assets}.")
    if exclude_cash:
        selected_assets = [asset for asset in selected_assets if asset != "CASH"]
    if not selected_assets:
        raise ValueError("At least one non-CASH asset is required for GARCH features.")
    return selected_assets


def _primary_fitted_garch_backend() -> str:
    if _arch_model is not None:
        return "arch_model"
    if minimize is not None:
        return "scipy_mle_zero_mean_normal"
    return "unavailable"


def _fit_one_step_garch_forecast(
    returns: pd.Series,
) -> tuple[dict[str, float] | None, str, str]:
    """Fit one-step GARCH forecast, preferring ``arch_model`` over scipy fallback."""
    fallback_reason = ""
    if _arch_model is not None:
        try:
            return _fit_arch_zero_mean_garch_11(returns), "arch_model", fallback_reason
        except Exception as exc:  # pragma: no cover - exact arch errors vary by version.
            fallback_reason = f"arch_fit_failed:{type(exc).__name__}"
    else:
        fallback_reason = "arch_unavailable"

    if minimize is None:
        return None, _primary_fitted_garch_backend(), _append_reason(
            fallback_reason,
            "scipy_unavailable",
        )

    try:
        fit = _fit_zero_mean_garch_11(returns)
        latest_return = float(returns.iloc[-1])
        latest_variance = float(fit["last_variance"])
        forecast_variance = (
            fit["omega"] + fit["alpha"] * latest_return**2 + fit["beta"] * latest_variance
        )
        return (
            {
                "omega": fit["omega"],
                "alpha": fit["alpha"],
                "beta": fit["beta"],
                "forecast_variance": forecast_variance,
            },
            "scipy_mle_zero_mean_normal",
            fallback_reason,
        )
    except Exception as exc:  # pragma: no cover - exact optimizer errors vary.
        return None, "scipy_mle_zero_mean_normal", _append_reason(
            fallback_reason,
            f"scipy_fit_failed:{type(exc).__name__}",
        )


def _fit_arch_zero_mean_garch_11(returns: pd.Series) -> dict[str, float]:
    """Fit zero-mean normal GARCH(1,1) with ``arch_model`` on past returns only."""
    if _arch_model is None:
        raise ImportError("arch is not available.")
    values = pd.to_numeric(returns, errors="raise").to_numpy(dtype=float)
    if len(values) < 2:
        raise ValueError("at least two returns are required for GARCH fitting.")
    sample_variance = float(np.var(values, ddof=1))
    if not np.isfinite(sample_variance) or sample_variance <= 0.0:
        raise ValueError("return variance must be positive for GARCH fitting.")

    # Scale decimal weekly returns to percentages for numerical stability in arch.
    scaled_values = values * 100.0
    model = _arch_model(
        scaled_values,
        mean="Zero",
        vol="GARCH",
        p=1,
        o=0,
        q=1,
        dist="normal",
        rescale=False,
    )
    fitted = model.fit(update_freq=0, disp="off", show_warning=False)
    forecast = fitted.forecast(horizon=1, reindex=False)
    forecast_variance_pct = float(forecast.variance.iloc[-1, 0])
    forecast_variance = forecast_variance_pct / 10000.0
    params = fitted.params
    return {
        "omega": float(params.get("omega", np.nan)) / 10000.0,
        "alpha": float(params.get("alpha[1]", np.nan)),
        "beta": float(params.get("beta[1]", np.nan)),
        "forecast_variance": forecast_variance,
    }


def _fit_zero_mean_garch_11(returns: pd.Series) -> dict[str, float]:
    """Fit zero-mean normal GARCH(1,1) using scipy MLE on decimal returns."""
    values = pd.to_numeric(returns, errors="raise").to_numpy(dtype=float)
    if len(values) < 2:
        raise ValueError("at least two returns are required for GARCH fitting.")
    sample_variance = float(np.var(values, ddof=1))
    if not np.isfinite(sample_variance) or sample_variance <= 0.0:
        raise ValueError("return variance must be positive for GARCH fitting.")

    # Optimize on transformed parameters to enforce positivity and alpha+beta<1.
    initial = np.array(
        [
            np.log(max(sample_variance * 0.05, 1e-12)),
            _logit(0.05),
            _logit(0.90),
        ],
        dtype=float,
    )
    result = minimize(
        lambda params: _garch_negative_log_likelihood(values, params),
        initial,
        method="L-BFGS-B",
        options={"maxiter": 200, "ftol": 1e-8},
    )
    if not result.success or not np.isfinite(result.fun):
        raise ValueError("GARCH optimizer did not converge.")
    omega, alpha, beta = _unpack_garch_params(result.x)
    variances = _garch_variance_path(values, omega, alpha, beta)
    return {
        "omega": omega,
        "alpha": alpha,
        "beta": beta,
        "last_variance": float(variances[-1]),
    }


def _garch_negative_log_likelihood(values: np.ndarray, params: np.ndarray) -> float:
    omega, alpha, beta = _unpack_garch_params(params)
    variances = _garch_variance_path(values, omega, alpha, beta)
    if not np.isfinite(variances).all() or (variances <= 0.0).any():
        return 1e12
    likelihood = 0.5 * (np.log(2.0 * math.pi) + np.log(variances) + values**2 / variances)
    total = float(np.sum(likelihood))
    return total if np.isfinite(total) else 1e12


def _unpack_garch_params(params: np.ndarray) -> tuple[float, float, float]:
    omega = float(np.exp(np.clip(params[0], -50.0, 10.0)))
    alpha_raw = _sigmoid(float(params[1]))
    beta_raw = _sigmoid(float(params[2]))
    total = max(alpha_raw + beta_raw, 1e-12)
    persistence = 0.999 * _sigmoid(float(params[1]) + float(params[2]))
    alpha = persistence * alpha_raw / total
    beta = persistence * beta_raw / total
    return omega, alpha, beta


def _garch_variance_path(
    values: np.ndarray,
    omega: float,
    alpha: float,
    beta: float,
) -> np.ndarray:
    variances = np.empty(len(values), dtype=float)
    sample_variance = max(float(np.var(values, ddof=1)), 1e-12)
    variances[0] = sample_variance
    for index in range(1, len(values)):
        variances[index] = omega + alpha * values[index - 1] ** 2 + beta * variances[index - 1]
    return variances


def _rolling_realized_variance(history: pd.Series, window: int | None) -> float:
    if len(history) < 2:
        return np.nan
    fit_history = history if window is None else history.tail(window)
    if len(fit_history) < 2:
        return np.nan
    variance = float(np.var(fit_history.to_numpy(dtype=float), ddof=1))
    return variance if np.isfinite(variance) and variance > 0.0 else np.nan


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _logit(value: float) -> float:
    clipped = min(max(value, 1e-8), 1.0 - 1e-8)
    return math.log(clipped / (1.0 - clipped))


def _append_reason(existing: str, reason: str) -> str:
    if not existing:
        return reason
    return f"{existing};{reason}"


def _validate_numeric(value: float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a numeric, non-boolean value.")
