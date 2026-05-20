"""Dynamic allocation benchmark rules for diagnostic policy comparison.

These rules are intentionally simple. They provide transparent dominant-asset
switching baselines that can be compared against concentrated TD3 policies
without changing the existing training, reward, or benchmark pipeline.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest.evaluate_policy import summary_metrics
from src.backtest.performance_metrics import extended_summary_metrics


def compute_rolling_momentum(
    returns: pd.DataFrame,
    window: int = 12,
) -> pd.DataFrame:
    """Compute rolling compounded returns over a fixed lookback window."""
    _validate_returns(returns)
    _validate_window(window, "window")

    return (1.0 + returns).rolling(window).apply(np.prod, raw=True) - 1.0


def compute_rolling_volatility(
    returns: pd.DataFrame,
    window: int = 12,
) -> pd.DataFrame:
    """Compute rolling standard deviation over a fixed lookback window."""
    _validate_returns(returns)
    _validate_window(window, "window")

    return returns.rolling(window).std()


def build_momentum_winner_weights(
    returns: pd.DataFrame,
    window: int = 12,
    eligible_assets: list[str] | None = None,
) -> pd.DataFrame:
    """Allocate to the asset with the highest prior rolling momentum."""
    eligible = _resolve_eligible_assets(returns, eligible_assets)
    momentum = compute_rolling_momentum(returns[eligible], window)

    return _winner_weights_from_scores(momentum, returns.columns)


def build_risk_adjusted_momentum_winner_weights(
    returns: pd.DataFrame,
    momentum_window: int = 12,
    volatility_window: int = 12,
    eligible_assets: list[str] | None = None,
    volatility_floor: float = 1e-8,
) -> pd.DataFrame:
    """Allocate to the asset with the highest prior momentum-to-volatility score."""
    _validate_window(momentum_window, "momentum_window")
    _validate_window(volatility_window, "volatility_window")
    if volatility_floor <= 0.0:
        raise ValueError("volatility_floor must be positive.")

    eligible = _resolve_eligible_assets(returns, eligible_assets)
    momentum = compute_rolling_momentum(returns[eligible], momentum_window)
    volatility = compute_rolling_volatility(returns[eligible], volatility_window)
    volatility = volatility.where(volatility > volatility_floor)
    scores = momentum / volatility

    return _winner_weights_from_scores(scores, returns.columns)


def build_trend_following_spy_cash_weights(
    returns: pd.DataFrame,
    market_asset: str = "SPY",
    cash_asset: str = "CASH",
    window: int = 12,
) -> pd.DataFrame:
    """Allocate to SPY during positive prior trend and CASH otherwise."""
    _validate_required_assets(returns, [market_asset, cash_asset])
    momentum = compute_rolling_momentum(returns[[market_asset]], window)[market_asset]

    selected = pd.Series(index=returns.index, dtype=object)
    selected.loc[momentum > 0.0] = market_asset
    selected.loc[momentum <= 0.0] = cash_asset

    return _weights_from_selected_assets(selected, returns.columns)


def build_defensive_risk_off_weights(
    returns: pd.DataFrame,
    market_asset: str = "SPY",
    defensive_assets: list[str] | None = None,
    cash_asset: str = "CASH",
    window: int = 12,
) -> pd.DataFrame:
    """Allocate to SPY in positive trend, otherwise to the best defensive asset."""
    _validate_required_assets(returns, [market_asset, cash_asset])
    if defensive_assets is None:
        defensive_assets = [
            asset for asset in ["GLD", "TLT", "CASH"] if asset in returns.columns
        ]
    _validate_required_assets(returns, defensive_assets)
    if not defensive_assets:
        raise ValueError("defensive_assets must contain at least one available asset.")

    signal_assets = [market_asset, *defensive_assets]
    momentum = compute_rolling_momentum(returns[signal_assets], window)
    defensive_winner = _idxmax_excluding_all_nan(momentum[defensive_assets])

    selected = pd.Series(index=returns.index, dtype=object)
    selected.loc[momentum[market_asset] > 0.0] = market_asset
    selected.loc[momentum[market_asset] <= 0.0] = defensive_winner.loc[
        momentum[market_asset] <= 0.0
    ]

    return _weights_from_selected_assets(selected, returns.columns)


def build_rolling_risk_parity_weights(
    returns: pd.DataFrame,
    window: int = 12,
    assets: list[str] | None = None,
    include_cash: bool = False,
    min_vol: float = 1e-8,
    max_weight: float | None = None,
) -> pd.DataFrame:
    """Build signal-lagged rolling inverse-volatility risk parity weights.

    This is not full equal-risk-contribution optimization. It uses rolling
    realized volatility from prior returns and assigns weights proportional to
    inverse volatility. CASH is excluded by default; if included, its near-zero
    volatility is clipped by min_vol and can receive a large allocation.
    """
    _validate_returns(returns)
    _validate_window(window, "window")
    if min_vol <= 0.0:
        raise ValueError("min_vol must be positive.")
    if max_weight is not None and (max_weight <= 0.0 or max_weight > 1.0):
        raise ValueError("max_weight must be in the range (0, 1].")

    eligible_assets = _resolve_risk_parity_assets(returns, assets, include_cash)
    if max_weight is not None and max_weight * len(eligible_assets) < 1.0 - 1e-12:
        raise ValueError("max_weight is too small to form a fully invested portfolio.")

    rolling_volatility = compute_rolling_volatility(returns[eligible_assets], window)
    lagged_volatility = rolling_volatility.shift(1)
    weights = pd.DataFrame(0.0, index=returns.index, columns=returns.columns)
    equal_weight = pd.Series(1.0 / len(eligible_assets), index=eligible_assets)

    for date, volatility_row in lagged_volatility.iterrows():
        if volatility_row.isna().any():
            selected_weights = equal_weight
        else:
            clipped_volatility = volatility_row.clip(lower=min_vol)
            inverse_volatility = 1.0 / clipped_volatility
            selected_weights = inverse_volatility / inverse_volatility.sum()
            if max_weight is not None:
                selected_weights = _cap_and_renormalize_weights(
                    selected_weights,
                    max_weight,
                )
        weights.loc[date, eligible_assets] = selected_weights.to_numpy(dtype=float)

    return weights


def build_rolling_markowitz_weights(
    returns: pd.DataFrame,
    window: int = 52,
    assets: list[str] | None = None,
    include_cash: bool = False,
    risk_aversion: float = 1.0,
    max_weight: float = 0.60,
    min_weight: float = 0.0,
    ridge: float = 1e-6,
    use_mean_returns: bool = True,
) -> pd.DataFrame:
    """Build signal-lagged constrained rolling mean-variance weights.

    The optimizer uses only the prior rolling window for each date. If SciPy is
    unavailable, the covariance matrix is degenerate, or optimization fails for
    a date, the date falls back to rolling inverse-volatility weights.
    """
    _validate_returns(returns)
    _validate_window(window, "window")
    _validate_markowitz_parameters(
        risk_aversion=risk_aversion,
        max_weight=max_weight,
        min_weight=min_weight,
        ridge=ridge,
    )

    eligible_assets = _resolve_markowitz_assets(returns, assets, include_cash)
    _validate_weight_bounds_feasible(
        n_assets=len(eligible_assets),
        min_weight=min_weight,
        max_weight=max_weight,
    )
    weights = pd.DataFrame(0.0, index=returns.index, columns=returns.columns)
    equal_weight = pd.Series(1.0 / len(eligible_assets), index=eligible_assets)

    for row_number, date in enumerate(returns.index):
        if row_number < window:
            selected_weights = equal_weight
        else:
            window_returns = returns[eligible_assets].iloc[row_number - window:row_number]
            selected_weights = _solve_markowitz_window(
                window_returns=window_returns,
                risk_aversion=risk_aversion,
                max_weight=max_weight,
                min_weight=min_weight,
                ridge=ridge,
                use_mean_returns=use_mean_returns,
            )
        weights.loc[date, eligible_assets] = selected_weights.to_numpy(dtype=float)

    return weights


def evaluate_weight_strategy(
    returns: pd.DataFrame,
    weights: pd.DataFrame,
    transaction_cost: float = 0.001,
    initial_value: float = 100000,
    initial_weights: pd.Series | None = None,
) -> dict:
    """Evaluate a precomputed weight strategy with simple turnover costs."""
    _validate_returns(returns)
    if weights.empty:
        raise ValueError("weights must not be empty.")
    if not isinstance(weights.index, pd.DatetimeIndex):
        raise TypeError("weights index must be a DatetimeIndex.")
    if transaction_cost < 0.0 or transaction_cost >= 1.0:
        raise ValueError("transaction_cost must be greater than or equal to 0 and less than 1.")
    if initial_value <= 0.0:
        raise ValueError("initial_value must be positive.")
    _validate_required_assets(weights, list(returns.columns))

    aligned_index = returns.index.intersection(weights.index)
    if aligned_index.empty:
        raise ValueError("returns and weights must share at least one date.")

    aligned_returns = returns.loc[aligned_index]
    aligned_weights = weights.loc[aligned_index, returns.columns]
    previous_weights = _prepare_initial_weights(aligned_returns, initial_weights)

    gross_returns = (aligned_weights * aligned_returns).sum(axis=1)
    turnover_values = []
    for _, current_weights in aligned_weights.iterrows():
        turnover_values.append(
            float((current_weights - previous_weights).abs().sum())
        )
        previous_weights = current_weights
    turnover = pd.Series(turnover_values, index=aligned_index, name="turnover")
    transaction_costs = transaction_cost * turnover
    net_returns = gross_returns - transaction_costs

    portfolio_value = initial_value * (1.0 + net_returns).cumprod()
    drawdown = portfolio_value / portfolio_value.cummax() - 1.0

    history = pd.DataFrame(
        {
            "date": aligned_index,
            "portfolio_return": gross_returns.to_numpy(dtype=float),
            "financial_net_return": net_returns.to_numpy(dtype=float),
            "gross_return": gross_returns.to_numpy(dtype=float),
            "net_return": net_returns.to_numpy(dtype=float),
            "portfolio_value": portfolio_value.to_numpy(dtype=float),
            "drawdown": drawdown.to_numpy(dtype=float),
            "turnover": turnover.to_numpy(dtype=float),
            "transaction_cost": transaction_costs.to_numpy(dtype=float),
        },
        index=aligned_index,
    )
    for asset in returns.columns:
        history[f"weight_{asset}"] = aligned_weights[asset].to_numpy(dtype=float)

    base_metrics = summary_metrics(net_returns)
    risk_adjusted_metrics = extended_summary_metrics(net_returns)
    sortino_value = risk_adjusted_metrics["sortino_ratio"]
    calmar_value = risk_adjusted_metrics["calmar_ratio"]
    max_drawdown_value = base_metrics["max_drawdown"]

    return {
        "history": history.reset_index(drop=True),
        "returns": net_returns.rename("net_return"),
        "final_value": float(portfolio_value.iloc[-1]),
        "cumulative_return": base_metrics["cumulative_return"],
        "annualized_return": base_metrics["annualized_return"],
        "annualized_volatility": base_metrics["annualized_volatility"],
        "sharpe_ratio": base_metrics["sharpe_ratio"],
        "sortino_ratio": sortino_value,
        "calmar_ratio": calmar_value,
        "max_drawdown": max_drawdown_value,
        "average_turnover": float(turnover.mean()),
        "sortino_ratio_is_finite": bool(np.isfinite(sortino_value)),
        "sortino_ratio_is_extreme": bool(
            not np.isfinite(sortino_value) or abs(sortino_value) > 10.0
        ),
        "calmar_ratio_is_finite": bool(np.isfinite(calmar_value)),
        "calmar_ratio_is_infinite": bool(np.isinf(calmar_value)),
        "max_drawdown_is_zero": bool(max_drawdown_value == 0.0),
    }


def build_dynamic_benchmark_suite(
    returns: pd.DataFrame,
    transaction_cost: float = 0.001,
    initial_value: float = 100000,
    momentum_window: int = 12,
    volatility_window: int = 12,
    markowitz_window: int = 52,
    market_asset: str = "SPY",
    cash_asset: str = "CASH",
) -> dict:
    """Build and evaluate the default dynamic benchmark rule suite."""
    benchmark_builders = {
        f"momentum_winner_{momentum_window}p": lambda: build_momentum_winner_weights(
            returns,
            window=momentum_window,
        ),
        (
            f"risk_adjusted_momentum_winner_"
            f"{momentum_window}p_{volatility_window}p"
        ): lambda: build_risk_adjusted_momentum_winner_weights(
            returns,
            momentum_window=momentum_window,
            volatility_window=volatility_window,
        ),
        f"trend_spy_cash_{momentum_window}p": lambda: build_trend_following_spy_cash_weights(
            returns,
            market_asset=market_asset,
            cash_asset=cash_asset,
            window=momentum_window,
        ),
        f"defensive_risk_off_{momentum_window}p": lambda: build_defensive_risk_off_weights(
            returns,
            market_asset=market_asset,
            cash_asset=cash_asset,
            window=momentum_window,
        ),
        f"rolling_risk_parity_inverse_vol_{volatility_window}p": lambda: (
            build_rolling_risk_parity_weights(
                returns,
                window=volatility_window,
                include_cash=False,
            )
        ),
        f"rolling_markowitz_long_only_{markowitz_window}p": lambda: (
            build_rolling_markowitz_weights(
                returns,
                window=markowitz_window,
                include_cash=False,
            )
        ),
    }

    suite = {}
    for benchmark_name, build_weights in benchmark_builders.items():
        try:
            weights = build_weights()
            evaluation = evaluate_weight_strategy(
                returns,
                weights,
                transaction_cost=transaction_cost,
                initial_value=initial_value,
            )
        except (KeyError, ValueError):
            continue

        suite[benchmark_name] = {
            "weights": weights,
            **evaluation,
        }

    return suite


def build_benchmark_timing_audit_summary() -> pd.DataFrame:
    """Return the benchmark timing and cost conventions used for protocol review."""
    rows = [
        _static_audit_row(
            "Equal_Weight",
            turnover_convention="gross row-wise equal weights; turnover not modeled",
            transaction_cost_convention="none in equal_weight_returns",
            comparable_with_td3=False,
        ),
        _static_audit_row(
            "Equal_Weight_Risky",
            turnover_convention="requires explicit weight-strategy evaluation",
            transaction_cost_convention="depends on caller",
            comparable_with_td3=False,
        ),
        _static_audit_row(
            "60_40_SPY_TLT",
            turnover_convention="requires explicit weight-strategy evaluation",
            transaction_cost_convention="depends on caller",
            comparable_with_td3=False,
        ),
    ]
    for asset in ("GLD", "SPY", "TLT", "BTC-USD"):
        rows.append(
            _static_audit_row(
                f"BuyHold_{asset}",
                turnover_convention="gross asset return; turnover not modeled",
                transaction_cost_convention="none in individual_buy_and_hold_returns",
                comparable_with_td3=False,
            )
        )

    for benchmark_name in (
        "momentum_winner_12p",
        "risk_adjusted_momentum_winner_12p_12p",
        "trend_spy_cash_12p",
        "defensive_risk_off_12p",
        "rolling_risk_parity_inverse_vol_12p",
        "rolling_markowitz_long_only_52p",
    ):
        rows.append(
            {
                "benchmark_name": benchmark_name,
                "signal_lagged": True,
                "applies_return_t_after_weight_t": True,
                "turnover_convention": (
                    "sum(abs(weight_t - previous_weight)); previous weights "
                    "default to equal weight as in PortfolioEnv"
                ),
                "transaction_cost_convention": (
                    "financial_net_return_t = portfolio_return_t - "
                    "transaction_cost_rate * turnover_t"
                ),
                "comparable_with_td3": True,
            }
        )

    return pd.DataFrame(rows)


def save_dynamic_benchmark_suite(
    benchmark_suite: dict,
    output_dir: str,
) -> dict:
    """Save dynamic benchmark histories and an aggregate summary table."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    paths = {}
    for benchmark_name, benchmark_result in benchmark_suite.items():
        history_path = destination / f"{benchmark_name}_history.csv"
        benchmark_result["history"].to_csv(history_path, index=False)
        paths[f"{benchmark_name}_history"] = str(history_path)
        summary_rows.append(
            {
                "benchmark_name": benchmark_name,
                "final_value": benchmark_result["final_value"],
                "cumulative_return": benchmark_result["cumulative_return"],
                "annualized_return": benchmark_result["annualized_return"],
                "annualized_volatility": benchmark_result["annualized_volatility"],
                "sharpe_ratio": benchmark_result["sharpe_ratio"],
                "sortino_ratio": benchmark_result["sortino_ratio"],
                "calmar_ratio": benchmark_result["calmar_ratio"],
                "max_drawdown": benchmark_result["max_drawdown"],
                "average_turnover": benchmark_result["average_turnover"],
                "sortino_ratio_is_finite": benchmark_result["sortino_ratio_is_finite"],
                "sortino_ratio_is_extreme": benchmark_result["sortino_ratio_is_extreme"],
                "calmar_ratio_is_finite": benchmark_result["calmar_ratio_is_finite"],
                "calmar_ratio_is_infinite": benchmark_result["calmar_ratio_is_infinite"],
                "max_drawdown_is_zero": benchmark_result["max_drawdown_is_zero"],
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary_path = destination / "dynamic_benchmark_summary.csv"
    summary.to_csv(summary_path, index=False)
    paths["summary"] = str(summary_path)

    return paths


def _winner_weights_from_scores(
    scores: pd.DataFrame,
    all_assets: pd.Index,
) -> pd.DataFrame:
    selected = _idxmax_excluding_all_nan(scores)

    return _weights_from_selected_assets(selected, all_assets)


def _idxmax_excluding_all_nan(scores: pd.DataFrame) -> pd.Series:
    selected = pd.Series(np.nan, index=scores.index, dtype=object)
    valid_rows = ~scores.isna().all(axis=1)
    selected.loc[valid_rows] = scores.loc[valid_rows].idxmax(axis=1)

    return selected


def _weights_from_selected_assets(
    selected_assets: pd.Series,
    all_assets: pd.Index,
) -> pd.DataFrame:
    shifted_selection = selected_assets.shift(1).dropna()
    weights = pd.DataFrame(0.0, index=shifted_selection.index, columns=all_assets)
    for date, asset in shifted_selection.items():
        weights.loc[date, asset] = 1.0

    return weights


def _resolve_eligible_assets(
    returns: pd.DataFrame,
    eligible_assets: list[str] | None,
) -> list[str]:
    _validate_returns(returns)
    if eligible_assets is None:
        return list(returns.columns)

    _validate_required_assets(returns, eligible_assets)
    if not eligible_assets:
        raise ValueError("eligible_assets must contain at least one asset.")

    return eligible_assets


def _resolve_risk_parity_assets(
    returns: pd.DataFrame,
    assets: list[str] | None,
    include_cash: bool,
) -> list[str]:
    if assets is None:
        resolved_assets = list(returns.columns)
        if not include_cash:
            resolved_assets = [asset for asset in resolved_assets if asset != "CASH"]
    else:
        _validate_required_assets(returns, assets)
        if not include_cash and "CASH" in assets:
            raise ValueError("assets must not include CASH unless include_cash is True.")
        resolved_assets = list(assets)

    if not resolved_assets:
        raise ValueError("risk parity requires at least one eligible asset.")

    return resolved_assets


def _resolve_markowitz_assets(
    returns: pd.DataFrame,
    assets: list[str] | None,
    include_cash: bool,
) -> list[str]:
    if assets is None:
        resolved_assets = list(returns.columns)
        if not include_cash:
            resolved_assets = [asset for asset in resolved_assets if asset != "CASH"]
    else:
        _validate_required_assets(returns, assets)
        if not include_cash and "CASH" in assets:
            raise ValueError("assets must not include CASH unless include_cash is True.")
        resolved_assets = list(assets)

    if not resolved_assets:
        raise ValueError("Markowitz requires at least one eligible asset.")

    return resolved_assets


def _validate_required_assets(
    data: pd.DataFrame,
    required_assets: list[str],
) -> None:
    missing_assets = [asset for asset in required_assets if asset not in data.columns]
    if missing_assets:
        raise ValueError(f"Missing required assets: {missing_assets}")


def _prepare_initial_weights(
    returns: pd.DataFrame,
    initial_weights: pd.Series | None,
) -> pd.Series:
    if initial_weights is None:
        return pd.Series(1.0 / len(returns.columns), index=returns.columns)

    weights = initial_weights.reindex(returns.columns)
    if weights.isna().any():
        missing_assets = weights[weights.isna()].index.tolist()
        raise KeyError(f"Missing initial weights for assets: {missing_assets}")
    if (weights < 0.0).any():
        raise ValueError("initial_weights must be non-negative.")
    if not np.isclose(float(weights.sum()), 1.0):
        raise ValueError("initial_weights must sum to 1.")

    return weights.astype(float)


def _cap_and_renormalize_weights(
    weights: pd.Series,
    max_weight: float,
) -> pd.Series:
    capped = weights.copy().astype(float)
    free_assets = pd.Series(True, index=capped.index)

    while True:
        over_cap = (capped > max_weight) & free_assets
        if not over_cap.any():
            break
        capped.loc[over_cap] = max_weight
        free_assets.loc[over_cap] = False
        remaining_weight = 1.0 - float(capped.loc[~free_assets].sum())
        if remaining_weight <= 0.0 or not free_assets.any():
            break
        free_original = weights.loc[free_assets]
        capped.loc[free_assets] = (
            free_original / free_original.sum() * remaining_weight
        )

    total_weight = float(capped.sum())
    if total_weight <= 0.0:
        raise ValueError("capped weights must have positive total weight.")

    return capped / total_weight


def _validate_markowitz_parameters(
    risk_aversion: float,
    max_weight: float,
    min_weight: float,
    ridge: float,
) -> None:
    if not _is_number(risk_aversion) or risk_aversion < 0.0:
        raise ValueError("risk_aversion must be a non-negative number.")
    if not _is_number(min_weight) or min_weight < 0.0:
        raise ValueError("min_weight must be a non-negative number.")
    if not _is_number(max_weight) or max_weight <= 0.0 or max_weight > 1.0:
        raise ValueError("max_weight must be in the range (0, 1].")
    if min_weight > max_weight:
        raise ValueError("min_weight must be less than or equal to max_weight.")
    if not _is_number(ridge) or ridge < 0.0:
        raise ValueError("ridge must be a non-negative number.")


def _validate_weight_bounds_feasible(
    n_assets: int,
    min_weight: float,
    max_weight: float,
) -> None:
    if min_weight * n_assets > 1.0 + 1e-12:
        raise ValueError("min_weight is too large to form a fully invested portfolio.")
    if max_weight * n_assets < 1.0 - 1e-12:
        raise ValueError("max_weight is too small to form a fully invested portfolio.")


def _solve_markowitz_window(
    window_returns: pd.DataFrame,
    risk_aversion: float,
    max_weight: float,
    min_weight: float,
    ridge: float,
    use_mean_returns: bool,
) -> pd.Series:
    fallback = _inverse_volatility_weights_from_window(
        window_returns,
        min_vol=max(ridge, 1e-8),
        min_weight=min_weight,
        max_weight=max_weight,
    )
    if window_returns.isna().any().any():
        return fallback

    covariance = window_returns.cov().to_numpy(dtype=float)
    covariance = np.nan_to_num(covariance, nan=0.0, posinf=0.0, neginf=0.0)
    covariance = covariance + np.eye(len(window_returns.columns)) * ridge
    if not np.isfinite(covariance).all():
        return fallback

    mean_returns = (
        window_returns.mean().to_numpy(dtype=float)
        if use_mean_returns
        else np.zeros(len(window_returns.columns), dtype=float)
    )
    if not np.isfinite(mean_returns).all():
        return fallback

    try:
        from scipy.optimize import minimize
    except ImportError:
        return fallback

    n_assets = len(window_returns.columns)
    initial_weights = np.full(n_assets, 1.0 / n_assets, dtype=float)
    bounds = [(min_weight, max_weight) for _ in range(n_assets)]
    constraints = ({"type": "eq", "fun": lambda weights: np.sum(weights) - 1.0},)

    def objective(weights: np.ndarray) -> float:
        portfolio_mean = float(np.dot(mean_returns, weights))
        portfolio_variance = float(weights @ covariance @ weights)
        return -(portfolio_mean - risk_aversion * portfolio_variance)

    result = minimize(
        objective,
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 200, "ftol": 1e-12, "disp": False},
    )
    if not result.success or not np.isfinite(result.x).all():
        return fallback

    optimized = pd.Series(result.x, index=window_returns.columns, dtype=float)
    optimized = optimized.clip(lower=min_weight, upper=max_weight)
    total_weight = float(optimized.sum())
    if total_weight <= 0.0:
        return fallback
    optimized = optimized / total_weight
    if optimized.max() > max_weight + 1e-8 or optimized.min() < min_weight - 1e-8:
        return fallback

    return optimized


def _inverse_volatility_weights_from_window(
    window_returns: pd.DataFrame,
    min_vol: float,
    min_weight: float,
    max_weight: float,
) -> pd.Series:
    volatility = window_returns.std().clip(lower=min_vol)
    inverse_volatility = 1.0 / volatility
    weights = inverse_volatility / inverse_volatility.sum()
    if min_weight > 0.0:
        weights = weights.clip(lower=min_weight)
        weights = weights / weights.sum()
    if weights.max() > max_weight:
        weights = _cap_and_renormalize_weights(weights, max_weight)

    return weights


def _static_audit_row(
    benchmark_name: str,
    turnover_convention: str,
    transaction_cost_convention: str,
    comparable_with_td3: bool,
) -> dict:
    return {
        "benchmark_name": benchmark_name,
        "signal_lagged": True,
        "applies_return_t_after_weight_t": True,
        "turnover_convention": turnover_convention,
        "transaction_cost_convention": transaction_cost_convention,
        "comparable_with_td3": comparable_with_td3,
    }


def _validate_returns(returns: pd.DataFrame) -> None:
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns must be a pandas DataFrame.")
    if returns.empty:
        raise ValueError("returns must not be empty.")
    if not isinstance(returns.index, pd.DatetimeIndex):
        raise TypeError("returns index must be a DatetimeIndex.")


def _validate_window(window: int, field_name: str) -> None:
    if not isinstance(window, int) or window < 2:
        raise ValueError(f"{field_name} must be an integer greater than or equal to 2.")


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)
