"""Validated baseline forecast of daily activity.

Why this exists. Gate 4 of the competition PDF asks "Did you use forecasting or
clustering?". Clustering was considered and rejected: the overdispersion test
found no merchant-level structure, so a clustering algorithm would partition
sampling noise into confident-looking groups. A forecast *can* be validated
honestly, so that is what this module adds.

Method. Four deliberately simple methods are compared on a time-ordered holdout
(the last 28 of 90 days), never on data they were fitted to:

    mean            the training mean
    naive           the last training value
    seasonal_naive  the same weekday one week earlier
    linear_trend    ordinary least squares on the day index

The method with the lowest holdout MAE is refitted on all 90 days and projected
14 days ahead, with a 95% band of ±1.96 residual standard deviations. The band
is itself validated: the report states what share of holdout days it contained.

Two diagnostics accompany every series, so the forecast is read in context:
a slope t-test for trend, and a chi-square test for a day-of-week pattern.

Limitations, stated rather than hidden: 90 days is a short history, the data is
synthetic, and a projection beyond the observed window is a capacity-planning
baseline — not a prediction of fraud or of any individual transaction.
"""
from __future__ import annotations

from math import erfc, sqrt

import numpy as np
import pandas as pd

from src.analytics.shrinkage import chi_square_sf

HOLDOUT_DAYS = 28
HORIZON_DAYS = 14
SEASON = 7
Z95 = 1.96

SERIES = {
    "transactions": ("Transactions per day", "count"),
    "disputed_transactions": ("Disputed transactions per day", "count"),
    "transaction_value": ("Transaction value per day", "INR"),
}
METHODS = ("mean", "naive", "seasonal_naive", "linear_trend")
METHOD_LABELS = {
    "mean": "flat mean",
    "naive": "last value carried forward",
    "seasonal_naive": "same weekday last week",
    "linear_trend": "linear trend",
}


def predict(method: str, history: np.ndarray, horizon: int) -> np.ndarray:
    n = len(history)
    if method == "mean":
        return np.full(horizon, float(history.mean()))
    if method == "naive":
        return np.full(horizon, float(history[-1]))
    if method == "seasonal_naive":
        return np.array([history[n - SEASON + (i % SEASON)] for i in range(horizon)], dtype=float)
    if method == "linear_trend":
        x = np.arange(n)
        slope, intercept = np.polyfit(x, history, 1)
        return intercept + slope * np.arange(n, n + horizon)
    raise ValueError(f"unknown forecasting method {method!r}")


def residual_sd(method: str, history: np.ndarray) -> float:
    """In-sample one-step residual spread for a method, used for the band."""
    n = len(history)
    if method == "mean":
        resid = history - history.mean()
    elif method == "naive":
        resid = np.diff(history)
    elif method == "seasonal_naive":
        resid = history[SEASON:] - history[:-SEASON]
    else:
        x = np.arange(n)
        slope, intercept = np.polyfit(x, history, 1)
        resid = history - (intercept + slope * x)
    return float(np.std(resid, ddof=1)) if len(resid) > 1 else float("nan")


def errors(actual: np.ndarray, pred: np.ndarray) -> tuple[float, float, float]:
    err = actual - pred
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    nonzero = actual != 0
    mape = float(np.mean(np.abs(err[nonzero] / actual[nonzero])) * 100) if nonzero.any() else float("nan")
    return mae, rmse, mape


def trend_test(values: np.ndarray) -> tuple[float, float, float]:
    """OLS slope per day, its t statistic and a two-sided normal-approximation p."""
    n = len(values)
    x = np.arange(n)
    slope, intercept = np.polyfit(x, values, 1)
    resid = values - (intercept + slope * x)
    denom = float(np.sum((x - x.mean()) ** 2))
    se = sqrt(float(np.sum(resid ** 2)) / (n - 2) / denom) if n > 2 and denom > 0 else 0.0
    t = slope / se if se > 0 else 0.0
    return float(slope), float(t), float(min(1.0, erfc(abs(t) / sqrt(2))))


def weekday_test(dates: pd.Series, values: np.ndarray) -> tuple[float, int, float]:
    """Chi-square goodness of fit of weekday totals against weekday day-counts.

    Only meaningful for count series; value series return NaN.
    """
    frame = pd.DataFrame({"weekday": pd.to_datetime(dates).dt.dayofweek, "v": values})
    observed = frame.groupby("weekday")["v"].sum()
    days = frame.groupby("weekday").size()
    expected = values.sum() * days / days.sum()
    chi = float(((observed - expected) ** 2 / expected).sum())
    dof = len(observed) - 1
    return chi, dof, chi_square_sf(chi, dof)


def build(agg_daily: pd.DataFrame) -> dict[str, pd.DataFrame]:
    daily = agg_daily.copy()
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values("date").reset_index(drop=True)
    if len(daily) <= HOLDOUT_DAYS + SEASON:
        raise ValueError("not enough days of history to backtest a forecast")

    backtests, forecasts, diagnostics = [], [], []
    for series, (label, unit) in SERIES.items():
        values = daily[series].astype(float).to_numpy()
        dates = daily["date"]
        train, test = values[:-HOLDOUT_DAYS], values[-HOLDOUT_DAYS:]

        results = {}
        for method in METHODS:
            pred = predict(method, train, HOLDOUT_DAYS)
            mae, rmse, mape = errors(test, pred)
            sd = residual_sd(method, train)
            coverage = float(np.mean(np.abs(test - pred) <= Z95 * sd) * 100)
            results[method] = (mae, rmse, mape, coverage)

        selected = min(METHODS, key=lambda m: (results[m][0], METHODS.index(m)))
        for method in METHODS:
            mae, rmse, mape, coverage = results[method]
            backtests.append({
                "series": series, "label": label, "method": method,
                "method_label": METHOD_LABELS[method], "mae": round(mae, 3),
                "rmse": round(rmse, 3), "mape_pct": round(mape, 3),
                "holdout_band_coverage_pct": round(coverage, 1),
                "selected": method == selected,
            })

        sd_full = residual_sd(selected, values)
        future = predict(selected, values, HORIZON_DAYS)
        lower = np.maximum(future - Z95 * sd_full, 0.0)
        upper = future + Z95 * sd_full
        future_dates = pd.date_range(dates.iloc[-1] + pd.Timedelta(days=1),
                                     periods=HORIZON_DAYS, freq="D")

        for d, v in zip(dates, values):
            forecasts.append({"series": series, "date": d, "kind": "actual", "value": v,
                              "lower_95": np.nan, "upper_95": np.nan, "method": selected})
        for d, v, lo, hi in zip(future_dates, future, lower, upper):
            forecasts.append({"series": series, "date": d, "kind": "forecast",
                              "value": round(float(v), 3), "lower_95": round(float(lo), 3),
                              "upper_95": round(float(hi), 3), "method": selected})

        slope, t, slope_p = trend_test(values)
        if unit == "count":
            chi, dof, weekday_p = weekday_test(dates, values)
        else:
            chi, dof, weekday_p = float("nan"), 0, float("nan")
        mean = float(values.mean())
        sd = float(values.std(ddof=1))
        diagnostics.append({
            "series": series, "label": label, "unit": unit,
            "history_days": len(values), "train_days": len(train),
            "holdout_days": HOLDOUT_DAYS, "horizon_days": HORIZON_DAYS,
            "first_date": dates.iloc[0], "last_date": dates.iloc[-1],
            "mean": round(mean, 3), "sd": round(sd, 3),
            "cv": round(sd / mean, 4) if mean else float("nan"),
            "slope_per_day": round(slope, 4), "slope_t": round(t, 3),
            "slope_p": round(slope_p, 4),
            "weekday_chi2": round(chi, 3) if chi == chi else float("nan"),
            "weekday_df": dof, "weekday_p": round(weekday_p, 4) if weekday_p == weekday_p else float("nan"),
            "selected_method": selected, "selected_method_label": METHOD_LABELS[selected],
            "selected_mae": round(results[selected][0], 3),
            "mean_baseline_mae": round(results["mean"][0], 3),
            "holdout_band_coverage_pct": round(results[selected][3], 1),
            "forecast_mean": round(float(future.mean()), 3),
            "band_lower_mean": round(float(lower.mean()), 3),
            "band_upper_mean": round(float(upper.mean()), 3),
        })

    return {
        "agg_forecast": pd.DataFrame(forecasts),
        "agg_forecast_backtest": pd.DataFrame(backtests),
        "agg_forecast_diagnostics": pd.DataFrame(diagnostics),
    }
