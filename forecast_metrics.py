import numpy as np


def spec_score(actual, predicted, stock_cost=0.25, lost_cost=1.0):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if actual.size == 0 or actual.size != predicted.size:
        return float("nan")

    cumulative_actual = np.cumsum(actual)
    cumulative_forecast = np.cumsum(predicted)
    horizon, period = np.arange(actual.size)[:, None], np.arange(actual.size)[None, :]
    weights = (period - horizon + 1) * (horizon <= period)

    lost_sales = np.clip(
        np.minimum(actual[:, None], cumulative_actual[:, None] - cumulative_forecast[None, :]),
        0,
        None,
    )
    excess_stock = np.clip(
        np.minimum(predicted[:, None], cumulative_forecast[:, None] - cumulative_actual[None, :]),
        0,
        None,
    )
    return float((weights * (lost_cost * lost_sales + stock_cost * excess_stock)).sum() / actual.size)
