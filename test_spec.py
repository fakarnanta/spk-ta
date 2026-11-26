import numpy as np

from forecast_metrics import spec_score


def spec_score_manual(actual, predicted, stock_cost=0.25, lost_cost=1.0):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    total = 0.0
    n = len(actual)
    for i in range(n):
        for t in range(n):
            if i <= t:
                weight = t - i + 1
                cumulative_actual_at_i = np.sum(actual[:i + 1])
                cumulative_forecast_at_t = np.sum(predicted[:t + 1])
                cumulative_forecast_at_i = np.sum(predicted[:i + 1])
                cumulative_actual_at_t = np.sum(actual[:t + 1])
                lost = min(actual[i], cumulative_actual_at_i - cumulative_forecast_at_t)
                stock = min(predicted[i], cumulative_forecast_at_i - cumulative_actual_at_t)
                total += weight * (lost_cost * max(lost, 0.0) + stock_cost * max(stock, 0.0))
    return total / n


def spec_score_wrong_axis(actual, predicted, stock_cost=0.25):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    cy = np.cumsum(actual)
    cf = np.cumsum(predicted)
    i, t = np.arange(len(actual))[:, None], np.arange(len(actual))[None, :]
    w = (t - i + 1) * (i <= t)
    lost = np.clip(np.minimum(actual[:, None], cy[:, None] - cf[None, :]), 0, None)
    hold = np.clip(np.minimum(predicted[:, None], cf[None, :] - cy[None, :]), 0, None)
    return float((w * (lost + stock_cost * hold)).sum() / len(actual))


def test_spec_correct_excess_stock_axis_matches_manual_calculation():
    actual = np.array([0, 0, 1], dtype=float)
    predicted = np.array([0, 1, 1], dtype=float)

    reference = spec_score_manual(actual, predicted)
    corrected = spec_score(actual, predicted)
    old = spec_score_wrong_axis(actual, predicted)

    assert corrected == reference
    assert old != corrected


def calculate_reorder_point_pure(sales, lead_time, service_level, future_forecast):
    sales = np.asarray(sales, dtype=float)
    lt_demand = np.convolve(sales, np.ones(lead_time, dtype=float), "valid")
    empirical_rop = float(np.quantile(lt_demand, service_level)) if len(lt_demand) else 0.0
    fc = np.asarray(future_forecast, dtype=float)
    if len(fc) < lead_time:
        fc = np.pad(fc, (0, lead_time - len(fc)), mode="edge")
    expected_lt_demand = float(fc[:lead_time].sum())
    reorder_point = max(empirical_rop, expected_lt_demand)
    return reorder_point, max(reorder_point - expected_lt_demand, 0.0)


def test_reorder_point_pads_short_forecast_to_lead_time():
    sparse_sales = np.zeros(20, dtype=float)
    reorder_point, safety_stock = calculate_reorder_point_pure(
        sparse_sales,
        lead_time=3,
        service_level=0.80,
        future_forecast=np.array([5.0]),
    )

    assert reorder_point == 15.0
    assert safety_stock == 0.0


def test_reorder_point_respects_forecast_floor():
    sparse_sales = np.zeros(20, dtype=float)
    reorder_point, safety_stock = calculate_reorder_point_pure(
        sparse_sales,
        lead_time=2,
        service_level=0.80,
        future_forecast=np.array([5.0, 5.0]),
    )

    assert reorder_point == 10.0
    assert safety_stock == 0.0


if __name__ == "__main__":
    test_spec_correct_excess_stock_axis_matches_manual_calculation()
    test_reorder_point_pads_short_forecast_to_lead_time()
    test_reorder_point_respects_forecast_floor()
    print("SPEC and reorder-point tests passed")
