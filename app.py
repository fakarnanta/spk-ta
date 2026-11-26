import math
from itertools import product
from statistics import NormalDist
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

from forecast_metrics import spec_score

st.set_page_config(page_title="Forecast Penjualan", page_icon="📈", layout="wide")

def show_df(obj, **kwargs):
    try:
        return st.dataframe(obj, width="stretch", hide_index=True, **kwargs)
    except TypeError:
        return st.dataframe(obj, use_container_width=True, hide_index=True, **kwargs)


def show_button(label, **kwargs):
    try:
        return st.button(label, width="stretch", **kwargs)
    except TypeError:
        return st.button(label, use_container_width=True, **kwargs)


def show_download(label, data, file_name, mime, **kwargs):
    try:
        return st.download_button(label, data=data, file_name=file_name, mime=mime, width="stretch", **kwargs)
    except TypeError:
        return st.download_button(label, data=data, file_name=file_name, mime=mime, use_container_width=True, **kwargs)


def show_chart(fig):
    try:
        return st.plotly_chart(fig, width="stretch")
    except TypeError:
        return st.plotly_chart(fig, use_container_width=True)


def shos_states(values, alpha=0.2, beta=0.1, k=0.1):
    values = np.asarray(values, dtype=float)
    count = len(values)
    positive = values > 0
    positive_mean = float(values[positive].mean()) if positive.any() else 0.0
    initial_count = min(3, count)
    initial_positive = int(positive[:initial_count].sum())
    q = np.zeros(count + 1, dtype=float)
    z = np.zeros(count + 1, dtype=float)
    q[0] = initial_positive / initial_count if initial_count else 0.0
    z[0] = float(values[:initial_count][positive[:initial_count]].mean()) if initial_positive else positive_mean

    observed_positive = 0
    for index in range(1, count + 1):
        demand = values[index - 1]
        is_positive = demand > 0
        observed_positive += int(is_positive)
        sparsity_scale = 1.0 - k * (1.0 - observed_positive / index)
        adjusted_alpha = alpha * sparsity_scale
        adjusted_beta = beta * sparsity_scale
        q[index] = adjusted_beta * float(is_positive) + (1.0 - adjusted_beta) * q[index - 1]
        z[index] = (
            adjusted_alpha * demand + (1.0 - adjusted_alpha) * z[index - 1]
            if is_positive else z[index - 1]
        )
    return q, z


def tsb_forecast(history, alpha=0.2, beta=0.1, k=0.0):
    q, z = shos_states(history, alpha=alpha, beta=beta, k=k)
    return max(float(q[-1] * z[-1]), 0.0)


def croston_sba_forecast(history, alpha=0.1):
    history = np.asarray(history, dtype=float)
    positive_indices = np.flatnonzero(history > 0)
    if not len(positive_indices):
        return 0.0

    first_index = int(positive_indices[0])
    demand_size = float(history[first_index])
    interval = float(first_index + 1)
    last_occurrence = first_index
    for index in positive_indices[1:]:
        index = int(index)
        observed_interval = float(index - last_occurrence)
        demand_size = (1.0 - alpha) * demand_size + alpha * float(history[index])
        interval = (1.0 - alpha) * interval + alpha * observed_interval
        last_occurrence = index
    return max((1.0 - alpha / 2.0) * demand_size / max(interval, 1e-9), 0.0)


CONFIG_VERSION = "2026-11-25-dynamic-exog-hurdle-shos"


@st.cache_data(show_spinner=False, max_entries=64)
def compute_product(
    sku,
    values,
    promos,
    horizon,
    heavy,
    cfg_hash,
    next_promo=0.0,
    extra_cols=(),
    extra_values=(),
):
    sales = np.asarray(values, dtype=float)
    promotion_values = np.asarray(promos, dtype=float)
    extra_df = (
        pd.DataFrame(list(extra_values), columns=list(extra_cols), dtype=float)
        if extra_cols
        else None
    )
    return fit_predict_series(sales, promotion_values, horizon, heavy, next_promo, extra_df=extra_df)


def tune_smoothing_parameters(values, train_end, use_shos, heavy=False):
    values = np.asarray(values, dtype=float)
    step = 0.05 if heavy else 0.1
    alpha_values = np.round(np.arange(0.1, 0.9001, step), 2)
    beta_values = np.round(np.arange(0.1, 0.9001, step), 2)
    k_values = np.round(np.arange(0.0, 0.2001, step), 2) if use_shos else [0.0]
    validation_start = max(3, train_end - max(1, math.ceil(train_end * 0.2)))
    validation_periods = range(validation_start, train_end)
    if not validation_periods:
        validation_periods = range(max(1, train_end - 1), train_end)

    best = None
    for alpha, beta, k in product(alpha_values, beta_values, k_values):
        q, z = shos_states(values[:train_end], alpha, beta, k)
        forecast = q * z
        score = spec_score(
            values[validation_start:train_end],
            forecast[validation_start:train_end],
        )
        if best is None or score < best[0]:
            best = (score, float(alpha), float(beta), float(k))
    return {"alpha": best[1], "beta": best[2], "k": best[3], "SPEC": best[0]}


def make_features(
    history,
    promotion=0.0,
    use_shos=True,
    shos_params=None,
    shos_state=None,
    event_history=None,
    periods_ahead=0,
    extra_features=None,
):
    history = np.asarray(history, dtype=float)
    observed_history = np.asarray(event_history if event_history is not None else history, dtype=float)
    features = {f"lag_{lag}": float(history[-lag]) for lag in (1, 2, 3)}
    features["mean_history"] = float(history.mean())
    features["zero_fraction_history"] = float((observed_history == 0).mean())
    positive_positions = np.flatnonzero(observed_history > 0)
    since_last = len(observed_history) if not len(positive_positions) else len(observed_history) - 1 - int(positive_positions[-1])
    features["since_last"] = float(since_last + periods_ahead)
    recent_events = np.r_[observed_history[-3:], np.zeros(periods_ahead, dtype=float)][-3:]
    features["nz_roll_3"] = float(np.count_nonzero(recent_events > 0))
    if use_shos:
        if shos_state is not None:
            q, z = shos_state
            if np.isscalar(q) and np.isscalar(z):
                features["shos_q"] = float(q)
                features["shos_z"] = float(z)
            else:
                period = len(history)
                features["shos_q"] = float(q[period])
                features["shos_z"] = float(z[period])
        else:
            alpha, beta, k = shos_params or (0.2, 0.1, 0.1)
            q, z = shos_states(history, alpha=alpha, beta=beta, k=k)
            features["shos_q"] = float(q[-1])
            features["shos_z"] = float(z[-1])
    features["promosi"] = float(promotion)
    if extra_features:
        for key, value in extra_features.items():
            features[key] = 0.0 if value is None or pd.isna(value) else float(value)
    return features


def build_training_data(sales, promotions, use_shos=True, shos_params=None, extra_df=None):
    shos_state = None
    if use_shos:
        alpha, beta, k = shos_params or (0.2, 0.1, 0.1)
        shos_state = shos_states(sales, alpha=alpha, beta=beta, k=k)
    has_extra = extra_df is not None and extra_df.shape[1] > 0
    rows = [
        make_features(
            sales[:period],
            promotions[period],
            use_shos,
            shos_params,
            shos_state,
            extra_features=extra_df.iloc[period].to_dict() if has_extra else None,
        )
        for period in range(3, len(sales))
    ]
    return pd.DataFrame(rows), np.asarray(sales[3:], dtype=float)


REGULAR_PARAM_GRID = {
    "n_estimators": [50, 100],
    "max_depth": [3, 5],
    "learning_rate": [0.03, 0.1],
}
HEAVY_PARAM_GRID = {
    "n_estimators": [50, 100, 150],
    "max_depth": [2, 3, 5],
    "learning_rate": [0.02, 0.05, 0.1],
}


def new_lgbm_regressor(params=None):
    model_params = {
        "objective": "tweedie",
        "n_estimators": 50,
        "learning_rate": 0.05,
        "max_depth": 3,
        "min_child_samples": 5,
        "random_state": 42,
        "n_jobs": 1,
        "verbosity": -1,
    }
    model_params.update(params or {})
    return LGBMRegressor(**model_params)


def new_lgbm_classifier(params=None):
    model_params = {
        "objective": "binary",
        "n_estimators": 50,
        "learning_rate": 0.05,
        "max_depth": 3,
        "min_child_samples": 5,
        "random_state": 42,
        "n_jobs": 1,
        "verbosity": -1,
    }
    model_params.update(params or {})
    return LGBMClassifier(**model_params)


def spec_grid_scorer(estimator, features, targets):
    predictions = np.maximum(estimator.predict(features), 0.0)
    return -spec_score(targets, predictions)


def tune_lgbm_regressor(features, targets, heavy=False):
    targets = np.asarray(targets, dtype=float)
    if len(targets) < 4:
        params = {"n_estimators": 50, "max_depth": 3, "learning_rate": 0.05}
        model = new_lgbm_regressor(params)
        model.fit(features, targets)
        return model, params

    search = GridSearchCV(
        estimator=new_lgbm_regressor(),
        param_grid=HEAVY_PARAM_GRID if heavy else REGULAR_PARAM_GRID,
        scoring=spec_grid_scorer,
        cv=TimeSeriesSplit(n_splits=3),
        n_jobs=1,
        refit=True,
    )
    search.fit(features, targets)
    return search.best_estimator_, dict(search.best_params_)


def tune_lgbm_classifier(features, targets, heavy=False):
    targets = np.asarray(targets, dtype=int)
    fallback = {"n_estimators": 50, "max_depth": 3, "learning_rate": 0.05}
    if np.unique(targets).size < 2 or len(targets) < 4:
        return None, fallback

    splitter = TimeSeriesSplit(n_splits=3)
    folds = list(splitter.split(features))
    if any(np.unique(targets[train_indices]).size < 2 for train_indices, _ in folds):
        model = new_lgbm_classifier(fallback)
        model.fit(features, targets)
        return model, fallback

    search = GridSearchCV(
        estimator=new_lgbm_classifier(),
        param_grid=HEAVY_PARAM_GRID if heavy else REGULAR_PARAM_GRID,
        scoring="neg_log_loss",
        cv=splitter,
        n_jobs=1,
        refit=True,
    )
    search.fit(features, targets)
    return search.best_estimator_, dict(search.best_params_)


def split_hurdle_features(features, use_shos):
    standard_cols = {
        "lag_1", "lag_2", "lag_3", "mean_history", "zero_fraction_history",
        "since_last", "nz_roll_3", "promosi", "shos_q", "shos_z",
    }
    extra_cols = [col for col in features.columns if col not in standard_cols]
    lag_columns = ["lag_1", "lag_2", "lag_3"]
    occurrence_columns = lag_columns + ["zero_fraction_history", "since_last", "nz_roll_3", "promosi"] + extra_cols
    size_columns = lag_columns + ["mean_history", "since_last", "nz_roll_3", "promosi"] + extra_cols
    if use_shos:
        occurrence_columns.append("shos_q")
        size_columns.append("shos_z")
    return features[occurrence_columns], features[size_columns]


def fit_model_bundle(
    features,
    targets,
    use_hurdle,
    heavy=False,
    selected_params=None,
    occurrence_features=None,
    size_features=None,
):
    targets = np.asarray(targets, dtype=float)
    if float(np.sum(targets)) <= 0:
        return {
            "hurdle": False,
            "regressor": None,
            "params": {"regressor": {"fallback": "all_zero"}},
            "zero_demand": True,
        }
    if not use_hurdle:
        regressor_params = (selected_params or {}).get("regressor")
        if regressor_params is None:
            model, regressor_params = tune_lgbm_regressor(features, targets, heavy)
        else:
            model = new_lgbm_regressor(regressor_params)
            model.fit(features, targets)
        return {
            "hurdle": False,
            "regressor": model,
            "params": {"regressor": regressor_params},
        }

    occurrence = (targets > 0).astype(int)
    occurrence_rate = float(occurrence.mean()) if len(occurrence) else 0.0
    occurrence_features = occurrence_features if occurrence_features is not None else features
    size_features = size_features if size_features is not None else features
    classifier_params = (selected_params or {}).get("classifier")
    if classifier_params is None:
        classifier, classifier_params = tune_lgbm_classifier(occurrence_features, occurrence, heavy)
    elif np.unique(occurrence).size == 2:
        classifier = new_lgbm_classifier(classifier_params)
        classifier.fit(occurrence_features, occurrence)
    else:
        classifier = None

    positive_rows = targets > 0
    positive_targets = targets[positive_rows]
    demand_regressor = None
    positive_mean = float(positive_targets.mean()) if len(positive_targets) else 0.0
    regressor_params = {"fallback": "positive_mean"}
    if len(positive_targets) >= 4:
        regressor_params = (selected_params or {}).get("regressor")
        if regressor_params is None:
            demand_regressor, regressor_params = tune_lgbm_regressor(
                size_features.loc[positive_rows], positive_targets, heavy
            )
        else:
            demand_regressor = new_lgbm_regressor(regressor_params)
            demand_regressor.fit(size_features.loc[positive_rows], positive_targets)
    return {
        "hurdle": True,
        "classifier": classifier,
        "occurrence_rate": occurrence_rate,
        "demand_regressor": demand_regressor,
        "positive_mean": positive_mean,
        "occurrence_columns": list(occurrence_features.columns),
        "size_columns": list(size_features.columns),
        "params": {"classifier": classifier_params, "regressor": regressor_params},
    }


def predict_model_bundle(bundle, features):
    if bundle.get("zero_demand"):
        return np.zeros(len(features), dtype=float)
    if not bundle["hurdle"]:
        return np.maximum(bundle["regressor"].predict(features), 0.0)
    occurrence_features = features[bundle["occurrence_columns"]]
    size_features = features[bundle["size_columns"]]
    classifier = bundle["classifier"]
    if classifier is None:
        occurrence_probability = bundle["occurrence_rate"]
    else:
        positive_class = int(np.flatnonzero(classifier.classes_ == 1)[0])
        occurrence_probability = classifier.predict_proba(occurrence_features)[:, positive_class]
    regressor = bundle["demand_regressor"]
    if regressor is None:
        demand_size = np.full(len(size_features), bundle["positive_mean"], dtype=float)
    else:
        demand_size = np.maximum(regressor.predict(size_features), 0.0)
    return np.maximum(occurrence_probability * demand_size, 0.0)


def fit_predict_series(sales, promotions, horizon=7, heavy=False, next_promo=0.0, extra_df=None):
    sales = np.asarray(sales, dtype=float)
    promotions = np.asarray(promotions, dtype=float)
    if extra_df is not None:
        extra_df = extra_df.reset_index(drop=True)
    has_extra = extra_df is not None and extra_df.shape[1] > 0
    size = len(sales)
    holdout = min(max(1, math.ceil(size * 0.2)), max(size - 1, 0))
    split = size - holdout
    validation_actual = sales[split:]
    validation_periods = np.arange(split, size)
    tsb_params = tune_smoothing_parameters(sales, split, use_shos=False, heavy=heavy)
    shos_params = tune_smoothing_parameters(sales, split, use_shos=True, heavy=heavy)

    baseline_names = ["TSB", "SHOS", "Croston-SBA", "Naive", "Moving average (4)"]
    full_predictions = {name: np.full(size, np.nan) for name in baseline_names}
    q_tsb, z_tsb = shos_states(sales, tsb_params["alpha"], tsb_params["beta"], 0.0)
    q_shos, z_shos = shos_states(sales, shos_params["alpha"], shos_params["beta"], shos_params["k"])
    tsb_full_forecast = q_tsb * z_tsb
    shos_full_forecast = q_shos * z_shos
    for period in range(1, size):
        full_predictions["TSB"][period] = tsb_full_forecast[period]
        full_predictions["SHOS"][period] = shos_full_forecast[period]
        full_predictions["Croston-SBA"][period] = croston_sba_forecast(sales[:period])
        full_predictions["Naive"][period] = sales[period - 1]
        full_predictions["Moving average (4)"][period] = sales[max(0, period - 4):period].mean()

    candidate_specs = {}
    validation_predictions = {}
    future_predictions = {
        "TSB": np.full(horizon, tsb_forecast(
            sales, tsb_params["alpha"], tsb_params["beta"], 0.0
        ), dtype=float),
        "SHOS": np.full(horizon, tsb_forecast(
            sales, shos_params["alpha"], shos_params["beta"], shos_params["k"]
        ), dtype=float),
        "Croston-SBA": np.full(horizon, croston_sba_forecast(sales), dtype=float),
        "Naive": np.full(horizon, sales[-1], dtype=float),
        "Moving average (4)": np.full(horizon, sales[-min(4, size):].mean(), dtype=float),
    }
    tuning_details = {"TSB": tsb_params, "SHOS": shos_params}
    for model_name in baseline_names:
        validation_predictions[model_name] = full_predictions[model_name][split:]
        candidate_specs[model_name] = spec_score(validation_actual, validation_predictions[model_name])

    if size >= 30:
        variants = {
            "LightGBM": (False, False),
            "LightGBM + SHOS": (True, False),
            "Hurdle baseline": (False, True),
            "Hurdle + SHOS": (True, True),
        }
        train_extra_df = extra_df.iloc[:split] if has_extra else None
        last_extra_features = extra_df.iloc[-1].to_dict() if has_extra else None
        for model_name, (use_shos, use_hurdle) in variants.items():
            feature_shos_params = (
                shos_params["alpha"], shos_params["beta"], shos_params["k"]
            ) if use_shos else None
            train_features, train_targets = build_training_data(
                sales[:split], promotions[:split], use_shos, feature_shos_params, extra_df=train_extra_df
            )
            if use_hurdle:
                train_occ_features, train_size_features = split_hurdle_features(train_features, use_shos)
            else:
                train_occ_features, train_size_features = None, None
            bundle = fit_model_bundle(
                train_features,
                train_targets,
                use_hurdle,
                heavy,
                occurrence_features=train_occ_features,
                size_features=train_size_features,
            )
            tuning_details[model_name] = bundle["params"]
            history_prediction = np.full(size, np.nan)
            history_prediction[3:split] = predict_model_bundle(bundle, train_features)

            validation_shos_state = None
            if use_shos:
                validation_shos_state = shos_states(
                    sales,
                    feature_shos_params[0],
                    feature_shos_params[1],
                    feature_shos_params[2],
                )
            validation_features = pd.DataFrame([
                make_features(
                    sales[:period],
                    promotions[period],
                    use_shos,
                    feature_shos_params,
                    validation_shos_state,
                    extra_features=extra_df.iloc[period].to_dict() if has_extra else None,
                )
                for period in validation_periods
            ])
            validation_forecast = predict_model_bundle(bundle, validation_features)
            history_prediction[split:] = validation_forecast
            full_predictions[model_name] = history_prediction
            validation_predictions[model_name] = validation_forecast
            candidate_specs[model_name] = spec_score(validation_actual, validation_forecast)

            all_features, all_targets = build_training_data(
                sales, promotions, use_shos, feature_shos_params, extra_df=extra_df
            )
            if use_hurdle:
                all_occ_features, all_size_features = split_hurdle_features(all_features, use_shos)
            else:
                all_occ_features, all_size_features = None, None
            final_bundle = fit_model_bundle(
                all_features,
                all_targets,
                use_hurdle,
                heavy,
                bundle["params"],
                occurrence_features=all_occ_features,
                size_features=all_size_features,
            )
            frozen_shos_state = (float(q_shos[-1]), float(z_shos[-1])) if use_shos else None
            recursive_history = sales.tolist()
            future_values = []
            for step in range(horizon):
                future_features = pd.DataFrame([
                    make_features(
                        recursive_history,
                        next_promo if step == 0 else 0.0,
                        use_shos,
                        feature_shos_params,
                        shos_state=frozen_shos_state,
                        event_history=sales,
                        periods_ahead=step,
                        extra_features=last_extra_features,
                    )
                ])
                predicted_demand = float(predict_model_bundle(final_bundle, future_features)[0])
                future_values.append(predicted_demand)
                recursive_history.append(predicted_demand)
            future_predictions[model_name] = np.asarray(future_values, dtype=float)

    finite_specs = {name: score for name, score in candidate_specs.items() if np.isfinite(score)}
    best_model = min(finite_specs, key=finite_specs.get) if finite_specs else "TSB"
    return candidate_specs, validation_predictions, full_predictions, future_predictions, best_model, tuning_details


def calculate_reorder_point(sales, lead_time, service_level, future_forecast):
    sales = np.asarray(sales, dtype=float)
    lead_time = int(lead_time)
    if len(sales) >= 10:
        lt_demand = np.convolve(sales, np.ones(lead_time, dtype=float), "valid")
        empirical_rop = float(np.quantile(lt_demand, service_level)) if len(lt_demand) else 0.0
        fc = np.asarray(future_forecast, dtype=float)
        if len(fc) < lead_time:
            fc = np.pad(fc, (0, lead_time - len(fc)), mode="edge")
        expected_lt_demand = float(fc[:lead_time].sum())
        reorder_point = max(empirical_rop, expected_lt_demand)
        safety_stock = max(reorder_point - expected_lt_demand, 0.0)
        return reorder_point, safety_stock

    demand_std = float(np.std(sales, ddof=1)) if len(sales) > 1 else 0.0
    z_service = NormalDist().inv_cdf(service_level)
    fc = np.asarray(future_forecast, dtype=float)
    if len(fc) < lead_time:
        fc = np.pad(fc, (0, lead_time - len(fc)), mode="edge")
    expected_lt_demand = float(fc[:lead_time].sum())
    safety_stock = z_service * demand_std * math.sqrt(lead_time)
    return max(expected_lt_demand + safety_stock, 0.0), safety_stock


def evaluate_forecast(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if actual.size == 0:
        return {"MAE": float("nan"), "WAPE (%)": float("nan"), "SPEC": float("nan")}
    total_actual = float(actual.sum())
    wape = float(np.abs(actual - predicted).sum() / total_actual * 100) if total_actual else float("nan")
    return {
        "MAE": float(np.abs(actual - predicted).mean()),
        "WAPE (%)": wape,
        "SPEC": spec_score(actual, predicted),
    }


def make_future_dates(date_values, horizon):
    dates = pd.DatetimeIndex(date_values)
    if len(dates) >= 3 and dates.is_month_start.all():
        frequency = "MS"
    elif len(dates) >= 3 and dates.is_month_end.all():
        frequency = "ME"
    else:
        frequency = None
    if frequency:
        try:
            return pd.date_range(start=dates[-1], periods=horizon + 1, freq=frequency)[1:]
        except (ValueError, TypeError):
            pass

    intervals = dates.to_series().diff().dropna()
    interval = intervals.median() if len(intervals) else pd.Timedelta(days=1)
    if pd.isna(interval) or interval <= pd.Timedelta(0):
        interval = pd.Timedelta(days=1)
    return pd.date_range(start=dates[-1] + interval, periods=horizon, freq=interval)


def fill_missing_dates(frame):
    pieces = []
    filled_count = 0
    fallback_skus = []
    for product_id, product_frame in frame.groupby("id_produk", sort=False):
        product_frame = product_frame.sort_values("tanggal", kind="stable")
        product_frame = product_frame.drop_duplicates("tanggal", keep="last")
        if len(product_frame) >= 3 and product_frame["tanggal"].dt.is_month_start.all():
            frequency = "MS"
        elif len(product_frame) >= 3 and product_frame["tanggal"].dt.is_month_end.all():
            frequency = "ME"
        else:
            frequency = None
        if frequency:
            complete_dates = pd.date_range(
                product_frame["tanggal"].min(),
                product_frame["tanggal"].max(),
                freq=frequency,
            )
        else:
            intervals = product_frame["tanggal"].diff().dropna()
            interval = intervals.median() if len(intervals) else pd.Timedelta(days=1)
            if pd.isna(interval) or interval <= pd.Timedelta(0):
                interval = pd.Timedelta(days=1)
            complete_dates = pd.date_range(
                product_frame["tanggal"].min(),
                product_frame["tanggal"].max(),
                freq=interval,
            )

        indexed = product_frame.set_index("tanggal")
        original_sales = float(product_frame["penjualan"].sum())
        indexed = indexed.reindex(complete_dates)
        missing_count = int(indexed["penjualan"].isna().sum())
        indexed["penjualan"] = pd.to_numeric(indexed["penjualan"], errors="coerce").fillna(0.0)
        if not np.isclose(float(indexed["penjualan"].sum()), original_sales):
            fallback_skus.append(str(product_id))
            pieces.append(product_frame)
            continue
        filled_count += missing_count
        indexed["promosi"] = pd.to_numeric(indexed["promosi"], errors="coerce").fillna(0.0)
        indexed["id_produk"] = product_id
        if "extra_feature_columns" in product_frame.attrs:
            for col in product_frame.attrs["extra_feature_columns"]:
                indexed[col] = pd.to_numeric(indexed[col], errors="coerce").ffill().fillna(0.0)
        pieces.append(indexed.reset_index().rename(columns={"index": "tanggal"}))

    if not pieces:
        return frame, 0, []
    return pd.concat(pieces, ignore_index=True), filled_count, fallback_skus


def normalize_columns(frame):
    aliases = {
        "tanggal": {"tanggal", "date", "tanggal_penjualan", "ds"},
        "penjualan": {"penjualan", "sales", "target", "y"},
        "promosi": {"promosi", "promo", "promotion"},
        "id_produk": {"id produk", "id_produk", "produk", "product", "product_id", "item_id", "sku"},
        "stok": {"stok", "stock", "stok_tersedia", "inventory"},
        "lead_time": {"lead_time", "leadtime", "lead time", "lead_time_periode"},
    }
    normalized = {str(column).strip().lower(): column for column in frame.columns}
    rename = {}
    for canonical, names in aliases.items():
        match = next((normalized[name] for name in names if name in normalized), None)
        if match is not None:
            rename[match] = canonical
    return frame.rename(columns=rename)


def read_sales_file(uploaded_file):
    try:
        if uploaded_file.name.lower().endswith(".xlsx"):
            frame = pd.read_excel(uploaded_file, engine="openpyxl")
        else:
            frame = pd.read_csv(uploaded_file, sep=None, engine="python", encoding="utf-8-sig")
    except Exception as error:
        raise ValueError(f"File tidak dapat dibaca: {error}") from error
    frame = normalize_columns(frame)
    missing_columns = {"tanggal", "penjualan"} - set(frame.columns)
    if missing_columns:
        raise ValueError("Kolom wajib tidak ditemukan: " + ", ".join(sorted(missing_columns)) + ".")
    if frame.empty:
        raise ValueError("File tidak berisi baris data.")

    date_values = frame["tanggal"].astype("string").str.strip()
    first_date = date_values.dropna().iloc[0] if date_values.notna().any() else ""
    dayfirst = not (len(first_date) >= 5 and first_date[:4].isdigit() and first_date[4] in "-/")
    frame["tanggal"] = pd.to_datetime(date_values, errors="coerce", dayfirst=dayfirst)
    frame["penjualan"] = pd.to_numeric(frame["penjualan"], errors="coerce")
    if frame["tanggal"].isna().any():
        raise ValueError("Ada tanggal kosong atau tidak dikenali. Gunakan format tanggal yang konsisten.")
    if frame["penjualan"].isna().any() or not np.isfinite(frame["penjualan"]).all():
        raise ValueError("Kolom Penjualan harus berisi angka dan tidak boleh kosong.")
    if (frame["penjualan"] < 0).any():
        raise ValueError("Penjualan tidak boleh bernilai negatif.")

    if "promosi" in frame.columns:
        promotion_values = frame["promosi"].astype("string").str.strip().str.lower()
        promotion_values = promotion_values.replace({"ya": "1", "yes": "1", "true": "1", "tidak": "0", "no": "0", "false": "0"})
        frame["promosi"] = pd.to_numeric(promotion_values, errors="coerce")
        if frame["promosi"].isna().any() or not np.isfinite(frame["promosi"]).all():
            raise ValueError("Kolom Promosi harus berupa angka atau penanda ya/tidak.")
    else:
        frame["promosi"] = 0.0

    if "id_produk" not in frame.columns:
        frame["id_produk"] = Path(uploaded_file.name).stem or "Produk"
    frame["id_produk"] = frame["id_produk"].fillna("Tanpa ID").astype(str)
    if frame.duplicated(["id_produk", "tanggal"]).any():
        raise ValueError("Ada tanggal ganda untuk produk yang sama. Gabungkan atau hapus baris duplikat terlebih dahulu.")

    standard_columns = {"tanggal", "penjualan", "promosi", "id_produk", "stok", "lead_time"}
    detected_extra = []
    for column in frame.columns:
        if column in standard_columns:
            continue
        converted = pd.to_numeric(frame[column], errors="coerce")
        valid_ratio = float(converted.notna().mean()) if len(converted) else 0.0
        if valid_ratio >= 0.80:
            frame[column] = converted.fillna(0.0)
            detected_extra.append(column)
    frame.attrs["extra_feature_columns"] = detected_extra
    return frame.sort_values(["id_produk", "tanggal"], kind="stable").reset_index(drop=True)


def sample_sales_data():
    dates = pd.date_range("2025-01-01", periods=42, freq="D")
    intermittent_sales = np.zeros(42, dtype=int)
    intermittent_sales[[3, 7, 12, 18, 19, 27, 34, 35, 41]] = [4, 12, 7, 2, 9, 15, 5, 11, 8]
    smooth_sales = np.maximum(0, np.rint(8 + 2 * np.sin(2 * np.pi * np.arange(42) / 7))).astype(int)

    rows = []
    for product_id, sales in (("LUMPY-001", intermittent_sales), ("SMOOTH-001", smooth_sales)):
        rows.append(pd.DataFrame({
            "Tanggal": dates,
            "Penjualan": sales,
            "Promosi": (np.arange(42) % 14 == 5).astype(int),
            "ID Produk": product_id,
        }))
    return pd.concat(rows, ignore_index=True)


st.caption("Perencanaan permintaan")
st.title("Ramalkan penjualan. Siapkan stok.")
st.write("Unggah riwayat penjualan untuk membandingkan model forecast dan kebutuhan stok periode berikutnya.")
upload_column, sample_column = st.columns([1.5, 1], gap="large")
with upload_column:
    st.subheader("Mulai dari data Anda")
    st.caption("Wajib: Tanggal dan Penjualan. Opsional: Promosi dan ID Produk.")
    uploaded_file = st.file_uploader("Tarik file CSV atau Excel ke sini", type=["csv", "xlsx"])
with sample_column:
    st.subheader("Belum punya file?")
    st.write("Coba alur forecast dengan data demo 42 hari.")
    use_sample = show_button("Coba data contoh", type="primary")
    sample_data = sample_sales_data()
    show_download(
        "Unduh CSV contoh",
        data=sample_data.to_csv(index=False).encode("utf-8-sig"),
        file_name="contoh_penjualan.csv",
        mime="text/csv",
    )

if "use_sample_data" not in st.session_state:
    st.session_state["use_sample_data"] = False
if use_sample:
    st.session_state["use_sample_data"] = True
if uploaded_file is not None:
    st.session_state["use_sample_data"] = False

if uploaded_file is None and not st.session_state["use_sample_data"]:
    st.info("Unggah CSV atau pilih **Coba data contoh** untuk melihat hasilnya.")
    st.stop()

try:
    if uploaded_file is not None:
        sales_frame = read_sales_file(uploaded_file)
        st.success(f"File berhasil dibaca: {uploaded_file.name}")
        detected_extra = sales_frame.attrs.get("extra_feature_columns", [])
        if detected_extra:
            st.caption("Fitur numerik tambahan terdeteksi dan ikut dilatih: " + ", ".join(detected_extra))
    else:
        sales_frame = sample_data.rename(columns={
            "Tanggal": "tanggal",
            "Penjualan": "penjualan",
            "Promosi": "promosi",
            "ID Produk": "id_produk",
        })
        st.info("Sedang memakai data contoh. Unggah file Anda untuk menggantinya.")

    fill_missing_dates_option = st.sidebar.checkbox(
        "Isi tanggal kosong dengan penjualan 0",
        value=True,
        help="Mendeteksi interval tanggal dan mengisi periode yang hilang agar pola intermiten tetap terlihat.",
    )
    if fill_missing_dates_option:
        sales_frame, filled_count, fallback_skus = fill_missing_dates(sales_frame)
        if fallback_skus:
            st.warning("Tanggal SKU berikut gagal dilengkapi karena interval tidak konsisten; data asli dipakai: " + ", ".join(fallback_skus))
        if filled_count:
            st.warning(f"Ada {filled_count} periode tanggal kosong yang diisi dengan penjualan 0.")

    with st.expander("Pratinjau data", expanded=False):
        show_df(sales_frame.head(12))
    forecast_horizon = st.sidebar.select_slider(
        "Forecast berapa periode ke depan?",
        options=[1, 3, 7],
        value=7,
    )
    heavy_tuning = st.sidebar.toggle(
        "Pencarian parameter lengkap (Grid Search)",
        value=False,
        help="Kalibrasi lebih detail untuk SHOS dan Hurdle. Waktu pencarian lebih lama.",
    )
    service_level = st.sidebar.slider("Target tingkat layanan", min_value=0.80, max_value=0.99, value=0.95, step=0.01)
    if heavy_tuning and len(sales_frame["id_produk"].dropna().unique()) > 12:
        st.warning("Pencarian parameter lengkap pada lebih dari 12 SKU dapat membutuhkan waktu lama. Pertimbangkan menjalankan subset terlebih dahulu.")
    inventory_products = []
    for product_id in sorted(sales_frame["id_produk"].dropna().astype(str).unique().tolist(), key=str):
        product_rows = sales_frame[sales_frame["id_produk"].astype(str) == product_id]
        stock_value = product_rows["stok"].dropna().iloc[-1] if "stok" in sales_frame.columns and product_rows["stok"].notna().any() else 0
        lead_value = product_rows["lead_time"].dropna().iloc[-1] if "lead_time" in sales_frame.columns and product_rows["lead_time"].notna().any() else 1
        inventory_products.append({
            "ID Produk": product_id,
            "Stok tersedia": max(int(stock_value), 0),
            "Lead time (periode)": max(int(lead_value), 1),
            "Promo t+1": 0.0,
        })
    inventory_table = st.data_editor(
        pd.DataFrame(inventory_products),
        hide_index=True,
        num_rows="fixed",
        key="inventory_per_sku",
        column_config={
            "ID Produk": st.column_config.TextColumn(disabled=True),
            "Stok tersedia": st.column_config.NumberColumn(min_value=0, step=1),
            "Lead time (periode)": st.column_config.NumberColumn(
                min_value=1,
                step=1,
                help="Gunakan satuan periode data: hari, minggu, atau bulan sesuai tanggal CSV.",
            ),
            "Promo t+1": st.column_config.NumberColumn(min_value=0.0, max_value=1.0, step=0.1),
        },
    )

    results = []
    evaluation_rows = []
    chart_data = {}
    tuning_rows = []
    progress = st.progress(0, text="Membaca data...")
    product_groups = list(sales_frame.groupby("id_produk", sort=False))

    for group_index, (product_id, product_frame) in enumerate(product_groups, start=1):
        product_frame = product_frame.sort_values("tanggal", kind="stable")
        if len(product_frame) < 30:
            st.warning(
                f"SKU {product_id} memiliki {len(product_frame)} periode dan dilewati. "
                "Minimal 30 periode diperlukan untuk validasi dan data uji yang cukup stabil. Seri ini ditandai indikatif dan tidak layak menjadi dasar keputusan."
            )
            continue
        sales = product_frame["penjualan"].to_numpy(dtype=float)
        promotions = product_frame["promosi"].to_numpy(dtype=float)
        extra_cols = tuple(product_frame.attrs.get("extra_feature_columns", []))
        extra_values = tuple(
            tuple(float(value) for value in product_frame[col].to_numpy(dtype=float))
            for col in extra_cols
        ) if extra_cols else ()
        try:
            (
                specs,
                validation_predictions,
                full_predictions,
                future_predictions,
                best_model,
                tuning_details,
            ) = compute_product(
                str(product_id),
                tuple(float(value) for value in sales),
                tuple(float(value) for value in promotions),
                forecast_horizon,
                heavy_tuning,
                CONFIG_VERSION,
                float(inventory_table.loc[inventory_table["ID Produk"] == str(product_id), "Promo t+1"].iloc[0])
                if not inventory_table[inventory_table["ID Produk"] == str(product_id)].empty
                else 0.0,
                extra_cols=extra_cols,
                extra_values=extra_values,
            )
        except Exception as product_error:
            st.warning(f"SKU {product_id} dilewati karena model gagal diproses.")
            with st.expander(f"Detail error {product_id}"):
                st.exception(product_error)
            continue
        dates = product_frame["tanggal"].to_numpy()
        validation_length = len(validation_predictions["TSB"])
        validation_actual = sales[-validation_length:] if validation_length else np.asarray([])
        for model_name, predictions in validation_predictions.items():
            metrics = evaluate_forecast(validation_actual, predictions)
            evaluation_rows.append({
                "ID Produk": product_id,
                "Model": model_name,
                "MAE": metrics["MAE"],
                "WAPE (%)": metrics["WAPE (%)"],
                "SPEC": metrics["SPEC"],
                "Status": "Terbaik" if model_name == best_model else "Pembanding",
            })
        future_dates = make_future_dates(dates, forecast_horizon)
        next_forecast = float(future_predictions[best_model][0])
        inventory_row = inventory_table[inventory_table["ID Produk"] == str(product_id)]
        stock_available = int(inventory_row["Stok tersedia"].iloc[0]) if len(inventory_row) else 0
        lead_time = int(inventory_row["Lead time (periode)"].iloc[0]) if len(inventory_row) else 1
        reorder_point, safety_stock = calculate_reorder_point(
            sales,
            lead_time,
            service_level,
            future_predictions[best_model],
        )
        chart_data[product_id] = {
            "dates": dates,
            "sales": sales,
            "validation_dates": dates[len(sales) - validation_length:],
            "validation_predictions": validation_predictions,
            "full_predictions": full_predictions,
            "future_dates": future_dates,
            "future_predictions": future_predictions,
            "best_model": best_model,
            "reorder_point": reorder_point,
        }
        for model_name, parameters in tuning_details.items():
            tuning_rows.append({
                "ID Produk": product_id,
                "Model": model_name,
                "Parameter terbaik": str(parameters),
            })
        shos_tuned = tuning_details.get("SHOS", {})
        q_value, z_value = shos_states(
            sales,
            shos_tuned.get("alpha", 0.2),
            shos_tuned.get("beta", 0.1),
            shos_tuned.get("k", 0.1),
        )
        results.append({
            "ID Produk": product_id,
            "Ramalan t+1": round(next_forecast, 2),
            "Model Terbaik": best_model,
            "Rekomendasi Stok": int(math.ceil(max(next_forecast, 0.0))),
            "Stok Safety": int(math.ceil(safety_stock)),
            "Reorder Point": int(math.ceil(reorder_point)),
            "Butuh Pesan": "Ya" if int(stock_available) < int(math.ceil(reorder_point)) else "Tidak",
            "Jumlah Pesan": int(math.ceil(max(reorder_point - stock_available, 0.0))),
            "Target Risiko Stockout": f"{1.0 - service_level:.1%}",
            "q SHOS": round(float(q_value[-1]), 3),
            "z SHOS": round(float(z_value[-1]), 3),
        })
        progress.progress(group_index / len(product_groups), text=f"Memproses {product_id}...")

    progress.empty()
    if not results:
        st.error("Tidak ada SKU dengan minimal 30 periode yang dapat diproses. Periksa data atau isi tanggal kosong.")
        st.stop()

    result_table = pd.DataFrame(results)
    only_reorder = st.toggle("Hanya tampilkan SKU yang perlu pesan", value=False)
    display_table = result_table[result_table["Butuh Pesan"] == "Ya"] if only_reorder else result_table

    metric_columns = st.columns(4)
    metric_columns[0].metric("Total SKU dianalisis", len(result_table))
    reorder_count = int((result_table["Butuh Pesan"] == "Ya").sum())
    metric_columns[1].metric("SKU perlu reorder", reorder_count)
    dominant_model = result_table["Model Terbaik"].mode().iloc[0] if len(result_table) else "-"
    total_reorder_units = int(display_table["Jumlah Pesan"].sum())
    metric_columns[2].metric("Total unit reorder", total_reorder_units)
    metric_columns[3].metric("Model paling dominan", dominant_model)

    st.subheader("Tabel Rekomendasi")
    show_df(display_table)
    st.caption("Rekomendasi stok memadukan akumulasi ramalan sesuai lead time, stok tersedia, dan kuantil permintaan historis. Jumlah pesan dihitung dari reorder point dikurangi stok tersedia. Promo t+1 diambil dari tabel inventori.")
    show_download(
        "Unduh tabel hasil (CSV)",
        data=result_table.to_csv(index=False).encode("utf-8-sig"),
        file_name="rekomendasi_forecast.csv",
        mime="text/csv",
    )

    evaluation_matrix = pd.DataFrame(evaluation_rows)
    st.subheader("Matriks Evaluasi Model")
    st.caption("Metrik ini dihitung pada periode validasi yang dipakai untuk memilih model, bukan estimasi holdout out-of-sample. Periode validasi dihitung dari panjang data setiap SKU dan dapat berbeda antar SKU. Perbedaan SPEC antar kandidat perlu diperhatikan, bukan hanya pemenangnya. Data di bawah 30 periode ditandai indikatif dan LightGBM baru dilatih dari minimal 30 periode. SPEC menyesuaikan bobot TSB/SHOS berdasarkan hasil kalibrasi pada validasi, sementara Hurdle + SHOS memisahkan shos_q ke klasifier kejadian dan shos_z ke regresor ukuran permintaan.")
    show_df(evaluation_matrix)
    show_download(
        "Unduh matriks evaluasi (CSV)",
        data=evaluation_matrix.to_csv(index=False).encode("utf-8-sig"),
        file_name="matriks_evaluasi.csv",
        mime="text/csv",
    )

    with st.expander("Parameter terbaik model"):
        show_df(pd.DataFrame(tuning_rows))

    selected_product = st.selectbox("Produk pada grafik", list(chart_data))
    selected = chart_data[selected_product]
    available_models = list(selected["full_predictions"])
    comparison_default = selected["best_model"] if selected["best_model"] in available_models else available_models[0]
    comparison_model = st.selectbox(
        "Bandingkan dengan model",
        available_models,
        index=available_models.index(comparison_default),
        key=f"comparison_model_{selected_product}",
    )
    st.subheader(f"Aktual vs Prediksi: {selected_product}")
    figure = go.Figure()
    figure.add_trace(go.Scatter(
        x=selected["dates"], y=selected["sales"], mode="lines+markers",
        name="Aktual", line={"color": "#284b63", "width": 2},
    ))
    model_colors = {
        "TSB": "#237a61",
        "SHOS": "#7f9f45",
        "Croston-SBA": "#3b6e8c",
        "Naive": "#52616b",
        "Moving average (4)": "#e29b5d",
        "LightGBM": "#d66b4b",
        "LightGBM + SHOS": "#9b6b9e",
        "Hurdle baseline": "#4e86a6",
        "Hurdle + SHOS": "#c04d65",
    }
    selected_models = [comparison_model]
    if selected["best_model"] not in selected_models:
        selected_models.append(selected["best_model"])
    for model_name in selected_models:
        color = model_colors[model_name]
        validation_forecast = selected["validation_predictions"][model_name]
        figure.add_trace(go.Scatter(
            x=selected["validation_dates"], y=validation_forecast, mode="lines+markers",
            name=f"{model_name} - validasi", line={"color": color, "width": 2, "dash": "dot"}, marker={"size": 5},
        ))
        future_forecast = selected["future_predictions"][model_name]
        figure.add_trace(go.Scatter(
            x=[selected["dates"][-1], *selected["future_dates"]],
            y=[selected["sales"][-1], *future_forecast],
            mode="lines+markers", name=f"{model_name}",
            showlegend=True,
            line={"color": color, "width": 2.5}, marker={"size": 5},
        ))
    figure.add_hline(
        y=selected["reorder_point"],
        line_dash="dot",
        line_color="#8a4b2d",
        annotation_text="Reorder Point",
        annotation_position="bottom right",
    )
    figure.update_layout(template="plotly_white", height=390, hovermode="x unified",
                         xaxis_title="Tanggal", yaxis_title="Unit terjual",
                         legend={"orientation": "h", "groupclick": "togglegroup"})
    show_chart(figure)
    st.caption("Grafik menampilkan aktual, validasi bertitik-titik, dan forecast model pemenang serta satu pembanding. Untuk horizon > 1, state SHOS dibekuk pada nilai terakhir saat proyeksi dan forecast LightGBM bersifat rekursif dan tidak andal sebagai estimasi keputusan. Akumulasi ramalan sesuai durasi lead time dipakai untuk menghitung Reorder Point. Deteksi fitur numerik tambahan otomatis dari file pengguna diaplikasikan pada semua model LightGBM dan Hurdle + SHOS.")

    with st.expander("Tentang metode"):
        st.markdown(
            "**TSB/SHOS** memakai peluang permintaan q dan ukuran permintaan z. Perkiraan TSB/SHOS adalah q × z. Parameter dipilih pada jendela validasi menggunakan kalibrasi berbasis SPEC, dengan nilai SPEC lebih rendah sebagai indikator lebih baik.\\n\\n"
            "**SPEC** mengukur kehilangan penjualan dan biaya simpan lintas horizon; nilai lebih rendah lebih baik. Metrik pada validasi digunakan untuk memilih model dan belum merupakan estimasi holdout independen.\\n\\n"
            "**LightGBM** dilatih per SKU dengan fitur lag 1–3, riwayat, rasio nol, promosi, dan SHOS bila dipilih. **Hurdle** melatih kemungkinan permintaan positif dan ukuran permintaan secara terpisah. Pada **Hurdle + SHOS**, fitur `shos_q` khusus masuk ke klasifier kejadian dan `shos_z` khusus masuk ke regresor ukuran permintaan. Saat proyeksi horizon > 1, state SHOS dibekuk pada nilai terakhir sehingga prediksi tidak melonjak. Sistem juga mendeteksi kolom numerik tambahan apa pun dari CSV/Excel (minimal 80% nilai valid) dan melatihnya sebagai fitur eksternal pada LightGBM serta kedua tahap Hurdle. Hasil pada satu seri pendek berisiko tinggi dan tidak boleh diperlakukan sebagai bukti kausal."
        )
except Exception as error:
    st.error("Pemrosesan model gagal. Periksa data dan parameter, lalu buka detail error hanya untuk debugging.")
    with st.expander("Detail error (debug)"):
        st.exception(error)
