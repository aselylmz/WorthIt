"""
Phase 4 — Baseline Forecasting ve Evaluation testleri.

Tüm baseline modelleri (Naive, Drift, Seasonal Naive, Moving Average) ve
değerlendirme araçlarını (metrikler, PI coverage, CV, model karşılaştırma) test eder.
"""

import numpy as np
import pandas as pd
import pytest

from time_to_afford.forecasting.base import BaseForecaster
from time_to_afford.forecasting.baseline import (
    DriftForecaster,
    MovingAverageForecaster,
    NaiveForecaster,
    SeasonalNaiveForecaster,
    _generate_future_index,
)
from time_to_afford.forecasting.evaluation import (
    compare_models,
    compute_all_metrics,
    mean_absolute_error,
    mean_absolute_percentage_error,
    prediction_interval_coverage,
    root_mean_squared_error,
    symmetric_mean_absolute_percentage_error,
    time_series_cv,
)


# =====================================================================
# Test Fixtures
# =====================================================================

@pytest.fixture
def monthly_index():
    """60 aylık (5 yıl) test amaçlı DatetimeIndex."""
    return pd.date_range("2019-01-31", periods=60, freq="ME")


@pytest.fixture
def constant_series(monthly_index):
    """Sabit değerli seri (100.0). Naive modelin mükemmel çalışması gerekir."""
    return pd.Series(100.0, index=monthly_index, name="constant")


@pytest.fixture
def linear_trend_series(monthly_index):
    """Doğrusal trendli seri (100 + 2*t). Drift modelinin iyi çalışması gerekir."""
    values = 100.0 + 2.0 * np.arange(len(monthly_index))
    return pd.Series(values, index=monthly_index, name="linear_trend")


@pytest.fixture
def seasonal_series(monthly_index):
    """Mevsimsel desenli seri. SeasonalNaive'in iyi çalışması gerekir."""
    seasonal_pattern = np.tile([10, 12, 15, 18, 22, 25, 24, 20, 16, 13, 11, 10], 5)
    return pd.Series(seasonal_pattern, index=monthly_index, name="seasonal")


@pytest.fixture
def noisy_series(monthly_index):
    """Gürültülü seri (trend + noise)."""
    rng = np.random.default_rng(42)
    values = 100.0 + 0.5 * np.arange(len(monthly_index)) + rng.normal(0, 3, len(monthly_index))
    return pd.Series(values, index=monthly_index, name="noisy")


@pytest.fixture
def short_series():
    """Kısa seri (5 gözlem)."""
    idx = pd.date_range("2024-01-31", periods=5, freq="ME")
    return pd.Series([10, 12, 14, 16, 18], index=idx, name="short")


# =====================================================================
# _generate_future_index Testleri
# =====================================================================

class TestGenerateFutureIndex:
    def test_basic(self):
        last = pd.Timestamp("2024-06-30")
        idx = _generate_future_index(last, 3)
        assert len(idx) == 3
        assert idx[0] == pd.Timestamp("2024-07-31")
        assert idx[2] == pd.Timestamp("2024-09-30")

    def test_single_step(self):
        last = pd.Timestamp("2024-12-31")
        idx = _generate_future_index(last, 1)
        assert len(idx) == 1
        assert idx[0] == pd.Timestamp("2025-01-31")


# =====================================================================
# BaseForecaster ABC Testleri
# =====================================================================

class TestBaseForecaster:
    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            BaseForecaster()

    def test_subclass_repr(self):
        model = NaiveForecaster()
        assert "NaiveForecaster" in repr(model)
        assert "fitted=False" in repr(model)


# =====================================================================
# NaiveForecaster Testleri
# =====================================================================

class TestNaiveForecaster:
    def test_fit_predict_constant(self, constant_series):
        model = NaiveForecaster()
        model.fit(constant_series)
        assert model.is_fitted

        forecast = model.predict(12)
        assert len(forecast) == 12
        assert all(forecast == 100.0)

    def test_fit_predict_trend(self, linear_trend_series):
        model = NaiveForecaster()
        model.fit(linear_trend_series)
        forecast = model.predict(6)

        # Naive: tüm tahminler son değere eşit olmalı
        last_value = linear_trend_series.iloc[-1]
        assert all(forecast == last_value)

    def test_predict_interval_widens(self, noisy_series):
        model = NaiveForecaster()
        model.fit(noisy_series)
        pi = model.predict_interval(12, alpha=0.05)

        assert "lower" in pi.columns
        assert "upper" in pi.columns
        assert len(pi) == 12

        # Aralık h arttıkça genişlemeli
        widths = (pi["upper"] - pi["lower"]).values
        assert all(widths[i] <= widths[i + 1] for i in range(len(widths) - 1))

    def test_predict_interval_narrower_at_higher_alpha(self, noisy_series):
        model = NaiveForecaster()
        model.fit(noisy_series)

        pi_95 = model.predict_interval(6, alpha=0.05)
        pi_80 = model.predict_interval(6, alpha=0.20)

        width_95 = (pi_95["upper"] - pi_95["lower"]).mean()
        width_80 = (pi_80["upper"] - pi_80["lower"]).mean()
        assert width_95 > width_80

    def test_not_fitted_raises(self):
        model = NaiveForecaster()
        with pytest.raises(RuntimeError, match="eğitilmedi"):
            model.predict(5)

    def test_too_few_observations(self):
        idx = pd.date_range("2024-01-31", periods=1, freq="ME")
        short = pd.Series([100.0], index=idx)
        model = NaiveForecaster()
        with pytest.raises(ValueError, match="en az 2"):
            model.fit(short)

    def test_invalid_horizon(self, constant_series):
        model = NaiveForecaster()
        model.fit(constant_series)
        with pytest.raises(ValueError, match="horizon >= 1"):
            model.predict(0)

    def test_future_index_correct(self, constant_series):
        model = NaiveForecaster()
        model.fit(constant_series)
        forecast = model.predict(3)

        last_train_date = constant_series.index[-1]
        assert forecast.index[0] > last_train_date
        assert len(forecast.index) == 3


# =====================================================================
# DriftForecaster Testleri
# =====================================================================

class TestDriftForecaster:
    def test_perfect_on_linear_trend(self, linear_trend_series):
        """Doğrusal trendli seride Drift modeli mükemmel tahmin üretmeli."""
        model = DriftForecaster()
        model.fit(linear_trend_series)
        forecast = model.predict(6)

        # Beklenen: son değer + drift * h
        last = linear_trend_series.iloc[-1]
        drift = 2.0  # her adımda +2
        for h in range(1, 7):
            expected = last + drift * h
            assert abs(forecast.iloc[h - 1] - expected) < 1e-6

    def test_drift_on_constant(self, constant_series):
        """Sabit seride drift ~0 olmalı."""
        model = DriftForecaster()
        model.fit(constant_series)
        forecast = model.predict(6)
        # Tüm tahminler ~100 olmalı
        assert all(abs(forecast - 100.0) < 1e-6)

    def test_predict_interval(self, noisy_series):
        model = DriftForecaster()
        model.fit(noisy_series)
        pi = model.predict_interval(12, alpha=0.05)

        assert "lower" in pi.columns
        assert "upper" in pi.columns

        # Aralık genişlemeli
        widths = (pi["upper"] - pi["lower"]).values
        assert all(widths[i] <= widths[i + 1] + 1e-10 for i in range(len(widths) - 1))

    def test_too_few_observations(self):
        idx = pd.date_range("2024-01-31", periods=1, freq="ME")
        model = DriftForecaster()
        with pytest.raises(ValueError):
            model.fit(pd.Series([100.0], index=idx))


# =====================================================================
# SeasonalNaiveForecaster Testleri
# =====================================================================

class TestSeasonalNaiveForecaster:
    def test_perfect_on_pure_seasonal(self, seasonal_series):
        """Tam mevsimsel seride Seasonal Naive mükemmel tahmin üretmeli."""
        model = SeasonalNaiveForecaster(period=12)
        model.fit(seasonal_series)
        forecast = model.predict(12)

        # Son sezonun değerleri tekrarlanmalı
        last_season = seasonal_series.iloc[-12:].values
        np.testing.assert_array_almost_equal(forecast.values, last_season)

    def test_cyclic_repeat_beyond_one_season(self, seasonal_series):
        """24 aylık tahminde sezon 2 kez tekrarlanmalı."""
        model = SeasonalNaiveForecaster(period=12)
        model.fit(seasonal_series)
        forecast = model.predict(24)

        last_season = seasonal_series.iloc[-12:].values
        np.testing.assert_array_almost_equal(forecast.values[:12], last_season)
        np.testing.assert_array_almost_equal(forecast.values[12:], last_season)

    def test_predict_interval(self):
        """Gürültülü mevsimsel seride PI ikinci sezonda genişlemeli."""
        idx = pd.date_range("2015-01-31", periods=60, freq="ME")
        rng = np.random.default_rng(123)
        seasonal_pattern = np.tile([10, 12, 15, 18, 22, 25, 24, 20, 16, 13, 11, 10], 5)
        noisy_seasonal = pd.Series(
            seasonal_pattern + rng.normal(0, 1.5, 60), index=idx, name="noisy_seasonal"
        )

        model = SeasonalNaiveForecaster(period=12)
        model.fit(noisy_seasonal)
        pi = model.predict_interval(24, alpha=0.05)

        # İkinci sezon tekrarında aralık birinciden geniş olmalı
        width_first_season = (pi["upper"].iloc[:12] - pi["lower"].iloc[:12]).mean()
        width_second_season = (pi["upper"].iloc[12:] - pi["lower"].iloc[12:]).mean()
        assert width_second_season > width_first_season

    def test_custom_period(self):
        """period=4 (çeyreklik) ile çalışmalı."""
        idx = pd.date_range("2020-01-31", periods=20, freq="ME")
        values = np.tile([10, 20, 30, 40], 5)
        series = pd.Series(values, index=idx, name="quarterly")

        model = SeasonalNaiveForecaster(period=4)
        model.fit(series)
        forecast = model.predict(8)

        expected = np.tile([10, 20, 30, 40], 2)
        np.testing.assert_array_almost_equal(forecast.values, expected)

    def test_too_few_observations(self):
        idx = pd.date_range("2024-01-31", periods=10, freq="ME")
        series = pd.Series(range(10), index=idx)
        model = SeasonalNaiveForecaster(period=12)
        with pytest.raises(ValueError, match="en az 13"):
            model.fit(series)


# =====================================================================
# MovingAverageForecaster Testleri
# =====================================================================

class TestMovingAverageForecaster:
    def test_constant_series(self, constant_series):
        """Sabit seride MA = sabit olmalı."""
        model = MovingAverageForecaster(window=6)
        model.fit(constant_series)
        forecast = model.predict(12)
        assert all(abs(forecast - 100.0) < 1e-6)

    def test_ma_value_correct(self, short_series):
        """MA değeri son k gözlemin ortalamasına eşit olmalı."""
        model = MovingAverageForecaster(window=3)
        model.fit(short_series)
        forecast = model.predict(6)

        expected_ma = short_series.iloc[-3:].mean()  # (14+16+18)/3 = 16
        assert all(abs(forecast - expected_ma) < 1e-6)

    def test_flat_forecast(self, noisy_series):
        """MA tahminleri sabit olmalı (tüm horizon için aynı)."""
        model = MovingAverageForecaster(window=6)
        model.fit(noisy_series)
        forecast = model.predict(12)
        assert all(forecast == forecast.iloc[0])

    def test_predict_interval_constant_width(self, noisy_series):
        """MA aralığı sabit genişlikte olmalı."""
        model = MovingAverageForecaster(window=6)
        model.fit(noisy_series)
        pi = model.predict_interval(12, alpha=0.05)

        widths = (pi["upper"] - pi["lower"]).values
        # Tüm genişlikler aynı olmalı
        assert all(abs(widths - widths[0]) < 1e-10)

    def test_invalid_window(self):
        with pytest.raises(ValueError, match="window >= 1"):
            MovingAverageForecaster(window=0)

    def test_too_few_observations(self):
        idx = pd.date_range("2024-01-31", periods=5, freq="ME")
        series = pd.Series(range(5), index=idx)
        model = MovingAverageForecaster(window=6)
        with pytest.raises(ValueError, match="en az 7"):
            model.fit(series)


# =====================================================================
# Evaluation: Hata Metrikleri Testleri
# =====================================================================

class TestMetrics:
    def test_mae_perfect(self):
        idx = pd.date_range("2024-01-31", periods=5, freq="ME")
        actual = pd.Series([10, 20, 30, 40, 50], index=idx, dtype=float)
        predicted = pd.Series([10, 20, 30, 40, 50], index=idx, dtype=float)
        assert mean_absolute_error(actual, predicted) == pytest.approx(0.0)

    def test_mae_known_value(self):
        idx = pd.date_range("2024-01-31", periods=4, freq="ME")
        actual = pd.Series([10, 20, 30, 40], index=idx, dtype=float)
        predicted = pd.Series([12, 18, 33, 37], index=idx, dtype=float)
        # MAE = (2+2+3+3)/4 = 2.5
        assert mean_absolute_error(actual, predicted) == pytest.approx(2.5)

    def test_rmse_known_value(self):
        idx = pd.date_range("2024-01-31", periods=4, freq="ME")
        actual = pd.Series([10, 20, 30, 40], index=idx, dtype=float)
        predicted = pd.Series([12, 18, 33, 37], index=idx, dtype=float)
        # MSE = (4+4+9+9)/4 = 6.5, RMSE = sqrt(6.5)
        assert root_mean_squared_error(actual, predicted) == pytest.approx(np.sqrt(6.5))

    def test_mape_known_value(self):
        idx = pd.date_range("2024-01-31", periods=3, freq="ME")
        actual = pd.Series([100, 200, 50], index=idx, dtype=float)
        predicted = pd.Series([110, 190, 55], index=idx, dtype=float)
        # MAPE = 100 * mean(10/100, 10/200, 5/50) = 100 * mean(0.1, 0.05, 0.1) = 8.333...
        expected = 100.0 * np.mean([0.1, 0.05, 0.1])
        assert mean_absolute_percentage_error(actual, predicted) == pytest.approx(expected)

    def test_smape_symmetric(self):
        """SMAPE simetrik olmalı: SMAPE(a,p) == SMAPE(p,a)."""
        idx = pd.date_range("2024-01-31", periods=5, freq="ME")
        a = pd.Series([10, 20, 30, 40, 50], index=idx, dtype=float)
        p = pd.Series([12, 18, 33, 37, 55], index=idx, dtype=float)
        assert symmetric_mean_absolute_percentage_error(a, p) == pytest.approx(
            symmetric_mean_absolute_percentage_error(p, a)
        )

    def test_compute_all_metrics(self):
        idx = pd.date_range("2024-01-31", periods=5, freq="ME")
        actual = pd.Series([10, 20, 30, 40, 50], index=idx, dtype=float)
        predicted = pd.Series([11, 19, 31, 39, 51], index=idx, dtype=float)
        metrics = compute_all_metrics(actual, predicted)
        assert set(metrics.keys()) == {"mae", "rmse", "mape", "smape"}
        assert metrics["mae"] == pytest.approx(1.0)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            mean_absolute_error(pd.Series(dtype=float), pd.Series(dtype=float))


# =====================================================================
# Evaluation: Prediction Interval Coverage Testleri
# =====================================================================

class TestPredictionIntervalCoverage:
    def test_perfect_coverage(self):
        idx = pd.date_range("2024-01-31", periods=5, freq="ME")
        actual = pd.Series([10, 20, 30, 40, 50], index=idx, dtype=float)
        lower = pd.Series([5, 15, 25, 35, 45], index=idx, dtype=float)
        upper = pd.Series([15, 25, 35, 45, 55], index=idx, dtype=float)

        result = prediction_interval_coverage(actual, lower, upper)
        assert result["coverage"] == pytest.approx(1.0)
        assert result["avg_width"] == pytest.approx(10.0)

    def test_partial_coverage(self):
        idx = pd.date_range("2024-01-31", periods=4, freq="ME")
        actual = pd.Series([10, 20, 30, 40], index=idx, dtype=float)
        lower = pd.Series([5, 25, 25, 45], index=idx, dtype=float)  # 2. ve 4. dışarıda
        upper = pd.Series([15, 35, 35, 55], index=idx, dtype=float)

        result = prediction_interval_coverage(actual, lower, upper)
        assert result["coverage"] == pytest.approx(0.5)  # 2/4


# =====================================================================
# Evaluation: Time-Series CV Testleri
# =====================================================================

class TestTimeSeriesCV:
    def test_cv_runs_with_naive(self, noisy_series):
        """CV, NaiveForecaster ile sorunsuz çalışmalı."""
        cv_results = time_series_cv(
            y=noisy_series,
            forecaster=NaiveForecaster(),
            horizon=6,
            initial_window=36,
            step_size=6,
            expanding=True,
        )
        assert isinstance(cv_results, pd.DataFrame)
        assert len(cv_results) > 0
        assert "mae" in cv_results.columns
        assert "rmse" in cv_results.columns
        assert all(cv_results["mae"] >= 0)

    def test_cv_expanding_vs_rolling(self, noisy_series):
        """Expanding ve rolling CV farklı sonuç üretmeli ama ikisi de çalışmalı."""
        cv_exp = time_series_cv(
            y=noisy_series,
            forecaster=NaiveForecaster(),
            horizon=6,
            initial_window=36,
            step_size=6,
            expanding=True,
        )
        cv_roll = time_series_cv(
            y=noisy_series,
            forecaster=NaiveForecaster(),
            horizon=6,
            initial_window=36,
            step_size=6,
            expanding=False,
        )
        assert len(cv_exp) > 0
        assert len(cv_roll) > 0

    def test_cv_insufficient_data_raises(self):
        idx = pd.date_range("2024-01-31", periods=10, freq="ME")
        series = pd.Series(range(10), index=idx, dtype=float)
        with pytest.raises(ValueError, match="Yeterli veri yok"):
            time_series_cv(
                y=series,
                forecaster=NaiveForecaster(),
                horizon=6,
                initial_window=60,
            )


# =====================================================================
# Evaluation: Model Karşılaştırma Testleri
# =====================================================================

class TestCompareModels:
    def test_compare_two_models(self, noisy_series):
        comparison = compare_models(
            y=noisy_series,
            forecasters=[NaiveForecaster(), DriftForecaster()],
            horizon=6,
            initial_window=36,
            step_size=6,
        )
        assert isinstance(comparison, pd.DataFrame)
        assert len(comparison) == 2
        assert "model" in comparison.columns
        assert "mae_mean" in comparison.columns

    def test_compare_all_baseline_models(self):
        """Tüm baseline modelleri karşılaştırabilmeli."""
        idx = pd.date_range("2015-01-31", periods=120, freq="ME")
        rng = np.random.default_rng(42)
        seasonal = np.tile([10, 12, 15, 18, 22, 25, 24, 20, 16, 13, 11, 10], 10)
        values = seasonal + 0.3 * np.arange(120) + rng.normal(0, 1, 120)
        series = pd.Series(values, index=idx, name="test_series")

        comparison = compare_models(
            y=series,
            forecasters=[
                NaiveForecaster(),
                DriftForecaster(),
                SeasonalNaiveForecaster(period=12),
                MovingAverageForecaster(window=6),
            ],
            horizon=12,
            initial_window=60,
            step_size=12,
        )
        assert len(comparison) == 4
        # Sıralı olmalı (MAE'ye göre)
        mae_values = comparison["mae_mean"].dropna().values
        assert all(mae_values[i] <= mae_values[i + 1] for i in range(len(mae_values) - 1))


# =====================================================================
# Entegrasyon: Fit → Predict → Evaluate Akışı
# =====================================================================

class TestEndToEndFlow:
    def test_full_pipeline(self, noisy_series):
        """Tam akış: fit → predict → evaluate → PI coverage."""
        # Veriyi train/test böl
        train = noisy_series.iloc[:48]
        test = noisy_series.iloc[48:]
        horizon = len(test)

        # Model eğit ve tahmin et
        model = DriftForecaster()
        model.fit(train)
        forecast = model.predict(horizon)
        pi = model.predict_interval(horizon, alpha=0.10)

        # Metrikleri hesapla
        metrics = compute_all_metrics(test, forecast)
        assert metrics["mae"] > 0
        assert metrics["rmse"] >= metrics["mae"]  # RMSE >= MAE her zaman

        # PI coverage hesapla
        coverage = prediction_interval_coverage(test, pi["lower"], pi["upper"])
        assert 0.0 <= coverage["coverage"] <= 1.0
        assert coverage["avg_width"] > 0
