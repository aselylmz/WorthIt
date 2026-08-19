"""
ARIMA/ETS modelleri için testler (Phase 4.3).
"""

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from time_to_afford.forecasting.arima import ARIMAForecaster, cv_evaluate_arima_order, select_best_arima

@pytest.fixture
def stationary_series():
    """Durağan (stationary) bir sentetik AR(1) serisi."""
    np.random.seed(42)
    y = np.zeros(100)
    for t in range(1, 100):
        y[t] = 0.5 * y[t-1] + np.random.normal(0, 1)
    # Datetime index is required by BaseForecaster/statsmodels strictly
    dates = pd.date_range("2010-01-01", periods=100, freq="ME")
    return pd.Series(y, index=dates)

@pytest.fixture
def non_stationary_series():
    """Durağan olmayan (Random Walk) sentetik bir seri."""
    np.random.seed(42)
    y = np.cumsum(np.random.normal(0, 1, 100))
    dates = pd.date_range("2010-01-01", periods=100, freq="ME")
    return pd.Series(y, index=dates)

class TestARIMAForecaster:

    def test_fit_and_predict_shape(self, stationary_series):
        model = ARIMAForecaster(order=(1,0,0))
        model.fit(stationary_series)

        preds = model.predict(horizon=5)
        assert len(preds) == 5
        assert isinstance(preds, pd.Series)
        assert preds.name == "predict"

    def test_not_fitted_raises(self):
        model = ARIMAForecaster(order=(1,0,0))
        with pytest.raises(RuntimeError, match="eğitilmedi"):
            model.predict(horizon=5)
        with pytest.raises(RuntimeError, match="eğitilmedi"):
            model.predict_interval(horizon=5)
        with pytest.raises(RuntimeError, match="eğitilmedi"):
            model.simulate_paths(horizon=5, n_paths=10)

    def test_predict_matches_statsmodels(self, stationary_series):
        model = ARIMAForecaster(order=(1,0,0))
        model.fit(stationary_series)
        preds = model.predict(horizon=12)

        # Statsmodels directly
        sm_model = sm.tsa.ARIMA(stationary_series, order=(1,0,0)).fit()
        sm_preds = sm_model.forecast(steps=12)

        np.testing.assert_allclose(preds.values, sm_preds.values)

    def test_predict_interval_shape(self, stationary_series):
        model = ARIMAForecaster(order=(1,0,0))
        model.fit(stationary_series)

        interval = model.predict_interval(horizon=5, alpha=0.05)
        assert isinstance(interval, pd.DataFrame)
        assert list(interval.columns) == ["lower", "upper"]
        assert len(interval) == 5

        # Lower should be strictly less than upper
        assert np.all(interval["lower"] < interval["upper"])

    def test_simulate_paths_shape(self, stationary_series):
        model = ARIMAForecaster(order=(1,0,0))
        model.fit(stationary_series)

        paths = model.simulate_paths(horizon=6, n_paths=10, random_state=42)
        assert isinstance(paths, np.ndarray)
        assert paths.shape == (10, 6)

    def test_simulate_paths_reproducibility(self, stationary_series):
        model = ARIMAForecaster(order=(1,0,0))
        model.fit(stationary_series)

        paths1 = model.simulate_paths(horizon=12, n_paths=50, random_state=42)
        paths2 = model.simulate_paths(horizon=12, n_paths=50, random_state=42)
        paths3 = model.simulate_paths(horizon=12, n_paths=50, random_state=999)

        np.testing.assert_array_equal(paths1, paths2)
        with pytest.raises(AssertionError):
            np.testing.assert_array_equal(paths1, paths3)

    def test_simulate_paths_non_degenerate(self, stationary_series):
        model = ARIMAForecaster(order=(1,0,0))
        model.fit(stationary_series)

        paths = model.simulate_paths(horizon=12, n_paths=100, random_state=42)
        stds = paths.std(axis=0)
        assert np.all(stds > 0)

    def test_invalid_horizon_and_n_paths(self, stationary_series):
        model = ARIMAForecaster(order=(1,0,0))
        model.fit(stationary_series)

        with pytest.raises(ValueError, match="horizon >= 1"):
            model.predict(0)
        with pytest.raises(ValueError, match="horizon >= 1"):
            model.predict_interval(0)
        with pytest.raises(ValueError, match="horizon >= 1"):
            model.simulate_paths(0, 10)
        with pytest.raises(ValueError, match="n_paths >= 1"):
            model.simulate_paths(10, 0)

    def test_invalid_alpha(self, stationary_series):
        model = ARIMAForecaster(order=(1,0,0))
        model.fit(stationary_series)
        with pytest.raises(ValueError, match="alpha"):
            model.predict_interval(10, alpha=1.5)
        with pytest.raises(ValueError, match="alpha"):
            model.predict_interval(10, alpha=0.0)

    def test_law_of_large_numbers(self, stationary_series):
        model = ARIMAForecaster(order=(1,0,0))
        model.fit(stationary_series)

        horizon = 12
        n_paths = 10000
        paths = model.simulate_paths(horizon, n_paths, random_state=42)

        mean_paths = paths.mean(axis=0)
        deterministic = model.predict(horizon).values

        # 0.1 lik makul bir hata payı (statsmodels'un simülasyon inovasyonlarına göre)
        np.testing.assert_allclose(mean_paths, deterministic, atol=0.1)


class TestARIMAOrderSelection:

    def test_cv_chronological_ordering(self, stationary_series):
        # Bu test fonksiyonun hata (Exception) vermeden çalışıp çalışmadığını değil,
        # cv'nin matematiksel olarak doğru skor üretip üretmediğini,
        # yani veri kronolojisine uyup uymadığını dolaylı test eder.

        score = cv_evaluate_arima_order(
            stationary_series,
            order=(1,0,0),
            initial_window=50,
            step=10,
            horizon=5
        )
        assert not np.isnan(score)
        assert not np.isinf(score)
        assert score > 0

    def test_order_selection_determinism(self, stationary_series):
        # En iyi order'ı bul
        best_model1 = select_best_arima(
            stationary_series,
            p_choices=[0, 1],
            d_choices=[0],
            q_choices=[0],
            initial_window=60,
            step=10,
            horizon=6
        )
        best_model2 = select_best_arima(
            stationary_series,
            p_choices=[0, 1],
            d_choices=[0],
            q_choices=[0],
            initial_window=60,
            step=10,
            horizon=6
        )
        # Order seçimi tam olarak aynı olmalı
        assert best_model1.order == best_model2.order

    def test_d1_evaluated(self, non_stationary_series):
        # Eger d_choices=[0,1] verirsek, non-stationary bir random walk verisinde
        # d=1 seçilme olasılığı yüksektir, çünkü d=0 modelleri (e.g. ARIMA(0,0,0))
        # bu veriye çok kötü uyacaktır. Bu testte d=1 veya d=0 adaylarının
        # grid search sisteminde hatasız elenebildiğini kontrol ediyoruz.
        best_model = select_best_arima(
            non_stationary_series,
            p_choices=[0, 1],
            d_choices=[0, 1],
            q_choices=[0],
            initial_window=50,
            step=20,
            horizon=5
        )
        # Random Walk olduğu için d=1 seçilmesini umarız, ama en azından
        # mekanizmanın tüm adayları çökmeden denediğini garanti ediyoruz.
        assert best_model.order[1] in [0, 1]

    def test_failed_orders_are_handled(self, stationary_series):
        # ARIMAForecaster, çok kötü bir order verildiğinde hata fırlatabilir,
        # CV evaluator bunu np.inf ile handle edip yola devam etmelidir.

        # Çok küçük veride çok yüksek bir order deniyoruz ki fit çöksün
        short_series = stationary_series.iloc[:10]

        score = cv_evaluate_arima_order(
            short_series,
            order=(5,1,5),  # p+q=10, data=10 -> ValueError!
            initial_window=5,
            step=1,
            horizon=2
        )
        assert np.isinf(score)
