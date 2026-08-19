"""
ARIMA/ETS modelleri.
Phase 4.3 Statistical Forecasting Models.
"""
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import warnings

from time_to_afford.forecasting.base import BaseForecaster
import logging

logger = logging.getLogger(__name__)

class ARIMAForecaster(BaseForecaster):
    """ARIMA (p, d, q) istatistiksel tahmin modeli."""

    def __init__(self, order: Tuple[int, int, int] = (1, 0, 0), name: str = "ARIMA") -> None:
        super().__init__(name=name)
        self.order = order
        self._model_fit = None

    def fit(self, y: pd.Series, **kwargs: Any) -> "ARIMAForecaster":
        """Modeli eğit.

        Parametreler statiktir.
        Auto-selection için statik metot (select_best_arima) kullanılmalıdır.
        """
        y_valid = y.dropna()
        if len(y_valid) < max(self.order) * 2 + 3:
            raise ValueError(f"Seri ARIMA{self.order} modelini tahmin etmek için çok kısa.")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            warnings.filterwarnings("ignore")
            try:
                # model = sm.tsa.ARIMA(y_valid, order=self.order)
                # Note: For strict adherence to no-future-data, the fit is in-sample only.
                model = sm.tsa.ARIMA(y_valid, order=self.order)
                self._model_fit = model.fit()
            except Exception as e:
                raise RuntimeError(f"ARIMA fit başarısız (order={self.order}): {e}")

        self._is_fitted = True
        return self

    def predict(self, horizon: int, **kwargs: Any) -> pd.Series:
        self._check_is_fitted()
        if horizon < 1:
            raise ValueError(f"horizon >= 1 olmalıdır, alınan: {horizon}")

        forecast = self._model_fit.forecast(steps=horizon)
        return pd.Series(forecast.values, name="predict")

    def predict_interval(
        self, horizon: int, alpha: float = 0.05, **kwargs: Any
    ) -> pd.DataFrame:
        self._check_is_fitted()
        if horizon < 1:
            raise ValueError(f"horizon >= 1 olmalıdır, alınan: {horizon}")
        if not (0 < alpha < 1):
            raise ValueError(f"alpha (0, 1) aralığında olmalıdır, alınan: {alpha}")

        pred_res = self._model_fit.get_forecast(steps=horizon)
        conf_int = pred_res.conf_int(alpha=alpha)

        # statsmodels conf_int column names: "lower {series_name}", "upper {series_name}"
        # We need "lower" and "upper"
        return pd.DataFrame({
            "lower": conf_int.iloc[:, 0].values,
            "upper": conf_int.iloc[:, 1].values
        })

    def simulate_paths(
        self, horizon: int, n_paths: int, random_state: int | None = None
    ) -> np.ndarray:
        self._check_is_fitted()
        if horizon < 1:
            raise ValueError(f"horizon >= 1 olmalıdır, alınan: {horizon}")
        if n_paths < 1:
            raise ValueError(f"n_paths >= 1 olmalıdır, alınan: {n_paths}")

        sim = self._model_fit.simulate(
            nsimulations=horizon,
            repetitions=n_paths,
            anchor='end',
            random_state=random_state
        )

        # sim shape is (horizon, n_paths). We need (n_paths, horizon)
        sim_array = np.asarray(sim)

        # statsmodels.simulate returns (horizon,) if repetitions=1 or (horizon, repetitions) if >1
        if sim_array.ndim == 1:
            sim_array = sim_array.reshape(horizon, 1)

        return sim_array.T


def cv_evaluate_arima_order(y: pd.Series, order: Tuple[int, int, int], initial_window: int, step: int, horizon: int) -> float:
    """Tek bir ARIMA order kombinasyonunu expanding window CV ile test eder.

    Veri sızıntısını engellemek için, fit işlemi yalnızca geçmiş veri üzerinde
    yapılır ve gelecekteki değerler yalnızca test hatası (RMSE) hesaplanırken okunur.
    """
    n = len(y)
    if initial_window + horizon > n:
        return np.inf

    errors = []

    start_idx = initial_window
    while start_idx + horizon <= n:
        # Leakage prevention: strictly subset data before fitting
        train = y.iloc[:start_idx]
        test = y.iloc[start_idx:start_idx+horizon]

        model = ARIMAForecaster(order=order)
        try:
            model.fit(train)
            pred = model.predict(horizon)

            # calculate metrics
            mse = np.mean((test.values - pred.values) ** 2)
            errors.append(np.sqrt(mse)) # RMSE

        except Exception:
            # Fit başarısız olursa aşırı penaltılandır.
            # (Bu modelin candidate listesinden düşmesi sağlanır)
            return np.inf

        start_idx += step

    if not errors:
        return np.inf

    return float(np.mean(errors))


def select_best_arima(
    y: pd.Series,
    p_choices: List[int] = [0, 1, 2],
    d_choices: List[int] = [0, 1],
    q_choices: List[int] = [0, 1, 2],
    initial_window: int = 36,
    step: int = 12,
    horizon: int = 6
) -> ARIMAForecaster:
    """Data-Driven olarak Expanding-Window CV kullanarak en iyi ARIMA modelini seçer.

    DİKKAT: Log-return dönüşümü yapıldığında d=0 önceliklidir (zaten birinci fark alınmıştır).
    Eğer d=1 daha iyi performans gösterirse ikinci fark anlamında kabul edilir.
    Eşit RMSE skorunda (lexicographical order), her zaman d=0 ve daha az karmaşık olan (küçük p, q)
    tercih edilir (Ockham'ın usturası).
    """
    best_order = None
    best_score = np.inf

    for d in d_choices:
        for p in p_choices:
            for q in q_choices:
                order = (p, d, q)
                score = cv_evaluate_arima_order(y, order, initial_window, step, horizon)

                if score < best_score:
                    best_score = score
                    best_order = order

    if best_order is None or best_score == np.inf:
        raise RuntimeError("Hiçbir ARIMA konfigürasyonu verilere fit edilemedi.")

    logger.info(f"CV ile seçilen en iyi ARIMA order: {best_order}, RMSE: {best_score:.4f}")

    final_model = ARIMAForecaster(order=best_order)
    final_model.fit(y)
    return final_model
