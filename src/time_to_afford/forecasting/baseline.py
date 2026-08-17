"""
Baseline forecasting modelleri.

Naive, Drift, Seasonal Naive ve Moving Average gibi basit benchmark modeller.
Tüm modeller BaseForecaster arayüzünü uygular ve Phase 5'teki
karmaşık modeller (ARIMA, ETS) için referans performans sağlar.
"""

from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from time_to_afford.forecasting.base import BaseForecaster
from time_to_afford.utils.logging import get_logger

logger = get_logger(__name__)


def _generate_future_index(last_date: pd.Timestamp, horizon: int) -> pd.DatetimeIndex:
    """Eğitim serisinin son tarihinden itibaren aylık gelecek tarih indeksi üretir.

    Parameters
    ----------
    last_date : pd.Timestamp
        Eğitim serisinin son tarihi.
    horizon : int
        Kaç ay ileriye tahmin yapılacağı.

    Returns
    -------
    pd.DatetimeIndex
        Aylık dönem sonu (ME) indeksi.
    """
    return pd.date_range(
        start=last_date + pd.offsets.MonthEnd(1),
        periods=horizon,
        freq="ME",
    )


# =====================================================================
# 1. Naive Forecaster
# =====================================================================

class NaiveForecaster(BaseForecaster):
    """Son gözlem değerini düz olarak ileriye taşıyan en basit benchmark model.

    Tahmin: ŷ_{T+h} = y_T  (∀ h)
    Prediction Interval: ŷ ± z_{α/2} · σ̂ · √h
    burada σ̂ = std(e_t), e_t = y_t - y_{t-1} (naive residuals)

    Bu model, daha karmaşık modellerin yenip yenemediğini ölçmek için
    alt sınır (lower bound) referansı olarak kullanılır.
    """

    def __init__(self) -> None:
        super().__init__(name="NaiveForecaster")
        self._last_value: Optional[float] = None
        self._residual_std: float = 0.0
        self._last_date: Optional[pd.Timestamp] = None
        self._n_train: int = 0

    def fit(self, y: pd.Series, **kwargs: Any) -> "NaiveForecaster":
        """Modeli eğit: son değeri ve naive residual std'yi kaydet.

        Parameters
        ----------
        y : pd.Series
            Eğitim zaman serisi. DatetimeIndex olmalı.

        Returns
        -------
        self
        """
        if y is None or len(y) < 2:
            raise ValueError("NaiveForecaster en az 2 gözlem gerektirir.")

        clean_y = y.dropna()
        self._last_value = float(clean_y.iloc[-1])
        self._last_date = clean_y.index[-1]
        self._n_train = len(clean_y)

        # Naive residuals: e_t = y_t - y_{t-1}
        residuals = clean_y.diff().dropna()
        self._residual_std = float(residuals.std(ddof=1)) if len(residuals) > 1 else 0.0

        self._is_fitted = True
        logger.info(f"NaiveForecaster eğitildi: last_value={self._last_value:.4f}, residual_std={self._residual_std:.4f}")
        return self

    def predict(self, horizon: int, **kwargs: Any) -> pd.Series:
        """Nokta tahmini üret: son değeri horizon adım tekrarla.

        Parameters
        ----------
        horizon : int
            Kaç adım ileri tahmin.

        Returns
        -------
        pd.Series
            Sabit tahmin serisi.
        """
        self._check_is_fitted()
        if horizon < 1:
            raise ValueError(f"horizon >= 1 olmalıdır, alınan: {horizon}")

        future_index = _generate_future_index(self._last_date, horizon)
        forecast = pd.Series(
            data=np.full(horizon, self._last_value),
            index=future_index,
            name="naive_forecast",
        )
        return forecast

    def predict_interval(
        self, horizon: int, alpha: float = 0.05, **kwargs: Any
    ) -> pd.DataFrame:
        """Tahmin aralığı üret.

        Naive PI: ŷ ± z_{α/2} · σ̂ · √h

        Parameters
        ----------
        horizon : int
            Kaç adım ileri tahmin.
        alpha : float
            Anlamlılık düzeyi.

        Returns
        -------
        pd.DataFrame
            'lower' ve 'upper' sütunları.
        """
        self._check_is_fitted()
        forecast = self.predict(horizon)
        z = sp_stats.norm.ppf(1 - alpha / 2)

        h_steps = np.arange(1, horizon + 1)
        margin = z * self._residual_std * np.sqrt(h_steps)

        return pd.DataFrame(
            {"lower": forecast.values - margin, "upper": forecast.values + margin},
            index=forecast.index,
        )


# =====================================================================
# 2. Drift Forecaster
# =====================================================================

class DriftForecaster(BaseForecaster):
    """Doğrusal drift (eğilim) ile ileriye taşıyan model.

    Tahmin: ŷ_{T+h} = y_T + h · d,  d = (y_T - y_1) / (T - 1)
    Prediction Interval: ŷ ± z_{α/2} · σ̂ · √(h · (1 + h/T))
    burada σ̂ = std(e_t), e_t = y_t - y_{t-1} - d

    Eğer serinin belirgin bir trendi varsa Naive'den daha iyi
    performans göstermesi beklenir.
    """

    def __init__(self) -> None:
        super().__init__(name="DriftForecaster")
        self._last_value: Optional[float] = None
        self._drift: float = 0.0
        self._residual_std: float = 0.0
        self._last_date: Optional[pd.Timestamp] = None
        self._n_train: int = 0

    def fit(self, y: pd.Series, **kwargs: Any) -> "DriftForecaster":
        """Modeli eğit: drift katsayısını ve residual std'yi hesapla.

        Parameters
        ----------
        y : pd.Series
            Eğitim zaman serisi.

        Returns
        -------
        self
        """
        if y is None or len(y) < 2:
            raise ValueError("DriftForecaster en az 2 gözlem gerektirir.")

        clean_y = y.dropna()
        n = len(clean_y)
        self._last_value = float(clean_y.iloc[-1])
        self._last_date = clean_y.index[-1]
        self._n_train = n

        # Drift = (y_T - y_1) / (T - 1)
        self._drift = float((clean_y.iloc[-1] - clean_y.iloc[0]) / (n - 1))

        # Drift residuals: e_t = (y_t - y_{t-1}) - drift
        naive_diffs = clean_y.diff().dropna()
        drift_residuals = naive_diffs - self._drift
        self._residual_std = float(drift_residuals.std(ddof=1)) if len(drift_residuals) > 1 else 0.0

        self._is_fitted = True
        logger.info(f"DriftForecaster eğitildi: drift={self._drift:.6f}, residual_std={self._residual_std:.4f}")
        return self

    def predict(self, horizon: int, **kwargs: Any) -> pd.Series:
        """Nokta tahmini üret: son değere drift ekle.

        Parameters
        ----------
        horizon : int
            Kaç adım ileri tahmin.

        Returns
        -------
        pd.Series
            Drift tabanlı tahmin serisi.
        """
        self._check_is_fitted()
        if horizon < 1:
            raise ValueError(f"horizon >= 1 olmalıdır, alınan: {horizon}")

        future_index = _generate_future_index(self._last_date, horizon)
        h_steps = np.arange(1, horizon + 1)
        forecast_values = self._last_value + self._drift * h_steps

        return pd.Series(
            data=forecast_values,
            index=future_index,
            name="drift_forecast",
        )

    def predict_interval(
        self, horizon: int, alpha: float = 0.05, **kwargs: Any
    ) -> pd.DataFrame:
        """Tahmin aralığı üret.

        Drift PI: ŷ ± z_{α/2} · σ̂ · √(h · (1 + h/T))

        Parameters
        ----------
        horizon : int
            Kaç adım ileri tahmin.
        alpha : float
            Anlamlılık düzeyi.

        Returns
        -------
        pd.DataFrame
            'lower' ve 'upper' sütunları.
        """
        self._check_is_fitted()
        forecast = self.predict(horizon)
        z = sp_stats.norm.ppf(1 - alpha / 2)

        h_steps = np.arange(1, horizon + 1)
        margin = z * self._residual_std * np.sqrt(h_steps * (1.0 + h_steps / self._n_train))

        return pd.DataFrame(
            {"lower": forecast.values - margin, "upper": forecast.values + margin},
            index=forecast.index,
        )


# =====================================================================
# 3. Seasonal Naive Forecaster
# =====================================================================

class SeasonalNaiveForecaster(BaseForecaster):
    """Geçen sezonun aynı dönemini tekrarlayan mevsimsel benchmark model.

    Tahmin: ŷ_{T+h} = y_{T+h-m·⌈h/m⌉}  (son sezonun ilgili ayını tekrarlar)
    burada m = mevsimsel periyot (varsayılan 12 ay).

    Prediction Interval: ŷ ± z_{α/2} · σ̂ · √⌈h/m⌉
    burada σ̂ = std(mevsimsel residuals), e_t = y_t - y_{t-m}

    Güçlü mevsimsel desen gösteren seriler (ör: TÜFE) için etkilidir.
    """

    def __init__(self, period: int = 12) -> None:
        super().__init__(name="SeasonalNaiveForecaster")
        self._period = period
        self._seasonal_values: Optional[np.ndarray] = None
        self._residual_std: float = 0.0
        self._last_date: Optional[pd.Timestamp] = None
        self._n_train: int = 0

    def fit(self, y: pd.Series, **kwargs: Any) -> "SeasonalNaiveForecaster":
        """Modeli eğit: son sezonun değerlerini ve mevsimsel residual std'yi kaydet.

        Parameters
        ----------
        y : pd.Series
            Eğitim zaman serisi. En az `period` + 1 gözlem gerektirir.

        Returns
        -------
        self
        """
        if y is None or len(y) < self._period + 1:
            raise ValueError(
                f"SeasonalNaiveForecaster en az {self._period + 1} gözlem gerektirir "
                f"(period={self._period}), alınan: {len(y) if y is not None else 0}."
            )

        clean_y = y.dropna()
        self._last_date = clean_y.index[-1]
        self._n_train = len(clean_y)

        # Son sezonun değerlerini al
        self._seasonal_values = clean_y.iloc[-self._period:].values.copy()

        # Mevsimsel residuals: e_t = y_t - y_{t-m}
        seasonal_residuals = clean_y.values[self._period:] - clean_y.values[:-self._period]
        self._residual_std = float(np.std(seasonal_residuals, ddof=1)) if len(seasonal_residuals) > 1 else 0.0

        self._is_fitted = True
        logger.info(
            f"SeasonalNaiveForecaster eğitildi: period={self._period}, "
            f"residual_std={self._residual_std:.4f}"
        )
        return self

    def predict(self, horizon: int, **kwargs: Any) -> pd.Series:
        """Nokta tahmini üret: son sezonu döngüsel olarak tekrarla.

        Parameters
        ----------
        horizon : int
            Kaç adım ileri tahmin.

        Returns
        -------
        pd.Series
            Mevsimsel naive tahmin serisi.
        """
        self._check_is_fitted()
        if horizon < 1:
            raise ValueError(f"horizon >= 1 olmalıdır, alınan: {horizon}")

        future_index = _generate_future_index(self._last_date, horizon)

        # Son sezonu döngüsel tekrarla
        forecast_values = np.array([
            self._seasonal_values[i % self._period] for i in range(horizon)
        ])

        return pd.Series(
            data=forecast_values,
            index=future_index,
            name="seasonal_naive_forecast",
        )

    def predict_interval(
        self, horizon: int, alpha: float = 0.05, **kwargs: Any
    ) -> pd.DataFrame:
        """Tahmin aralığı üret.

        Seasonal Naive PI: ŷ ± z_{α/2} · σ̂ · √⌈h/m⌉

        Parameters
        ----------
        horizon : int
            Kaç adım ileri tahmin.
        alpha : float
            Anlamlılık düzeyi.

        Returns
        -------
        pd.DataFrame
            'lower' ve 'upper' sütunları.
        """
        self._check_is_fitted()
        forecast = self.predict(horizon)
        z = sp_stats.norm.ppf(1 - alpha / 2)

        h_steps = np.arange(1, horizon + 1)
        # Kaçıncı tam sezon tekrarı olduğunu hesapla
        k_repeats = np.ceil(h_steps / self._period)
        margin = z * self._residual_std * np.sqrt(k_repeats)

        return pd.DataFrame(
            {"lower": forecast.values - margin, "upper": forecast.values + margin},
            index=forecast.index,
        )


# =====================================================================
# 4. Moving Average Forecaster
# =====================================================================

class MovingAverageForecaster(BaseForecaster):
    """Son k gözlemin ortalamasını ileriye taşıyan model.

    Tahmin: ŷ_{T+h} = (1/k) · Σ_{i=0}^{k-1} y_{T-i}  (∀ h, sabit)
    Prediction Interval: ŷ ± z_{α/2} · σ̂ · √(1 + 1/k)
    burada σ̂ = std(MA residuals)

    Parameters
    ----------
    window : int, default 6
        Hareketli ortalama pencere boyutu (ay).
    """

    def __init__(self, window: int = 6) -> None:
        super().__init__(name=f"MovingAverageForecaster(k={window})")
        if window < 1:
            raise ValueError(f"window >= 1 olmalıdır, alınan: {window}")
        self._window = window
        self._ma_value: Optional[float] = None
        self._residual_std: float = 0.0
        self._last_date: Optional[pd.Timestamp] = None
        self._n_train: int = 0

    def fit(self, y: pd.Series, **kwargs: Any) -> "MovingAverageForecaster":
        """Modeli eğit: son k gözlemin ortalamasını ve MA residual std'yi hesapla.

        Parameters
        ----------
        y : pd.Series
            Eğitim zaman serisi. En az `window` + 1 gözlem gerektirir.

        Returns
        -------
        self
        """
        if y is None or len(y) < self._window + 1:
            raise ValueError(
                f"MovingAverageForecaster en az {self._window + 1} gözlem gerektirir "
                f"(window={self._window}), alınan: {len(y) if y is not None else 0}."
            )

        clean_y = y.dropna()
        self._last_date = clean_y.index[-1]
        self._n_train = len(clean_y)

        # Son k gözlemin ortalaması
        self._ma_value = float(clean_y.iloc[-self._window:].mean())

        # MA residuals: her t için, o anki k-pencereli MA ile gerçek değer farkı
        ma_series = clean_y.rolling(window=self._window).mean()
        residuals = (clean_y - ma_series).dropna()
        self._residual_std = float(residuals.std(ddof=1)) if len(residuals) > 1 else 0.0

        self._is_fitted = True
        logger.info(
            f"MovingAverageForecaster eğitildi: window={self._window}, "
            f"ma_value={self._ma_value:.4f}, residual_std={self._residual_std:.4f}"
        )
        return self

    def predict(self, horizon: int, **kwargs: Any) -> pd.Series:
        """Nokta tahmini üret: MA değerini sabit olarak tekrarla.

        Parameters
        ----------
        horizon : int
            Kaç adım ileri tahmin.

        Returns
        -------
        pd.Series
            Sabit MA tahmin serisi.
        """
        self._check_is_fitted()
        if horizon < 1:
            raise ValueError(f"horizon >= 1 olmalıdır, alınan: {horizon}")

        future_index = _generate_future_index(self._last_date, horizon)
        forecast_values = np.full(horizon, self._ma_value)

        return pd.Series(
            data=forecast_values,
            index=future_index,
            name="ma_forecast",
        )

    def predict_interval(
        self, horizon: int, alpha: float = 0.05, **kwargs: Any
    ) -> pd.DataFrame:
        """Tahmin aralığı üret.

        MA PI: ŷ ± z_{α/2} · σ̂ · √(1 + 1/k)  (sabit genişlik)

        Parameters
        ----------
        horizon : int
            Kaç adım ileri tahmin.
        alpha : float
            Anlamlılık düzeyi.

        Returns
        -------
        pd.DataFrame
            'lower' ve 'upper' sütunları.
        """
        self._check_is_fitted()
        forecast = self.predict(horizon)
        z = sp_stats.norm.ppf(1 - alpha / 2)

        margin = z * self._residual_std * np.sqrt(1.0 + 1.0 / self._window)

        return pd.DataFrame(
            {
                "lower": forecast.values - margin,
                "upper": forecast.values + margin,
            },
            index=forecast.index,
        )
