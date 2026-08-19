"""
Forecasting temel sınıfı.

Tüm forecasting modellerinin ortak interface'ini tanımlayan
abstract base class.
"""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd


class BaseForecaster(ABC):
    """Tüm forecasting modellerinin uyması gereken ortak arayüz.

    Simulation engine bu arayüz üzerinden çalışır; böylece
    altındaki model değiştirildiğinde simulation kodunun
    değişmesine gerek kalmaz.
    """

    def __init__(self, name: str = "BaseForecaster") -> None:
        self.name = name
        self._is_fitted: bool = False

    @abstractmethod
    def fit(self, y: pd.Series, **kwargs: Any) -> "BaseForecaster":
        """Modeli eğit.

        Parameters
        ----------
        y : pd.Series
            Eğitim zaman serisi. Index datetime olmalı.

        Returns
        -------
        self
        """

    @abstractmethod
    def predict(self, horizon: int, **kwargs: Any) -> pd.Series:
        """Nokta tahmini üret.

        Parameters
        ----------
        horizon : int
            Kaç adım ileri tahmin yapılacağı.

        Returns
        -------
        pd.Series
            Tahmin değerleri. Index gelecek tarihleri içermeli.
        """

    @abstractmethod
    def predict_interval(
        self, horizon: int, alpha: float = 0.05, **kwargs: Any
    ) -> pd.DataFrame:
        """Tahmin aralığı üret.

        Parameters
        ----------
        horizon : int
            Kaç adım ileri tahmin yapılacağı.
        alpha : float
            Anlamlılık düzeyi. Örn: 0.05 → %95 güven aralığı.

        Returns
        -------
        pd.DataFrame
            'lower' ve 'upper' sütunlarını içeren DataFrame.
        """
    @abstractmethod
    def simulate_paths(
        self, horizon: int, n_paths: int, random_state: int | None = None
    ) -> np.ndarray:
        """Belirtilen ufuk ve sayıda stokastik simülasyon patikası üretir.

        Monte Carlo simülasyonu için kullanılır. Modelin kendi istatistiksel
        dağılımını ve belirsizliğini (uncertainty) baz alır.

        Parameters
        ----------
        horizon : int
            Kaç adım ileri tahmin yapılacağı.
        n_paths : int
            Üretilecek rassal patika sayısı.
        random_state : int | None, optional
            Tekrarlanabilirlik için seed.

        Returns
        -------
        np.ndarray
            (n_paths, horizon) boyutunda numpy dizisi.
        """
    @property
    def is_fitted(self) -> bool:
        """Modelin eğitilip eğitilmediğini döndürür."""
        return self._is_fitted

    def _check_is_fitted(self) -> None:
        """Model eğitilmemişse hata fırlat."""
        if not self._is_fitted:
            raise RuntimeError(
                f"{self.name} henüz eğitilmedi. Önce fit() çağırın."
            )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', fitted={self._is_fitted})"
