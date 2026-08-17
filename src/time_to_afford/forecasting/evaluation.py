"""
Forecasting model değerlendirme araçları.

Hata metrikleri (MAE, RMSE, MAPE, SMAPE), prediction interval coverage,
zaman serisi cross-validation (rolling/expanding window) ve
model karşılaştırma tablosu üretim fonksiyonları.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union

import numpy as np
import pandas as pd

from time_to_afford.forecasting.base import BaseForecaster
from time_to_afford.utils.logging import get_logger

logger = get_logger(__name__)


# =====================================================================
# 1. Nokta Tahmin Hata Metrikleri
# =====================================================================

def mean_absolute_error(actual: pd.Series, predicted: pd.Series) -> float:
    """Ortalama mutlak hata (MAE) hesaplar.

    MAE = (1/n) · Σ |y_t - ŷ_t|

    Parameters
    ----------
    actual : pd.Series
        Gerçekleşen değerler.
    predicted : pd.Series
        Tahmin değerleri.

    Returns
    -------
    float
        MAE değeri.
    """
    a, p = _align_series(actual, predicted)
    return float(np.mean(np.abs(a - p)))


def root_mean_squared_error(actual: pd.Series, predicted: pd.Series) -> float:
    """Kök ortalama kare hatası (RMSE) hesaplar.

    RMSE = √((1/n) · Σ (y_t - ŷ_t)²)

    Parameters
    ----------
    actual : pd.Series
        Gerçekleşen değerler.
    predicted : pd.Series
        Tahmin değerleri.

    Returns
    -------
    float
        RMSE değeri.
    """
    a, p = _align_series(actual, predicted)
    return float(np.sqrt(np.mean((a - p) ** 2)))


def mean_absolute_percentage_error(actual: pd.Series, predicted: pd.Series) -> float:
    """Ortalama mutlak yüzde hatası (MAPE) hesaplar.

    MAPE = (100/n) · Σ |y_t - ŷ_t| / |y_t|
    Sıfır gerçekleşen değerler hesaptan çıkarılır.

    Parameters
    ----------
    actual : pd.Series
        Gerçekleşen değerler (sıfır olmamalı).
    predicted : pd.Series
        Tahmin değerleri.

    Returns
    -------
    float
        MAPE değeri (yüzde, ör: 5.3 = %5.3).
    """
    a, p = _align_series(actual, predicted)
    # Sıfır olan gerçekleşen değerleri maskele
    mask = np.abs(a) > 1e-10
    if not mask.any():
        return np.inf
    return float(100.0 * np.mean(np.abs((a[mask] - p[mask]) / a[mask])))


def symmetric_mean_absolute_percentage_error(
    actual: pd.Series, predicted: pd.Series
) -> float:
    """Simetrik ortalama mutlak yüzde hatası (SMAPE) hesaplar.

    SMAPE = (200/n) · Σ |y_t - ŷ_t| / (|y_t| + |ŷ_t|)
    Her iki tarafın da sıfır olduğu gözlemler çıkarılır.

    Parameters
    ----------
    actual : pd.Series
        Gerçekleşen değerler.
    predicted : pd.Series
        Tahmin değerleri.

    Returns
    -------
    float
        SMAPE değeri (yüzde).
    """
    a, p = _align_series(actual, predicted)
    denominator = np.abs(a) + np.abs(p)
    mask = denominator > 1e-10
    if not mask.any():
        return np.inf
    return float(200.0 * np.mean(np.abs(a[mask] - p[mask]) / denominator[mask]))


def compute_all_metrics(actual: pd.Series, predicted: pd.Series) -> Dict[str, float]:
    """Tüm standart hata metriklerini tek seferde hesaplar.

    Parameters
    ----------
    actual : pd.Series
        Gerçekleşen değerler.
    predicted : pd.Series
        Tahmin değerleri.

    Returns
    -------
    dict
        {'mae', 'rmse', 'mape', 'smape'} metrik sözlüğü.
    """
    return {
        "mae": mean_absolute_error(actual, predicted),
        "rmse": root_mean_squared_error(actual, predicted),
        "mape": mean_absolute_percentage_error(actual, predicted),
        "smape": symmetric_mean_absolute_percentage_error(actual, predicted),
    }


def _align_series(
    actual: pd.Series, predicted: pd.Series
) -> Tuple[np.ndarray, np.ndarray]:
    """İki seriyi hizalar ve numpy array'e dönüştürür.

    Parameters
    ----------
    actual, predicted : pd.Series
        Hizalanacak seriler.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Hizalanmış (actual, predicted) çifti.

    Raises
    ------
    ValueError
        Seriler boş veya hizalanamıyorsa.
    """
    if actual is None or predicted is None:
        raise ValueError("actual ve predicted serileri None olamaz.")

    if isinstance(actual.index, pd.DatetimeIndex) and isinstance(predicted.index, pd.DatetimeIndex):
        common_idx = actual.index.intersection(predicted.index)
        if len(common_idx) == 0:
            raise ValueError("actual ve predicted serileri arasında ortak tarih bulunamadı.")
        a = actual.loc[common_idx].values.astype(float)
        p = predicted.loc[common_idx].values.astype(float)
    else:
        min_len = min(len(actual), len(predicted))
        if min_len == 0:
            raise ValueError("Seriler boş olamaz.")
        a = actual.values[:min_len].astype(float)
        p = predicted.values[:min_len].astype(float)

    # NaN'ları çıkar
    valid_mask = ~(np.isnan(a) | np.isnan(p))
    if not valid_mask.any():
        raise ValueError("Hizalama sonrası geçerli gözlem kalmadı (tümü NaN).")

    return a[valid_mask], p[valid_mask]


# =====================================================================
# 2. Prediction Interval Değerlendirme
# =====================================================================

def prediction_interval_coverage(
    actual: pd.Series,
    lower: pd.Series,
    upper: pd.Series,
) -> Dict[str, float]:
    """Tahmin aralığının kapsama oranını ve ortalama genişliğini hesaplar.

    Parameters
    ----------
    actual : pd.Series
        Gerçekleşen değerler.
    lower : pd.Series
        Alt sınır.
    upper : pd.Series
        Üst sınır.

    Returns
    -------
    dict
        'coverage': gerçekleşen değerlerin aralık içinde kalma oranı (0-1),
        'avg_width': ortalama aralık genişliği.
    """
    # Ortak indeks üzerinde hizala
    common_idx = actual.index.intersection(lower.index).intersection(upper.index)
    if len(common_idx) == 0:
        raise ValueError("Seriler arasında ortak tarih bulunamadı.")

    a = actual.loc[common_idx].values.astype(float)
    lo = lower.loc[common_idx].values.astype(float)
    up = upper.loc[common_idx].values.astype(float)

    valid_mask = ~(np.isnan(a) | np.isnan(lo) | np.isnan(up))
    a, lo, up = a[valid_mask], lo[valid_mask], up[valid_mask]

    if len(a) == 0:
        raise ValueError("Geçerli gözlem bulunamadı.")

    within = (a >= lo) & (a <= up)
    coverage = float(np.mean(within))
    avg_width = float(np.mean(up - lo))

    return {"coverage": coverage, "avg_width": avg_width}


# =====================================================================
# 3. Time-Series Cross-Validation
# =====================================================================

def time_series_cv(
    y: pd.Series,
    forecaster: BaseForecaster,
    horizon: int = 12,
    initial_window: int = 60,
    step_size: int = 1,
    expanding: bool = True,
) -> pd.DataFrame:
    """Zaman serisi cross-validation (rolling veya expanding window).

    Zaman serisi verisinde gelecek bilgisinin sızmaması (leakage) için
    yalnızca geçmiş veriler eğitim, gelecek veriler test seti olarak kullanılır.

    Parameters
    ----------
    y : pd.Series
        Tam zaman serisi.
    forecaster : BaseForecaster
        Değerlendirilecek model (her fold'da yeniden eğitilir).
    horizon : int, default 12
        Her fold'da kaç adım ileri tahmin yapılacağı.
    initial_window : int, default 60
        İlk eğitim penceresi boyutu (ay).
    step_size : int, default 1
        Pencere kayma adımı (ay).
    expanding : bool, default True
        True: pencere genişler (expanding). False: sabit boyut kalır (rolling).

    Returns
    -------
    pd.DataFrame
        Her fold için hata metrikleri tablosu.
        Sütunlar: ['fold', 'train_end', 'test_start', 'test_end', 'n_test',
                    'mae', 'rmse', 'mape', 'smape']
    """
    n = len(y)
    if n < initial_window + horizon:
        raise ValueError(
            f"Yeterli veri yok: serinin uzunluğu ({n}) en az "
            f"initial_window ({initial_window}) + horizon ({horizon}) = "
            f"{initial_window + horizon} olmalıdır."
        )

    results = []
    fold = 0

    for split_point in range(initial_window, n - horizon + 1, step_size):
        # Eğitim penceresi
        if expanding:
            train = y.iloc[:split_point]
        else:
            train = y.iloc[split_point - initial_window : split_point]

        # Test penceresi
        test = y.iloc[split_point : split_point + horizon]
        if len(test) == 0:
            break

        # Model eğit ve tahmin et
        try:
            model = forecaster.__class__(**_get_init_params(forecaster))
            model.fit(train)
            predicted = model.predict(len(test))

            # Metrik hesapla
            metrics = compute_all_metrics(test, predicted)

            results.append({
                "fold": fold,
                "train_end": str(train.index[-1].date()),
                "test_start": str(test.index[0].date()),
                "test_end": str(test.index[-1].date()),
                "n_test": len(test),
                **metrics,
            })
            fold += 1

        except Exception as e:
            logger.warning(f"CV fold {fold} başarısız: {e}")
            fold += 1
            continue

    if not results:
        raise ValueError("Hiçbir CV fold başarıyla tamamlanamadı.")

    return pd.DataFrame(results)


def _get_init_params(forecaster: BaseForecaster) -> Dict[str, Any]:
    """Forecaster'ın __init__ parametrelerini çıkarır (yeniden oluşturma için).

    Parameters
    ----------
    forecaster : BaseForecaster
        Kaynak model.

    Returns
    -------
    dict
        __init__ parametreleri.
    """
    params: Dict[str, Any] = {}

    # SeasonalNaiveForecaster → period
    if hasattr(forecaster, "_period"):
        params["period"] = forecaster._period

    # MovingAverageForecaster → window
    if hasattr(forecaster, "_window"):
        params["window"] = forecaster._window

    return params


# =====================================================================
# 4. Model Karşılaştırma Tablosu
# =====================================================================

def compare_models(
    y: pd.Series,
    forecasters: Sequence[BaseForecaster],
    horizon: int = 12,
    initial_window: int = 60,
    step_size: int = 1,
    expanding: bool = True,
) -> pd.DataFrame:
    """Birden fazla modeli aynı CV ayarlarıyla değerlendirir ve karşılaştırma tablosu üretir.

    Parameters
    ----------
    y : pd.Series
        Tam zaman serisi.
    forecasters : sequence of BaseForecaster
        Karşılaştırılacak modeller.
    horizon : int, default 12
        Her fold'da kaç adım ileri tahmin.
    initial_window : int, default 60
        İlk eğitim penceresi boyutu.
    step_size : int, default 1
        Pencere kayma adımı.
    expanding : bool, default True
        Expanding (True) veya rolling (False) pencere.

    Returns
    -------
    pd.DataFrame
        Model bazlı ortalama metrikler tablosu.
        Sütunlar: ['model', 'n_folds', 'mae_mean', 'mae_std',
                    'rmse_mean', 'rmse_std', 'mape_mean', 'smape_mean']
    """
    comparison_rows = []

    for forecaster in forecasters:
        model_name = forecaster.name
        logger.info(f"Model değerlendiriliyor: {model_name}")

        try:
            cv_results = time_series_cv(
                y=y,
                forecaster=forecaster,
                horizon=horizon,
                initial_window=initial_window,
                step_size=step_size,
                expanding=expanding,
            )

            comparison_rows.append({
                "model": model_name,
                "n_folds": len(cv_results),
                "mae_mean": float(cv_results["mae"].mean()),
                "mae_std": float(cv_results["mae"].std()),
                "rmse_mean": float(cv_results["rmse"].mean()),
                "rmse_std": float(cv_results["rmse"].std()),
                "mape_mean": float(cv_results["mape"].mean()),
                "smape_mean": float(cv_results["smape"].mean()),
            })

        except Exception as e:
            logger.warning(f"{model_name} değerlendirmesi başarısız: {e}")
            comparison_rows.append({
                "model": model_name,
                "n_folds": 0,
                "mae_mean": np.nan,
                "mae_std": np.nan,
                "rmse_mean": np.nan,
                "rmse_std": np.nan,
                "mape_mean": np.nan,
                "smape_mean": np.nan,
            })

    result_df = pd.DataFrame(comparison_rows)

    # MAE'ye göre sırala (en düşük = en iyi)
    result_df = result_df.sort_values("mae_mean", ascending=True, na_position="last").reset_index(drop=True)
    result_df.index.name = "rank"

    return result_df
