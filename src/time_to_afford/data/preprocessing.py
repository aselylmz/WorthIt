"""
Veri ön işleme, dönüştürme ve standartlaştırma modülü.

Zaman serilerinin aylık frekansa hizalanması (resampling), log-return hesaplamaları,
efektif mevduat faizi dönüşümü, sentetik gram altın üretimi ve
yalnızca gözlemlenmiş verileri içeren 'macro_monthly.parquet' üretim boru hattını (pipeline) yönetir.
"""

from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd

from time_to_afford.config.settings import get_settings
from time_to_afford.data.validators import (
    validate_data_contract,
    validate_numeric_bounds,
    validate_time_series_structure,
)
from time_to_afford.utils.logging import get_logger

logger = get_logger(__name__)

# 1 Troy Ons = 31.1034768 Gram
TROY_OUNCE_TO_GRAM = 31.1034768


# =====================================================================
# Aylık Frekans Hizalama Fonksiyonları (Resampling)
# =====================================================================

def resample_to_monthly_close(df: pd.DataFrame, column: str) -> pd.Series:
    """Günlük veya düzensiz fiyat serisini ayın son gerçekleşen kapanış değerine indirger.

    Piyasa tatilleri durumunda ayın son işlem günündeki geçerli kapanış değeri seçilir.
    Yapay fiyat uydurulmaz veya ileriye doğru doldurma yapılmaz.

    Parameters
    ----------
    df : pd.DataFrame
        Tarih indeksli fiyat verisi.
    column : str
        Fiyat sütun adı.

    Returns
    -------
    pd.Series
        Aylık dönem sonu (`ME`) indeksli fiyat serisi.
    """
    validate_time_series_structure(df)
    validate_numeric_bounds(df, column=column, allow_zero=False, allow_negative=False)

    series = df[column].dropna()
    monthly_series = series.resample("ME").last()
    monthly_series.name = column
    return monthly_series.dropna()


def resample_rates_to_monthly(df: pd.DataFrame, column: str) -> pd.Series:
    """Günlük veya haftalık faiz serisini ilgili ayın aritmetik ortalamasına indirger.

    Parameters
    ----------
    df : pd.DataFrame
        Tarih indeksli faiz verisi.
    column : str
        Faiz sütun adı.

    Returns
    -------
    pd.Series
        Aylık dönem sonu (`ME`) indeksli ortalama faiz serisi.
    """
    validate_time_series_structure(df)
    validate_numeric_bounds(df, column=column, allow_zero=True, allow_negative=False)

    series = df[column].dropna()
    monthly_series = series.resample("ME").mean()
    monthly_series.name = column
    return monthly_series.dropna()


# =====================================================================
# Matematiksel Dönüşümler (Leakage-Safe)
# =====================================================================

def compute_log_returns(prices: pd.Series) -> pd.Series:
    """Fiyat veya endeks serisinin aylık logaritmik getirisini hesaplar.

    Formül: r_t = ln(P_t / P_{t-1}) = ln(P_t) - ln(P_{t-1})
    İlk değer (t=0) zorunlu olarak NaN olur; geriye doğru bilgi sızdırılmaz.

    Parameters
    ----------
    prices : pd.Series
        Zaman serisi fiyatları.

    Returns
    -------
    pd.Series
        Aylık log-return serisi.

    Raises
    ------
    ValueError
        Sıfır veya negatif fiyat girdisi varsa.
    """
    valid_prices = prices.dropna()
    if (valid_prices <= 0).any():
        raise ValueError("Log-return hesabı için tüm fiyat değerleri kesinlikle pozitif (> 0) olmalıdır.")

    log_prices = np.log(prices.astype(float))
    returns = log_prices.diff()
    returns.name = f"{prices.name}_return" if prices.name else "return"
    return returns


def compute_effective_deposit_rate(annual_rate_percent: Union[float, pd.Series]) -> Union[float, pd.Series]:
    """Yıllıklandırılmış brüt basit faiz oranını aylık efektif getiri oranına dönüştürür.

    Formül: r_monthly = (1 + i_annual / 100)^(1/12) - 1

    Parameters
    ----------
    annual_rate_percent : float or pd.Series
        Yıllık faiz oranı (yüzde cinsinden, örn: 25.0).

    Returns
    -------
    float or pd.Series
        Aylık efektif getiri oranı (ondalık cinsinden, örn: 0.0187).
    """
    return (1.0 + annual_rate_percent / 100.0) ** (1.0 / 12.0) - 1.0


def align_and_compute_synthetic_gold(
    gold_ons_usd: pd.Series,
    usd_try: pd.Series,
) -> pd.Series:
    """Ons altın (USD) ve USD/TRY kurunu güvenli eşleyerek sentetik gram altın (TL) hesaplar.

    Formül: Gram Altın (TL) = (Ons Altın USD * USD/TRY) / 31.1034768
    Farklı piyasa tatilleri durumunda güvenli inner-join uygulanır.

    Parameters
    ----------
    gold_ons_usd : pd.Series
        Ons altın fiyatı (USD).
    usd_try : pd.Series
        USD/TRY döviz kuru.

    Returns
    -------
    pd.Series
        Sentetik gram altın serisi (TL/Gram).
    """
    # Ortak tarihler üzerinde hizala (inner join)
    aligned_df = pd.DataFrame({
        "ons": gold_ons_usd,
        "usd_try": usd_try,
    }).dropna()

    synthetic_gram_tl = (aligned_df["ons"] * aligned_df["usd_try"]) / TROY_OUNCE_TO_GRAM
    synthetic_gram_tl.name = "synthetic_gram_gold_try"
    return synthetic_gram_tl


# =====================================================================
# Gözlemlenmiş Aylık Veri Seti Oluşturucu (Dataset Builder)
# =====================================================================

def build_macro_monthly_dataset(raw_series_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Tüm ham aylık serileri birleştirir, deterministik dönüşümleri uygular ve doğrular.

    Katman İzolasyon Kuralı: Bu fonksiyon yalnızca gözlemlenmiş geçmiş verileri ve
    deterministik türevlerini içerir. Gelecekteki maaş tahminleri veya Monte Carlo
    çıktıları kesinlikle bu fonksiyona dahil edilmez.

    Parameters
    ----------
    raw_series_dict : dict of str -> pd.DataFrame
        Ham aylık seriler sözlüğü. Beklenen anahtarlar:
        'cpi_index', 'house_price_index', 'deposit_rate_3m', 'usd_try',
        'policy_rate', 'bist100_close', 'gold_ons_usd', 'vehicle_price_proxy'.

    Returns
    -------
    pd.DataFrame
        MVP Data Contract'a tam uyumlu gözlemlenmiş aylık veri tablosu.
    """
    logger.info("Aylık makroekonomik veri seti derleniyor...")

    # Ortak tarih havuzunu oluştur
    df = pd.DataFrame(index=pd.DatetimeIndex([]))

    # Resampling işlemleri (Phase 2 Sorumluluğu)
    df["cpi_index"] = resample_to_monthly_close(raw_series_dict["cpi_index"], "value")
    df["house_price_index"] = resample_to_monthly_close(raw_series_dict["house_price_index"], "value")
    df["usd_try"] = resample_to_monthly_close(raw_series_dict["usd_try"], "value")
    df["vehicle_price_proxy"] = resample_to_monthly_close(raw_series_dict["vehicle_price_proxy"], "value")

    df["deposit_rate_3m"] = resample_rates_to_monthly(raw_series_dict["deposit_rate_3m"], "value")
    df["policy_rate"] = resample_rates_to_monthly(raw_series_dict["policy_rate"], "value")

    df["bist100_close"] = resample_to_monthly_close(raw_series_dict["bist100_close"], "value")

    # Sentetik gram altın hesabı (Günlük veriler üzerinden güvenli hesaplanıp sonra aylıklaştırılır)
    daily_synthetic_gold = align_and_compute_synthetic_gold(
        raw_series_dict["gold_ons_usd"]["value"],
        raw_series_dict["usd_try"]["value"],
    )
    df["synthetic_gram_gold_try"] = resample_to_monthly_close(daily_synthetic_gold.to_frame("value"), "value")

    # Deterministik Dönüşümler (Log-returns & Efektif Faiz)
    df["cpi_return"] = compute_log_returns(df["cpi_index"])
    df["house_price_return"] = compute_log_returns(df["house_price_index"])
    df["deposit_monthly_return"] = compute_effective_deposit_rate(df["deposit_rate_3m"])
    df["usd_try_return"] = compute_log_returns(df["usd_try"])
    df["bist100_return"] = compute_log_returns(df["bist100_close"])
    df["gold_return"] = compute_log_returns(df["synthetic_gram_gold_try"])
    df["vehicle_proxy_return"] = compute_log_returns(df["vehicle_price_proxy"])

    # Tarihleri ME standardına hizala ve sırala
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    df = df.sort_index()

    # Tüm sayısal sütunları standart float64 yap
    for col in df.columns:
        df[col] = df[col].astype(np.float64)

    # Data Contract doğrulaması
    validate_data_contract(df)
    logger.info(f"Makro aylık veri seti başarıyla oluşturuldu: {df.shape[0]} gözlem, {df.shape[1]} değişken.")

    return df


def run_data_pipeline(
    raw_series_dict: Optional[Dict[str, pd.DataFrame]] = None,
    processed_dir: Optional[Path] = None,
) -> Path:
    """Veri boru hattını çalıştırır ve 'macro_monthly.parquet' ile 'macro_monthly.csv' üretir.

    Parameters
    ----------
    raw_series_dict : dict of str -> pd.DataFrame, optional
        Hazır ham seriler sözlüğü. Verilmezse raw snapshot'lardan okunur.
    processed_dir : Path, optional
        Çıktı dizini. Verilmezse settings.processed_data_dir kullanılır.

    Returns
    -------
    Path
        Oluşturulan parquet dosyasının yolu.
    """
    if processed_dir is None:
        processed_dir = get_settings().processed_data_dir

    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    if raw_series_dict is None:
        raise ValueError(
            "Gerçek API veya snapshot verisi sağlanmalıdır. "
            "Pipeline sessizce sahte veri uydurmaz."
        )

    df_processed = build_macro_monthly_dataset(raw_series_dict)

    parquet_path = processed_dir / "macro_monthly.parquet"
    csv_path = processed_dir / "macro_monthly.csv"

    df_processed.to_parquet(parquet_path)
    df_processed.to_csv(csv_path)

    logger.info(f"Processed dataset kaydedildi: {parquet_path}")
    return parquet_path
