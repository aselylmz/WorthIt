"""
Veri doğrulama ve veri sözleşmesi (Data Contract) kontrol modülü.

Zaman serilerinin yapısal geçerliliğini, sayısal sınırlarını,
eksik veri durumlarını ve MVP Data Contract uyumunu denetler.
"""

from typing import List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from time_to_afford.utils.logging import get_logger

logger = get_logger(__name__)


# =====================================================================
# Özel Hata Sınıfları
# =====================================================================

class ValidationError(Exception):
    """Veri tipi, aralık ve mantıksal kural hataları."""
    pass


class DataIntegrityError(Exception):
    """Mükerrer tarih, sıra bozukluğu ve beklenmeyen null gibi veri bütünlüğü ihlalleri."""
    pass


class DataContractError(Exception):
    """Nihai işlenmiş veri setinin MVP Data Contract şemasına uymaması durumu."""
    pass


# =====================================================================
# Zaman Serisi Yapı ve Tarih Doğrulama
# =====================================================================

def validate_time_series_structure(df: pd.DataFrame) -> None:
    """Zaman serisi DataFrame'inin yapısal geçerliliğini denetler.

    Parameters
    ----------
    df : pd.DataFrame
        Doğrulanacak zaman serisi DataFrame'i.

    Raises
    ------
    ValidationError
        İndeks DatetimeIndex değilse, boşsa veya gelecek tarih içeriyorsa.
    DataIntegrityError
        Mükerrer tarih veya kronolojik sıra bozukluğu varsa.
    """
    if df is None or not isinstance(df, pd.DataFrame):
        raise ValidationError(f"Girdi geçerli bir pandas DataFrame olmalıdır, alınan: {type(df)}")

    if df.empty:
        raise ValidationError("DataFrame boştur, en az bir gözlem içermelidir.")

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValidationError(
            f"DataFrame indeksi pd.DatetimeIndex olmalıdır, alınan: {type(df.index)}"
        )

    if df.index.has_duplicates:
        duplicate_dates = df.index[df.index.duplicated()].tolist()
        raise DataIntegrityError(
            f"Mükerrer (duplicate) tarihler tespit edildi: {duplicate_dates[:5]}"
        )

    if not df.index.is_monotonic_increasing:
        raise DataIntegrityError(
            "Zaman serisi indeksindeki tarihler kesin kronolojik sırada (strictly monotonic increasing) değildir."
        )

    # Gelecek tarih kontrolü (küçük tolerans: bugünün sonu)
    now = pd.Timestamp.now().normalize() + pd.DateOffset(days=1)
    if (df.index > now).any():
        future_dates = df.index[df.index > now].tolist()
        raise ValidationError(
            f"Gelecek tarihli gözlemler tespit edildi (veri bozukluğu): {future_dates[:3]}"
        )


# =====================================================================
# Sayısal Sınır ve Mantık Doğrulama
# =====================================================================

def validate_numeric_bounds(
    df: pd.DataFrame,
    column: str,
    allow_zero: bool = False,
    allow_negative: bool = False,
) -> None:
    """Belirtilen sayısal sütunun mantıksal değer sınırlarına uygunluğunu denetler.

    Parameters
    ----------
    df : pd.DataFrame
        Veri tablosu.
    column : str
        Kontrol edilecek sütun adı.
    allow_zero : bool, default False
        Sıfır değerine izin verilip verilmediği.
    allow_negative : bool, default False
        Negatif değerlere izin verilip verilmediği.

    Raises
    ------
    ValidationError
        Sütun sayısal değilse veya sınır ihlali varsa.
    """
    if column not in df.columns:
        raise ValidationError(f"'{column}' sütunu DataFrame içinde bulunamadı.")

    series = df[column].dropna()
    if series.empty:
        return

    if not pd.api.types.is_numeric_dtype(series):
        raise ValidationError(f"'{column}' sayısal bir veri tipinde (numeric) olmalıdır, alınan: {series.dtype}")

    if not allow_negative and (series < 0).any():
        negative_vals = series[series < 0].head().to_dict()
        raise ValidationError(
            f"'{column}' serisinde negatif değerler tespit edildi (değerler negatif olamaz): {negative_vals}"
        )

    if not allow_zero and not allow_negative and (series <= 0).any():
        zero_or_neg = series[series <= 0].head().to_dict()
        raise ValidationError(
            f"'{column}' serisinde sıfır veya negatif fiyat tespit edildi (fiyatlar pozitif olmalıdır): {zero_or_neg}"
        )


# =====================================================================
# Seri Bazlı Eksik Veri Doğrulama (Valid Date Window)
# =====================================================================

def validate_missing_values(
    df: pd.DataFrame,
    column: str,
    start_date: Optional[pd.Timestamp] = None,
    max_consecutive_nulls: int = 0,
) -> None:
    """Belirtilen serinin geçerli tarih penceresi (active window) içindeki eksikliklerini denetler.

    Serinin başlangıç tarihinden önceki dönemler yapısal olarak mevcut olmadığı için
    eksik veri sayılmaz. Ancak başlangıç tarihinden sonraki dönemlerde null bulunamaz.

    Parameters
    ----------
    df : pd.DataFrame
        Veri tablosu.
    column : str
        Kontrol edilecek sütun adı.
    start_date : pd.Timestamp, optional
        Serinin resmî başlangıç tarihi.
    max_consecutive_nulls : int, default 0
        İzin verilen maksimum ardışık boş ay sayısı.

    Raises
    ------
    DataIntegrityError
        Aktif pencere içinde beklenmeyen eksiklik tespit edilirse.
    """
    if column not in df.columns:
        raise ValidationError(f"'{column}' sütunu DataFrame içinde bulunamadı.")

    if start_date is not None:
        active_series = df.loc[df.index >= start_date, column]
    else:
        # Serinin ilk geçerli değerinden itibaren aktif pencere kabul et
        first_valid = df[column].first_valid_index()
        if first_valid is None:
            raise DataIntegrityError(f"'{column}' serisi tamamen boştur.")
        active_series = df.loc[df.index >= first_valid, column]

    null_count = active_series.isna().sum()
    if null_count > max_consecutive_nulls:
        null_dates = active_series[active_series.isna()].index.tolist()
        raise DataIntegrityError(
            f"'{column}' serisinde aktif tarih penceresinde ({start_date or 'başlangıç'} sonrası) "
            f"beklenmeyen eksik değer (null) tespit edildi: {null_dates[:5]} (Toplam: {null_count})"
        )


# =====================================================================
# MVP Data Contract Doğrulayıcı
# =====================================================================

# MVP Data Contract'a göre zorunlu gözlemlenmiş sütunlar
REQUIRED_CONTRACT_COLUMNS: Sequence[str] = [
    "cpi_index",
    "cpi_return",
    "house_price_index",
    "house_price_return",
    "deposit_rate_3m",
    "deposit_monthly_return",
    "usd_try",
    "usd_try_return",
    "policy_rate",
    "bist100_close",
    "bist100_return",
    "synthetic_gram_gold_try",
    "gold_return",
    "vehicle_price_proxy",
    "vehicle_proxy_return",
]

# Katman ayrımı gereği bu aşamada processed dataset içinde YER ALAMAZ
FORBIDDEN_SIMULATED_COLUMNS: Sequence[str] = [
    "salary_growth_rate",
    "simulated_salary",
    "monte_carlo_path",
    "target_price_forecast",
]


def validate_data_contract(df: pd.DataFrame) -> None:
    """Nihai 'macro_monthly.parquet' tablosunun MVP Data Contract'a uyumunu doğrular.

    Parameters
    ----------
    df : pd.DataFrame
        İşlenmiş aylık makroekonomik veri seti.

    Raises
    ------
    DataContractError
        Zorunlu sütun eksikse, yasaklı simülasyon sütunu varsa veya indeks uyumsuzsa.
    """
    validate_time_series_structure(df)

    # 1. Yasaklı simüle sütun kontrolü (Katman İzolasyon Kuralı)
    for forbidden in FORBIDDEN_SIMULATED_COLUMNS:
        if forbidden in df.columns or any(forbidden in str(c).lower() for c in df.columns):
            raise DataContractError(
                f"Katman İzolasyon İhlali: Simüle/tahmin edilmiş '{forbidden}' sütunu "
                f"Phase 2 gözlemlenmiş macro dataset içine dahil edilemez."
            )

    # 2. Zorunlu sütun kontrolü
    missing_columns = [col for col in REQUIRED_CONTRACT_COLUMNS if col not in df.columns]
    if missing_columns:
        raise DataContractError(
            f"MVP Data Contract ihlali: Eksik sütunlar tespit edildi: {missing_columns}"
        )

    # 3. Sayısal veri tipi kontrolü
    for col in REQUIRED_CONTRACT_COLUMNS:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise DataContractError(
                f"'{col}' sütunu sayısal (numeric/float64) veri tipinde olmalıdır, alınan: {df[col].dtype}"
            )

    logger.info("MVP Data Contract doğrulaması başarıyla tamamlandı.")
