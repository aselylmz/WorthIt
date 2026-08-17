"""
Veri yükleme ve harici API istemcileri modülü.

TCMB EVDS API ve Yahoo Finance üzerinden veri çekme,
ham verileri değiştirilemez (immutable) snapshot olarak saklama ve okuma işlemlerini yönetir.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import requests

from time_to_afford.config.settings import get_settings
from time_to_afford.utils.logging import get_logger

logger = get_logger(__name__)


# =====================================================================
# Özel Hata Sınıfları
# =====================================================================

class DataDownloadError(Exception):
    """Veri indirme veya API çağrısı sırasında oluşan hatalar."""
    pass


class DataFormatError(Exception):
    """API'den gelen verinin beklenen formatta olmaması durumunda oluşan hatalar."""
    pass


# =====================================================================
# TCMB EVDS İstemcisi
# =====================================================================

class EVDSClient:
    """TCMB Elektronik Veri Dağıtım Sistemi (EVDS) REST API İstemcisi.

    05 Nisan 2024 güvenlik standardına uygun olarak API anahtarını
    HTTP isteğinin 'key' başlığında (header) iletir.
    """

    BASE_URL = "https://evds2.tcmb.gov.tr/service/evds/"

    def __init__(self, api_key: Optional[str] = None) -> None:
        """EVDS İstemcisini başlat.

        Parameters
        ----------
        api_key : str, optional
            TCMB EVDS API anahtarı. Verilmezse settings.evds_api_key kullanılır.

        Raises
        ------
        ValueError
            API anahtarı bulunamazsa fırlatılır.
        """
        if api_key is None:
            api_key = get_settings().evds_api_key

        if not api_key or not isinstance(api_key, str) or not api_key.strip():
            raise ValueError(
                "TCMB EVDS API anahtarı zorunludur. Lütfen .env dosyasında "
                "EVDS_API_KEY tanımlayın veya istemciye geçerli bir api_key geçin."
            )

        self.api_key = api_key.strip()
        self.session = requests.Session()

    def fetch_series(
        self,
        series_code: str,
        start_date: str,
        end_date: str,
        frequency: Optional[str] = None,
        aggregation_type: Optional[str] = None,
    ) -> pd.DataFrame:
        """Belirtilen seri koduna ait verileri EVDS API üzerinden çeker.

        Parameters
        ----------
        series_code : str
            EVDS seri kodu (ör: 'TP.FG.J0', 'TP.KFE.TR').
        start_date : str
            Başlangıç tarihi (Format: 'DD-MM-YYYY', ör: '01-01-2020').
        end_date : str
            Bitiş tarihi (Format: 'DD-MM-YYYY', ör: '31-12-2023').
        frequency : str, optional
            Frekans (1: Günlük, 2: İşgünü, 3: Haftalık, 4: Ayda 2 Kez, 5: Aylık, 6: Çeyreklik, 7: 6 Aylık, 8: Yıllık).
        aggregation_type : str, optional
            Toplulaştırma yöntemi (avg, min, max, first, last).

        Returns
        -------
        pd.DataFrame
            Tarih indeksli zaman serisi DataFrame'i.

        Raises
        ------
        DataDownloadError
            Ağ veya HTTP yetkilendirme hatalarında.
        DataFormatError
            Yanıt formatı bozuk veya boş olduğunda.
        """
        params: Dict[str, Any] = {
            "series": series_code,
            "startDate": start_date,
            "endDate": end_date,
            "type": "json",
        }
        if frequency is not None:
            params["frequency"] = frequency
        if aggregation_type is not None:
            params["aggregationTypes"] = aggregation_type

        headers = {"key": self.api_key}

        logger.info(f"EVDS API isteği gönderiliyor: series={series_code}, start={start_date}, end={end_date}")

        try:
            response = self.session.get(self.BASE_URL, params=params, headers=headers, timeout=30)
        except requests.RequestException as e:
            raise DataDownloadError(f"EVDS API bağlantı hatası: {e}") from e

        if response.status_code != 200:
            raise DataDownloadError(
                f"EVDS API çağrısı başarısız oldu (HTTP {response.status_code}): {response.text[:200]}"
            )

        try:
            data = response.json()
        except Exception as e:
            raise DataFormatError(f"EVDS API yanıtı geçerli bir JSON formatı bozuk: {e}") from e

        if not isinstance(data, dict) or "items" not in data:
            raise DataFormatError(f"EVDS API beklenmeyen veri yapısı döndürdü: {list(data.keys()) if isinstance(data, dict) else type(data)}")

        items = data.get("items", [])
        if not items:
            raise DataFormatError(f"EVDS API seri '{series_code}' için boş veri döndü.")

        df = pd.DataFrame(items)

        # Tarih sütununu indeks yap
        date_col = "Tarih" if "Tarih" in df.columns else "tarih"
        if date_col not in df.columns:
            raise DataFormatError(f"EVDS yanıtında tarih sütunu bulunamadı: {df.columns.tolist()}")

        # EVDS tarihleri genellikle DD-MM-YYYY veya YYYY-MM formatındadır
        df["date"] = pd.to_datetime(df[date_col], format="%d-%m-%Y", errors="coerce")
        if df["date"].isna().all():
            df["date"] = pd.to_datetime(df[date_col], errors="coerce")

        df = df.dropna(subset=["date"]).sort_values("date").set_index("date")
        if date_col in df.columns:
            df = df.drop(columns=[date_col])

        # UNIXTIME gibi gereksiz meta sütunları temizle
        meta_cols = [c for c in df.columns if "UNIXTIME" in c.upper()]
        if meta_cols:
            df = df.drop(columns=meta_cols)

        # Sayısal sütunları dönüştür
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        return df


# =====================================================================
# Yahoo Finance Yükleyicisi
# =====================================================================

class YahooFinanceLoader:
    """Yahoo Finance üzerinden borsa ve emtia verilerini çeken yükleyici."""

    def fetch_ticker(
        self,
        ticker: str,
        start_date: str,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Yahoo Finance üzerinden belirtilen sembol için günlük veri indirir.

        Parameters
        ----------
        ticker : str
            Yahoo Finance sembolü (ör: 'XU100.IS', 'GC=F', 'USDTRY=X').
        start_date : str
            Başlangıç tarihi ('YYYY-MM-DD').
        end_date : str, optional
            Bitiş tarihi ('YYYY-MM-DD'). Verilmezse bugüne kadar çeker.

        Returns
        -------
        pd.DataFrame
            Tarih indeksli fiyat verileri (küçük harf sütun isimleriyle).

        Raises
        ------
        DataDownloadError
            İndirme başarısız olduğunda veya veri bulunamadığında.
        """
        import yfinance as yf

        logger.info(f"Yahoo Finance verisi indiriliyor: ticker={ticker}, start={start_date}, end={end_date}")

        try:
            df = yf.download(ticker, start=start_date, end=end_date, interval="1d", progress=False)
        except Exception as e:
            raise DataDownloadError(f"Yahoo Finance indirme hatası ({ticker}): {e}") from e

        if df is None or df.empty:
            raise DataDownloadError(f"Yahoo Finance üzerinde '{ticker}' sembolü için veri bulunamadı.")

        # MultiIndex sütun yapısını düzleştir (yfinance v0.2+ sürümleri için)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]

        df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
        df.index = pd.to_datetime(df.index)
        df.index.name = "date"
        df = df.sort_index()

        return df


# =====================================================================
# Raw Snapshot Saklama & Yükleme (Immutability Standardı)
# =====================================================================

def save_raw_snapshot(
    df: pd.DataFrame,
    source: str,
    series_name: str,
    raw_dir: Optional[Path] = None,
) -> Path:
    """Ham veri DataFrame'ini zaman damgasıyla değiştirilemez (immutable) snapshot olarak kaydeder.

    Parameters
    ----------
    df : pd.DataFrame
        Kaydedilecek ham veri.
    source : str
        Veri kaynağı (ör: 'evds', 'yfinance', 'tuik').
    series_name : str
        Seri adı (ör: 'cpi_index', 'bist100', 'kfe').
    raw_dir : Path, optional
        Kayıt dizini. Verilmezse settings.raw_data_dir kullanılır.

    Returns
    -------
    Path
        Oluşturulan snapshot dosyasının tam yolu.
    """
    if raw_dir is None:
        raw_dir = get_settings().raw_data_dir

    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{source}_{series_name}_{timestamp_str}.parquet"
    file_path = raw_dir / filename

    df.to_parquet(file_path)
    logger.info(f"Raw snapshot kaydedildi: {file_path}")
    return file_path


def load_latest_raw_snapshot(
    source: str,
    series_name: str,
    raw_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Belirtilen kaynak ve seriye ait en güncel raw snapshot dosyasını okur.

    Parameters
    ----------
    source : str
        Veri kaynağı ('evds', 'yfinance' vb.).
    series_name : str
        Seri adı ('cpi_index' vb.).
    raw_dir : Path, optional
        Arama dizini. Verilmezse settings.raw_data_dir kullanılır.

    Returns
    -------
    pd.DataFrame
        En güncel ham veri DataFrame'i.

    Raises
    ------
    FileNotFoundError
        Hiçbir snapshot dosyası bulunamazsa.
    """
    if raw_dir is None:
        raw_dir = get_settings().raw_data_dir

    raw_dir = Path(raw_dir)
    pattern = f"{source}_{series_name}_*.parquet"
    matching_files = sorted(raw_dir.glob(pattern))

    if not matching_files:
        raise FileNotFoundError(
            f"'{raw_dir}' dizininde '{source}_{series_name}' kalıbına uygun hiçbir raw snapshot bulunamadı."
        )

    latest_file = matching_files[-1]
    logger.info(f"En güncel raw snapshot okunuyor: {latest_file}")
    return pd.read_parquet(latest_file)
