"""
Ham makroekonomik verileri TCMB EVDS ve Yahoo Finance'ten çeker, her seriyi
değiştirilemez (immutable) raw snapshot olarak `data/raw/` altına kaydeder
ve `run_data_pipeline` ile `data/processed/macro_monthly.parquet` üretir.

Gereksinimler:
    .env dosyasında EVDS_API_KEY tanımlı olmalıdır (bkz. .env.example).
    TCMB EVDS API anahtarı https://evds2.tcmb.gov.tr/index.php?/evds/login
    adresinden ücretsiz alınabilir.

Kullanım:
    python scripts/fetch_data.py
    python scripts/fetch_data.py --start 2010-01-01 --end 2026-08-01

Kaynak kodları için bkz. docs/data_sources.md.
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta
from typing import Dict, List, Tuple

import pandas as pd

from time_to_afford.data.loaders import (
    DataFormatError,
    EVDSClient,
    YahooFinanceLoader,
    save_raw_snapshot,
)
from time_to_afford.data.preprocessing import run_data_pipeline
from time_to_afford.utils.logging import get_logger

logger = get_logger(__name__)

# EVDS API, tek istekte döndürdüğü satır sayısını sessizce (hata vermeden)
# ~1000 ile sınırlıyor: uzun tarih aralığı istenen günlük/haftalık frekanslı
# serilerde en eski gözlemler kırpılıp yalnızca en güncel ~1000 satır
# döner. Bu, yüksek frekanslı serilerin (günlük/haftalık) tarih aralığının
# parçalara bölünerek (chunked) çekilmesini gerektirir. Aylık seriler
# (16 yılda ~200 satır) bu sınırı hiçbir zaman aşmaz.
EVDS_CHUNK_DAYS = 650

# Chunked çekilmesi gereken (günlük veya haftalık frekanslı) dataset anahtarları.
_HIGH_FREQUENCY_DATASETS = {"usd_try", "policy_rate", "deposit_rate_3m"}

# =====================================================================
# Seri Kayıt Defteri (bkz. docs/data_sources.md)
# =====================================================================

# dataset anahtarı -> EVDS seri kodu
EVDS_LEVEL_SERIES: Dict[str, str] = {
    "cpi_index": "TP.FG.J0",
    "house_price_index": "TP.KFE.TR",
    "usd_try": "TP.DK.USD.A.YTL",
}

# Faiz oranı serileri: ayın aritmetik ortalamasıyla aylıklaştırılır
#
# NOT: docs/data_sources.md'de "TP.KT.IFJ01" olarak belgelenen politika
# faizi kodu, gerçek EVDS API'sinde (evds3, Ağustos 2026) HTTP 400 ile
# reddediliyor — kod ya kaldırılmış ya da yanlış belgelenmiş. EVDS'de
# tek bir "resmi ilan edilen politika faizi" serisi bulunmuyor; bunun
# yerine TCMB'nin fiilen uyguladığı ortalama fonlama maliyeti kullanılır.
# TP.APIFON4 (Ağırlıklı Ortalama Fonlama Maliyeti) gerçek veriyle
# doğrulandı: Ocak-Mart 2024 için %42.50 -> %45.00 gösteriyor, bu da
# TCMB'nin 25 Ocak 2024 faiz kararıyla birebir örtüşüyor.
EVDS_RATE_SERIES: Dict[str, str] = {
    "deposit_rate_3m": "TP.TRY.MT02",
    "policy_rate": "TP.APIFON4",
}

# NOT: Bu kod YANLIŞ/GÜVENİLMEZ olduğu doğrulandı (bkz. docs/data_sources.md
# § 2.5). "TP.TUFE1YI" ön eki TÜFE alt grupları değil, Yİ-ÜFE (üretici
# fiyat endeksi) ailesine ait — sonuç seride ekonomik olarak anlamsız
# aylık sıçramalar var (ör. -%49 / +%41). scripts/calibrate_distributions.py
# bu seriyi kasıtlı olarak dışlıyor. Çalıştırmadan önce
# https://evds3.tcmb.gov.tr adresindeki Veri Kataloğu > TÜFE Alt Grupları
# bölümünden "Ulaştırma" grubunun doğru seri kodunu bulup güncelleyin.
EVDS_LEVEL_SERIES["vehicle_price_proxy"] = "TP.TUFE1YI.T7"

# dataset anahtarı -> Yahoo Finance sembolü (bkz. docs/data_sources.md § 2.3, 2.4)
YAHOO_TICKERS: Dict[str, str] = {
    "bist100_close": "XU100.IS",
    "gold_ons_usd": "GC=F",
}

DEFAULT_START = "2010-01-01"


# =====================================================================
# Çekme Yardımcıları
# =====================================================================


def _date_chunks(start: date, end: date, chunk_days: int) -> List[Tuple[date, date]]:
    """[start, end] aralığını chunk_days boyutunda ardışık, çakışmayan parçalara böler."""
    chunks: List[Tuple[date, date]] = []
    cur = start
    while cur <= end:
        chunk_end = min(cur + timedelta(days=chunk_days), end)
        chunks.append((cur, chunk_end))
        cur = chunk_end + timedelta(days=1)
    return chunks


def _fetch_evds_series(
    client: EVDSClient,
    dataset_key: str,
    series_code: str,
    start: date,
    end: date,
    chunked: bool,
) -> pd.DataFrame:
    """EVDS'ten bir seriyi çeker, raw snapshot kaydeder ve raw DataFrame döner.

    `chunked=True` olduğunda tarih aralığı EVDS_CHUNK_DAYS boyutunda
    parçalara bölünerek çekilir (bkz. EVDS_CHUNK_DAYS yorumu — API'nin
    sessiz ~1000 satır sınırını aşmak için).

    Parameters
    ----------
    chunked : bool
        True ise günlük/haftalık frekanslı seriler için tarih aralığı
        parçalanır. Aylık seriler için False yeterlidir (tek istek).
    """
    column = series_code.replace(".", "_")
    date_ranges = _date_chunks(start, end, EVDS_CHUNK_DAYS) if chunked else [(start, end)]

    parts: List[pd.DataFrame] = []
    for chunk_start, chunk_end in date_ranges:
        try:
            df_chunk = client.fetch_series(
                series_code,
                start_date=chunk_start.strftime("%d-%m-%Y"),
                end_date=chunk_end.strftime("%d-%m-%Y"),
            )
            parts.append(df_chunk)
        except DataFormatError:
            # Serinin gerçek başlangıcından önceki boş parçalar (ör. seri
            # istenen aralığın bir kısmında henüz mevcut değilse) — atla.
            logger.info(
                f"{dataset_key}: {chunk_start}–{chunk_end} aralığında veri yok, atlanıyor."
            )
            continue

    if not parts:
        raise DataFormatError(
            f"{dataset_key} ({series_code}) için hiçbir tarih parçasında veri bulunamadı."
        )

    df = pd.concat(parts)
    df = df[~df.index.duplicated(keep="last")].sort_index()

    save_raw_snapshot(df, source="evds", series_name=dataset_key)
    return df[[column]].rename(columns={column: "value"})


def _fetch_yahoo_series(
    loader: YahooFinanceLoader, dataset_key: str, ticker: str, start: str, end: str
) -> pd.DataFrame:
    """Yahoo Finance'ten bir seri çeker, raw snapshot kaydeder ve raw DataFrame döner."""
    df = loader.fetch_ticker(ticker, start_date=start, end_date=end)
    save_raw_snapshot(df, source="yfinance", series_name=dataset_key)
    return df[["close"]].rename(columns={"close": "value"})


# =====================================================================
# Ana Orkestrasyon
# =====================================================================


def fetch_all_raw_series(start_iso: str, end_iso: str) -> Dict[str, pd.DataFrame]:
    """Tüm EVDS ve Yahoo Finance serilerini çeker ve dataset anahtarına göre sözlük döner.

    Parameters
    ----------
    start_iso : str
        Başlangıç tarihi ('YYYY-MM-DD').
    end_iso : str
        Bitiş tarihi ('YYYY-MM-DD').

    Returns
    -------
    dict of str -> pd.DataFrame
        `build_macro_monthly_dataset` / `run_data_pipeline` için hazır
        ham günlük/haftalık seriler sözlüğü.
    """
    start_date = date.fromisoformat(start_iso)
    end_date = date.fromisoformat(end_iso)

    raw_series: Dict[str, pd.DataFrame] = {}

    evds_client = EVDSClient()
    for dataset_key, series_code in {**EVDS_LEVEL_SERIES, **EVDS_RATE_SERIES}.items():
        chunked = dataset_key in _HIGH_FREQUENCY_DATASETS
        logger.info(
            f"EVDS serisi çekiliyor: {dataset_key} ({series_code})"
            + (" [parçalı]" if chunked else "")
        )
        raw_series[dataset_key] = _fetch_evds_series(
            evds_client, dataset_key, series_code, start_date, end_date, chunked=chunked
        )

    yahoo_loader = YahooFinanceLoader()
    for dataset_key, ticker in YAHOO_TICKERS.items():
        logger.info(f"Yahoo Finance serisi çekiliyor: {dataset_key} ({ticker})")
        raw_series[dataset_key] = _fetch_yahoo_series(
            yahoo_loader, dataset_key, ticker, start_iso, end_iso
        )

    return raw_series


def main(start_iso: str, end_iso: str) -> None:
    """Tüm veri çekme ve işleme boru hattını çalıştırır."""
    logger.info(f"Veri çekme başlıyor: {start_iso} -> {end_iso}")
    raw_series = fetch_all_raw_series(start_iso, end_iso)
    parquet_path = run_data_pipeline(raw_series_dict=raw_series)
    logger.info(f"Tamamlandı. Çıktı: {parquet_path}")
    print(f"macro_monthly.parquet oluşturuldu: {parquet_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--start",
        default=DEFAULT_START,
        help=f"Başlangıç tarihi (YYYY-MM-DD). Varsayılan: {DEFAULT_START}",
    )
    parser.add_argument(
        "--end",
        default=datetime.now().date().isoformat(),
        help="Bitiş tarihi (YYYY-MM-DD). Varsayılan: bugün.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(args.start, args.end)
