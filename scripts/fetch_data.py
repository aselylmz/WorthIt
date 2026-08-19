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
from datetime import date, datetime
from typing import Dict

import pandas as pd

from time_to_afford.data.loaders import EVDSClient, YahooFinanceLoader, save_raw_snapshot
from time_to_afford.data.preprocessing import (
    resample_rates_to_monthly,
    resample_to_monthly_close,
    run_data_pipeline,
)
from time_to_afford.utils.logging import get_logger

logger = get_logger(__name__)

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
EVDS_RATE_SERIES: Dict[str, str] = {
    "deposit_rate_3m": "TP.TRY.MT02",
    "policy_rate": "TP.KT.IFJ01",
}

# NOT: Bu kod DOĞRULANMAMIŞTIR. docs/data_sources.md § 2.5'te belirtildiği
# gibi TÜİK'in TÜFE Ulaştırma alt grubuna karşılık gelen resmi bir EVDS
# seri kodu bu depoda henüz teyit edilmedi. Çalıştırmadan önce
# https://evds2.tcmb.gov.tr adresindeki Veri Kataloğu > TÜFE Alt Grupları
# bölümünden "Ulaştırma" grubunun güncel seri kodunu doğrulayın ve
# gerekirse aşağıdaki değeri güncelleyin.
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


def _fetch_evds_level_series(
    client: EVDSClient, dataset_key: str, series_code: str, start: str, end: str
) -> pd.Series:
    """EVDS'ten bir seviye serisi çeker, raw snapshot kaydeder ve aylık kapanışa indirger."""
    df = client.fetch_series(
        series_code,
        start_date=start,
        end_date=end,
    )
    column = series_code.replace(".", "_")
    save_raw_snapshot(df, source="evds", series_name=dataset_key)
    return resample_to_monthly_close(df, column=column)


def _fetch_evds_rate_series(
    client: EVDSClient, dataset_key: str, series_code: str, start: str, end: str
) -> pd.Series:
    """EVDS'ten bir faiz oranı serisi çeker, raw snapshot kaydeder ve aylık ortalamaya indirger."""
    df = client.fetch_series(
        series_code,
        start_date=start,
        end_date=end,
    )
    column = series_code.replace(".", "_")
    save_raw_snapshot(df, source="evds", series_name=dataset_key)
    return resample_rates_to_monthly(df, column=column)


def _fetch_yahoo_series(
    loader: YahooFinanceLoader, dataset_key: str, ticker: str, start: str, end: str
) -> pd.Series:
    """Yahoo Finance'ten bir seri çeker, raw snapshot kaydeder ve aylık kapanışa indirger."""
    df = loader.fetch_ticker(ticker, start_date=start, end_date=end)
    save_raw_snapshot(df, source="yfinance", series_name=dataset_key)
    return resample_to_monthly_close(df, column="close")


# =====================================================================
# Ana Orkestrasyon
# =====================================================================


def fetch_all_raw_series(start_iso: str, end_iso: str) -> Dict[str, pd.Series]:
    """Tüm EVDS ve Yahoo Finance serilerini çeker ve dataset anahtarına göre sözlük döner.

    Parameters
    ----------
    start_iso : str
        Başlangıç tarihi ('YYYY-MM-DD').
    end_iso : str
        Bitiş tarihi ('YYYY-MM-DD').

    Returns
    -------
    dict of str -> pd.Series
        `build_macro_monthly_dataset` / `run_data_pipeline` için hazır
        ham aylık seriler sözlüğü.
    """
    start_date = date.fromisoformat(start_iso)
    end_date = date.fromisoformat(end_iso)
    evds_start = start_date.strftime("%d-%m-%Y")
    evds_end = end_date.strftime("%d-%m-%Y")

    raw_series: Dict[str, pd.Series] = {}

    evds_client = EVDSClient()
    for dataset_key, series_code in EVDS_LEVEL_SERIES.items():
        logger.info(f"EVDS seviye serisi çekiliyor: {dataset_key} ({series_code})")
        raw_series[dataset_key] = _fetch_evds_level_series(
            evds_client, dataset_key, series_code, evds_start, evds_end
        )

    for dataset_key, series_code in EVDS_RATE_SERIES.items():
        logger.info(f"EVDS faiz serisi çekiliyor: {dataset_key} ({series_code})")
        raw_series[dataset_key] = _fetch_evds_rate_series(
            evds_client, dataset_key, series_code, evds_start, evds_end
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
