"""
Data modülü kapsamlı birim ve edge-case testleri.

Testler tamamen offline / mockable olarak tasarlanmıştır.
"""

from datetime import date, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from time_to_afford.data.loaders import (
    DataDownloadError,
    DataFormatError,
    EVDSClient,
    YahooFinanceLoader,
    load_latest_raw_snapshot,
    save_raw_snapshot,
)
from time_to_afford.data.preprocessing import (
    align_and_compute_synthetic_gold,
    build_macro_monthly_dataset,
    compute_effective_deposit_rate,
    compute_log_returns,
    resample_rates_to_monthly,
    resample_to_monthly_close,
)
from time_to_afford.data.validators import (
    DataContractError,
    DataIntegrityError,
    ValidationError,
    validate_data_contract,
    validate_missing_values,
    validate_numeric_bounds,
    validate_time_series_structure,
)


# =====================================================================
# 1. EVDS Client & Loader Testleri
# =====================================================================

class TestEVDSClient:
    """TCMB EVDS API İstemcisi testleri."""

    @patch("time_to_afford.data.loaders.get_settings")
    def test_init_without_key_raises(self, mock_get_settings):
        """API anahtarı verilmediğinde açık ValueError üretilmeli."""
        mock_settings = MagicMock()
        mock_settings.evds_api_key = None
        mock_get_settings.return_value = mock_settings
        with pytest.raises(ValueError, match="API anahtarı zorunludur"):
            EVDSClient(api_key=None)

    def test_init_with_empty_key_raises(self):
        """Boş API anahtarı hata vermeli."""
        with pytest.raises(ValueError, match="API anahtarı zorunludur"):
            EVDSClient(api_key="")

    @patch("requests.Session.get")
    def test_fetch_series_success(self, mock_get):
        """Başarılı API isteğinde JSON parse edilmeli ve URL/header doğrulanmalı."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "totalCount": 2,
            "items": [
                {"Tarih": "01-01-2023", "TP_FG_J0": "1203.4"},
                {"Tarih": "01-02-2023", "TP_FG_J0": "1241.3"},
            ],
        }
        mock_get.return_value = mock_response

        client = EVDSClient(api_key="test_api_key")
        df = client.fetch_series("TP.FG.J0", start_date="01-01-2023", end_date="01-02-2023")

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "TP_FG_J0" in df.columns

        # URL'in path parametreleriyle dogru uretildigini dogrula
        mock_get.assert_called_once()
        call_url = mock_get.call_args[0][0]
        assert "series=TP.FG.J0&startDate=01-01-2023&endDate=01-02-2023&type=json" in call_url

        # Header'da key gönderildiği doğrulanmalı
        headers = mock_get.call_args[1].get("headers", {})
        assert headers.get("key") == "test_api_key"

    @patch("requests.Session.get")
    def test_fetch_series_http_error_raises(self, mock_get):
        """401/403/500 gibi HTTP hatalarında DataDownloadError üretilmeli."""
        mock_response = MagicMock()
        mock_response.status_code = 403
        mock_response.text = "Forbidden / Invalid Key"
        mock_get.return_value = mock_response

        client = EVDSClient(api_key="invalid_key")
        with pytest.raises(DataDownloadError, match="HTTP 403"):
            client.fetch_series("TP.FG.J0", start_date="01-01-2023", end_date="01-02-2023")

    @patch("requests.Session.get")
    def test_fetch_series_empty_items_raises(self, mock_get):
        """API boş veya geçersiz liste döndüğünde DataFormatError üretilmeli."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"totalCount": 0, "items": []}
        mock_get.return_value = mock_response

        client = EVDSClient(api_key="test_key")
        with pytest.raises(DataFormatError):
            client.fetch_series("TP.FG.J0", start_date="01-01-2023", end_date="01-02-2023")

    @patch("requests.Session.get")
    def test_fetch_series_malformed_json_raises(self, mock_get):
        """Bozuk JSON yanıtında DataFormatError üretilmeli."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.side_effect = ValueError("Invalid JSON")
        mock_get.return_value = mock_response

        client = EVDSClient(api_key="test_key")
        with pytest.raises(DataFormatError, match="JSON formatı bozuk"):
            client.fetch_series("TP.FG.J0", start_date="01-01-2023", end_date="01-02-2023")


class TestYahooFinanceLoader:
    """Yahoo Finance Loader testleri."""

    @patch("yfinance.download")
    def test_download_success(self, mock_yf_download):
        """Başarılı indirmede DataFrame dönmeli."""
        mock_df = pd.DataFrame(
            {"Close": [5000.0, 5100.0]},
            index=pd.to_datetime(["2023-01-02", "2023-01-03"]),
        )
        mock_yf_download.return_value = mock_df

        loader = YahooFinanceLoader()
        df = loader.fetch_ticker("XU100.IS", start_date="2023-01-01", end_date="2023-01-05")

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "close" in df.columns

    @patch("yfinance.download")
    def test_download_empty_raises(self, mock_yf_download):
        """Boş dönen indirmede DataDownloadError üretilmeli."""
        mock_yf_download.return_value = pd.DataFrame()

        loader = YahooFinanceLoader()
        with pytest.raises(DataDownloadError):
            loader.fetch_ticker("INVALID_TICKER", start_date="2023-01-01", end_date="2023-01-05")


class TestRawSnapshots:
    """Raw snapshot immutability testleri."""

    def test_save_and_load_raw_snapshot(self, tmp_path):
        """Raw snapshot zaman damgasıyla kaydedilmeli ve okunabilmeli."""
        df = pd.DataFrame(
            {"value": [10.0, 20.0]},
            index=pd.to_datetime(["2023-01-31", "2023-02-28"]),
        )

        saved_path = save_raw_snapshot(
            df=df,
            source="test_source",
            series_name="test_series",
            raw_dir=tmp_path,
        )

        assert saved_path.exists()
        assert "test_source_test_series" in saved_path.name

        loaded_df = load_latest_raw_snapshot(
            source="test_source",
            series_name="test_series",
            raw_dir=tmp_path,
        )
        assert len(loaded_df) == 2
        assert list(loaded_df["value"]) == [10.0, 20.0]


# =====================================================================
# 2. Validator Testleri
# =====================================================================

class TestValidators:
    """Veri doğrulama ve bütünlük testleri."""

    def test_validate_time_series_structure_valid(self):
        """Geçerli yapı doğrulamadan geçmeli."""
        df = pd.DataFrame(
            {"val": [1.0, 2.0]},
            index=pd.to_datetime(["2023-01-01", "2023-02-01"]),
        )
        validate_time_series_structure(df)

    def test_validate_non_datetime_index_raises(self):
        """Datetime olmayan indeks hata vermeli."""
        df = pd.DataFrame({"val": [1.0, 2.0]}, index=[1, 2])
        with pytest.raises(ValidationError, match="DatetimeIndex"):
            validate_time_series_structure(df)

    def test_validate_duplicate_dates_raises(self):
        """Mükerrer tarih indeksi DataIntegrityError vermeli."""
        df = pd.DataFrame(
            {"val": [1.0, 2.0]},
            index=pd.to_datetime(["2023-01-01", "2023-01-01"]),
        )
        with pytest.raises(DataIntegrityError, match="Mükerrer"):
            validate_time_series_structure(df)

    def test_validate_unsorted_dates_raises(self):
        """Kronolojik sırada olmayan tarihler hata vermeli."""
        df = pd.DataFrame(
            {"val": [1.0, 2.0]},
            index=pd.to_datetime(["2023-02-01", "2023-01-01"]),
        )
        with pytest.raises(DataIntegrityError, match="kronolojik"):
            validate_time_series_structure(df)

    def test_validate_future_dates_raises(self):
        """Bugünden ileri tarih içeren veri hata vermeli."""
        future_date = datetime.now() + timedelta(days=365)
        df = pd.DataFrame(
            {"val": [1.0]},
            index=pd.to_datetime([future_date]),
        )
        with pytest.raises(ValidationError, match="Gelecek tarihli"):
            validate_time_series_structure(df)

    def test_validate_numeric_bounds_positive_price(self):
        """Sıfır veya negatif fiyat ValidationError vermeli."""
        df = pd.DataFrame(
            {"price": [100.0, -5.0]},
            index=pd.to_datetime(["2023-01-01", "2023-02-01"]),
        )
        with pytest.raises(ValidationError):
            validate_numeric_bounds(df, column="price", allow_zero=False, allow_negative=False)

    def test_validate_numeric_bounds_negative_interest_rate(self):
        """Negatif faiz oranı ValidationError vermeli."""
        df = pd.DataFrame(
            {"rate": [15.0, -0.5]},
            index=pd.to_datetime(["2023-01-01", "2023-02-01"]),
        )
        with pytest.raises(ValidationError):
            validate_numeric_bounds(df, column="rate", allow_zero=True, allow_negative=False)

    def test_validate_missing_values_internal_nan_fails(self):
        """Internal NaN'ler reddedilmeli, ancak leading ve trailing NaN'ler kabul edilmelidir."""
        dates = pd.date_range("2020-01-01", periods=5, freq="D")

        # 1. Leading NaN: İlk değer NaN, diğerleri dolu. Kabul edilmeli.
        df_leading = pd.DataFrame({"col": [np.nan, 2.0, 3.0, 4.0, 5.0]}, index=dates)
        validate_missing_values(df_leading, "col")  # Should not raise

        # 2. Trailing NaN: Son değer NaN, diğerleri dolu. Kabul edilmeli.
        df_trailing = pd.DataFrame({"col": [1.0, 2.0, 3.0, 4.0, np.nan]}, index=dates)
        validate_missing_values(df_trailing, "col")  # Should not raise

        # 3. Internal NaN: Ortada NaN var. Reddedilmeli.
        df_internal = pd.DataFrame({"col": [1.0, 2.0, np.nan, 4.0, 5.0]}, index=dates)
        with pytest.raises(DataIntegrityError, match="ortasında delik olamaz"):
            validate_missing_values(df_internal, "col")


# =====================================================================
# 3. Preprocessing ve Matematiksel Dönüşüm Testleri
# =====================================================================

class TestPreprocessing:
    """Zaman serisi ön işleme ve dönüşüm testleri."""

    def test_resample_to_monthly_close(self):
        """Günlük seriden ayın son gerçekleşen kapanış günü seçilmeli."""
        # Cuma ay sonu ise Cuma günü değeri alınmalı
        df = pd.DataFrame(
            {"close": [10.0, 11.0, 12.0]},
            index=pd.to_datetime(["2023-01-29", "2023-01-30", "2023-01-31"]),
        )
        res = resample_to_monthly_close(df, column="close")
        assert len(res) == 1
        assert res.iloc[0] == 12.0
        assert res.index[0] == pd.Timestamp("2023-01-31")

    def test_resample_rates_to_monthly(self):
        """Faiz serilerinde ilgili ayın aritmetik ortalaması alınmalı."""
        df = pd.DataFrame(
            {"rate": [20.0, 24.0]},
            index=pd.to_datetime(["2023-01-10", "2023-01-20"]),
        )
        res = resample_rates_to_monthly(df, column="rate")
        assert len(res) == 1
        assert res.iloc[0] == 22.0

    def test_compute_log_returns_math(self):
        """Log-return: r_t = ln(P_t / P_{t-1}) elle hesaplanmış değerle doğrulanmalı."""
        prices = pd.Series([100.0, 110.0, 121.0], index=pd.date_range("2023-01-31", periods=3, freq="ME"))
        returns = compute_log_returns(prices)

        assert np.isnan(returns.iloc[0])  # İlk değer NaN olmalı (leakage yok)
        expected_r1 = np.log(110.0 / 100.0)  # ~0.095310
        expected_r2 = np.log(121.0 / 110.0)  # ~0.095310
        assert np.isclose(returns.iloc[1], expected_r1)
        assert np.isclose(returns.iloc[2], expected_r2)

    def test_compute_log_returns_zero_or_negative_raises(self):
        """Sıfır veya negatif fiyatta log-return ValueError vermeli."""
        prices = pd.Series([100.0, 0.0], index=pd.date_range("2023-01-31", periods=2, freq="ME"))
        with pytest.raises(ValueError):
            compute_log_returns(prices)

    def test_compute_effective_deposit_rate(self):
        """Yıllık faiz -> Aylık getiri dönüşümü: (1 + i/100)^(1/12) - 1 test edilmeli."""
        annual_rate = 12.0  # %12 yıllık faiz
        expected_monthly = (1.0 + 12.0 / 100.0) ** (1.0 / 12.0) - 1.0  # ~0.00948879
        calculated = compute_effective_deposit_rate(annual_rate)
        assert np.isclose(calculated, expected_monthly, atol=1e-7)

    def test_compute_synthetic_gram_gold_math(self):
        """Sentetik gram altın formülü: (Ons * USD/TRY) / 31.1034768 elle test edilmeli."""
        ons_usd = 2000.0
        usd_try = 30.0
        expected_gram_tl = (2000.0 * 30.0) / 31.1034768  # ~1929.0448

        gold_series = pd.Series([ons_usd], index=[pd.Timestamp("2023-01-31")])
        fx_series = pd.Series([usd_try], index=[pd.Timestamp("2023-01-31")])

        synthetic = align_and_compute_synthetic_gold(gold_series, fx_series)
        assert len(synthetic) == 1
        assert np.isclose(synthetic.iloc[0], expected_gram_tl, atol=1e-3)

    def test_align_and_compute_synthetic_gold_mismatched_dates(self):
        """Ons ve USD/TRY farklı tatil günlerine sahip olduğunda güvenli inner join yapılmalı."""
        # 1. gün: ortak, 2. gün: sadece altın, 3. gün: sadece fx
        gold_series = pd.Series([2000.0, 2010.0], index=pd.to_datetime(["2023-01-30", "2023-01-31"]))
        fx_series = pd.Series([30.0, 30.5], index=pd.to_datetime(["2023-01-30", "2023-02-01"]))

        synthetic = align_and_compute_synthetic_gold(gold_series, fx_series)
        assert len(synthetic) == 1
        assert synthetic.index[0] == pd.Timestamp("2023-01-30")


# =====================================================================
# 4. Pipeline Runner & Data Contract Uyum Testi
# =====================================================================

class TestMacroPipelineRunner:
    """Nihai dataset üretim pipeline'ı ve Data Contract testleri."""

    def test_build_macro_monthly_dataset_contract_compliance(self):
        """Pipeline çıktısı Data Contract'a tam uyum sağlamalı ve salary_growth içermemelidir."""
        daily_dates = pd.date_range("2020-01-01", "2020-12-31", freq="D")

        def make_df(val_start, val_end):
            return pd.DataFrame({"value": np.linspace(val_start, val_end, len(daily_dates))}, index=daily_dates)

        raw_data = {
            "cpi_index": make_df(100, 150),
            "house_price_index": make_df(200, 350),
            "deposit_rate_3m": make_df(25.0, 25.0),
            "usd_try": make_df(10, 20),
            "policy_rate": make_df(15.0, 15.0),
            "bist100_close": make_df(2000, 5000),
            "gold_ons_usd": make_df(1800, 2000),
            "vehicle_price_proxy": make_df(100, 180),
        }

        df_processed = build_macro_monthly_dataset(raw_data)

        # 1. salary_growth_rate OLMAMALI (katman ayrımı kuralı)
        assert "salary_growth_rate" not in df_processed.columns
        assert "salary" not in str(df_processed.columns).lower()

        # 2. Zorunlu sütunlar mevcut olmalı
        expected_cols = [
            "cpi_index", "cpi_return",
            "house_price_index", "house_price_return",
            "deposit_rate_3m", "deposit_monthly_return",
            "usd_try", "usd_try_return",
            "policy_rate",
            "bist100_close", "bist100_return",
            "synthetic_gram_gold_try", "gold_return",
            "vehicle_price_proxy", "vehicle_proxy_return",
        ]
        for col in expected_cols:
            assert col in df_processed.columns, f"Eksik sütun: {col}"

        # 3. İndeks ME olmalı
        assert isinstance(df_processed.index, pd.DatetimeIndex)

        # 4. Data contract doğrulayıcı geçerli kabul etmeli
        validate_data_contract(df_processed)

    def test_validate_data_contract_missing_column_raises(self):
        """Eksik sütun içeren dataframe DataContractError vermeli."""
        df = pd.DataFrame(
            {"cpi_index": [100.0]},
            index=pd.to_datetime(["2023-01-31"]),
        )
        with pytest.raises(DataContractError, match="Eksik sütunlar"):
            validate_data_contract(df)
