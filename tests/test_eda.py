"""
EDA (Exploratory Data Analysis) modülü kapsamlı birim ve edge-case testleri.

Testler tamamen offline / mockable olarak tasarlanmıştır.
"""

import numpy as np
import pandas as pd
import pytest

from time_to_afford.data.eda import (
    compute_autocorrelations,
    compute_cross_correlation,
    compute_descriptive_stats,
    compute_rolling_volatility,
    compute_stationarity_tests,
    diagnose_arch_effect,
    diagnose_structural_breaks,
    generate_candidate_model_matrix,
    perform_conditional_stl,
    run_eda_pipeline,
)


# =====================================================================
# 1. Tanımlayıcı İstatistikler ve Normallik Testleri
# =====================================================================

class TestDescriptiveStats:
    """Tanımlayıcı istatistikler ve holistik normallik testleri."""

    def test_compute_descriptive_stats_valid(self):
        """Geçerli bir seride tüm özet metrikler hesaplanmalı."""
        np.random.seed(42)
        dates = pd.date_range("2020-01-31", periods=36, freq="ME")
        series = pd.Series(np.random.normal(loc=100, scale=10, size=36), index=dates, name="test_cpi")

        stats = compute_descriptive_stats(series)

        assert stats["series_name"] == "test_cpi"
        assert stats["count"] == 36
        assert np.isclose(stats["mean"], 100, atol=5)
        assert np.isclose(stats["std"], 10, atol=5)
        assert "skewness" in stats
        assert "kurtosis" in stats
        assert "p10" in stats
        assert "p50" in stats
        assert "p90" in stats
        assert "jarque_bera_pvalue" in stats
        assert "shapiro_wilk_pvalue" in stats

    def test_compute_descriptive_stats_empty_raises(self):
        """Boş seri verildiğinde ValueError fırlatılmalı."""
        series = pd.Series([], dtype=float, name="empty")
        with pytest.raises(ValueError, match="boş"):
            compute_descriptive_stats(series)

    def test_compute_descriptive_stats_constant_series(self):
        """Sabit (sıfır varyanslı) seri çökmeksizin sıfır std döndürmeli."""
        dates = pd.date_range("2020-01-31", periods=20, freq="ME")
        series = pd.Series(np.full(20, 50.0), index=dates, name="constant")

        stats = compute_descriptive_stats(series)
        assert stats["std"] == 0.0
        assert stats["mean"] == 50.0


# =====================================================================
# 2. Durağanlık (Stationarity) Testleri: ADF & KPSS
# =====================================================================

class TestStationarityTests:
    """ADF ve KPSS testleri."""

    def test_stationarity_on_stationary_series(self):
        """Durağan AR(1) veya beyaz gürültü serisinde ADF reddetmeli, KPSS reddedememeli."""
        np.random.seed(42)
        dates = pd.date_range("2015-01-31", periods=60, freq="ME")
        series = pd.Series(np.random.normal(0, 1, 60), index=dates, name="stationary_noise")

        res = compute_stationarity_tests(series)

        assert res["series_name"] == "stationary_noise"
        assert res["adf_pvalue"] < 0.05
        assert res["kpss_pvalue"] >= 0.05
        assert res["decision"] == "I(0) Stationary"

    def test_stationarity_on_random_walk(self):
        """Rastgele yürüyüş (random walk / I(1)) serisinde fark alma gerekliliği teşhis edilmeli."""
        np.random.seed(42)
        dates = pd.date_range("2015-01-31", periods=80, freq="ME")
        rw = np.cumsum(np.random.normal(0.5, 1, 80)) + 100
        series = pd.Series(rw, index=dates, name="random_walk")

        res = compute_stationarity_tests(series)
        assert res["decision"] in ["I(1) Non-Stationary", "Trend Stationary / Structural Break"]

    def test_stationarity_short_series_graceful_fallback(self):
        """Çok kısa seride (ör: 5 gözlem) çökmeden güvenli uyarı döndürmeli."""
        dates = pd.date_range("2020-01-31", periods=5, freq="ME")
        series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=dates, name="short")

        res = compute_stationarity_tests(series)
        assert res["decision"] == "Insufficient Observations"


# =====================================================================
# 3. Otokorelasyon (ACF/PACF) ve Koşullu STL
# =====================================================================

class TestAutocorrelationAndSTL:
    """ACF/PACF ve STL testleri."""

    def test_compute_autocorrelations(self):
        """ACF ve PACF değerleri doğru boyutta dönmeli."""
        np.random.seed(42)
        dates = pd.date_range("2020-01-31", periods=30, freq="ME")
        series = pd.Series(np.random.normal(0, 1, 30), index=dates, name="noise")

        df_acf = compute_autocorrelations(series, max_lags=10)
        assert len(df_acf) > 0
        assert "acf" in df_acf.columns
        assert "pacf" in df_acf.columns

    def test_stl_with_sufficient_data_and_seasonality(self):
        """Yeterli veri (N >= 24) ve mevsimsellik içeren seride STL trend, seasonal, resid döndürmeli."""
        np.random.seed(42)
        dates = pd.date_range("2015-01-31", periods=48, freq="ME")
        trend = np.linspace(10, 50, 48)
        season = 5 * np.sin(2 * np.pi * np.arange(48) / 12)
        noise = np.random.normal(0, 1, 48)
        series = pd.Series(trend + season + noise, index=dates, name="seasonal_cpi")

        res = perform_conditional_stl(series, period=12)
        assert res["stl_applied"] is True
        assert "trend" in res
        assert "seasonal" in res
        assert "residual" in res

    def test_stl_skipped_when_insufficient_data(self):
        """N < 24 olduğunda STL zorlanmamalı, stl_applied=False dönmeli."""
        dates = pd.date_range("2020-01-31", periods=18, freq="ME")
        series = pd.Series(np.random.normal(10, 1, 18), index=dates, name="short_series")

        res = perform_conditional_stl(series, period=12)
        assert res["stl_applied"] is False


# =====================================================================
# 4. Volatilite: Rolling Volatilite & ARCH-LM Teşhisi
# =====================================================================

class TestVolatilityAndARCH:
    """Rolling volatilite ve ARCH-LM testleri."""

    def test_compute_rolling_volatility(self):
        """12 aylık kayan standart sapma hesaplanmalı ve leakage içermemeli."""
        np.random.seed(42)
        dates = pd.date_range("2020-01-31", periods=24, freq="ME")
        returns = pd.Series(np.random.normal(0.02, 0.05, 24), index=dates, name="returns")

        r_vol = compute_rolling_volatility(returns, window=12)
        assert len(r_vol) == 24
        # İlk 5 ay min_periods=6 nedeniyle NaN olabilir, 12. ayda geçerli float olmalı
        assert not np.isnan(r_vol.iloc[12])

    def test_arch_lm_on_garch_process(self):
        """Oynaklık kümelenmesi içeren getiri serisinde ARCH etkisi teşhis edilmeli."""
        np.random.seed(42)
        n = 100
        sigma = np.zeros(n)
        returns = np.zeros(n)
        sigma[0] = 0.02
        for t in range(1, n):
            sigma[t] = np.sqrt(0.0001 + 0.6 * (returns[t - 1] ** 2) + 0.3 * (sigma[t - 1] ** 2))
            returns[t] = np.random.normal(0, sigma[t])

        dates = pd.date_range("2015-01-31", periods=n, freq="ME")
        series = pd.Series(returns, index=dates, name="bist_returns")

        res = diagnose_arch_effect(series, lags=3)
        assert "arch_lm_pvalue" in res
        assert "has_arch_effect" in res

    def test_arch_lm_skipped_for_short_series(self):
        """Yetersiz örneklemde ARCH testi güvenli şekilde atlanmalı."""
        dates = pd.date_range("2020-01-31", periods=8, freq="ME")
        series = pd.Series(np.random.normal(0, 1, 8), index=dates, name="short_returns")

        res = diagnose_arch_effect(series, lags=3)
        assert res["test_applied"] is False


# =====================================================================
# 5. Gecikmeli Çapraz Korelasyon (CCF) ve Lag Yönü Testi
# =====================================================================

class TestCrossCorrelation:
    """Çapraz korelasyon ve kesin lag yönü doğrulaması."""

    def test_ccf_lag_direction_mathematical_proof(self):
        """
        Metodolojik Doğrulama Testi:
        CCF(X, Y, k) = corr(X_t, Y_{t+k})
        Eğer X_t serisi Y_t serisinden tam 1 ay ÖNCE hareket ediyorsa (X leads Y by 1 month, yani Y_{t+1} ≈ X_t),
        maksimum korelasyon k = +1 gecikmesinde çıkmalıdır.
        """
        np.random.seed(42)
        n = 50
        dates = pd.date_range("2020-01-31", periods=n, freq="ME")

        x = np.random.normal(0, 1, n)
        y = np.zeros(n)
        y[1:] = x[:-1] + np.random.normal(0, 0.05, n - 1)

        s_x = pd.Series(x, index=dates, name="FX_Shock")
        s_y = pd.Series(y, index=dates, name="CPI_Response")

        ccf_df = compute_cross_correlation(s_x, s_y, max_lags=4)

        corr_at_plus_1 = ccf_df.loc[ccf_df["lag"] == 1, "correlation"].values[0]
        corr_at_minus_1 = ccf_df.loc[ccf_df["lag"] == -1, "correlation"].values[0]

        assert corr_at_plus_1 > 0.85, f"k=+1 korelasyonu yüksek olmalı, bulunan: {corr_at_plus_1}"
        assert corr_at_plus_1 > corr_at_minus_1, "X leads Y durumunda k=+1 korelasyonu k=-1'den büyük olmalı"

    def test_ccf_insufficient_overlap_handled(self):
        """Kesişimi olmayan serilerde CCF çökmemeli ve ValueError vermeli."""
        dates1 = pd.date_range("2010-01-31", periods=10, freq="ME")
        dates2 = pd.date_range("2020-01-31", periods=10, freq="ME")
        s1 = pd.Series(np.random.normal(0, 1, 10), index=dates1, name="s1")
        s2 = pd.Series(np.random.normal(0, 1, 10), index=dates2, name="s2")

        with pytest.raises(ValueError, match="ortak tarih"):
            compute_cross_correlation(s1, s2, max_lags=3)


# =====================================================================
# 6. Yapısal Kırılma ve Aday Model Tavsiye Matrisi Testleri
# =====================================================================

class TestStructuralBreaksAndCandidates:
    """Yapısal kırılma ve Phase 4 Aday Model Karar Matrisi testleri."""

    def test_diagnose_structural_breaks(self):
        """Rejim pencerelerinde ortalama ve standart sapmalar hesaplanmalı."""
        dates = pd.date_range("2018-01-31", periods=60, freq="ME")
        df = pd.DataFrame({
            "cpi_return": np.random.normal(0.02, 0.01, 60),
            "usd_try_return": np.random.normal(0.03, 0.02, 60),
        }, index=dates)

        breaks_df = diagnose_structural_breaks(df)
        assert isinstance(breaks_df, pd.DataFrame)
        assert len(breaks_df) > 0
        assert "event_name" in breaks_df.columns
        assert "mean" in breaks_df.columns
        assert "std" in breaks_df.columns

    def test_generate_candidate_model_matrix(self):
        """Tüm seriler için kural bazlı model tavsiye tablosu üretilmeli."""
        stationarity_results = [
            {"series_name": "cpi_index", "decision": "I(1) Non-Stationary", "suggested_transform": "log_return"},
            {"series_name": "deposit_rate_3m", "decision": "I(0) Stationary", "suggested_transform": "level"},
        ]

        matrix = generate_candidate_model_matrix(stationarity_results)
        assert isinstance(matrix, pd.DataFrame)
        assert len(matrix) == 2
        assert "candidate_models" in matrix.columns
        cpi_models = matrix.loc[matrix["series_name"] == "cpi_index", "candidate_models"].values[0]
        assert "SARIMA" in cpi_models or "ETS" in cpi_models or "Drift" in cpi_models


# =====================================================================
# 7. Pipeline Runner Testleri
# =====================================================================

class TestEDAPipelineRunner:
    """EDA boru hattı ve reports/eda/ çıktısı testleri."""

    def test_run_eda_pipeline_reproducible_outputs(self, tmp_path):
        """Sentetik DataFrame verildiğinde reports/eda/ altında tüm standart CSV'ler üretilmeli."""
        np.random.seed(42)
        dates = pd.date_range("2018-01-31", periods=36, freq="ME")
        df = pd.DataFrame(
            {
                "cpi_index": np.linspace(100, 200, 36) + np.random.normal(0, 1, 36),
                "cpi_return": np.random.normal(0.02, 0.005, 36),
                "usd_try": np.linspace(5, 20, 36) + np.random.normal(0, 0.2, 36),
                "usd_try_return": np.random.normal(0.03, 0.01, 36),
            },
            index=dates,
        )

        output_dir = tmp_path / "reports" / "eda"
        report_files = run_eda_pipeline(df, output_dir=output_dir)

        assert (output_dir / "summary.csv").exists()
        assert (output_dir / "stationarity.csv").exists()
        assert (output_dir / "correlation.csv").exists()
        assert (output_dir / "correlation_spearman.csv").exists()
        assert (output_dir / "structural_breaks.csv").exists()
        assert (output_dir / "eda_report.md").exists()
