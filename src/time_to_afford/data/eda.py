"""
Keşifsel Veri Analizi (EDA - Exploratory Data Analysis) ve Teşhis Modülü.

Gözlemlenmiş makroekonomik serilerin dağılım, otokorelasyon (ACF/PACF), mevsimsellik (STL),
çift yönlü durağanlık (ADF/KPSS), ARCH oynaklık kümelenmesi, 12 aylık kayan oynaklık,
gecikmeli çapraz korelasyon (CCF) ve yapısal kırılma/rejim teşhislerini gerçekleştirir.

Katman Sorumluluk Sınırı: Bu modül kesinlikle model fit etmez, tahmin üretmez.
Yalnızca istatistiksel analiz yapar ve Phase 4 için aday model tavsiye matrisi üretir.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf, adfuller, kpss, pacf

from time_to_afford.config.settings import get_settings
from time_to_afford.utils.logging import get_logger

logger = get_logger(__name__)


# =====================================================================
# 1. Tanımlayıcı İstatistikler & Holistik Normallik Teşhisi
# =====================================================================

def compute_descriptive_stats(series: pd.Series) -> Dict[str, Any]:
    """Tek bir zaman serisi için konum, yayılım, basıklık ve normallik metriklerini hesaplar.

    Parameters
    ----------
    series : pd.Series
        Analiz edilecek sayısal seri.

    Returns
    -------
    dict
        Tanımlayıcı istatistikler ve normallik test sonuçları sözlüğü.

    Raises
    ------
    ValueError
        Seri boşsa veya tamamen NaN içeriyorsa.
    """
    clean_series = series.dropna()
    if clean_series.empty:
        raise ValueError(f"'{series.name or 'Seri'}' için geçerli gözlem bulunamadı (seri boş).")

    n = len(clean_series)
    val_mean = float(clean_series.mean())
    val_std = float(clean_series.std(ddof=1)) if n > 1 else 0.0

    skew_val = float(stats.skew(clean_series)) if n >= 3 and val_std > 0 else 0.0
    kurt_val = float(stats.kurtosis(clean_series, fisher=True)) if n >= 4 and val_std > 0 else 0.0

    # Jarque-Bera Normallik Testi (Asimetri ve basıklık testi)
    if n >= 8 and val_std > 0:
        jb_stat, jb_pvalue = stats.jarque_bera(clean_series)
        jb_stat, jb_pvalue = float(jb_stat), float(jb_pvalue)
    else:
        jb_stat, jb_pvalue = np.nan, np.nan

    # Shapiro-Wilk (Yardımcı test, n <= 5000)
    if 3 <= n <= 5000 and val_std > 0:
        shapiro_stat, shapiro_pvalue = stats.shapiro(clean_series)
        shapiro_stat, shapiro_pvalue = float(shapiro_stat), float(shapiro_pvalue)
    else:
        shapiro_stat, shapiro_pvalue = np.nan, np.nan

    return {
        "series_name": series.name or "unnamed_series",
        "count": n,
        "mean": val_mean,
        "std": val_std,
        "min": float(clean_series.min()),
        "p10": float(clean_series.quantile(0.10)),
        "p25": float(clean_series.quantile(0.25)),
        "p50": float(clean_series.quantile(0.50)),
        "p75": float(clean_series.quantile(0.75)),
        "p90": float(clean_series.quantile(0.90)),
        "max": float(clean_series.max()),
        "skewness": skew_val,
        "kurtosis": kurt_val,
        "jarque_bera_stat": jb_stat,
        "jarque_bera_pvalue": jb_pvalue,
        "shapiro_wilk_stat": shapiro_stat,
        "shapiro_wilk_pvalue": shapiro_pvalue,
        "annualized_std": val_std * np.sqrt(12.0),
    }


# =====================================================================
# 2. Çift Yönlü Durağanlık (Stationarity) Teşhisi (ADF & KPSS)
# =====================================================================

def compute_stationarity_tests(series: pd.Series) -> Dict[str, Any]:
    """ADF (Birim Kök) ve KPSS (Durağanlık) testlerini birlikte uygular ve karar matrisi üretir.

    Parameters
    ----------
    series : pd.Series
        Test edilecek zaman serisi.

    Returns
    -------
    dict
        ADF p-value, KPSS p-value, test istatistikleri ve ekonometrik karar.
    """
    clean_series = series.dropna()
    series_name = series.name or "series"

    if len(clean_series) < 12 or clean_series.std() == 0:
        return {
            "series_name": series_name,
            "adf_stat": np.nan,
            "adf_pvalue": np.nan,
            "kpss_stat": np.nan,
            "kpss_pvalue": np.nan,
            "decision": "Insufficient Observations",
            "interpretation": "Gözlem sayısı ekonometrik testler için yetersizdir.",
        }

    # 1. ADF Testi (H0: Birim kök var / durağan değil)
    try:
        adf_res = adfuller(clean_series, autolag="AIC")
        adf_stat, adf_pvalue = float(adf_res[0]), float(adf_res[1])
    except Exception as e:
        logger.warning(f"ADF testi başarısız ({series_name}): {e}")
        adf_stat, adf_pvalue = np.nan, np.nan

    # 2. KPSS Testi (H0: Düzey durağandır)
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kpss_res = kpss(clean_series, regression="c", nlags="auto")
            kpss_stat, kpss_pvalue = float(kpss_res[0]), float(kpss_res[1])
    except Exception as e:
        logger.warning(f"KPSS testi başarısız ({series_name}): {e}")
        kpss_stat, kpss_pvalue = np.nan, np.nan

    # Karar Matrisi
    is_adf_stationary = (adf_pvalue < 0.05) if not np.isnan(adf_pvalue) else False
    is_kpss_stationary = (kpss_pvalue >= 0.05) if not np.isnan(kpss_pvalue) else False

    if is_adf_stationary and is_kpss_stationary:
        decision = "I(0) Stationary"
        interp = "Seri kesinlikle durağandır (fark alma gerektirmez)."
    elif not is_adf_stationary and not is_kpss_stationary:
        decision = "I(1) Non-Stationary"
        interp = "Seri birim kök içermektedir; birinci fark (log-return) alınmalıdır."
    elif is_adf_stationary and not is_kpss_stationary:
        decision = "Trend Stationary / Structural Break"
        interp = "Seri trend durağandır veya yapısal kırılma içermektedir."
    else:
        decision = "Inconclusive"
        interp = "Testler sınır durumdadır; serinin ekonomik doğası dikkate alınmalıdır."

    return {
        "series_name": series_name,
        "adf_stat": adf_stat,
        "adf_pvalue": adf_pvalue,
        "kpss_stat": kpss_stat,
        "kpss_pvalue": kpss_pvalue,
        "decision": decision,
        "interpretation": interp,
    }


# =====================================================================
# 3. Otokorelasyon (ACF / PACF) ve Koşullu STL Ayrıştırması
# =====================================================================

def compute_autocorrelations(series: pd.Series, max_lags: int = 24) -> pd.DataFrame:
    """Zaman serisinin ACF (Otokorelasyon) ve PACF (Kısmi Otokorelasyon) değerlerini hesaplar.

    Parameters
    ----------
    series : pd.Series
        Analiz edilecek zaman serisi.
    max_lags : int, default 24
        Hesaplanacak maksimum gecikme sayısı.

    Returns
    -------
    pd.DataFrame
        ['lag', 'acf', 'pacf'] sütunlu tablo.
    """
    clean_series = series.dropna()
    n = len(clean_series)
    nlags = min(max_lags, n // 2 - 1) if n > 4 else 1

    if nlags < 1:
        return pd.DataFrame({"lag": [], "acf": [], "pacf": []})

    acf_vals = acf(clean_series, nlags=nlags, fft=True)
    try:
        pacf_vals = pacf(clean_series, nlags=nlags, method="yule_walker")
    except Exception:
        pacf_vals = np.full(len(acf_vals), np.nan)

    return pd.DataFrame({
        "lag": list(range(len(acf_vals))),
        "acf": acf_vals,
        "pacf": pacf_vals,
    })


def perform_conditional_stl(series: pd.Series, period: int = 12) -> Dict[str, Any]:
    """Yeterli gözlem (N >= 2*period) varsa STL ayrıştırması yapar.

    Parameters
    ----------
    series : pd.Series
        Ayrıştırılacak seri.
    period : int, default 12
        Mevsimsel periyot (aylık seriler için 12).

    Returns
    -------
    dict
        'stl_applied': bool, 'trend', 'seasonal', 'residual' serileri.
    """
    clean_series = series.dropna()
    if len(clean_series) < 2 * period:
        return {
            "stl_applied": False,
            "reason": f"Yetersiz gözlem sayısı ({len(clean_series)} < {2 * period}). STL için en az 2 tam dönem gereklidir.",
        }

    try:
        stl_model = STL(clean_series, period=period, robust=True)
        res = stl_model.fit()
        return {
            "stl_applied": True,
            "trend": res.trend,
            "seasonal": res.seasonal,
            "residual": res.resid,
            "seasonal_strength": max(0.0, float(1.0 - np.var(res.resid) / np.var(res.seasonal + res.resid)))
            if np.var(res.seasonal + res.resid) > 0 else 0.0,
        }
    except Exception as e:
        logger.warning(f"STL ayrıştırması uygulanamadı ({series.name}): {e}")
        return {"stl_applied": False, "reason": str(e)}


# =====================================================================
# 4. Volatilite: 12 Aylık Rolling Oynaklık & ARCH-LM Teşhisi
# =====================================================================

def compute_rolling_volatility(series: pd.Series, window: int = 12) -> pd.Series:
    """Geriye dönük (strictly backward-looking, leakage-safe) kayan standart sapmayı hesaplar.

    Parameters
    ----------
    series : pd.Series
        Getiri serisi.
    window : int, default 12
        Geriye dönük pencere boyutu (12 ay).

    Returns
    -------
    pd.Series
        Kayan standart sapma serisi.
    """
    clean_series = series.dropna()
    rolling_std = clean_series.rolling(window=window, min_periods=max(3, window // 2)).std()
    rolling_std.name = f"{series.name or 'series'}_rolling_vol_{window}m"
    return rolling_std


def diagnose_arch_effect(series: pd.Series, lags: int = 3) -> Dict[str, Any]:
    """Durağanlaştırılmış getiri serisinde ARCH oynaklık kümelenmesi etkisini test eder.

    Parameters
    ----------
    series : pd.Series
        Durağan getiri / fark serisi.
    lags : int, default 3
        Test edilecek gecikme sayısı.

    Returns
    -------
    dict
        ARCH-LM test istatistiği, p-value ve kümelenme tespiti.
    """
    clean_series = series.dropna()
    if len(clean_series) < 20 or clean_series.std() == 0:
        return {
            "test_applied": False,
            "reason": "Yetersiz gözlem veya sıfır varyans.",
        }

    # Basit ARCH-LM Regresyonu: e_t^2 = a_0 + a_1*e_{t-1}^2 + ... + a_p*e_{t-p}^2
    mean_adjusted = clean_series - clean_series.mean()
    sq_resid = (mean_adjusted ** 2).values
    n = len(sq_resid)

    # Gecikmeli matrisi oluştur
    y = sq_resid[lags:]
    x_mat = np.column_stack([np.ones(n - lags)] + [sq_resid[lags - i - 1 : n - i - 1] for i in range(lags)])

    try:
        # OLS çözümü
        beta, residuals, rank, s = np.linalg.lstsq(x_mat, y, rcond=None)
        y_hat = x_mat @ beta
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        ss_reg = np.sum((y_hat - np.mean(y)) ** 2)
        r_squared = (ss_reg / ss_tot) if ss_tot > 0 else 0.0

        lm_stat = (n - lags) * r_squared
        p_val = float(1.0 - stats.chi2.cdf(lm_stat, df=lags))

        return {
            "test_applied": True,
            "arch_lm_stat": float(lm_stat),
            "arch_lm_pvalue": p_val,
            "has_arch_effect": bool(p_val < 0.05),
            "lags": lags,
        }
    except Exception as e:
        logger.warning(f"ARCH-LM testi hatası: {e}")
        return {"test_applied": False, "reason": str(e)}


# =====================================================================
# 5. Gecikmeli Çapraz Korelasyon (CCF / Lead-Lag)
# =====================================================================

def compute_cross_correlation(
    s_x: pd.Series,
    s_y: pd.Series,
    max_lags: int = 6,
) -> pd.DataFrame:
    """İki zaman serisi arasında gecikmeli çapraz korelasyonu (CCF) hesaplar.

    Formül: CCF(X, Y, k) = corr(X_t, Y_{t+k})
    * k > 0: X serisi Y serisini k dönem ÖNCEDEN izler (X leads Y).
    * k < 0: X serisi Y serisini k dönem GECİKMELİ takip eder (X lags Y).
    * k = 0: Eşzamanlı korelasyon.

    Sorumluluk Notu: CCF tanısal bir ilişkidir, nedensellik (causality) kanıtı değildir.

    Parameters
    ----------
    s_x : pd.Series
        Öncü aday seri (X).
    s_y : pd.Series
        Hedef / takipçi aday seri (Y).
    max_lags : int, default 6
        Hesaplanacak maksimum ileri ve geri gecikme sayısı.

    Returns
    -------
    pd.DataFrame
        ['lag', 'correlation', 'interpretation'] sütunlu tablo.

    Raises
    ------
    ValueError
        İki seri arasında yeterli ortak tarih kesişimi yoksa.
    """
    df_pair = pd.DataFrame({"x": s_x, "y": s_y}).dropna()
    if len(df_pair) < max(5, max_lags + 2):
        raise ValueError(
            f"Seriler arasında yetersiz ortak tarih kesişimi: {len(df_pair)} gözlem (en az {max_lags + 2} gereklidir)."
        )

    results = []
    x_vals = df_pair["x"]
    y_vals = df_pair["y"]

    for k in range(-max_lags, max_lags + 1):
        if k > 0:
            # X_t vs Y_{t+k} -> X leads Y by k
            x_slice = x_vals.iloc[:-k]
            y_slice = y_vals.iloc[k:]
            interp = f"{s_x.name or 'X'} leads {s_y.name or 'Y'} by {k} months"
        elif k < 0:
            # X_t vs Y_{t-|k|} -> X lags Y by |k|
            abs_k = abs(k)
            x_slice = x_vals.iloc[abs_k:]
            y_slice = y_vals.iloc[:-abs_k]
            interp = f"{s_x.name or 'X'} lags {s_y.name or 'Y'} by {abs_k} months"
        else:
            x_slice = x_vals
            y_slice = y_vals
            interp = "Simultaneous correlation"

        if len(x_slice) >= 3 and x_slice.std() > 0 and y_slice.std() > 0:
            r = float(np.corrcoef(x_slice, y_slice)[0, 1])
        else:
            r = np.nan

        results.append({"lag": k, "correlation": r, "interpretation": interp})

    return pd.DataFrame(results)


# =====================================================================
# 6. Yapısal Kırılma ve Rejim Teşhisi (Event Windows)
# =====================================================================

DEFAULT_EVENT_WINDOWS: Dict[str, Tuple[str, str]] = {
    "2018_Currency_Shock": ("2018-06-01", "2019-06-30"),
    "2020_COVID_Pandemic": ("2020-03-01", "2021-06-30"),
    "2021_2023_Negative_Real_Rate_Regime": ("2021-09-01", "2023-05-31"),
}


def diagnose_structural_breaks(
    df: pd.DataFrame,
    event_windows: Optional[Dict[str, Tuple[str, str]]] = None,
) -> pd.DataFrame:
    """Belirli ekonomik kriz / rejim pencerelerinde serilerin ortalama ve oynaklık değişimlerini inceler.

    Parameters
    ----------
    df : pd.DataFrame
        Tarih indeksli makroekonomik veri tablosu.
    event_windows : dict, optional
        Rejim pencereleri {'rejim_adi': ('baslangic', 'bitis')}.

    Returns
    -------
    pd.DataFrame
        Rejim bazlı ortalama getiri ve oynaklık tablosu.
    """
    if event_windows is None:
        event_windows = DEFAULT_EVENT_WINDOWS

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    rows = []

    for event_name, (start_dt, end_dt) in event_windows.items():
        sub_df = df.loc[(df.index >= pd.Timestamp(start_dt)) & (df.index <= pd.Timestamp(end_dt)), numeric_cols]
        if sub_df.empty:
            continue

        for col in numeric_cols:
            series = sub_df[col].dropna()
            if len(series) >= 2:
                rows.append({
                    "event_name": event_name,
                    "series_name": col,
                    "start_date": start_dt,
                    "end_date": end_dt,
                    "obs_count": len(series),
                    "mean": float(series.mean()),
                    "std": float(series.std()),
                    "annualized_std": float(series.std() * np.sqrt(12.0)),
                })

    return pd.DataFrame(rows)


# =====================================================================
# 7. Aday Model Tavsiye Matrisi (Phase 4 Hazırlığı)
# =====================================================================

def generate_candidate_model_matrix(
    stationarity_results: Sequence[Dict[str, Any]],
) -> pd.DataFrame:
    """Durağanlık ve seri doğasına göre Phase 4 için aday model tablosu üretir.

    Katman Sorumluluk Sınırı: Model fit etmez; yalnızca ekonometrik kurallara göre
    tavsiye modelleri listeler.

    Parameters
    ----------
    stationarity_results : list of dict
        compute_stationarity_tests sonuçları.

    Returns
    -------
    pd.DataFrame
        Aday modeller ve tavsiye edilen dönüşümler tablosu.
    """
    rows = []
    for item in stationarity_results:
        s_name = item.get("series_name", "")
        decision = item.get("decision", "")

        if "cpi" in s_name.lower():
            suggested_tf = "log_return"
            models = "SARIMA, ETS, Baseline Drift, ARIMA"
        elif "house" in s_name.lower():
            suggested_tf = "log_return"
            models = "ARIMA, ETS, Drift, Smooth Trend"
        elif "deposit" in s_name.lower() or "policy" in s_name.lower():
            suggested_tf = "level" if "Stationary" in decision else "first_difference"
            models = "Vasicek / Mean-Reverting AR(1), Moving Average, ARIMA"
        elif "bist" in s_name.lower():
            suggested_tf = "log_return"
            models = "Random Walk with Drift, Student-t GARCH, ARIMA"
        elif "gold" in s_name.lower() or "usd" in s_name.lower():
            suggested_tf = "log_return"
            models = "Random Walk with Drift, ARIMA, GARCH"
        elif "vehicle" in s_name.lower():
            suggested_tf = "log_return"
            models = "SARIMA, Kompozit Enflasyon Modeli, ETS"
        else:
            suggested_tf = "log_return" if "I(1)" in decision else "level"
            models = "ARIMA, Baseline Drift, ETS"

        rows.append({
            "series_name": s_name,
            "stationarity_decision": decision,
            "suggested_transformation": suggested_tf,
            "candidate_models": models,
        })

    return pd.DataFrame(rows)


# =====================================================================
# 8. Reproducible EDA Boru Hattı Runner
# =====================================================================

def run_eda_pipeline(
    df: pd.DataFrame,
    output_dir: Optional[Path] = None,
) -> Dict[str, Path]:
    """İşlenmiş dataset üzerinde EDA analizlerini çalıştırır ve standart rapor dosyalarını üretir.

    Parameters
    ----------
    df : pd.DataFrame
        İşlenmiş aylık makroekonomik dataset ('macro_monthly.parquet').
    output_dir : Path, optional
        Rapor çıktı dizini. Verilmezse 'reports/eda/' kullanılır.

    Returns
    -------
    dict of str -> Path
        Oluşturulan rapor dosyalarının yolları.
    """
    if df is None or df.empty:
        raise ValueError("EDA boru hattı için geçerli bir DataFrame sağlanmalıdır.")

    if output_dir is None:
        _PROJECT_ROOT = Path(__file__).resolve().parents[3]
        output_dir = _PROJECT_ROOT / "reports" / "eda"

    output_dir = Path(output_dir)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"EDA boru hattı çalıştırılıyor. Gözlem sayısı: {len(df)}, Sütun sayısı: {len(df.columns)}")

    # 1. Tanımlayıcı İstatistikler & Durağanlık
    summary_list = []
    stationarity_list = []
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    for col in numeric_cols:
        col_series = df[col].dropna()
        if len(col_series) > 0:
            summary_list.append(compute_descriptive_stats(col_series))
            stationarity_list.append(compute_stationarity_tests(col_series))

    summary_df = pd.DataFrame(summary_list)
    summary_path = output_dir / "summary.csv"
    summary_df.to_csv(summary_path, index=False)

    # 2. Durağanlık Karar Tablosu
    stationarity_df = pd.DataFrame(stationarity_list)
    stationarity_path = output_dir / "stationarity.csv"
    stationarity_df.to_csv(stationarity_path, index=False)

    # 3. Korelasyon Matrisleri (Pearson & Spearman)
    corr_pearson = df[numeric_cols].corr(method="pearson")
    corr_path = output_dir / "correlation.csv"
    corr_pearson.to_csv(corr_path)

    corr_spearman = df[numeric_cols].corr(method="spearman")
    corr_spearman_path = output_dir / "correlation_spearman.csv"
    corr_spearman.to_csv(corr_spearman_path)

    # 4. Yapısal Kırılma Tablosu
    breaks_df = diagnose_structural_breaks(df)
    breaks_path = output_dir / "structural_breaks.csv"
    breaks_df.to_csv(breaks_path, index=False)

    # 5. Aday Model Matrisi
    candidate_matrix = generate_candidate_model_matrix(stationarity_list)

    # 6. Sentez Rapor Dokümanı (eda_report.md)
    start_date_str = str(df.index.min().date()) if isinstance(df.index, pd.DatetimeIndex) else "N/A"
    end_date_str = str(df.index.max().date()) if isinstance(df.index, pd.DatetimeIndex) else "N/A"

    report_md_content = f"""# Keşifsel Veri Analizi (EDA) Sentez Raporu

**Analiz Tarihi:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Veri Kapsamı:** {start_date_str} – {end_date_str} ({len(df)} Aylık Gözlem)  
**İncelenen Değişken Sayısı:** {len(numeric_cols)}

---

## 1. Veri Seti Özeti ve Dağılım Teşhisi
* Toplam gözlem sayısı: **{len(df)} ay**.
* Detaylı tanımlayıcı istatistikler ve normallik testleri için bkz: `summary.csv`.

## 2. Çift Yönlü Durağanlık (ADF & KPSS) Bulguları
* Detaylı test istatistikleri ve $d$ kararları için bkz: `stationarity.csv`.

## 3. Korelasyon Matrisleri
* Pearson doğrusal korelasyon matrisi için bkz: `correlation.csv`.
* Spearman monotonik korelasyon matrisi için bkz: `correlation_spearman.csv`.

## 4. Yapısal Kırılma ve Rejim Analizi
* Ekonomik olay pencereleri bazlı ortalama ve oynaklık değişimleri için bkz: `structural_breaks.csv`.

## 5. Phase 4 İçin Aday Model Tavsiyeleri

| Seri Adı | Durağanlık | Önerilen Dönüşüm | Aday Modeller |
| :--- | :--- | :--- | :--- |
"""
    for _, row in candidate_matrix.iterrows():
        report_md_content += f"| `{row['series_name']}` | {row['stationarity_decision']} | `{row['suggested_transformation']}` | {row['candidate_models']} |\n"

    report_path = output_dir / "eda_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md_content)

    logger.info(f"EDA raporları başarıyla üretildi: {output_dir}")

    return {
        "summary": summary_path,
        "stationarity": stationarity_path,
        "correlation": corr_path,
        "correlation_spearman": corr_spearman_path,
        "structural_breaks": breaks_path,
        "report": report_path,
    }
