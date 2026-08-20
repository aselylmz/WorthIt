"""
Time to Afford — Streamlit Kullanıcı Arayüzü.
"""

import sys
from pathlib import Path

import numpy as np
import streamlit as st

# Uygulama modüllerine erişebilmek için src klasörünü yola ekliyoruz
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from time_to_afford.affordability.time_to_afford import summarize_simulation
from time_to_afford.config.settings import get_settings
from time_to_afford.models.schemas import InvestmentType, TargetType
from time_to_afford.simulation.scenarios import PRESETS, get_preset, run_scenario

# ── Sayfa yapılandırması ─────────────────────────────────────────────
st.set_page_config(
    page_title="WorthIt | Time to Afford",
    page_icon="🏠",
    layout="wide",
)

# ── Yardımcı fonksiyonlar ─────────────────────────────────────────────

INVESTMENT_LABELS = {
    "gold": "🥇 Altın",
    "bist": "📈 BIST-100",
    "deposit": "🏦 Mevduat",
}

TARGET_LABELS = {
    "house": "🏠 Ev",
    "car": "🚗 Araba",
}


def format_tl(value: float) -> str:
    """TL cinsinden büyük sayıları okunabilir formata çevirir."""
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M ₺"
    if value >= 1_000:
        return f"{value / 1_000:.0f}K ₺"
    return f"{value:.0f} ₺"


def duration_label(years: int, months: int) -> str:
    """Yıl ve ayı okunabilir Türkçe metne dönüştürür."""
    parts = []
    if years:
        parts.append(f"{years} yıl")
    if months:
        parts.append(f"{months} ay")
    return " ".join(parts) if parts else "< 1 ay"


# ── Başlık ────────────────────────────────────────────────────────────
st.title("🏠 WorthIt: Time to Afford")
st.markdown(
    "**Belirli bir evi veya arabayı ne zaman satın alabileceğinizi "
    "Monte Carlo simülasyonu ile tahmin edin.**"
)
st.divider()

# ── Giriş formu ───────────────────────────────────────────────────────
st.header("📋 Finansal Bilgileriniz")

with st.form("user_input_form"):
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Mevcut Durumunuz")
        initial_savings = st.number_input(
            "Mevcut Birikim (TL)",
            min_value=0.0,
            step=10_000.0,
            format="%.0f",
            help="Bugün itibarıyla toplam birikimleriniz.",
        )
        monthly_saving = st.number_input(
            "Aylık Düzenli Tasarruf (TL)",
            min_value=0.0,
            step=1_000.0,
            format="%.0f",
            help="Her ay düzenli olarak biriktirdiğiniz miktar.",
        )
        investment_type = st.selectbox(
            "Birikiminizin Bulunduğu Yatırım Aracı",
            options=[i.value for i in InvestmentType],
            format_func=lambda x: INVESTMENT_LABELS.get(x, x.upper()),
        )

    with col2:
        st.subheader("Hedefiniz")
        target_type = st.selectbox(
            "Hedef Varlık Türü",
            options=[t.value for t in TargetType],
            format_func=lambda x: TARGET_LABELS.get(x, x),
        )
        target_price = st.number_input(
            "Hedef Varlığın Bugünkü Fiyatı (TL)",
            min_value=1.0,
            value=3_000_000.0,
            step=100_000.0,
            format="%.0f",
            help="Satın almak istediğiniz ev veya arabanın güncel piyasa değeri.",
        )

        st.markdown("---")
        st.markdown("**Simülasyon Ayarları**")
        scenario_key = st.selectbox(
            "Ekonomik Senaryo",
            options=list(PRESETS.keys()),
            index=0,
            format_func=lambda k: PRESETS[k].name,
            help=(
                "Farklı enflasyon/getiri varsayımları altında sonucu görün. "
                "Temel = tarihsel kaba kalibrasyon, İyimser/Kötümser = alternatif "
                "ekonomik koşullar."
            ),
        )
        n_paths = st.select_slider(
            "Senaryo Sayısı",
            options=[1_000, 2_000, 5_000, 10_000],
            value=5_000,
            help="Daha fazla senaryo = daha doğru tahmin, daha uzun süre.",
        )
        horizon_years = st.slider(
            "Simülasyon Ufku (Yıl)",
            min_value=5,
            max_value=30,
            value=20,
            step=5,
        )

    submitted = st.form_submit_button(
        "🚀 Simülasyonu Başlat",
        type="primary",
        use_container_width=True,
    )

# ── Simülasyon & Sonuçlar ─────────────────────────────────────────────
if submitted:
    if target_price <= 0:
        st.error("⚠️ Hedef varlık fiyatı sıfırdan büyük olmalıdır.")
        st.stop()

    with st.spinner("Monte Carlo simülasyonu çalışıyor…"):
        settings = get_settings()
        n_steps = horizon_years * 12
        scenario = get_preset(scenario_key)

        sim_output = run_scenario(
            scenario,
            initial_savings=initial_savings,
            initial_monthly_saving=monthly_saving,
            investment_type=investment_type,
            target_type=target_type,
            target_price=target_price,
            n_paths=n_paths,
            n_steps=n_steps,
            seed=settings.simulation_random_seed,
        )

        try:
            result = summarize_simulation(sim_output)
        except ValueError:
            st.error(
                "❌ Simülasyon ufku içinde hedefe ulaşılabilecek bir senaryo bulunamadı. "
                "Lütfen birikimlerinizi, aylık tasarrufunuzu artırın ya da "
                "simülasyon ufkunu genişletin."
            )
            st.stop()

    st.divider()
    st.header("📊 Simülasyon Sonuçları")
    st.caption(
        f"Senaryo: **{scenario.name}** · "
        f"Hedef: **{TARGET_LABELS[target_type]} — {format_tl(target_price)}** · "
        f"Yatırım: **{INVESTMENT_LABELS[investment_type]}** · "
        f"{n_paths:,} senaryo · {horizon_years} yıl ufuk"
    )

    # ── 3 Olasılık Dilimi Metrik Kartları ─────────────────────────────
    st.subheader(f"🎯 '{scenario.name}' Senaryosunda Ne Zaman Satın Alabilirsiniz?")
    c1, c2, c3 = st.columns(3)

    with c1:
        label = duration_label(
            result.optimistic_time.years, result.optimistic_time.months
        )
        st.metric(
            label="✅ En İyi %10 İhtimal (P10)",
            value=label,
            help="Seçili senaryo içindeki 10.000 yol arasında en hızlı %10'unda ulaşılan süre.",
        )

    with c2:
        label = duration_label(
            result.median_time.years, result.median_time.months
        )
        st.metric(
            label="📍 Medyan Sonuç (P50)",
            value=label,
            help="Seçili senaryo içindeki yolların yarısında ulaşılan süre.",
        )

    with c3:
        label = duration_label(
            result.pessimistic_time.years, result.pessimistic_time.months
        )
        st.metric(
            label="⚠️ En Kötü %10 İhtimal (P90)",
            value=label,
            help="Seçili senaryo içindeki yolların en kötü %10'unda bile ulaşılan süre.",
        )

    # ── Olasılık Çubuğu ──────────────────────────────────────────────
    st.divider()
    st.subheader("📅 Belirli Süre İçinde Satın Alma Olasılığı")

    prob_cols = st.columns(3)
    for col, (yil, prob) in zip(
        prob_cols,
        [
            (5, result.probability_5y),
            (10, result.probability_10y),
            (15, result.probability_15y),
        ],
    ):
        with col:
            pct = prob * 100
            color = "green" if pct >= 60 else "orange" if pct >= 30 else "red"
            st.metric(label=f"{yil} Yıl İçinde", value=f"%{pct:.1f}")
            st.progress(min(prob, 1.0))

    # ── Fan Chart (Quantile Bandı) ─────────────────────────────────────
    st.divider()
    st.subheader("📈 Olasılık Dağılımı (Ay Bazında)")

    affordable = sim_output.affordability_months[
        ~np.isnan(sim_output.affordability_months)
    ]

    if len(affordable) > 0:
        import pandas as pd

        # Ay bazında birikimli dağılım (CDF)
        sorted_months = np.sort(affordable)
        cdf = np.arange(1, len(sorted_months) + 1) / sim_output.n_paths

        chart_df = pd.DataFrame(
            {"Ay": sorted_months, "Olasılık (%)": cdf * 100}
        )
        st.line_chart(chart_df.set_index("Ay"), y="Olasılık (%)", use_container_width=True)
        st.caption(
            "Grafik: Her ayda veya öncesinde hedefe ulaşabilme olasılığı (birikimli). "
            "Hiç ulaşılamayan senaryolar dahil edilmemiştir."
        )
    else:
        st.info("Ulaşılabilen senaryo bulunmadığından grafik gösterilemiyor.")

    # ── Özet İstatistikler ────────────────────────────────────────────
    st.divider()
    st.subheader("📋 Özet İstatistikler")

    never_pct = result.never_affordable_pct * 100
    reached_pct = 100 - never_pct

    col_s1, col_s2, col_s3 = st.columns(3)
    col_s1.metric("Toplam Senaryo", f"{result.num_simulations:,}")
    col_s2.metric(
        "Hedefe Ulaşan",
        f"%{reached_pct:.1f}",
        help=f"Simülasyon ufku içinde hedefe ulaşabilen senaryo oranı.",
    )
    col_s3.metric(
        "Hedefe Ulaşamayan",
        f"%{never_pct:.1f}",
        help=f"{horizon_years} yıl içinde hiç hedefe ulaşamayan senaryo oranı.",
    )

    # ── Yasal Uyarı ───────────────────────────────────────────────────
    st.divider()
    st.caption(f"⚠️ {result.disclaimer}")
