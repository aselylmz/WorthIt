"""
Time to Afford — Streamlit Kullanıcı Arayüzü.
"""

import sys
from pathlib import Path

import streamlit as st

# Uygulama modüllerine erişebilmek için src klasörünü yola ekliyoruz
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

try:
    from time_to_afford.models.schemas import InvestmentType, TargetType
    investment_options = [i.value for i in InvestmentType]
    target_options = [t.value for t in TargetType]
except ImportError:
    # Geliştirme aşamasında modül import edilemezse kullanılacak yedek değerler
    investment_options = ["gold", "bist", "deposit"]
    target_options = ["house", "car"]

st.set_page_config(
    page_title="WorthIt | Time to Afford",
    page_icon="🏠",
    layout="centered"
)

st.title("🏠 WorthIt: Time to Afford")
st.markdown("**Belirli bir evi veya arabayı ne zaman satın alabileceğinizi simüle edin.**")

st.divider()

st.header("Finansal Bilgileriniz")
st.markdown("Lütfen mevcut durumunuzu ve hedefinizi girin:")

with st.form("user_input_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        profession = st.text_input(
            "Mesleğiniz",
            placeholder="Örn: Yazılım Mühendisi",
            help="Gelir artışınız bu meslek profiline göre tahmin edilecektir."
        )
        initial_savings = st.number_input(
            "Mevcut Birikim (TL)",
            min_value=0.0,
            step=10000.0,
            format="%.2f"
        )
        investment_type = st.selectbox(
            "Birikiminizin Bulunduğu Yatırım Aracı",
            options=investment_options,
            format_func=lambda x: x.upper()
        )
    
    with col2:
        target_type = st.selectbox(
            "Hedef Varlık Türü",
            options=target_options,
            format_func=lambda x: "🏠 Ev" if x == "house" else "🚗 Araba"
        )
        target_price = st.number_input(
            "Hedef Varlığın Fiyatı (TL)",
            min_value=0.0,
            value=1000000.0,
            step=50000.0,
            format="%.2f"
        )
        monthly_saving = st.number_input(
            "Aylık Düzenli Tasarruf (TL)",
            min_value=0.0,
            step=1000.0,
            format="%.2f"
        )

    submitted = st.form_submit_button("Simülasyonu Başlat 🚀", type="primary")

if submitted:
    if not profession.strip():
        st.error("Lütfen simülasyon için bir meslek giriniz.")
    else:
        st.success("Girdiler başarıyla alındı! (Monte Carlo simülasyonu entegre edildiğinde sonuçlar burada görünecektir.)")
        # Girdileri kontrol amaçlı ekrana basalım
        st.json({
            "profession": profession,
            "initial_savings": initial_savings,
            "investment_type": investment_type,
            "target_type": target_type,
            "target_price": target_price,
            "monthly_saving": monthly_saving
        })
