# 🏠 WorthIt

**Belirli bir evi veya arabayı ne zaman satın alabileceğinizi Monte Carlo simülasyonu ile tahmin edin.**

[![Tests](https://github.com/WorthIt/WorthIt/actions/workflows/tests.yml/badge.svg)](https://github.com/WorthIt/WorthIt/actions)

---

## Proje Hakkında

WorthIt, kullanıcının mevcut finansal durumundan yola çıkarak belirli
bir evi veya arabayı gelecekte yaklaşık ne zaman satın alabileceğini tahmin
eden kişiselleştirilmiş bir **multi-series forecasting + stochastic simulation**
uygulamasıdır.

Kullanıcı minimum bilgi girer:

- Mevcut birikim miktarı
- Aylık düzenli tasarruf miktarı
- Birikimin bulunduğu yatırım aracı (altın, BIST, mevduat)
- Hedef türü (ev veya araba) ve mevcut fiyatı
- Ekonomik senaryo (Temel / İyimser / Kötümser)

Sistem, enflasyon ile ilişkilendirilmiş (korelasyonlu) ekonomik değişkenler
üzerinden binlerce Monte Carlo senaryosu üretir ve her senaryoda kullanıcının
servetinin hedef fiyatı ilk kez geçtiği zamanı hesaplar.

## Motivasyon

Finansal hedef planlaması genellikle deterministik hesaplamalara dayanır:
_"Ayda X lira biriktirirsem, Y yılda hedefe ulaşırım."_ Ancak gerçek dünyada
enflasyon, yatırım getirileri, maaş artışları ve varlık fiyatları stokastik
süreçlerdir. Bu proje, bu belirsizliği sistematik olarak modeller ve
kullanıcıya olasılıklı bir tahmin sunar.

## Metodoloji

### Matematiksel Tanım

```
T = min{t : W_t >= P_t}
```

- **W_t**: Kullanıcının t zamanındaki toplam serveti
- **P_t**: Hedef varlığın t zamanındaki fiyatı
- **T**: Tahmini satın alma zamanı

### Modellenen Değişkenler

| Değişken | Açıklama | Durum |
|----------|----------|-------|
| Enflasyon | TÜFE bazlı tüketici fiyat endeksi | ✅ Gerçek veriyle kalibre |
| Konut fiyatları | Konut fiyat endeksi (KFE) | ✅ Gerçek veriyle kalibre |
| Altın getirisi | Sentetik gram altın (TL) getiri dağılımı | ✅ Gerçek veriyle kalibre |
| BIST getirisi | Borsa endeks getirisi | ✅ Gerçek veriyle kalibre |
| Mevduat getirisi | Faiz bazlı getiri | ✅ Gerçek veriyle kalibre |
| Otomobil fiyatları | Araç fiyat serisi | ⚠️ Kaba varsayım (güvenilir resmi seri henüz bulunamadı) |
| Gelir / Maaş | Meslek bazlı gelir ve artış oranları | ⚠️ Kaba varsayım (gerçek bireysel maaş verisi mevcut değil) |

Tüm değişkenler ayrıca ortak bir enflasyon faktörü üzerinden birbiriyle
ilişkilendirilir (bkz. [Metodoloji](docs/methodology.md)) — bağımsız
örnekleme yapılmaz.

### Monte Carlo Simülasyonu

Her simülasyon path'inde:
1. Gelecekteki maaş üretilir
2. Tasarruf hesaplanır
3. Yatırım getirisi üretilir
4. Kullanıcının serveti güncellenir
5. Hedef varlığın fiyatı güncellenir
6. `W_t >= P_t` kontrolü yapılır

### Sonuç Çıktıları

- **Medyan sonuç** (P50), **en iyi %10 ihtimal** (P10), **en kötü %10 ihtimal** (P90)
- 5/10/15 yıl içinde satın alma olasılığı
- Ay bazında birikimli olasılık dağılımı grafiği
- Hedefe ulaşan/ulaşamayan senaryo oranı

## Mimari

```
Data Pipeline → Simulation (Monte Carlo) → Affordability Engine → UI (Streamlit)
```

Detaylar için: [docs/architecture.md](docs/architecture.md)

## Kurulum

### Gereksinimler

- Python >= 3.10
- pip

### Adımlar

```bash
# Repository'yi klonla
git clone https://github.com/<user>/WorthIt.git
cd WorthIt

# Virtual environment oluştur
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Paketi editable modda kur
pip install -e ".[dev]"

# Testleri çalıştır
pytest tests/ -v
```

### Environment Variables

`.env.example` dosyasını `.env` olarak kopyalayıp doldurun:

```bash
cp .env.example .env
```

Gerçek veriyle çalışmak için `EVDS_API_KEY` gereklidir — TCMB EVDS'ten
ücretsiz alınabilir: https://evds2.tcmb.gov.tr/index.php?/evds/login

## Proje Yapısı

```
WorthIt/
├── README.md
├── .gitignore
├── .env.example
├── pyproject.toml
├── requirements.txt
│
├── data/
│   ├── raw/              # Ham veri dosyaları
│   ├── processed/        # İşlenmiş veriler
│   ├── external/         # Dış kaynak verileri
│   └── README.md
│
├── notebooks/            # EDA ve deneysel analizler
│
├── src/
│   └── time_to_afford/
│       ├── config/       # Uygulama konfigürasyonu
│       ├── data/         # Veri yükleme, doğrulama, ön işleme
│       ├── forecasting/  # Forecasting modelleri
│       ├── simulation/   # Monte Carlo simülasyonu
│       ├── affordability/# Servet ve time-to-afford hesaplama
│       ├── models/       # Pydantic şemaları
│       └── utils/        # Yardımcı araçlar
│
├── app/                  # Streamlit UI
├── scripts/              # Veri çekme ve kalibrasyon script'leri
├── tests/                # Testler
├── docs/                 # Dokümantasyon
└── .github/workflows/    # CI/CD
```

## Kullanım

### 1. Gerçek veriyi çek

```bash
python scripts/fetch_data.py
```

TCMB EVDS ve Yahoo Finance'ten TÜFE, KFE, USD/TRY, mevduat/politika
faizi, BIST-100 ve altın verilerini çeker; `data/raw/`e değiştirilemez
(immutable) anlık görüntüler kaydeder ve `data/processed/macro_monthly.parquet`
dosyasını üretir. Belirli bir tarih aralığı için:

```bash
python scripts/fetch_data.py --start 2015-01-01 --end 2026-08-01
```

### 2. Simülasyon parametrelerini kalibre et (opsiyonel)

Yeni veri çektikten sonra dağılım parametrelerini (mu, sigma) ve
enflasyon korelasyonlarını yeniden hesaplamak için:

```bash
python scripts/calibrate_distributions.py
```

Çıktıyı gözden geçirip `src/time_to_afford/simulation/distributions.py`
içindeki sabitleri elle güncelleyin (script kaynak kodu otomatik
değiştirmez).

### 3. Uygulamayı başlat

```bash
streamlit run app/streamlit_app.py
```

Formu doldurup ("Mevcut Birikim", "Aylık Tasarruf", yatırım aracı, hedef
varlık ve ekonomik senaryo) simülasyonu başlatın; P10/P50/P90 tahminleri,
yıllara göre satın alma olasılığı ve olasılık dağılımı grafiği anında
görüntülenir.

> **Not:** `data/processed/macro_monthly.parquet` olmadan da uygulama
> çalışır — bu durumda `distributions.py`'deki mevcut (önceden kalibre
> edilmiş) parametreler kullanılır.

## Limitasyonlar

- Bu sistem bir **tahmin aracıdır**, kesin bir finansal planlama aracı değildir.
- Model geçmiş verilere dayanır; yapısal kırılmalar (ekonomik krizler, politika
  değişiklikleri) tahmin doğruluğunu önemli ölçüde etkileyebilir.
- Otomobil fiyat artışı ve maaş artışı hâlâ kaba varsayımlara dayanır —
  gerçek veriyle kalibre edilmemiştir (bkz. [docs/data_sources.md](docs/data_sources.md) § 2.5, § 2.6).
- MVP'de sınırlı sayıda yatırım aracı (altın, BIST, mevduat) desteklenmektedir.
- Vergiler, kredi ve borçlanma gibi faktörler ilk sürümde dahil değildir.

## ⚠️ Disclaimer

**Bu uygulama yalnızca eğitim ve bilgilendirme amaçlıdır. Sunulan sonuçlar
istatistiksel bir tahmindir ve kesin bir finansal tavsiye veya garanti
niteliği taşımaz. Yatırım kararlarınızı bu araca dayanarak vermeyiniz.
Gerçek sonuçlar ekonomik koşullara bağlı olarak önemli ölçüde farklılık
gösterebilir.**
