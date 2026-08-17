# 🏠 Time to Afford

**Belirli bir evi veya arabayı ne zaman satın alabileceğinizi Monte Carlo simülasyonu ile tahmin edin.**

[![Tests](https://github.com/WorthIt/WorthIt/actions/workflows/tests.yml/badge.svg)](https://github.com/WorthIt/WorthIt/actions)

---

## Proje Hakkında

Time to Afford, kullanıcının mevcut finansal durumundan yola çıkarak belirli
bir evi veya arabayı gelecekte yaklaşık ne zaman satın alabileceğini tahmin
eden kişiselleştirilmiş bir **multi-series forecasting + stochastic simulation**
uygulamasıdır.

Kullanıcı minimum bilgi girer:

- Meslek
- Mevcut birikim miktarı
- Birikimin bulunduğu yatırım aracı (altın, BIST, mevduat)
- Hedef türü (ev veya araba)
- Hedef varlığın mevcut fiyatı
- Aylık düzenli tasarruf miktarı

Sistem, farklı ekonomik zaman serilerini ayrı ayrı modelleyerek 10.000 Monte
Carlo senaryosu üretir ve her senaryoda kullanıcının servetinin hedef fiyatı
ilk kez geçtiği zamanı hesaplar.

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

| Değişken | Açıklama |
|----------|----------|
| Gelir / Maaş | Meslek bazlı gelir ve artış oranları |
| Enflasyon | TÜFE bazlı tüketici fiyat endeksi |
| Konut fiyatları | Konut fiyat endeksi |
| Otomobil fiyatları | Araç fiyat serisi |
| Altın getirisi | Altın fiyat ve getiri dağılımı |
| BIST getirisi | Borsa endeks getirisi |
| Mevduat getirisi | Faiz bazlı getiri |

### Monte Carlo Simülasyonu

Her simülasyon path'inde:
1. Gelecekteki maaş üretilir
2. Tasarruf hesaplanır
3. Yatırım getirisi üretilir
4. Kullanıcının serveti güncellenir
5. Hedef varlığın fiyatı güncellenir
6. `W_t >= P_t` kontrolü yapılır

### Sonuç Çıktıları

- **Temel senaryo** (median)
- **İyimser senaryo** (P10)
- **Kötümser senaryo** (P90)
- Belirli yıllar içinde satın alma olasılığı
- Servet vs. hedef fiyat grafiği
- Sonucu en fazla etkileyen faktörler

## Mimari

```
Data Pipeline → Forecasting → Monte Carlo Simulation → Affordability Engine → API → UI
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
├── tests/                # Testler
├── docs/                 # Dokümantasyon
└── .github/workflows/    # CI/CD
```

## Kullanım

> **Not:** Proje geliştirme aşamasındadır. Kullanım talimatları ilerleyen
> fazlarda güncellenecektir.

## Limitasyonlar

- Bu sistem bir **tahmin aracıdır**, kesin bir finansal planlama aracı değildir.
- Model geçmiş verilere dayanır; yapısal kırılmalar (ekonomik krizler, politika
  değişiklikleri) tahmin doğruluğunu önemli ölçüde etkileyebilir.
- MVP'de sınırlı sayıda meslek ve yatırım aracı desteklenmektedir.
- Vergiler, kredi ve borçlanma gibi faktörler ilk sürümde dahil değildir.

## ⚠️ Disclaimer

**Bu uygulama yalnızca eğitim ve bilgilendirme amaçlıdır. Sunulan sonuçlar
istatistiksel bir tahmindir ve kesin bir finansal tavsiye veya garanti
niteliği taşımaz. Yatırım kararlarınızı bu araca dayanarak vermeyiniz.
Gerçek sonuçlar ekonomik koşullara bağlı olarak önemli ölçüde farklılık
gösterebilir.**
