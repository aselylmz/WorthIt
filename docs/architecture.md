# Mimari

## Genel Bakış

Time to Afford aşağıdaki pipeline üzerinden çalışır:

```
Data Pipeline
     ↓
Time-Series Forecasting
     ↓
Forecast Uncertainty
     ↓
Stochastic / Monte Carlo Simulation
     ↓
Financial Affordability Engine
     ↓
API
     ↓
User Interface
```

## Package Yapısı

```
src/time_to_afford/
├── config/         → Uygulama konfigürasyonu (pydantic-settings)
├── data/           → Veri yükleme, doğrulama, ön işleme
├── forecasting/    → Forecasting modelleri ve değerlendirme
├── simulation/     → Monte Carlo simülasyonu ve dağılımlar
├── affordability/  → Servet, hedef fiyat ve time-to-afford hesaplama
├── models/         → Pydantic şemaları (input/output)
└── utils/          → Logging, tarih yardımcıları
```

## Tasarım İlkeleri

1. **Modülerlik**: Her modül bağımsız olarak test edilebilir.
2. **Interface odaklı**: Forecasting modelleri ortak bir ABC üzerinden çalışır.
3. **Separation of concerns**: Business logic UI'dan ayrılmıştır.
4. **Type safety**: Pydantic ile input/output doğrulaması.
5. **Reproducibility**: Simulation seed kontrolü.
6. **Genişletilebilirlik**: Yeni modeller, meslekler ve yatırım araçları kolayca eklenebilir.

## Katmanlar

### 1. Data Layer (`data/`)
Ham verileri yükler, doğrular ve ön işleme yapar.

### 2. Forecasting Layer (`forecasting/`)
Her zaman serisini ayrı modeller ve gelecek dağılımı üretir.

### 3. Simulation Layer (`simulation/`)
Monte Carlo yöntemiyle stokastik senaryolar üretir.

### 4. Affordability Layer (`affordability/`)
Servet ile hedef fiyatı karşılaştırarak satın alma zamanını hesaplar.

### 5. API Layer (`app/`)
FastAPI backend + Streamlit frontend.

---

*Bu doküman proje geliştikçe güncellenecektir.*
