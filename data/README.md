# Data Dizini ve Veri Depolama Standartları

Bu dizin, **Time to Afford** projesinde kullanılan tüm ekonomik zaman serilerini, harici veri referanslarını ve işlenmiş simülasyon tablolarını barındırır.

## Dizin Yapısı

```
data/
├── raw/          # Ham veri dosyaları (TCMB EVDS API yanıtları, yfinance indirmeleri)
├── processed/    # Aylık standarda getirilmiş veri sözleşmesine uygun Parquet/CSV dosyaları
│                 # Örn: macro_monthly.parquet
└── external/     # TÜİK ISCO-08 meslek tanımları, parametrik katsayı referans tabloları
```

## Veri Prensipleri

1. **Değiştirilemez Ham Veri (Immutability):** `data/raw/` dizini altındaki ham dosyalar her zaman kaynak API veya siteden indirildiği orijinal haliyle korunur.
2. **Yeniden Üretilebilirlik (Reproducibility):** `data/processed/` altındaki dosyalar `time_to_afford.data` modülü scriptleri ile ham verilerden deterministik olarak yeniden üretilebilmelidir.
3. **Veri Sözleşmesine Bağlılık:** İşlenmiş tüm tablolardaki sütun isimleri, tipleri ve frekansları [docs/data_sources.md](../../docs/data_sources.md) içindeki **MVP Data Contract** tablosuna birebir uymak zorundadır.
4. **Git Kuralları:** Büyük boyutlu veri dosyaları `.gitignore` ile repo dışında tutulur.
