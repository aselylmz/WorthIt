# Data Dizini

Bu dizin projenin veri dosyalarını barındırır.

## Yapı

```
data/
├── raw/          # Ham, işlenmemiş veri dosyaları
├── processed/    # İşlenmiş, modellemeye hazır veri dosyaları
└── external/     # Dış kaynaklardan alınan referans verileri
```

## Kurallar

- `raw/` içindeki dosyalar **hiçbir zaman** değiştirilmez. Orijinal haliyle korunur.
- `processed/` içindeki dosyalar data pipeline tarafından üretilir ve yeniden oluşturulabilir olmalıdır.
- `external/` dış API'ler veya üçüncü parti kaynaklardan indirilen verileri içerir.
- Büyük veri dosyaları `.gitignore` ile repo dışında tutulur.
- Her veri dosyasının kaynağı ve formatı `docs/data_sources.md` içinde dokümante edilmelidir.
