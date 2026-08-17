# İstatistiksel Metodoloji

## Genel Bakış

Time to Afford, kullanıcının mevcut finansal durumundan yola çıkarak belirli
bir evi veya arabayı gelecekte yaklaşık ne zaman satın alabileceğini tahmin
eden bir **multi-series forecasting + stochastic simulation** sistemidir.

## Matematiksel Tanım

Sistem, gelecekte kullanıcının finansal varlığının hedef varlığın fiyatını
ilk kez geçtiği zamanı hesaplar:

```
T = min{t : W_t >= P_t}
```

Burada:
- **W_t** = Kullanıcının t zamanındaki toplam serveti
- **P_t** = Hedef varlığın t zamanındaki fiyatı
- **T** = Tahmini satın alma zamanı

## Servet Modeli (W_t)

Kullanıcının serveti üç bileşenden oluşur:

1. **Mevcut birikim**: Başlangıç varlığı
2. **Kümülatif tasarruf**: Aylık tasarrufların birikimi
3. **Yatırım getirisi**: Birikimlerin yatırım aracına göre kazancı

```
W_t = (W_0 + Σ S_i) × Π (1 + r_i)
```

## Hedef Fiyat Modeli (P_t)

Hedef varlığın gelecekteki fiyatı, ilgili fiyat endeksinin
forecasting modeli tarafından belirlenir.

## Monte Carlo Simülasyonu

Sistem deterministik tek bir tahmin üretmez. Her değişken için
stokastik örnekleme yapılarak 10.000 senaryo üretilir.

Her senaryoda:
1. Gelecekteki maaş üretilir
2. Tasarruf hesaplanır
3. Yatırım getirisi üretilir
4. Servet güncellenir
5. Hedef fiyat güncellenir
6. W_t >= P_t kontrolü yapılır

## Senaryo Tanımları

- **İyimser**: Forecast distribution'ın P10 quantile'ı (kullanıcı için en kısa süre)
- **Temel**: Median (P50)
- **Kötümser**: P90 quantile'ı (kullanıcı için en uzun süre)

## Forecasting Yaklaşımı

Her zaman serisi ayrı modellenir. Önce basit modeller (Naive, ETS,
ARIMA) değerlendirilir; karmaşık modeller ancak anlamlı iyileşme
gösterirse kullanılır.

## Belirsizlik ve Limitasyonlar

Bu sistem bir tahmin aracıdır, kesin bir finansal tavsiye değildir.
Modelin başarısı geçmiş verilerin geleceği temsil etme gücüne bağlıdır.
Yapısal kırılmalar (krizler, politika değişiklikleri) modelin doğruluğunu
önemli ölçüde etkileyebilir.

---

*Bu doküman proje geliştikçe güncellenecektir.*
