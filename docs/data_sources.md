# Veri Kaynakları ve Veri Sözleşmesi (Data Sources & Data Contract)

Bu doküman, **Time to Afford** projesinde zaman serisi tahminleme (forecasting) ve Monte Carlo simülasyonu için kullanılacak veri kaynaklarını, doğrulama durumlarını, sınırlılıklarını ve standart **MVP Data Contract** tanımını içerir.

---

## 1. Doğrulanmış Veri Kaynakları Özeti

| Seri Adı | Kaynak Kurum | Sistem / Seri Kodu | Frekans | Durum | Resmî Kaynak mı? |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **TÜFE (Enflasyon)** | TCMB / TÜİK | EVDS: `TP.FG.J0` (2003=100) / `TP.TUKFIY2025` | Aylık | ✅ Doğrulandı | Evet |
| **Konut Fiyat Endeksi (KFE)** | TCMB | EVDS: `TP.KFE.TR` | Aylık | ✅ Doğrulandı | Evet |
| **Mevduat Faizi (3 Ay Vadeli)** | TCMB | EVDS: `TP.TRY.MT02` | Haftalık $\to$ Aylık | ✅ Doğrulandı | Evet |
| **Döviz Kuru (USD/TRY)** | TCMB | EVDS: `TP.DK.USD.A.YTL` | Günlük $\to$ Aylık | ✅ Doğrulandı | Evet |
| **Politika Faizi (Fonlama Maliyeti)** | TCMB | EVDS: `TP.APIFON4` | Günlük $\to$ Aylık | ✅ Doğrulandı (Ağustos 2026) | Evet |
| **BIST 100 Endeksi** | Dış Sağlayıcı (Yahoo Fin.) | `XU100.IS` | Günlük $\to$ Aylık | ✅ Doğrulandı (Dış Sağlayıcı) | Hayır (3. Parti Proxy) |
| **Sentetik Gram Altın** | Sentetik Hesaplama | `synthetic_gram_gold_try` | Günlük $\to$ Aylık | ✅ Matematiksel Proxy | Kısmen (TCMB Kur + Ons) |
| **Taşıt Fiyat Göstergesi** | TÜİK | `vehicle_price_proxy` (Ulaştırma TÜFE) | Aylık | ⚠️ Proxy Olarak Doğrulandı | Evet (Ama Araç Fiyatı Değil) |
| **Parametrik Gelir Modeli** | Parametrik Büyüme Modeli | `salary_growth_rate` | Aylık Simülasyon | ⚠️ Varsayımsal Model | Hayır (Kalibrasyon Bekliyor) |

---

## 2. Detaylı Kaynak Analizleri ve Sınırlılıklar

### 2.1. Makroekonomik Seriler (TCMB EVDS)
* **TÜFE:** `TP.FG.J0` (2003=100 genel tüketici fiyat endeksi) kesintisiz, doğrulanmış resmi seridir. Güncel baz yılı serisi `TP.TUKFIY2025` olarak yayımlanmaktadır.
* **Konut Fiyat Endeksi (KFE):** `TP.KFE.TR` Türkiye geneli hedonik konut fiyat endeksidir. 2010 yılından itibaren aylık olarak mevcuttur. Yaklaşık 45-50 gün gecikmeyle yayımlanır.
* **Mevduat Faizi:** `TP.TRY.MT02` (3 aya kadar vadeli TL mevduat ağırlıklı ortalama faiz oranı). Haftalık akım veri olup aylık ortalaması alınarak kullanılır.
* **Döviz Kuru:** `TP.DK.USD.A.YTL` TCMB gösterge niteliğindeki döviz alış kurudur.
* **Politika Faizi:** `TP.APIFON4` (TCMB Ağırlıklı Ortalama Fonlama Maliyeti). EVDS'de tek bir "resmi ilan edilen politika faizi" serisi bulunmuyor; bu seri TCMB'nin fiilen uyguladığı ortalama fonlama maliyetini yansıtır ve yaygın olarak politika faizi proxy'si olarak kullanılır. (Daha önce bu depoda `TP.KT.IFJ01` olarak belgelenmişti; bu kod evds3 API'sinde HTTP 400 ile reddediliyor — kaldırılmış veya yanlış belgelenmiş olabilir. `TP.APIFON4`, Ocak-Mart 2024 için gerçek veriyle doğrulandı: %42.50 → %45.00, TCMB'nin 25 Ocak 2024 faiz kararıyla birebir örtüşüyor.)

### 2.2. Faiz $\to$ Aylık Getiri Dönüşümü ve Sınırlılıkları
* **Dönüşüm Formülü (Basitleştirilmiş Varsayım):**
  $$r_{deposit, t} = \left(1 + \frac{i_{annual, t}}{100}\right)^{1/12} - 1$$
* **Metodolojik Sınırlılıklar ve Uyarılar:**
  1. `TP.TRY.MT02` bankacılık sektöründeki **brüt ağırlıklı ortalama mevduat faizidir**; bireysel bir yatırımcının mevduat tutarına, pazarlık gücüne veya banka kampanyalarına göre fiilen elde edeceği getiriyle birebir aynı değildir.
  2. **Vergi ve Stopaj:** Mevduat faizi gelirlerinden kesilen gelir vergisi stopajı (vadelerine göre %5 - %10 vb.) ve masraflar MVP simülasyonunda modellenmemiştir. Modeldeki getiri brüt piyasa göstergesidir.

### 2.3. BIST 100 Endeksi Kaynağı ve Durumu
* **Resmî Durum:** Borsa İstanbul endeks verilerinin mülkiyeti Borsa İstanbul A.Ş.'ye aittir. Resmî ve anlık API erişimi BISTECH / lisanslı veri sağlayıcıları (Matriks, Foreks, Finnet vb.) üzerinden ücretli/kurumsal lisanslara tabidir.
* **MVP Yaklaşımı:** MVP kapsamında maliyetsiz ve açık analitik geliştirme için **Yahoo Finance (`yfinance` kütüphanesi, `XU100.IS` sembolü)** dış veri sağlayıcısı olarak kullanılacaktır.
* **Sınırlılık:** `yfinance` resmî bir borsa servisi değildir; üçüncü parti bir veri toplayıcıdır.

### 2.4. Altın: Sentetik Gram Altın (`synthetic_gram_gold_try`)
* **Hesaplama Formülü:** 
  $$\text{Gram Altın (TL)} = \frac{\text{Ons Altın (USD)} \times \text{USD/TRY}}{31.1034768}$$
* **Sınırlılık ve Uyarı:** Bu seri uluslararası ons spot fiyatı ile resmî kurun çarpımıyla üretilen teorik/sentetik bir fiyattır. Kapalıçarşı serbest piyasa fiziki primlerini, banka alış-satış makas aralıklarını ve Darphane basım farklarını **içermez**. MVP'de `synthetic_gram_gold_try` adı altında takip edilecektir.

### 2.5. Otomobil Fiyat Modeli ve `vehicle_price_proxy`
* **Problem:** Türkiye'de konut için TCMB'nin yayımladığı KFE benzeri resmî, merkezi ve herkese açık bir "Otomobil Fiyat Endeksi API'si" **bulunmamaktadır**.
  * ODMD (Otomotiv Distribütörleri Derneği) yalnızca adet bazlı satış raporları yayımlar.
  * Indicata ve Cardata gibi özel veri tabanları ticari lisansa tabidir.
  * BETAM *sahibindex* raporları akademik PDF formatındadır, doğrudan API sağlamaz.
* **MVP Yaklaşımı:** Kullanıcı başlangıç hedef araç fiyatını ($P_0$) kendisi girer. Gelecekteki fiyat artışı ($P_t$) için TÜİK Ulaştırma harcama grubu alt kalemi doğrudan otomobil fiyatı olarak adlandırılmayacak, **`vehicle_price_proxy`** olarak kompozit bir artış oranı referansı olarak kullanılacaktır.

### 2.6. Nominal Gelir Büyüme Modeli (Parametrik Gelir Varsayımı)
* **Model Tanımı:** Bu aşamada gerçek bir bireysel maaş tahmini yerine, aylık frekansta çalışan parametrik bir **nominal gelir büyüme modeli (nominal salary growth model)** kullanılır:
  $$g_{salary, t} = \pi_t + \alpha_{prof} + \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, \sigma_{salary}^2)$$
  * $\pi_t$: Aylık enflasyon oranı / beklentisi ($t$ ayı için).
  * $\alpha_{prof}$: İlgili meslek grubunun **aylık frekanstaki** reel büyüme/kıdem/verimlilik primi varsayımı.
  * $\epsilon_t$: Bireysel kariyer şoku ve piyasa dalgalanması (aylık stokastik hata terimi).
* **Metodolojik Sınırlılıklar:**
  1. **Kesikli Ücret Artışları:** Türkiye iş gücü piyasasında maaşlar her ay düzenli enflasyon kadar artmaz; genellikle 6 ayda veya yılda bir toplu olarak güncellenir (asgari ücret artışları, toplu iş sözleşmeleri, memur zam rejimleri). MVP'deki sürekli aylık büyüme formülasyonu stokastik bir modelleme kolaylığıdır.
  2. **Kalibrasyon İhtiyacı:** $\alpha_{prof}$ ve $\sigma_{salary}$ parametreleri başlangıçta TÜİK ISCO-08 ana meslek gruplarına dayalı parametrik varsayımlardır. İlerleyen fazlarda gerçek anket veya SGK idari kayıt verileriyle kalibre edilecektir.

---

## 3. MVP Veri Frekansı ve Dönüşüm Kuralları

MVP'nin temel hesaplama ve simülasyon frekansı **Aylık (Monthly)** olarak belirlenmiştir. Farklı frekanstaki ham seriler aşağıdaki kurallarla aylık seriye dönüştürülür:

1. **Fiyat ve Endeks Seviyeleri (TÜFE, KFE, BIST 100, Altın, Döviz):**
   * *Aylık Temsil:* Ayın son iş günü kapanış seviyesi (`ME` - Month End).
   * *Getiri:* Aylık log-return: $r_t = \ln(P_t / P_{t-1})$.
2. **Faiz Oranları (Mevduat, Politika Faizi):**
   * *Aylık Temsil:* İlgili ayın haftalık/günlük değerlerinin aritmetik ortalaması.
   * *Aylık Getiri Temsili:* Basitleştirilmiş bileşik faiz formülü: $r_{monthly} = (1 + i_{annual}/100)^{1/12} - 1$.

---

## 4. MVP Data Contract (Veri Sözleşmesi)

Aşağıdaki tablo, `data/processed/macro_monthly.parquet` veri kümesinde bulunacak her bir serinin resmi teknik sözleşmesidir:

| series_name | source | source_id | frequency | unit | start_date | update_frequency | is_official | transformation | limitations |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `cpi_index` | TCMB EVDS / TÜİK | `TP.FG.J0` | Monthly | Index (2003=100) | 2003-01-01 | Monthly | Yes | Level & Log-return | Baz yılı revizyonları kontrol edilmelidir |
| `house_price_index` | TCMB EVDS | `TP.KFE.TR` | Monthly | Index (2023=100) | 2010-01-01 | Monthly | Yes | Level & Log-return | ~45 gün gecikmeli; 2010 öncesi yok |
| `deposit_rate_3m` | TCMB EVDS | `TP.TRY.MT02` | Weekly $\to$ Monthly | % (Annual) | 2002-01-01 | Monthly (Avg) | Yes | Level & Simplified effective monthly rate | Brüt piyasa ortalamasıdır; stopaj ve bireysel makasları içermez |
| `usd_try` | TCMB EVDS | `TP.DK.USD.A.YTL` | Daily $\to$ Monthly | TRY / USD | 2000-01-01 | Monthly (Close) | Yes | Month-end close & Log-return | Gösterge kurdur; serbest piyasa makasını içermez |
| `policy_rate` | TCMB EVDS | `TP.APIFON4` | Daily $\to$ Monthly | % (Annual) | 2010-05-01 | Monthly (Avg) | Yes | Level | Ağırlıklı ortalama fonlama maliyetidir (tek bir resmi "politika faizi" serisi EVDS'de yok) |
| `bist100_close` | Yahoo Finance (External) | `XU100.IS` | Daily $\to$ Monthly | Index Pts | 2000-01-01 | Monthly (Close) | No (3rd Party) | Month-end close & Log-return | Analitik amaçlı üçüncü parti veridir |
| `synthetic_gram_gold_try` | Synthetic (`GC=F` $\times$ `USDTRY`) | Internal Formula | Daily $\to$ Monthly | TRY / Gram | 2005-01-01 | Monthly (Close) | Partial | Month-end close & Log-return | Fiziki piyasa prim ve makaslarını içermez |
| `vehicle_price_proxy` | TÜİK (TÜFE Ulaştırma) | TÜİK Harcama Grubu | Monthly | Index | 2003-01-01 | Monthly | Yes | Level & Log-return | Araç satış fiyatı değil, TÜFE alt bileşenidir |
| `salary_growth_rate` | Parametric Formula | $g_{salary} = \pi + \alpha + \epsilon$ | Monthly | % (Monthly) | Simulated | Per Simulation | No | Stochastic simulation | Parametrik varsayımdır; kesikli zam rejimlerini içermez |
