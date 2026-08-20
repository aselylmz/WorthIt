"""
Simülasyonda kullanılan olasılık dağılımları ve örnekleme fonksiyonları.

Her ekonomik değişken için:
  - Dağılım tipi (log-normal, normal, truncated-normal vb.)
  - Türkiye ekonomisine dayalı varsayılan parametreler
  - Tekrarlanabilir örnekleme fonksiyonu

Tüm getiriler aylık bazda ifade edilir.
Fiyat serileri aylık yüzde değişim olarak döndürülür.

Parametreler hakkında not
--------------------------
Parametreler, 2018-2024 dönemi Türkiye verisine (TÜFE, BIST-100,
altın/TL, mevduat faizi) dayalı kaba kalibrasyondur.
Gerçek verilerle yeniden kalibre edilmeleri tavsiye edilir.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.random import Generator

# =====================================================================
# 1. Dağılım Parametresi Veri Sınıfları
# =====================================================================


@dataclass(frozen=True)
class LogNormalParams:
    """Log-normal dağılım parametreleri.

    X ~ LogNormal(mu, sigma)
    E[X] = exp(mu + sigma**2 / 2)

    Parameters
    ----------
    mu : float
        Log-ölçeğinde ortalama.
    sigma : float
        Log-ölçeğinde standart sapma (>= 0).
    """

    mu: float
    sigma: float

    def __post_init__(self) -> None:
        if self.sigma < 0:
            raise ValueError(f"sigma >= 0 olmalıdır, alınan: {self.sigma}")


@dataclass(frozen=True)
class NormalParams:
    """Normal dağılım parametreleri.

    X ~ N(mu, sigma**2)

    Parameters
    ----------
    mu : float
        Ortalama.
    sigma : float
        Standart sapma (>= 0).
    """

    mu: float
    sigma: float

    def __post_init__(self) -> None:
        if self.sigma < 0:
            raise ValueError(f"sigma >= 0 olmalıdır, alınan: {self.sigma}")


# =====================================================================
# 2. Varsayılan Türkiye Parametreleri
# =====================================================================

# Aylık TÜFE değişimi (log-normal).
# 2018-2024 ortalaması ~%3.5/ay; sigma yüksek çünkü Türkiye enflasyonu oynak.
INFLATION_PARAMS = LogNormalParams(
    mu=0.030,    # aylık log-ort (~%3.0)
    sigma=0.015,
)

# BIST-100 aylık log-getiri (normal).
# 2018-2024 aylık medyan log-getiri yaklaşık %2.5, volatilite yüksek.
BIST_PARAMS = NormalParams(
    mu=0.025,    # aylık ort. log-getiri
    sigma=0.070, # aylık log-getiri std
)

# Altın (TL bazlı) aylık log-getiri (normal).
# Dolar altını + kur etkisiyle TL bazında yüksek getiri ve yüksek oynaklık.
GOLD_PARAMS = NormalParams(
    mu=0.030,    # aylık ort. log-getiri
    sigma=0.055, # aylık log-getiri std
)

# Mevduat aylık faiz oranı (normal, dar bant).
# Yıllık ~%40 faiz → aylık ~%2.8; dar volatilite çünkü düzenlenmiş faiz.
DEPOSIT_PARAMS = NormalParams(
    mu=0.028,    # aylık ort. getiri
    sigma=0.005, # dar bant
)

# Maaş artışı (aylık log-normal).
SALARY_GROWTH_PARAMS = LogNormalParams(
    mu=0.018,
    sigma=0.012,
)

# Konut fiyat artışı (aylık log-normal).
HOUSE_PRICE_PARAMS = LogNormalParams(
    mu=0.032,
    sigma=0.020,
)

# Araç fiyat artışı (aylık log-normal).
CAR_PRICE_PARAMS = LogNormalParams(
    mu=0.028,
    sigma=0.018,
)


# =====================================================================
# 3. Örnekleme Fonksiyonları
# =====================================================================


def sample_log_normal(
    params: LogNormalParams,
    n_steps: int,
    n_paths: int,
    rng: Generator,
) -> np.ndarray:
    """Log-normal dağılımdan örnekler.

    Parameters
    ----------
    params : LogNormalParams
        Dağılım parametreleri.
    n_steps : int
        Zaman adımı sayısı (ay).
    n_paths : int
        Simülasyon path sayısı.
    rng : numpy.random.Generator
        Rastgele sayı üreteci.

    Returns
    -------
    np.ndarray
        (n_steps, n_paths) boyutunda log-normal örnekler.
        Her değer pozitiftir (ör: 1.035 -> %3.5 artış faktörü).
    """
    z = rng.standard_normal(size=(n_steps, n_paths))
    return np.exp(params.mu + params.sigma * z)


def sample_normal(
    params: NormalParams,
    n_steps: int,
    n_paths: int,
    rng: Generator,
) -> np.ndarray:
    """Normal dağılımdan örnekler (log-getiri için).

    Parameters
    ----------
    params : NormalParams
        Dağılım parametreleri.
    n_steps : int
        Zaman adımı sayısı (ay).
    n_paths : int
        Simülasyon path sayısı.
    rng : numpy.random.Generator
        Rastgele sayı üreteci.

    Returns
    -------
    np.ndarray
        (n_steps, n_paths) boyutunda normal örnekler.
        Log-getiriden fiyat çarpanına dönüştürmek için np.exp kullanın.
    """
    return rng.normal(loc=params.mu, scale=params.sigma, size=(n_steps, n_paths))


def sample_investment_return(
    investment_type: str,
    n_steps: int,
    n_paths: int,
    rng: Generator,
) -> np.ndarray:
    """Yatırım aracına göre aylık brüt getiri çarpanı örnekler.

    Her hücre, o ay için (1 + net_return) çarpanıdır.
    Örneğin 1.025 -> o ay %2.5 getiri.

    Parameters
    ----------
    investment_type : str
        'gold', 'bist' veya 'deposit'.
    n_steps : int
        Zaman adımı sayısı (ay).
    n_paths : int
        Simülasyon path sayısı.
    rng : numpy.random.Generator
        Rastgele sayı üreteci.

    Returns
    -------
    np.ndarray
        (n_steps, n_paths) boyutunda brüt getiri çarpanları (>0).

    Raises
    ------
    ValueError
        Bilinmeyen yatırım türü girilirse.
    """
    inv = investment_type.lower().strip()

    if inv == "gold":
        log_returns = sample_normal(GOLD_PARAMS, n_steps, n_paths, rng)
        return np.exp(log_returns)

    elif inv == "bist":
        log_returns = sample_normal(BIST_PARAMS, n_steps, n_paths, rng)
        return np.exp(log_returns)

    elif inv == "deposit":
        # Mevduat log-getiri değil doğrusal getiri; negatif olamaz
        raw = sample_normal(DEPOSIT_PARAMS, n_steps, n_paths, rng)
        return 1.0 + np.clip(raw, a_min=-0.005, a_max=None)

    else:
        raise ValueError(
            f"Bilinmeyen yatırım türü: '{investment_type}'. "
            "Desteklenenler: 'gold', 'bist', 'deposit'."
        )


def sample_inflation(
    n_steps: int,
    n_paths: int,
    rng: Generator,
    params: Optional[LogNormalParams] = None,
) -> np.ndarray:
    """Aylık TÜFE değişim çarpanı örnekler.

    Parameters
    ----------
    n_steps : int
        Zaman adımı sayısı (ay).
    n_paths : int
        Simülasyon path sayısı.
    rng : numpy.random.Generator
        Rastgele sayı üreteci.
    params : LogNormalParams, optional
        Özel parametreler. None ise varsayılan INFLATION_PARAMS kullanılır.

    Returns
    -------
    np.ndarray
        (n_steps, n_paths) boyutunda aylık TÜFE çarpanları (>1 beklenir).
        Örneğin 1.035 -> o ay %3.5 enflasyon.
    """
    p = params if params is not None else INFLATION_PARAMS
    return sample_log_normal(p, n_steps, n_paths, rng)


def sample_salary_growth(
    n_steps: int,
    n_paths: int,
    rng: Generator,
    params: Optional[LogNormalParams] = None,
) -> np.ndarray:
    """Aylık maaş artış çarpanı örnekler.

    Parameters
    ----------
    n_steps : int
        Zaman adımı sayısı (ay).
    n_paths : int
        Simülasyon path sayısı.
    rng : numpy.random.Generator
        Rastgele sayı üreteci.
    params : LogNormalParams, optional
        Özel parametreler. None ise varsayılan SALARY_GROWTH_PARAMS kullanılır.

    Returns
    -------
    np.ndarray
        (n_steps, n_paths) boyutunda aylık maaş çarpanları (>0).
    """
    p = params if params is not None else SALARY_GROWTH_PARAMS
    return sample_log_normal(p, n_steps, n_paths, rng)


def sample_target_price_growth(
    target_type: str,
    n_steps: int,
    n_paths: int,
    rng: Generator,
) -> np.ndarray:
    """Hedef varlık fiyat artış çarpanı örnekler.

    Parameters
    ----------
    target_type : str
        'house' veya 'car'.
    n_steps : int
        Zaman adımı sayısı (ay).
    n_paths : int
        Simülasyon path sayısı.
    rng : numpy.random.Generator
        Rastgele sayı üreteci.

    Returns
    -------
    np.ndarray
        (n_steps, n_paths) boyutunda aylık fiyat çarpanları (>0).

    Raises
    ------
    ValueError
        Bilinmeyen hedef türü girilirse.
    """
    tt = target_type.lower().strip()

    if tt == "house":
        return sample_log_normal(HOUSE_PRICE_PARAMS, n_steps, n_paths, rng)
    elif tt == "car":
        return sample_log_normal(CAR_PRICE_PARAMS, n_steps, n_paths, rng)
    else:
        raise ValueError(
            f"Bilinmeyen hedef türü: '{target_type}'. "
            "Desteklenenler: 'house', 'car'."
        )


# =====================================================================
# 4. Korelasyonlu (Ortak Faktör) Örnekleme
# =====================================================================

# Değişkenlerin ortak enflasyon şokuyla (F) korelasyon katsayıları (rho).
#
# Tek-faktör modeli:
#     X_i = rho_i * F + sqrt(1 - rho_i**2) * epsilon_i
#     epsilon_i ~ iid N(0, 1), değişkenler arası ve F'den bağımsız
#
# Bu kurulum Corr(X_i, X_j) = rho_i * rho_j garantisiyle her zaman
# pozitif yarı tanımlı (geçerli) bir korelasyon yapısı üretir; ayrıca
# Cholesky ayrıştırması gerektirmez.
#
# rho değerleri Türkiye ekonomisine dair genel gözlemlere dayanan KABA
# varsayımlardır (ör: maaş zamları ve konut/araç fiyatları enflasyonu
# büyük ölçüde takip eder; BIST enflasyonla daha zayıf/gürültülü
# ilişkilidir). Gerçek veri kalibrasyonu tamamlanınca yeniden tahmin
# edilmelidir.
DEFAULT_INFLATION_RHOS: dict[str, float] = {
    "salary_growth": 0.55,
    "house_price": 0.50,
    "car_price": 0.55,
    "gold": 0.35,
    "bist": 0.15,
    "deposit": 0.60,
}


def sample_common_factor_shocks(
    variable_rhos: dict[str, float],
    n_steps: int,
    n_paths: int,
    rng: Generator,
) -> dict[str, np.ndarray]:
    """Tek ortak faktörlü (enflasyon) modelle korelasyonlu standart normal şoklar üretir.

    Her değişken için X_i = rho_i * F + sqrt(1 - rho_i**2) * epsilon_i
    hesaplanır; F ortak enflasyon faktörüdür ve ayrıca 'inflation'
    anahtarıyla döndürülür.

    Parameters
    ----------
    variable_rhos : dict of str -> float
        Değişken adı -> ortak faktörle korelasyonu ([-1, 1] aralığında).
    n_steps : int
        Zaman adımı sayısı (ay).
    n_paths : int
        Simülasyon path sayısı.
    rng : numpy.random.Generator
        Rastgele sayı üreteci.

    Returns
    -------
    dict of str -> np.ndarray
        'inflation' (ortak faktör) ve `variable_rhos` içindeki her
        değişken için (n_steps, n_paths) boyutunda standart normal
        (N(0,1)) şoklar.

    Raises
    ------
    ValueError
        Herhangi bir rho [-1, 1] aralığı dışındaysa.
    """
    for name, rho in variable_rhos.items():
        if not (-1.0 <= rho <= 1.0):
            raise ValueError(f"'{name}' için rho [-1, 1] aralığında olmalıdır, alınan: {rho}")

    common_factor = rng.standard_normal(size=(n_steps, n_paths))
    shocks: dict[str, np.ndarray] = {"inflation": common_factor}

    for name, rho in variable_rhos.items():
        idiosyncratic = rng.standard_normal(size=(n_steps, n_paths))
        shocks[name] = rho * common_factor + np.sqrt(1.0 - rho**2) * idiosyncratic

    return shocks


def sample_correlated_variables(
    investment_type: str,
    target_type: str,
    n_steps: int,
    n_paths: int,
    rng: Generator,
    inflation_rhos: Optional[dict[str, float]] = None,
) -> dict[str, np.ndarray]:
    """Ortak enflasyon faktörüyle ilişkilendirilmiş simülasyon değişkenlerini örnekler.

    Bağımsız örnekleme yapan `sample_investment_return` / `sample_salary_growth`
    / `sample_target_price_growth` fonksiyonlarının aksine, bu fonksiyon tüm
    değişkenleri ortak bir enflasyon şoku üzerinden ilişkilendirir. Böylece
    örneğin yüksek enflasyon senaryolarında maaş artışının ve konut/araç
    fiyat artışının da (gerçekçi biçimde) birlikte yüksek çıkması sağlanır.

    Her değişkenin marjinal dağılımı (mu, sigma) bağımsız örnekleme ile
    birebir aynı kalır; değişen yalnızca değişkenler arasındaki korelasyondur.

    Parameters
    ----------
    investment_type : str
        'gold', 'bist' veya 'deposit'.
    target_type : str
        'house' veya 'car'.
    n_steps : int
        Zaman adımı sayısı (ay).
    n_paths : int
        Simülasyon path sayısı.
    rng : numpy.random.Generator
        Rastgele sayı üreteci.
    inflation_rhos : dict of str -> float, optional
        Her değişkenin ortak enflasyon faktörüyle korelasyonu.
        None ise DEFAULT_INFLATION_RHOS kullanılır.

    Returns
    -------
    dict of str -> np.ndarray
        'inflation', 'salary_growth', 'investment_return',
        'target_price_growth' anahtarlarıyla (n_steps, n_paths)
        boyutunda aylık çarpan array'leri.

    Raises
    ------
    ValueError
        Bilinmeyen investment_type veya target_type girilirse.
    """
    rhos = inflation_rhos if inflation_rhos is not None else DEFAULT_INFLATION_RHOS

    inv = investment_type.lower().strip()
    if inv not in ("gold", "bist", "deposit"):
        raise ValueError(
            f"Bilinmeyen yatırım türü: '{investment_type}'. "
            "Desteklenenler: 'gold', 'bist', 'deposit'."
        )

    tt = target_type.lower().strip()
    if tt not in ("house", "car"):
        raise ValueError(
            f"Bilinmeyen hedef türü: '{target_type}'. "
            "Desteklenenler: 'house', 'car'."
        )

    shocks = sample_common_factor_shocks(rhos, n_steps, n_paths, rng)

    inflation_factors = np.exp(INFLATION_PARAMS.mu + INFLATION_PARAMS.sigma * shocks["inflation"])
    salary_factors = np.exp(
        SALARY_GROWTH_PARAMS.mu + SALARY_GROWTH_PARAMS.sigma * shocks["salary_growth"]
    )

    price_params = HOUSE_PRICE_PARAMS if tt == "house" else CAR_PRICE_PARAMS
    price_shock = shocks["house_price"] if tt == "house" else shocks["car_price"]
    price_growth_factors = np.exp(price_params.mu + price_params.sigma * price_shock)

    if inv == "gold":
        investment_factors = np.exp(GOLD_PARAMS.mu + GOLD_PARAMS.sigma * shocks["gold"])
    elif inv == "bist":
        investment_factors = np.exp(BIST_PARAMS.mu + BIST_PARAMS.sigma * shocks["bist"])
    else:  # deposit
        raw = DEPOSIT_PARAMS.mu + DEPOSIT_PARAMS.sigma * shocks["deposit"]
        investment_factors = 1.0 + np.clip(raw, a_min=-0.005, a_max=None)

    return {
        "inflation": inflation_factors,
        "salary_growth": salary_factors,
        "investment_return": investment_factors,
        "target_price_growth": price_growth_factors,
    }


# =====================================================================
# 5. Yardımcı: Kümülatif Çarpan -> Zaman Serisi
# =====================================================================


def cumulative_product(monthly_factors: np.ndarray) -> np.ndarray:
    """Aylık çarpan matrisinden kümülatif fiyat endeksi üretir.

    Parameters
    ----------
    monthly_factors : np.ndarray
        (n_steps, n_paths) boyutunda aylık çarpanlar.
        İlk satır t=1 ayına karşılık gelir.

    Returns
    -------
    np.ndarray
        (n_steps, n_paths) boyutunda kümülatif çarpan.
        Her hücre başlangıç değerine göre toplam değişimi temsil eder.

    Examples
    --------
    >>> import numpy as np
    >>> factors = np.array([[1.03, 1.02], [1.02, 1.03]])
    >>> cumulative_product(factors)
    array([[1.03  , 1.02  ],
           [1.0506, 1.0506]])
    """
    return np.cumprod(monthly_factors, axis=0)

