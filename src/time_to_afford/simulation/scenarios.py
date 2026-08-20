"""
Senaryo tanımları.

Her senaryo, ekonomik değişkenlere ait dağılım parametrelerinin
belirli bir kombinasyonunu temsil eder:

  - BASELINE   : Varsayılan Türkiye parametreleri (2018-2024 kalibrasyonu)
  - OPTIMISTIC : Daha yüksek maaş artışı, daha yüksek yatırım getirisi,
                 daha düşük enflasyon ve varlık fiyat artışı
  - PESSIMISTIC: Daha düşük maaş artışı, daha düşük yatırım getirisi,
                 daha yüksek enflasyon ve varlık fiyat artışı

Kullanım:
    from time_to_afford.simulation.scenarios import OPTIMISTIC, run_scenario
    result = run_scenario(OPTIMISTIC, initial_savings=..., ...)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from time_to_afford.simulation.distributions import (
    BIST_PARAMS,
    CAR_PRICE_PARAMS,
    DEPOSIT_PARAMS,
    GOLD_PARAMS,
    HOUSE_PRICE_PARAMS,
    INFLATION_PARAMS,
    SALARY_GROWTH_PARAMS,
    LogNormalParams,
    NormalParams,
)
from time_to_afford.simulation.monte_carlo import SimulationOutput

# =====================================================================
# 1. Senaryo Parametresi Veri Sınıfı
# =====================================================================


@dataclass(frozen=True)
class ScenarioParams:
    """Bir ekonomik senaryoya ait tüm dağılım parametreleri.

    Attributes
    ----------
    name : str
        Senaryo adı (ör: 'Temel', 'İyimser', 'Kötümser').
    salary_growth : LogNormalParams
        Aylık maaş artış çarpanı parametreleri.
    inflation : LogNormalParams
        Aylık TÜFE değişim çarpanı parametreleri.
    gold : NormalParams
        Altın aylık log-getiri parametreleri.
    bist : NormalParams
        BIST-100 aylık log-getiri parametreleri.
    deposit : NormalParams
        Mevduat aylık getiri parametreleri.
    house_price : LogNormalParams
        Konut fiyat artış çarpanı parametreleri.
    car_price : LogNormalParams
        Araç fiyat artış çarpanı parametreleri.
    """

    name: str
    salary_growth: LogNormalParams
    inflation: LogNormalParams
    gold: NormalParams
    bist: NormalParams
    deposit: NormalParams
    house_price: LogNormalParams
    car_price: LogNormalParams

    def __str__(self) -> str:
        return f"ScenarioParams(name={self.name!r})"


# =====================================================================
# 2. Hazır Senaryo Presetleri
# =====================================================================

#: Temel senaryo — 2018-2024 Türkiye kalibrasyonu (değiştirilmemiş varsayılanlar)
BASELINE = ScenarioParams(
    name="Temel",
    salary_growth=SALARY_GROWTH_PARAMS,
    inflation=INFLATION_PARAMS,
    gold=GOLD_PARAMS,
    bist=BIST_PARAMS,
    deposit=DEPOSIT_PARAMS,
    house_price=HOUSE_PRICE_PARAMS,
    car_price=CAR_PRICE_PARAMS,
)

#: İyimser senaryo — yüksek maaş artışı, yüksek yatırım getirisi,
#  düşük enflasyon ve varlık fiyat artışı
OPTIMISTIC = ScenarioParams(
    name="İyimser",
    salary_growth=LogNormalParams(mu=0.028, sigma=0.010),   # %~2.8/ay, düşük oynaklık
    inflation=LogNormalParams(mu=0.015, sigma=0.008),        # %~1.5/ay enflasyon
    gold=NormalParams(mu=0.035, sigma=0.045),                # daha yüksek getiri
    bist=NormalParams(mu=0.035, sigma=0.060),                # daha yüksek getiri
    deposit=NormalParams(mu=0.032, sigma=0.004),             # yüksek mevduat faizi
    house_price=LogNormalParams(mu=0.018, sigma=0.012),      # yavaş konut artışı
    car_price=LogNormalParams(mu=0.015, sigma=0.010),        # yavaş araç artışı
)

#: Kötümser senaryo — düşük maaş artışı, düşük yatırım getirisi,
#  yüksek enflasyon ve varlık fiyat artışı
PESSIMISTIC = ScenarioParams(
    name="Kötümser",
    salary_growth=LogNormalParams(mu=0.010, sigma=0.015),   # %~1.0/ay, yüksek oynaklık
    inflation=LogNormalParams(mu=0.045, sigma=0.020),        # %~4.5/ay enflasyon
    gold=NormalParams(mu=0.020, sigma=0.070),                # düşük getiri, yüksek oynaklık
    bist=NormalParams(mu=0.010, sigma=0.090),                # düşük getiri, yüksek oynaklık
    deposit=NormalParams(mu=0.020, sigma=0.006),             # düşük mevduat faizi
    house_price=LogNormalParams(mu=0.045, sigma=0.025),      # hızlı konut artışı
    car_price=LogNormalParams(mu=0.038, sigma=0.022),        # hızlı araç artışı
)

#: Tüm hazır presetlerin isim bazlı haritası
PRESETS: dict[str, ScenarioParams] = {
    "baseline": BASELINE,
    "optimistic": OPTIMISTIC,
    "pessimistic": PESSIMISTIC,
}


# =====================================================================
# 3. Senaryo Çalıştırıcı
# =====================================================================


def run_scenario(
    scenario: ScenarioParams,
    initial_savings: float,
    initial_monthly_saving: float,
    investment_type: str,
    target_type: str,
    target_price: float,
    n_paths: int = 10_000,
    n_steps: int = 240,
    seed: Optional[int] = None,
) -> SimulationOutput:
    """Belirli bir senaryo parametresiyle Monte Carlo simülasyonu çalıştırır.

    Bu fonksiyon, distributions modülündeki örnekleme fonksiyonlarını
    doğrudan kullanarak ScenarioParams'taki parametreleri uygular.

    Parameters
    ----------
    scenario : ScenarioParams
        Kullanılacak senaryo parametresi (ör: OPTIMISTIC, BASELINE).
    initial_savings : float
        Başlangıç birikimi (TL).
    initial_monthly_saving : float
        Aylık tasarruf miktarı (TL).
    investment_type : str
        Yatırım aracı: 'gold', 'bist' veya 'deposit'.
    target_type : str
        Hedef varlık: 'house' veya 'car'.
    target_price : float
        Hedef varlığın bugünkü fiyatı (TL).
    n_paths : int, default 10_000
        Monte Carlo path sayısı.
    n_steps : int, default 240
        Simülasyon ufku (ay).
    seed : int, optional
        Tekrarlanabilirlik için tohum değeri.

    Returns
    -------
    SimulationOutput
        Ham simülasyon çıktısı.

    Notes
    -----
    Senaryonun tüm dağılım parametreleri (maaş, enflasyon, konut/araç fiyatı,
    altın/BIST/mevduat getirisi) `sample_correlated_variables` üzerinden
    ortak enflasyon faktörüyle ilişkilendirilerek örneklenir — `run_simulation`
    ile aynı korelasyon mantığı kullanılır.
    """
    from time_to_afford.affordability.target_price import compute_price_paths
    from time_to_afford.affordability.wealth import compute_wealth_paths
    from time_to_afford.simulation.distributions import sample_correlated_variables
    from time_to_afford.simulation.monte_carlo import (
        SimulationOutput,
        _find_first_affordable_month,
    )

    rng = np.random.default_rng(seed)

    # --- Senaryo parametreleriyle korelasyonlu örnekleme ---
    correlated = sample_correlated_variables(
        investment_type,
        target_type,
        n_steps,
        n_paths,
        rng,
        inflation_params=scenario.inflation,
        salary_growth_params=scenario.salary_growth,
        house_price_params=scenario.house_price,
        car_price_params=scenario.car_price,
        gold_params=scenario.gold,
        bist_params=scenario.bist,
        deposit_params=scenario.deposit,
    )
    investment_factors = correlated["investment_return"]
    salary_factors = correlated["salary_growth"]
    price_growth_factors = correlated["target_price_growth"]

    # --- Servet ve fiyat yolları ---
    wealth_paths = compute_wealth_paths(
        initial_savings=initial_savings,
        initial_monthly_saving=initial_monthly_saving,
        investment_return_factors=investment_factors,
        salary_growth_factors=salary_factors,
        saving_ratio=0.0,
    )
    price_paths = compute_price_paths(
        initial_price=target_price,
        monthly_growth_factors=price_growth_factors,
    )

    # --- T = min{t : W_t >= P_t} ---
    affordability_months = _find_first_affordable_month(
        wealth_paths=wealth_paths,
        price_paths=price_paths,
        n_steps=n_steps,
        n_paths=n_paths,
    )

    return SimulationOutput(
        affordability_months=affordability_months,
        n_paths=n_paths,
        n_steps=n_steps,
    )


def get_preset(name: str) -> ScenarioParams:
    """İsme göre hazır senaryo preset'ini döndürür.

    Parameters
    ----------
    name : str
        'baseline', 'optimistic' veya 'pessimistic' (büyük/küçük harf duyarsız).

    Returns
    -------
    ScenarioParams
        İlgili senaryo nesnesi.

    Raises
    ------
    ValueError
        Bilinmeyen senaryo adı girilirse.
    """
    key = name.lower().strip()
    if key not in PRESETS:
        raise ValueError(
            f"Bilinmeyen senaryo adı: '{name}'. "
            f"Desteklenenler: {list(PRESETS.keys())}"
        )
    return PRESETS[key]
