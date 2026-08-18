"""
Servet yolu (W_t) hesaplama modülü.

Her t adımında:
  W_t = W_{t-1} * yatırım_getirisi(t) + aylık_tasarruf(t)

Aylık tasarruf, maaş artışına göre güncellenir.
"""

from __future__ import annotations

import numpy as np


def compute_wealth_paths(
    initial_savings: float,
    initial_monthly_saving: float,
    investment_return_factors: np.ndarray,
    salary_growth_factors: np.ndarray,
    saving_ratio: float = 0.30,
) -> np.ndarray:
    """Kullanıcının servet yolunu (W_t) hesaplar.

    Her adımda:
      maaş(t) = maaş(t-1) * salary_growth_factors[t]
      tasarruf(t) = maaş(t) * saving_ratio
      W_t = W_{t-1} * investment_return_factors[t] + tasarruf(t)

    Not: Eğer maaş büyümesi yerine sabit aylık tasarruf kullanılmak
    isteniyorsa saving_ratio=0 ve initial_monthly_saving değeri
    yeterlidir (salary_growth_factors görmezden gelinir).

    Parameters
    ----------
    initial_savings : float
        t=0 anındaki toplam birikim (TL).
    initial_monthly_saving : float
        t=0 anındaki aylık tasarruf miktarı (TL).
        Maaş artışıyla birlikte büyür.
    investment_return_factors : np.ndarray
        (n_steps, n_paths) boyutunda aylık brüt getiri çarpanları.
        distributions.sample_investment_return() çıktısı kullanılabilir.
    salary_growth_factors : np.ndarray
        (n_steps, n_paths) boyutunda aylık maaş artış çarpanları.
        distributions.sample_salary_growth() çıktısı kullanılabilir.
    saving_ratio : float, default 0.30
        Maaşın ne kadarının tasarrufa gittiği (0-1 arası).
        0 verilirse tasarruf sabit kalır (initial_monthly_saving).

    Returns
    -------
    np.ndarray
        (n_steps, n_paths) boyutunda servet yolu.
        wealth_paths[t, i] → path i'de t. aydaki toplam servet (TL).
    """
    if initial_savings < 0:
        raise ValueError(f"initial_savings >= 0 olmalıdır, alınan: {initial_savings}")
    if initial_monthly_saving < 0:
        raise ValueError(f"initial_monthly_saving >= 0 olmalıdır, alınan: {initial_monthly_saving}")
    if not (0.0 <= saving_ratio <= 1.0):
        raise ValueError(f"saving_ratio 0-1 arasında olmalıdır, alınan: {saving_ratio}")
    if investment_return_factors.shape != salary_growth_factors.shape:
        raise ValueError("investment_return_factors ve salary_growth_factors aynı boyutta olmalıdır.")

    n_steps, n_paths = investment_return_factors.shape
    wealth = np.empty((n_steps, n_paths), dtype=np.float64)

    # Her path için başlangıç değerleri
    current_wealth = np.full(n_paths, initial_savings, dtype=np.float64)
    current_saving = np.full(n_paths, initial_monthly_saving, dtype=np.float64)

    for t in range(n_steps):
        # Maaş artışını uygula
        current_saving = current_saving * salary_growth_factors[t]

        # Aylık tasarruf: maaş artışına göre büyüyen miktar
        # saving_ratio == 0 ise initial_monthly_saving sabit kalır
        if saving_ratio > 0:
            monthly_contribution = current_saving * saving_ratio
        else:
            monthly_contribution = current_saving  # salary_growth ile büyür ama oran uygulanmaz

        # Servet güncelleme: yatırım getirisi + katkı
        current_wealth = current_wealth * investment_return_factors[t] + monthly_contribution
        wealth[t] = current_wealth

    return wealth

