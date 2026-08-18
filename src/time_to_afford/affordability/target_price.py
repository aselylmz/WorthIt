"""
Hedef varlık fiyat yolu (P_t) hesaplama modülü.

Başlangıç fiyatından yola çıkarak aylık fiyat artış çarpanlarını
uygular ve her t anındaki P_t fiyat yolunu döndürür.
"""

from __future__ import annotations

import numpy as np


def compute_price_paths(
    initial_price: float,
    monthly_growth_factors: np.ndarray,
) -> np.ndarray:
    """Hedef varlığın fiyat yolunu (P_t) hesaplar.

    P_t = P_0 * cumprod(monthly_growth_factors[:t])

    Parameters
    ----------
    initial_price : float
        t=0 anındaki başlangıç fiyatı (TL).
    monthly_growth_factors : np.ndarray
        (n_steps, n_paths) boyutunda aylık fiyat artış çarpanları.
        distributions.sample_target_price_growth() çıktısı kullanılabilir.

    Returns
    -------
    np.ndarray
        (n_steps, n_paths) boyutunda hedef varlık fiyat yolu.
        price_paths[t, i] → path i'de t. aydaki fiyat.
    """
    if initial_price <= 0:
        raise ValueError(f"initial_price > 0 olmalıdır, alınan: {initial_price}")
    if monthly_growth_factors.ndim != 2:
        raise ValueError("monthly_growth_factors (n_steps, n_paths) boyutunda olmalıdır.")

    cumulative = np.cumprod(monthly_growth_factors, axis=0)
    return initial_price * cumulative

