"""
Monte Carlo simülasyon motoru.

Birden fazla stokastik senaryo üreterek kullanıcının finansal
hedefine ulaşma süresini tahmin eder.

Ana fonksiyon:
    run_simulation() -> SimulationOutput

Her path'te:
  1. Yatırım getirisi, maaş artışı, hedef fiyat artışı örneklenir
  2. W_t ve P_t yolları hesaplanır
  3. T = min{t : W_t >= P_t} bulunur
  4. Hedefe ulaşılamayan path'lerde T = np.nan döner
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.random import Generator

from time_to_afford.affordability.target_price import compute_price_paths
from time_to_afford.affordability.wealth import compute_wealth_paths
from time_to_afford.simulation.distributions import sample_correlated_variables
from time_to_afford.utils.logging import get_logger

logger = get_logger(__name__)


# =====================================================================
# 1. Simülasyon Çıktı Veri Sınıfı
# =====================================================================


@dataclass
class SimulationOutput:
    """Monte Carlo simülasyon ham çıktısı.

    Attributes
    ----------
    affordability_months : np.ndarray
        (n_paths,) boyutunda array. Her eleman, o path'te hedefe
        ulaşılan ayı (1-indexed) veya np.nan (ulaşılamadı) içerir.
    n_paths : int
        Toplam path sayısı.
    n_steps : int
        Simülasyon ufku (ay).
    never_affordable_count : int
        Simülasyon ufku içinde hedefe hiç ulaşamayan path sayısı.
    """

    affordability_months: np.ndarray
    n_paths: int
    n_steps: int

    @property
    def never_affordable_count(self) -> int:
        """Hedefe ulaşılamayan path sayısı."""
        return int(np.sum(np.isnan(self.affordability_months)))

    @property
    def never_affordable_pct(self) -> float:
        """Hedefe ulaşılamayan path oranı (0-1)."""
        return self.never_affordable_count / self.n_paths

    def probability_within(self, months: int) -> float:
        """Verilen ay sayısı içinde hedefe ulaşma olasılığı.

        Parameters
        ----------
        months : int
            Eşik ay sayısı.

        Returns
        -------
        float
            0-1 arası olasılık.
        """
        reachable = self.affordability_months[~np.isnan(self.affordability_months)]
        return float(np.sum(reachable <= months) / self.n_paths)

    def quantile_months(self, q: float) -> Optional[float]:
        """Ulaşılabilen path'ler arasında q. quantile'ı döndürür.

        Parameters
        ----------
        q : float
            0-1 arası quantile (ör: 0.5 → medyan).

        Returns
        -------
        float or None
            Ay cinsinden süre; hiç ulaşılabilen path yoksa None.
        """
        reachable = self.affordability_months[~np.isnan(self.affordability_months)]
        if len(reachable) == 0:
            return None
        return float(np.quantile(reachable, q))


# =====================================================================
# 2. Ana Simülasyon Fonksiyonu
# =====================================================================


def run_simulation(
    initial_savings: float,
    initial_monthly_saving: float,
    investment_type: str,
    target_type: str,
    target_price: float,
    n_paths: int = 10_000,
    n_steps: int = 240,
    seed: Optional[int] = None,
) -> SimulationOutput:
    """Monte Carlo simülasyonunu çalıştırır.

    Her path'te stokastik değişkenler örneklenir, W_t ve P_t yolları
    hesaplanır ve T = min{t : W_t >= P_t} bulunur.

    Parameters
    ----------
    initial_savings : float
        Kullanıcının başlangıç birikimi (TL).
    initial_monthly_saving : float
        Başlangıçtaki aylık tasarruf miktarı (TL).
    investment_type : str
        Yatırım aracı: 'gold', 'bist' veya 'deposit'.
    target_type : str
        Hedef varlık: 'house' veya 'car'.
    target_price : float
        Hedef varlığın bugünkü fiyatı (TL).
    n_paths : int, default 10_000
        Monte Carlo path sayısı.
    n_steps : int, default 240
        Simülasyon ufku (ay). 240 = 20 yıl.
    seed : int, optional
        Tekrarlanabilirlik için rastgele sayı tohumu.

    Returns
    -------
    SimulationOutput
        Ham simülasyon sonuçları.
    """
    rng: Generator = np.random.default_rng(seed)

    logger.info(
        f"Simülasyon başlıyor: n_paths={n_paths}, n_steps={n_steps}, "
        f"investment={investment_type}, target={target_type}, "
        f"target_price={target_price:,.0f} TL"
    )

    # --- Stokastik değişkenleri ortak enflasyon faktörüyle ilişkilendirerek örnekle ---
    # (bkz. distributions.sample_correlated_variables — maaş, konut/araç fiyatı ve
    # yatırım getirisi artık bağımsız değil, ortak bir enflasyon şokuna göre birlikte hareket eder)
    correlated = sample_correlated_variables(investment_type, target_type, n_steps, n_paths, rng)
    investment_factors = correlated["investment_return"]
    salary_factors = correlated["salary_growth"]
    price_growth_factors = correlated["target_price_growth"]

    # --- W_t ve P_t yollarını hesapla ---
    # wealth_paths[t, i]: path i'de t. aydaki servet
    wealth_paths = compute_wealth_paths(
        initial_savings=initial_savings,
        initial_monthly_saving=initial_monthly_saving,
        investment_return_factors=investment_factors,
        salary_growth_factors=salary_factors,
        saving_ratio=0.0,  # initial_monthly_saving doğrudan tasarruf, ratio kullanılmıyor
    )

    # price_paths[t, i]: path i'de t. aydaki hedef fiyat
    price_paths = compute_price_paths(
        initial_price=target_price,
        monthly_growth_factors=price_growth_factors,
    )

    # --- T = min{t : W_t >= P_t} bul ---
    affordability_months = _find_first_affordable_month(
        wealth_paths=wealth_paths,
        price_paths=price_paths,
        n_steps=n_steps,
        n_paths=n_paths,
    )

    never_count = int(np.sum(np.isnan(affordability_months)))
    logger.info(
        f"Simülasyon tamamlandı. "
        f"Hedefe ulaşamayan: {never_count}/{n_paths} "
        f"(%{100 * never_count / n_paths:.1f})"
    )

    return SimulationOutput(
        affordability_months=affordability_months,
        n_paths=n_paths,
        n_steps=n_steps,
    )


# =====================================================================
# 3. İlk Karşılanabilir Ayı Bulan Yardımcı
# =====================================================================


def _find_first_affordable_month(
    wealth_paths: np.ndarray,
    price_paths: np.ndarray,
    n_steps: int,
    n_paths: int,
) -> np.ndarray:
    """Her path için W_t >= P_t koşulunun ilk sağlandığı ayı döndürür.

    Vektörize yaklaşım:
      1. W_t >= P_t maskesi oluştur  → (n_steps, n_paths) bool
      2. Her path'te ilk True'nun indeksini bul
      3. Hiç True yoksa np.nan ata

    Parameters
    ----------
    wealth_paths : np.ndarray
        (n_steps, n_paths) servet yolu.
    price_paths : np.ndarray
        (n_steps, n_paths) hedef fiyat yolu.
    n_steps : int
        Toplam adım sayısı.
    n_paths : int
        Toplam path sayısı.

    Returns
    -------
    np.ndarray
        (n_paths,) boyutunda float array.
        Hedefe ulaşılan ay (1-indexed) veya np.nan.
    """
    # (n_steps, n_paths) bool maskesi: hangi adımlarda servet >= fiyat?
    affordable_mask = wealth_paths >= price_paths  # (n_steps, n_paths)

    # Her path için (sütun) ilk True'nun satır indeksini bul
    # argmax, True olmadığında 0 döner; bunu nan ile ayırt edeceğiz
    first_true_idx = np.argmax(affordable_mask, axis=0)  # (n_paths,)

    # Hiç True olmayan path'leri tespit et
    ever_affordable = affordable_mask.any(axis=0)  # (n_paths,) bool

    # Sonuç array'i: 1-indexed ay veya nan
    result = np.where(
        ever_affordable,
        first_true_idx + 1,  # 0-indexed → 1-indexed (ay numarası)
        np.nan,
    ).astype(np.float64)

    return result

