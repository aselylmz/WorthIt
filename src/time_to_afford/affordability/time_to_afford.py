"""
Time-to-Afford hesaplama motoru.

SimulationOutput'tan istatistiksel özet üretir:
  - Temel senaryo  : median (P50)
  - İyimser senaryo: P10   (path'lerin %10'u bu ayda veya önce hedefe ulaşır)
  - Kötümser senaryo: P90  (path'lerin %90'ı bu ayda veya önce hedefe ulaşır)
  - Belirli yıllar içinde satın alma olasılıkları (5, 10, 15 yıl)

Ana fonksiyon:
    summarize_simulation(output, horizon_years_list) -> SimulationResult
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

from time_to_afford.models.schemas import ScenarioResult, SimulationResult
from time_to_afford.utils.dates import months_to_years_months
from time_to_afford.utils.logging import get_logger

if TYPE_CHECKING:
    from time_to_afford.simulation.monte_carlo import SimulationOutput

logger = get_logger(__name__)


# =====================================================================
# Dahili yardımcılar
# =====================================================================


def _months_to_scenario(label: str, total_months: float) -> ScenarioResult:
    """Float ay değerini ScenarioResult'a dönüştürür.

    Parameters
    ----------
    label : str
        Senaryo adı (ör: 'Temel', 'İyimser', 'Kötümser').
    total_months : float
        Ondalıklı ay değeri; int'e yuvarlanır.

    Returns
    -------
    ScenarioResult
        Yıl, ay ve toplam ay alanları dolu senaryo nesnesi.
    """
    rounded = round(total_months)
    years, months = months_to_years_months(rounded)
    return ScenarioResult(
        label=label,
        years=years,
        months=months,
        total_months=rounded,
    )


# =====================================================================
# Ana fonksiyon
# =====================================================================


def summarize_simulation(
    output: SimulationOutput,
    optimistic_q: float = 0.10,
    pessimistic_q: float = 0.90,
    probability_horizons_years: Optional[List[int]] = None,
) -> SimulationResult:
    """SimulationOutput'tan istatistiksel özet üretir.

    Ulaşılamayan path'ler (nan) olasılık hesaplamalarında
    "hedefe ulaşılamadı" olarak sayılır; quantile hesaplamalarında
    ise yalnızca ulaşılabilen path'ler kullanılır.

    Parameters
    ----------
    output : SimulationOutput
        run_simulation() çıktısı.
    optimistic_q : float, default 0.10
        İyimser senaryo quantile'ı (P10 → erken ulaşanlar).
    pessimistic_q : float, default 0.90
        Kötümser senaryo quantile'ı (P90 → geç ulaşanlar).
    probability_horizons_years : list of int, optional
        Olasılık hesaplanacak yıllar. None ise [5, 10, 15] kullanılır.

    Returns
    -------
    SimulationResult
        Median, iyimser, kötümser senaryolar ve olasılıklar.

    Raises
    ------
    ValueError
        Hiçbir path hedefe ulaşamadıysa (tüm değerler nan) — özet
        üretmek anlamlı değildir.
    """
    if probability_horizons_years is None:
        probability_horizons_years = [5, 10, 15]

    # --- Ulaşılabilen path'leri kontrol et ---
    median_months = output.quantile_months(0.50)
    if median_months is None:
        raise ValueError(
            "Simülasyon ufku içinde hiçbir path hedefe ulaşamadı. "
            "Ufuk süresini artırın veya finansal parametreleri gözden geçirin."
        )

    opt_months = output.quantile_months(optimistic_q)
    pes_months = output.quantile_months(pessimistic_q)

    logger.info(
        f"Özet üretiliyor: "
        f"P{int(optimistic_q*100)}={opt_months:.1f}ay, "
        f"P50={median_months:.1f}ay, "
        f"P{int(pessimistic_q*100)}={pes_months:.1f}ay | "
        f"Hiç ulaşamayan: %{output.never_affordable_pct * 100:.1f}"
    )

    # --- Senaryo nesneleri ---
    median_result = _months_to_scenario("Temel", median_months)

    # İyimser: P10 → erken ulaşanlar (en iyi %10)
    optimistic_result = _months_to_scenario(
        "İyimser",
        opt_months if opt_months is not None else median_months,
    )

    # Kötümser: P90 → geç ulaşanlar (en kötü %10'un eşiği)
    pessimistic_result = _months_to_scenario(
        "Kötümser",
        pes_months if pes_months is not None else median_months,
    )

    # --- Yıl bazlı olasılıklar ---
    horizons = {
        yr: output.probability_within(yr * 12)
        for yr in probability_horizons_years
    }

    return SimulationResult(
        median_time=median_result,
        optimistic_time=optimistic_result,
        pessimistic_time=pessimistic_result,
        probability_5y=horizons.get(5, 0.0),
        probability_10y=horizons.get(10, 0.0),
        probability_15y=horizons.get(15, 0.0),
        num_simulations=output.n_paths,
        never_affordable_pct=output.never_affordable_pct,
    )
