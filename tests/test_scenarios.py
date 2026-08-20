"""
Scenarios modülü testleri.

Kapsam:
  - ScenarioParams presetleri (BASELINE, OPTIMISTIC, PESSIMISTIC)
  - get_preset(): isim çözümleme
  - run_scenario(): korelasyonlu örnekleme ile SimulationOutput üretimi
"""

import numpy as np
import pytest

from time_to_afford.simulation.monte_carlo import SimulationOutput
from time_to_afford.simulation.scenarios import (
    BASELINE,
    OPTIMISTIC,
    PESSIMISTIC,
    PRESETS,
    ScenarioParams,
    get_preset,
    run_scenario,
)

N_PATHS_SMALL = 300
N_STEPS = 120


# =====================================================================
# 1. Presetler
# =====================================================================


class TestPresets:
    """Hazır senaryo presetlerinin tutarlılığını doğrular."""

    @pytest.mark.parametrize("preset", [BASELINE, OPTIMISTIC, PESSIMISTIC])
    def test_preset_is_scenario_params(self, preset):
        assert isinstance(preset, ScenarioParams)

    def test_presets_dict_contains_all_three(self):
        assert set(PRESETS.keys()) == {"baseline", "optimistic", "pessimistic"}

    def test_optimistic_has_lower_inflation_than_pessimistic(self):
        assert OPTIMISTIC.inflation.mu < PESSIMISTIC.inflation.mu

    def test_optimistic_has_higher_salary_growth_than_pessimistic(self):
        assert OPTIMISTIC.salary_growth.mu > PESSIMISTIC.salary_growth.mu


class TestGetPreset:
    """get_preset() isim çözümleme testleri."""

    def test_lowercase_name(self):
        assert get_preset("baseline") is BASELINE

    def test_case_insensitive(self):
        assert get_preset("OPTIMISTIC") is OPTIMISTIC
        assert get_preset("  Pessimistic  ") is PESSIMISTIC

    def test_unknown_name_raises(self):
        with pytest.raises(ValueError, match="Bilinmeyen senaryo adı"):
            get_preset("hyperinflation")


# =====================================================================
# 2. run_scenario()
# =====================================================================


class TestRunScenario:
    """run_scenario() çıktı yapısı ve davranışı."""

    @pytest.mark.parametrize("scenario", [BASELINE, OPTIMISTIC, PESSIMISTIC])
    def test_returns_simulation_output(self, scenario):
        out = run_scenario(
            scenario,
            initial_savings=200_000.0,
            initial_monthly_saving=10_000.0,
            investment_type="gold",
            target_type="house",
            target_price=2_000_000.0,
            n_paths=N_PATHS_SMALL,
            n_steps=N_STEPS,
            seed=0,
        )
        assert isinstance(out, SimulationOutput)
        assert out.affordability_months.shape == (N_PATHS_SMALL,)
        assert 0.0 <= out.never_affordable_pct <= 1.0

    def test_same_seed_reproducible(self):
        kwargs = dict(
            scenario=BASELINE,
            initial_savings=100_000.0,
            initial_monthly_saving=5_000.0,
            investment_type="bist",
            target_type="car",
            target_price=800_000.0,
            n_paths=N_PATHS_SMALL,
            n_steps=N_STEPS,
        )
        out_a = run_scenario(**kwargs, seed=11)
        out_b = run_scenario(**kwargs, seed=11)
        assert np.array_equal(
            out_a.affordability_months, out_b.affordability_months, equal_nan=True
        )

    def test_optimistic_reaches_goal_faster_than_pessimistic(self):
        """İyimser senaryo, kötümser senaryodan (aynı koşullarda) hedefe daha hızlı ulaşmalı."""
        shared_kwargs = dict(
            initial_savings=150_000.0,
            initial_monthly_saving=8_000.0,
            investment_type="deposit",
            target_type="house",
            target_price=2_500_000.0,
            n_paths=2_000,
            n_steps=N_STEPS,
            seed=1,
        )
        out_opt = run_scenario(OPTIMISTIC, **shared_kwargs)
        out_pes = run_scenario(PESSIMISTIC, **shared_kwargs)

        median_opt = out_opt.quantile_months(0.5)
        median_pes = out_pes.quantile_months(0.5)

        assert median_opt is not None
        if median_pes is not None:
            assert median_opt < median_pes
        # Pessimistic senaryoda hedefe ulaşamama oranı en az iyimser kadar yüksek olmalı
        assert out_pes.never_affordable_pct >= out_opt.never_affordable_pct

    def test_unknown_investment_type_raises(self):
        with pytest.raises(ValueError, match="Bilinmeyen yatırım türü"):
            run_scenario(
                BASELINE,
                initial_savings=100_000.0,
                initial_monthly_saving=5_000.0,
                investment_type="crypto",
                target_type="house",
                target_price=1_000_000.0,
                n_paths=10,
                n_steps=12,
            )

    def test_unknown_target_type_raises(self):
        with pytest.raises(ValueError, match="Bilinmeyen hedef türü"):
            run_scenario(
                BASELINE,
                initial_savings=100_000.0,
                initial_monthly_saving=5_000.0,
                investment_type="gold",
                target_type="yacht",
                target_price=1_000_000.0,
                n_paths=10,
                n_steps=12,
            )

    def test_negative_target_price_raises(self):
        with pytest.raises(ValueError):
            run_scenario(
                BASELINE,
                initial_savings=100_000.0,
                initial_monthly_saving=5_000.0,
                investment_type="gold",
                target_type="house",
                target_price=0.0,
                n_paths=10,
                n_steps=12,
            )
