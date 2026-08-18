"""
Affordability modülü testleri.

Kapsam:
  - _months_to_scenario(): ay → ScenarioResult dönüşümü
  - summarize_simulation(): senaryo etiketleri, quantile'lar,
    olasılıklar, all-nan hata durumu, özel parametreler
"""

import numpy as np
import pytest

from time_to_afford.affordability.time_to_afford import (
    _months_to_scenario,
    summarize_simulation,
)
from time_to_afford.models.schemas import ScenarioResult, SimulationResult
from time_to_afford.simulation.monte_carlo import SimulationOutput


# =====================================================================
# Yardımcı fabrika
# =====================================================================


def _make_sim_output(months_array: np.ndarray, n_steps: int = 240) -> SimulationOutput:
    """Test için SimulationOutput oluşturur."""
    return SimulationOutput(
        affordability_months=months_array,
        n_paths=len(months_array),
        n_steps=n_steps,
    )


# =====================================================================
# 1. _months_to_scenario — Birim Testleri
# =====================================================================


class TestMonthsToScenario:
    """_months_to_scenario() yardımcı fonksiyonunu test eder."""

    def test_returns_scenario_result_type(self):
        """Dönüş tipi ScenarioResult olmalı."""
        result = _months_to_scenario("Temel", 36.0)
        assert isinstance(result, ScenarioResult)

    def test_label_preserved(self):
        """Verilen etiket korunmalı."""
        result = _months_to_scenario("İyimser", 12.0)
        assert result.label == "İyimser"

    def test_total_months_rounded(self):
        """Ondalıklı ay değeri en yakın tam sayıya yuvarlanmalı."""
        result = _months_to_scenario("Test", 14.6)
        assert result.total_months == 15

    def test_years_and_months_correct(self):
        """100 ay = 8 yıl 4 ay olmalı."""
        result = _months_to_scenario("Test", 100.0)
        assert result.years == 8
        assert result.months == 4
        assert result.total_months == 100

    def test_exact_year_boundary(self):
        """24 ay = 2 yıl 0 ay olmalı."""
        result = _months_to_scenario("Test", 24.0)
        assert result.years == 2
        assert result.months == 0

    def test_less_than_one_year(self):
        """11 ay = 0 yıl 11 ay olmalı."""
        result = _months_to_scenario("Test", 11.0)
        assert result.years == 0
        assert result.months == 11

    def test_rounding_down(self):
        """14.4 → 14 ay yuvarlanmalı."""
        result = _months_to_scenario("Test", 14.4)
        assert result.total_months == 14

    def test_zero_months(self):
        """0 ay = 0 yıl 0 ay olmalı."""
        result = _months_to_scenario("Test", 0.0)
        assert result.years == 0
        assert result.months == 0
        assert result.total_months == 0


# =====================================================================
# 2. summarize_simulation() — Temel Çıktı Testleri
# =====================================================================


class TestSummarizeSimulationBasic:
    """summarize_simulation() temel davranışını doğrular."""

    @pytest.fixture
    def uniform_output(self) -> SimulationOutput:
        """Tüm path'ler eşit aralıklı: 1'den 100'e kadar."""
        arr = np.arange(1, 101, dtype=np.float64)   # 100 path: 1, 2, ..., 100 ay
        return _make_sim_output(arr, n_steps=120)

    def test_returns_simulation_result_type(self, uniform_output):
        """Dönüş tipi SimulationResult olmalı."""
        result = summarize_simulation(uniform_output)
        assert isinstance(result, SimulationResult)

    def test_median_label(self, uniform_output):
        """Temel senaryo etiketi 'Temel' olmalı."""
        result = summarize_simulation(uniform_output)
        assert result.median_time.label == "Temel"

    def test_optimistic_label(self, uniform_output):
        """İyimser senaryo etiketi 'İyimser' olmalı."""
        result = summarize_simulation(uniform_output)
        assert result.optimistic_time.label == "İyimser"

    def test_pessimistic_label(self, uniform_output):
        """Kötümser senaryo etiketi 'Kötümser' olmalı."""
        result = summarize_simulation(uniform_output)
        assert result.pessimistic_time.label == "Kötümser"

    def test_num_simulations_correct(self, uniform_output):
        """num_simulations n_paths ile eşleşmeli."""
        result = summarize_simulation(uniform_output)
        assert result.num_simulations == 100

    def test_never_affordable_pct_zero(self, uniform_output):
        """Hiç nan olmadığında never_affordable_pct == 0.0."""
        result = summarize_simulation(uniform_output)
        assert result.never_affordable_pct == 0.0

    def test_optimistic_le_median_le_pessimistic(self, uniform_output):
        """İyimser ≤ Temel ≤ Kötümser olmalı."""
        result = summarize_simulation(uniform_output)
        assert result.optimistic_time.total_months <= result.median_time.total_months
        assert result.median_time.total_months <= result.pessimistic_time.total_months

    def test_disclaimer_present(self, uniform_output):
        """Yasal uyarı metni boş olmamalı."""
        result = summarize_simulation(uniform_output)
        assert len(result.disclaimer) > 0


# =====================================================================
# 3. summarize_simulation() — Quantile Doğruluğu
# =====================================================================


class TestSummarizeSimulationQuantiles:
    """P10 / P50 / P90 quantile değerlerinin doğruluğunu test eder."""

    def test_median_is_p50(self):
        """Median (P50) değeri doğru hesaplanmalı."""
        # 1..100 → P50 = 50.5 → round = 50 (Python'un banker's rounding'i ile)
        arr = np.arange(1, 101, dtype=np.float64)
        out = _make_sim_output(arr)
        result = summarize_simulation(out)
        # np.quantile(arr, 0.5) = 50.5 → round(50.5) = 50 (banker's rounding)
        expected = round(float(np.quantile(arr, 0.5)))
        assert result.median_time.total_months == expected

    def test_optimistic_is_p10(self):
        """İyimser senaryo P10 quantile'ına eşleşmeli."""
        arr = np.arange(1, 101, dtype=np.float64)
        out = _make_sim_output(arr)
        result = summarize_simulation(out)
        expected = round(float(np.quantile(arr, 0.10)))
        assert result.optimistic_time.total_months == expected

    def test_pessimistic_is_p90(self):
        """Kötümser senaryo P90 quantile'ına eşleşmeli."""
        arr = np.arange(1, 101, dtype=np.float64)
        out = _make_sim_output(arr)
        result = summarize_simulation(out)
        expected = round(float(np.quantile(arr, 0.90)))
        assert result.pessimistic_time.total_months == expected

    def test_custom_quantiles(self):
        """Özel quantile parametreleri kullanılabilmeli."""
        arr = np.arange(1, 101, dtype=np.float64)
        out = _make_sim_output(arr)
        result = summarize_simulation(out, optimistic_q=0.20, pessimistic_q=0.80)
        expected_opt = round(float(np.quantile(arr, 0.20)))
        expected_pes = round(float(np.quantile(arr, 0.80)))
        assert result.optimistic_time.total_months == expected_opt
        assert result.pessimistic_time.total_months == expected_pes

    def test_quantiles_ignore_nan(self):
        """nan path'ler quantile hesaplamalarında sayılmamalı."""
        # 50 path ulaşmış [10..59], 50 path nan
        reached = np.arange(10, 60, dtype=np.float64)
        arr = np.concatenate([reached, np.full(50, np.nan)])
        out = _make_sim_output(arr, n_steps=120)
        result = summarize_simulation(out)
        # Median sadece reached array'inden hesaplanmalı
        expected_median = round(float(np.quantile(reached, 0.50)))
        assert result.median_time.total_months == expected_median


# =====================================================================
# 4. summarize_simulation() — Olasılık Testleri
# =====================================================================


class TestSummarizeSimulationProbabilities:
    """probability_5y / 10y / 15y değerlerini test eder."""

    def test_probability_5y_correct(self):
        """5 yıl (60 ay) içindeki olasılık doğru hesaplanmalı."""
        # 200 path: 1..200 ay
        arr = np.arange(1, 201, dtype=np.float64)
        out = _make_sim_output(arr, n_steps=240)
        result = summarize_simulation(out)
        # 60 ayda veya önce ulaşan: 1..60 → 60/200 = 0.30
        assert result.probability_5y == pytest.approx(60 / 200)

    def test_probability_10y_correct(self):
        """10 yıl (120 ay) içindeki olasılık doğru hesaplanmalı."""
        arr = np.arange(1, 201, dtype=np.float64)
        out = _make_sim_output(arr, n_steps=240)
        result = summarize_simulation(out)
        assert result.probability_10y == pytest.approx(120 / 200)

    def test_probability_15y_correct(self):
        """15 yıl (180 ay) içindeki olasılık doğru hesaplanmalı."""
        arr = np.arange(1, 201, dtype=np.float64)
        out = _make_sim_output(arr, n_steps=240)
        result = summarize_simulation(out)
        assert result.probability_15y == pytest.approx(180 / 200)

    def test_probabilities_are_monotone(self):
        """5y ≤ 10y ≤ 15y olasılık sıralaması korunmalı."""
        arr = np.arange(1, 181, dtype=np.float64)
        out = _make_sim_output(arr, n_steps=240)
        result = summarize_simulation(out)
        assert result.probability_5y <= result.probability_10y
        assert result.probability_10y <= result.probability_15y

    def test_probabilities_in_range(self):
        """Tüm olasılıklar [0, 1] arasında olmalı."""
        arr = np.arange(1, 101, dtype=np.float64)
        out = _make_sim_output(arr)
        result = summarize_simulation(out)
        assert 0.0 <= result.probability_5y <= 1.0
        assert 0.0 <= result.probability_10y <= 1.0
        assert 0.0 <= result.probability_15y <= 1.0

    def test_nan_paths_count_as_not_reached(self):
        """nan path'ler olasılık paydasına dahil edilmeli."""
        # 50 path hemen (1. ay), 50 path nan → 5y olasılığı = 50/100 = 0.5
        reached = np.ones(50, dtype=np.float64)
        arr = np.concatenate([reached, np.full(50, np.nan)])
        out = _make_sim_output(arr, n_steps=120)
        result = summarize_simulation(out)
        assert result.probability_5y == pytest.approx(0.5)

    def test_custom_horizon_years(self):
        """Özel ufuk yılları kullanıldığında varsayılan değerler yerine onlar kullanılmalı."""
        arr = np.arange(1, 101, dtype=np.float64)
        out = _make_sim_output(arr, n_steps=240)
        # Özel: sadece 3 ve 7 yıl → probability_5y/10y/15y varsayılan 0.0 dönmeli
        result = summarize_simulation(out, probability_horizons_years=[3, 7])
        # [5, 10, 15] horizons artık dict'te yok → .get() → 0.0
        assert result.probability_5y == pytest.approx(0.0)
        assert result.probability_10y == pytest.approx(0.0)
        assert result.probability_15y == pytest.approx(0.0)


# =====================================================================
# 5. summarize_simulation() — Hata Durumları
# =====================================================================


class TestSummarizeSimulationErrors:
    """Geçersiz girişlerde doğru hata fırlatıldığını doğrular."""

    def test_all_nan_raises_value_error(self):
        """Tüm path'ler nan ise ValueError fırlatılmalı."""
        arr = np.full(50, np.nan)
        out = _make_sim_output(arr)
        with pytest.raises(ValueError, match="hiçbir path hedefe ulaşamadı"):
            summarize_simulation(out)


# =====================================================================
# 6. summarize_simulation() — Entegrasyon: Gerçek Simülasyon ile
# =====================================================================


class TestSummarizeSimulationIntegration:
    """Gerçek run_simulation() çıktısıyla uçtan uca testi."""

    def test_end_to_end_with_run_simulation(self):
        """run_simulation() → summarize_simulation() pipeline hata vermemeli."""
        from time_to_afford.simulation.monte_carlo import run_simulation

        sim_out = run_simulation(
            initial_savings=500_000.0,
            initial_monthly_saving=15_000.0,
            investment_type="gold",
            target_type="house",
            target_price=3_000_000.0,
            n_paths=500,
            n_steps=180,
            seed=42,
        )
        result = summarize_simulation(sim_out)

        assert isinstance(result, SimulationResult)
        assert result.num_simulations == 500
        assert 0.0 <= result.never_affordable_pct <= 1.0
        assert result.optimistic_time.total_months <= result.pessimistic_time.total_months

    def test_result_is_pydantic_serializable(self):
        """SimulationResult JSON serileştirilebilir olmalı."""
        from time_to_afford.simulation.monte_carlo import run_simulation

        sim_out = run_simulation(
            initial_savings=200_000.0,
            initial_monthly_saving=10_000.0,
            investment_type="deposit",
            target_type="car",
            target_price=800_000.0,
            n_paths=200,
            n_steps=120,
            seed=0,
        )
        result = summarize_simulation(sim_out)
        json_str = result.model_dump_json()
        assert len(json_str) > 0
        assert "median_time" in json_str
        assert "optimistic_time" in json_str
        assert "pessimistic_time" in json_str
