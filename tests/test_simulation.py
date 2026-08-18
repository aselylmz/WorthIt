"""
Simulation modülü testleri.

Kapsam:
  - SimulationOutput: özellikler, quantile_months, probability_within
  - run_simulation(): çıktı şekli, tipi, seed tekrarlanabilirliği
  - Sınır durumlar: zaten karşılanabilir, asla karşılanamaz
  - Hatalı girişler: bilinmeyen investment_type / target_type
"""

import numpy as np
import pytest

from time_to_afford.simulation.monte_carlo import SimulationOutput, run_simulation


# =====================================================================
# Yardımcı sabitler
# =====================================================================

N_PATHS_SMALL = 200   # hızlı testler için küçük path sayısı
N_STEPS_SHORT = 24    # 2 yıl


# =====================================================================
# 1. SimulationOutput — Birim Testleri
# =====================================================================


class TestSimulationOutput:
    """SimulationOutput dataclass'ının özelliklerini test eder."""

    def _make_output(self, months_array: np.ndarray) -> SimulationOutput:
        """Verilen array'den SimulationOutput oluşturur."""
        return SimulationOutput(
            affordability_months=months_array,
            n_paths=len(months_array),
            n_steps=120,
        )

    def test_never_affordable_count_all_nan(self):
        """Tüm path'ler nan ise never_affordable_count == n_paths."""
        arr = np.full(100, np.nan)
        out = self._make_output(arr)
        assert out.never_affordable_count == 100

    def test_never_affordable_count_none_nan(self):
        """Hiç nan yoksa never_affordable_count == 0."""
        arr = np.array([10.0, 20.0, 30.0])
        out = self._make_output(arr)
        assert out.never_affordable_count == 0

    def test_never_affordable_count_mixed(self):
        """Karışık array'de nan sayısı doğru hesaplanmalı."""
        arr = np.array([5.0, np.nan, 15.0, np.nan, np.nan])
        out = self._make_output(arr)
        assert out.never_affordable_count == 3

    def test_never_affordable_pct_zero(self):
        """Hiç nan yoksa oran 0.0 olmalı."""
        arr = np.array([1.0, 2.0, 3.0, 4.0])
        out = self._make_output(arr)
        assert out.never_affordable_pct == 0.0

    def test_never_affordable_pct_one(self):
        """Tüm nan ise oran 1.0 olmalı."""
        arr = np.full(50, np.nan)
        out = self._make_output(arr)
        assert out.never_affordable_pct == 1.0

    def test_never_affordable_pct_half(self):
        """Yarısı nan ise oran 0.5 olmalı."""
        arr = np.array([1.0, 2.0, np.nan, np.nan])
        out = self._make_output(arr)
        assert out.never_affordable_pct == 0.5

    def test_quantile_months_median(self):
        """Medyan (q=0.5) doğru hesaplanmalı."""
        arr = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        out = self._make_output(arr)
        assert out.quantile_months(0.5) == pytest.approx(30.0)

    def test_quantile_months_ignores_nan(self):
        """quantile_months nan path'leri görmezden gelmeli."""
        arr = np.array([10.0, np.nan, 20.0, np.nan, 30.0])
        out = self._make_output(arr)
        # Sadece [10, 20, 30] kullanılır → medyan 20
        assert out.quantile_months(0.5) == pytest.approx(20.0)

    def test_quantile_months_all_nan_returns_none(self):
        """Ulaşılabilen path yoksa None döndürmeli."""
        arr = np.full(20, np.nan)
        out = self._make_output(arr)
        assert out.quantile_months(0.5) is None

    def test_probability_within_all_below_threshold(self):
        """Tüm path'ler eşik altındaysa olasılık 1.0 olmalı."""
        arr = np.array([5.0, 10.0, 15.0])
        out = self._make_output(arr)
        assert out.probability_within(20) == pytest.approx(1.0)

    def test_probability_within_none_below_threshold(self):
        """Hiçbir path eşik altında değilse olasılık 0.0 olmalı."""
        arr = np.array([50.0, 60.0, 70.0])
        out = self._make_output(arr)
        assert out.probability_within(10) == pytest.approx(0.0)

    def test_probability_within_nan_counts_as_not_reached(self):
        """nan path'ler eşiği aşamamış sayılmalı (payda n_paths)."""
        # 2 path ulaşmış (5, 10), 2 path nan → 2/4 = 0.5
        arr = np.array([5.0, 10.0, np.nan, np.nan])
        out = self._make_output(arr)
        assert out.probability_within(15) == pytest.approx(0.5)

    def test_probability_within_exact_threshold(self):
        """Eşit olan değer de dahil edilmeli (<= kontrolü)."""
        arr = np.array([12.0, 12.0, 12.0])
        out = self._make_output(arr)
        assert out.probability_within(12) == pytest.approx(1.0)


# =====================================================================
# 2. run_simulation() — Çıktı Yapısı Testleri
# =====================================================================


class TestRunSimulationOutput:
    """run_simulation() fonksiyonunun çıktı şekli ve tipini doğrular."""

    @pytest.fixture(scope="class")
    @classmethod
    def sim_output(cls) -> SimulationOutput:
        """Tekrar kullanılabilir simülasyon çıktısı fixture'ı."""
        return run_simulation(
            initial_savings=500_000.0,
            initial_monthly_saving=15_000.0,
            investment_type="gold",
            target_type="house",
            target_price=5_000_000.0,
            n_paths=N_PATHS_SMALL,
            n_steps=N_STEPS_SHORT,
            seed=42,
        )

    def test_returns_simulation_output_type(self, sim_output):
        """Dönüş tipi SimulationOutput olmalı."""
        assert isinstance(sim_output, SimulationOutput)

    def test_affordability_months_shape(self, sim_output):
        """affordability_months (n_paths,) boyutunda olmalı."""
        assert sim_output.affordability_months.shape == (N_PATHS_SMALL,)

    def test_affordability_months_dtype(self, sim_output):
        """affordability_months float64 tipinde olmalı."""
        assert sim_output.affordability_months.dtype == np.float64

    def test_n_paths_matches(self, sim_output):
        """n_paths alanı istenen değerle eşleşmeli."""
        assert sim_output.n_paths == N_PATHS_SMALL

    def test_n_steps_matches(self, sim_output):
        """n_steps alanı istenen değerle eşleşmeli."""
        assert sim_output.n_steps == N_STEPS_SHORT

    def test_finite_months_within_range(self, sim_output):
        """Nan olmayan ay değerleri [1, n_steps] aralığında olmalı."""
        finite = sim_output.affordability_months[
            ~np.isnan(sim_output.affordability_months)
        ]
        if len(finite) > 0:
            assert np.all(finite >= 1)
            assert np.all(finite <= N_STEPS_SHORT)

    def test_never_affordable_pct_is_valid_probability(self, sim_output):
        """never_affordable_pct 0 ile 1 arasında olmalı."""
        assert 0.0 <= sim_output.never_affordable_pct <= 1.0


# =====================================================================
# 3. run_simulation() — Tekrarlanabilirlik
# =====================================================================


class TestRunSimulationReproducibility:
    """Aynı seed ile aynı sonucun üretildiğini doğrular."""

    def test_same_seed_same_result(self):
        """Aynı seed ile iki çalıştırma aynı affordability_months vermeli."""
        kwargs = dict(
            initial_savings=300_000.0,
            initial_monthly_saving=10_000.0,
            investment_type="bist",
            target_type="car",
            target_price=1_500_000.0,
            n_paths=N_PATHS_SMALL,
            n_steps=N_STEPS_SHORT,
            seed=99,
        )
        out1 = run_simulation(**kwargs)
        out2 = run_simulation(**kwargs)
        np.testing.assert_array_equal(
            out1.affordability_months,
            out2.affordability_months,
        )

    def test_different_seeds_different_results(self):
        """Farklı seed'ler farklı sonuç üretmeli."""
        base_kwargs = dict(
            initial_savings=300_000.0,
            initial_monthly_saving=10_000.0,
            investment_type="deposit",
            target_type="house",
            target_price=2_000_000.0,
            n_paths=N_PATHS_SMALL,
            n_steps=N_STEPS_SHORT,
        )
        out_a = run_simulation(**base_kwargs, seed=1)
        out_b = run_simulation(**base_kwargs, seed=2)
        # İki array'in tamamen aynı olma ihtimali pratikte sıfıra yakın
        assert not np.array_equal(
            out_a.affordability_months,
            out_b.affordability_months,
        )

    def test_no_seed_runs_without_error(self):
        """seed=None ile çalıştırma hata vermemeli."""
        out = run_simulation(
            initial_savings=100_000.0,
            initial_monthly_saving=5_000.0,
            investment_type="gold",
            target_type="car",
            target_price=800_000.0,
            n_paths=50,
            n_steps=12,
            seed=None,
        )
        assert isinstance(out, SimulationOutput)


# =====================================================================
# 4. run_simulation() — Sınır Durumlar
# =====================================================================


class TestRunSimulationEdgeCases:
    """Aşırı uç değerlerle simülasyonun tutarlı davranışını test eder."""

    def test_already_affordable_at_start(self):
        """Birikim hedef fiyatından büyük olduğunda çok erken hedefe ulaşmalı."""
        out = run_simulation(
            initial_savings=10_000_000.0,   # çok yüksek birikim
            initial_monthly_saving=1_000.0,
            investment_type="deposit",
            target_type="house",
            target_price=1_000_000.0,       # düşük hedef
            n_paths=N_PATHS_SMALL,
            n_steps=60,
            seed=0,
        )
        # Neredeyse tüm path'ler çok erken (< 5. ay) hedefe ulaşmalı
        finite = out.affordability_months[~np.isnan(out.affordability_months)]
        assert len(finite) > 0, "Hiç hedefe ulaşılamamış — beklenen değil."
        assert np.median(finite) <= 5

    def test_never_affordable_scenario(self):
        """Çok yüksek hedef ve sıfır birikim/tasarrufta çoğu path nan olmalı."""
        out = run_simulation(
            initial_savings=0.0,
            initial_monthly_saving=0.0,
            investment_type="deposit",
            target_type="house",
            target_price=100_000_000.0,    # ulaşılamaz hedef
            n_paths=N_PATHS_SMALL,
            n_steps=N_STEPS_SHORT,
            seed=0,
        )
        # Büyük çoğunluğu (%90+) asla ulaşamaz olmalı
        assert out.never_affordable_pct > 0.9

    def test_zero_initial_savings_with_saving(self):
        """Başlangıç birikimsiz ama aylık tasarrufu olan senaryo hata vermemeli."""
        out = run_simulation(
            initial_savings=0.0,
            initial_monthly_saving=20_000.0,
            investment_type="gold",
            target_type="car",
            target_price=500_000.0,
            n_paths=N_PATHS_SMALL,
            n_steps=120,
            seed=7,
        )
        assert isinstance(out, SimulationOutput)
        assert out.affordability_months.shape == (N_PATHS_SMALL,)

    def test_n_paths_one(self):
        """Tek path'li simülasyon çalışmalı."""
        out = run_simulation(
            initial_savings=500_000.0,
            initial_monthly_saving=10_000.0,
            investment_type="bist",
            target_type="house",
            target_price=2_000_000.0,
            n_paths=1,
            n_steps=60,
            seed=42,
        )
        assert out.affordability_months.shape == (1,)

    def test_n_steps_one(self):
        """Tek adımlı simülasyon çalışmalı."""
        out = run_simulation(
            initial_savings=1_000_000.0,
            initial_monthly_saving=0.0,
            investment_type="deposit",
            target_type="car",
            target_price=500_000.0,
            n_paths=N_PATHS_SMALL,
            n_steps=1,
            seed=42,
        )
        assert out.n_steps == 1
        finite = out.affordability_months[~np.isnan(out.affordability_months)]
        if len(finite) > 0:
            assert np.all(finite == 1)  # tek adım, sadece ay-1 olabilir


# =====================================================================
# 5. run_simulation() — Geçersiz Girişler
# =====================================================================


class TestRunSimulationInvalidInputs:
    """Hatalı parametrelerde ValueError fırlatıldığını doğrular."""

    def test_unknown_investment_type_raises(self):
        """Bilinmeyen yatırım türü ValueError fırlatmalı."""
        with pytest.raises(ValueError, match="Bilinmeyen yatırım türü"):
            run_simulation(
                initial_savings=100_000.0,
                initial_monthly_saving=5_000.0,
                investment_type="crypto",    # geçersiz
                target_type="house",
                target_price=1_000_000.0,
                n_paths=10,
                n_steps=12,
            )

    def test_unknown_target_type_raises(self):
        """Bilinmeyen hedef türü ValueError fırlatmalı."""
        with pytest.raises(ValueError, match="Bilinmeyen hedef türü"):
            run_simulation(
                initial_savings=100_000.0,
                initial_monthly_saving=5_000.0,
                investment_type="gold",
                target_type="yacht",        # geçersiz
                target_price=1_000_000.0,
                n_paths=10,
                n_steps=12,
            )

    def test_negative_initial_savings_raises(self):
        """Negatif birikim ValueError fırlatmalı."""
        with pytest.raises(ValueError):
            run_simulation(
                initial_savings=-1.0,
                initial_monthly_saving=5_000.0,
                investment_type="gold",
                target_type="house",
                target_price=1_000_000.0,
                n_paths=10,
                n_steps=12,
            )

    def test_negative_target_price_raises(self):
        """Sıfır veya negatif hedef fiyat ValueError fırlatmalı."""
        with pytest.raises(ValueError):
            run_simulation(
                initial_savings=100_000.0,
                initial_monthly_saving=5_000.0,
                investment_type="gold",
                target_type="house",
                target_price=0.0,           # geçersiz
                n_paths=10,
                n_steps=12,
            )


# =====================================================================
# 6. run_simulation() — Tüm Geçerli Kombinasyonlar
# =====================================================================


class TestRunSimulationAllCombinations:
    """Her investment_type × target_type kombinasyonu çalışmalı."""

    @pytest.mark.parametrize("investment_type", ["gold", "bist", "deposit"])
    @pytest.mark.parametrize("target_type", ["house", "car"])
    def test_all_valid_combinations(self, investment_type, target_type):
        """Geçerli her kombinasyon SimulationOutput döndürmeli."""
        out = run_simulation(
            initial_savings=200_000.0,
            initial_monthly_saving=8_000.0,
            investment_type=investment_type,
            target_type=target_type,
            target_price=2_000_000.0,
            n_paths=50,
            n_steps=24,
            seed=0,
        )
        assert isinstance(out, SimulationOutput)
        assert out.affordability_months.shape == (50,)
        assert 0.0 <= out.never_affordable_pct <= 1.0
