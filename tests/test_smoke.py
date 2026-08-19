"""
Smoke testleri.

Package'ın import edilebilir olduğunu, konfigürasyonun çalıştığını
ve şemaların geçerli olduğunu doğrular.
"""

from datetime import date


class TestPackageImport:
    """Package import smoke testleri."""

    def test_import_root_package(self):
        """Root package import edilebilir olmalı."""
        import time_to_afford

        assert hasattr(time_to_afford, "__version__")
        assert hasattr(time_to_afford, "__app_name__")

    def test_version_format(self):
        """Version string geçerli formatta olmalı."""
        from time_to_afford import __version__

        parts = __version__.split(".")
        assert len(parts) == 3, f"Geçersiz version formatı: {__version__}"
        for part in parts:
            assert part.isdigit(), f"Version parçası sayısal değil: {part}"

    def test_import_config(self):
        """Config modülü import edilebilir olmalı."""
        from time_to_afford.config.settings import Settings, get_settings

        assert Settings is not None
        assert callable(get_settings)

    def test_import_schemas(self):
        """Schema modülü import edilebilir olmalı."""
        from time_to_afford.models.schemas import (
            InvestmentType,
            SimulationResult,
            TargetType,
            UserInput,
        )

        assert UserInput is not None
        assert SimulationResult is not None

    def test_import_forecasting_base(self):
        """Forecasting base modülü import edilebilir olmalı."""
        from time_to_afford.forecasting.base import BaseForecaster

        assert BaseForecaster is not None

    def test_import_utils(self):
        """Utils modülleri import edilebilir olmalı."""
        from time_to_afford.utils.dates import (
            add_months,
            format_duration_turkish,
            months_to_years_months,
        )
        from time_to_afford.utils.logging import get_logger, setup_logging

        assert callable(months_to_years_months)
        assert callable(setup_logging)

    def test_import_subpackages(self):
        """Tüm alt paketler import edilebilir olmalı."""
        import time_to_afford.data
        import time_to_afford.forecasting
        import time_to_afford.simulation
        import time_to_afford.affordability
        import time_to_afford.models
        import time_to_afford.utils


class TestConfig:
    """Konfigürasyon smoke testleri."""

    def test_settings_creation(self):
        """Settings nesnesi oluşturulabilmeli."""
        from time_to_afford.config.settings import get_settings

        settings = get_settings()
        assert settings is not None

    def test_settings_defaults(self, default_settings):
        """Varsayılan ayarlar makul değerlere sahip olmalı."""
        assert default_settings.log_level == "INFO"
        assert default_settings.simulation_num_paths == 10_000
        assert default_settings.simulation_random_seed == 42
        assert default_settings.simulation_horizon_years == 30

    def test_data_dirs_are_paths(self, default_settings):
        """Data dizin ayarları Path nesnesi olmalı."""
        from pathlib import Path

        assert isinstance(default_settings.data_dir, Path)
        assert isinstance(default_settings.raw_data_dir, Path)
        assert isinstance(default_settings.processed_data_dir, Path)
        assert isinstance(default_settings.external_data_dir, Path)


class TestSchemas:
    """Pydantic schema smoke testleri."""

    def test_valid_user_input(self, sample_user_input_data):
        """Geçerli kullanıcı girdisi kabul edilmeli."""
        from time_to_afford.models.schemas import UserInput

        user_input = UserInput(**sample_user_input_data)
        assert user_input.profession == "Yazılım Mühendisi"
        assert user_input.initial_savings == 500_000.0
        assert user_input.target_price == 5_000_000.0

    def test_invalid_target_price_rejected(self, sample_user_input_data):
        """Sıfır veya negatif hedef fiyat reddedilmeli."""
        import pytest
        from time_to_afford.models.schemas import UserInput

        sample_user_input_data["target_price"] = 0
        with pytest.raises(Exception):
            UserInput(**sample_user_input_data)

    def test_negative_savings_rejected(self, sample_user_input_data):
        """Negatif birikim reddedilmeli."""
        import pytest
        from time_to_afford.models.schemas import UserInput

        sample_user_input_data["initial_savings"] = -100
        with pytest.raises(Exception):
            UserInput(**sample_user_input_data)

    def test_investment_type_enum(self):
        """InvestmentType enum değerleri doğru olmalı."""
        from time_to_afford.models.schemas import InvestmentType

        assert InvestmentType.GOLD == "gold"
        assert InvestmentType.BIST == "bist"
        assert InvestmentType.DEPOSIT == "deposit"

    def test_target_type_enum(self):
        """TargetType enum değerleri doğru olmalı."""
        from time_to_afford.models.schemas import TargetType

        assert TargetType.HOUSE == "house"
        assert TargetType.CAR == "car"


class TestUtils:
    """Utils smoke testleri."""

    def test_months_to_years_months(self):
        """100 ay = 8 yıl 4 ay olmalı."""
        from time_to_afford.utils.dates import months_to_years_months

        years, months = months_to_years_months(100)
        assert years == 8
        assert months == 4

    def test_months_to_years_months_zero(self):
        """0 ay = 0 yıl 0 ay olmalı."""
        from time_to_afford.utils.dates import months_to_years_months

        years, months = months_to_years_months(0)
        assert years == 0
        assert months == 0

    def test_months_to_years_months_negative_raises(self):
        """Negatif ay değeri hata vermeli."""
        import pytest
        from time_to_afford.utils.dates import months_to_years_months

        with pytest.raises(ValueError):
            months_to_years_months(-5)

    def test_add_months(self):
        """Ay ekleme doğru çalışmalı."""
        from time_to_afford.utils.dates import add_months

        result = add_months(date(2025, 1, 15), 3)
        assert result == date(2025, 4, 15)

    def test_add_months_year_overflow(self):
        """Yıl taşması doğru çalışmalı."""
        from time_to_afford.utils.dates import add_months

        result = add_months(date(2025, 11, 15), 3)
        assert result == date(2026, 2, 15)

    def test_add_months_end_of_month(self):
        """Ay sonu taşması düzgün yuvarlanmalı."""
        from time_to_afford.utils.dates import add_months

        result = add_months(date(2025, 1, 31), 1)
        assert result == date(2025, 2, 28)

    def test_format_duration_turkish(self):
        """Türkçe formatlama doğru çalışmalı."""
        from time_to_afford.utils.dates import format_duration_turkish

        assert format_duration_turkish(8, 4) == "8 yıl 4 ay"
        assert format_duration_turkish(3, 0) == "3 yıl"
        assert format_duration_turkish(0, 7) == "7 ay"
        assert format_duration_turkish(0, 0) == "0 ay"

    def test_get_logger(self):
        """Logger oluşturulabilmeli."""
        from time_to_afford.utils.logging import get_logger

        logger = get_logger("test")
        assert logger.name == "time_to_afford.test"


class TestForecasting:
    """Forecasting base class smoke testleri."""

    def test_base_forecaster_is_abstract(self):
        """BaseForecaster doğrudan instantiate edilememeli."""
        import pytest
        from time_to_afford.forecasting.base import BaseForecaster

        with pytest.raises(TypeError):
            BaseForecaster()

    def test_base_forecaster_repr(self):
        """BaseForecaster alt sınıfı repr desteği vermeli."""
        from time_to_afford.forecasting.base import BaseForecaster
        import pandas as pd
        import numpy as np

        class DummyForecaster(BaseForecaster):
            def fit(self, y, **kwargs):
                self._is_fitted = True
                return self

            def predict(self, horizon, **kwargs):
                return pd.Series([0.0] * horizon)

            def predict_interval(self, horizon, alpha=0.05, **kwargs):
                return pd.DataFrame({
                    "lower": [0.0] * horizon,
                    "upper": [0.0] * horizon,
                })

            def simulate_paths(self, horizon, n_paths, random_state=None):
                return np.zeros((n_paths, horizon))

        fc = DummyForecaster(name="test")
        assert "test" in repr(fc)
        assert "False" in repr(fc)

        fc.fit(pd.Series([1, 2, 3]))
        assert fc.is_fitted
        assert "True" in repr(fc)

    def test_check_is_fitted_raises(self):
        """Eğitilmemiş model predict'te hata vermeli."""
        import pytest
        from time_to_afford.forecasting.base import BaseForecaster
        import pandas as pd

        class DummyForecaster(BaseForecaster):
            def fit(self, y, **kwargs):
                self._is_fitted = True
                return self

            def predict(self, horizon, **kwargs):
                self._check_is_fitted()
                return pd.Series([0.0] * horizon)

            def predict_interval(self, horizon, alpha=0.05, **kwargs):
                self._check_is_fitted()
                return pd.DataFrame()

            def simulate_paths(self, horizon, n_paths, random_state=None):
                return np.zeros((n_paths, horizon))

        fc = DummyForecaster(name="test")
        with pytest.raises(RuntimeError, match="eğitilmedi"):
            fc.predict(5)
