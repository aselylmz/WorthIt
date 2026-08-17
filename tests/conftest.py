"""
Ortak test fixture'ları.
"""

import pytest

from time_to_afford.config.settings import Settings


@pytest.fixture
def default_settings() -> Settings:
    """Varsayılan konfigürasyon nesnesi."""
    return Settings()


@pytest.fixture
def sample_user_input_data() -> dict:
    """Örnek kullanıcı girdi verisi (dict olarak)."""
    return {
        "profession": "Yazılım Mühendisi",
        "initial_savings": 500_000.0,
        "investment_type": "gold",
        "target_type": "house",
        "target_price": 5_000_000.0,
        "monthly_saving": 15_000.0,
    }
