"""
Uygulama konfigürasyonu.

Tüm ayarlar environment variable veya .env dosyasından okunur.
pydantic-settings (v2) kullanılır.
"""

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


# Proje kök dizini: WorthIt/
_PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    """Uygulama genelinde kullanılan konfigürasyon."""

    model_config = SettingsConfigDict(
        env_file=str(_PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Logging ──────────────────────────────────────────────
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    # ── Data Paths ───────────────────────────────────────────
    data_dir: Path = _PROJECT_ROOT / "data"
    raw_data_dir: Path = _PROJECT_ROOT / "data" / "raw"
    processed_data_dir: Path = _PROJECT_ROOT / "data" / "processed"
    external_data_dir: Path = _PROJECT_ROOT / "data" / "external"

    # ── Simulation ───────────────────────────────────────────
    simulation_num_paths: int = 10_000
    simulation_random_seed: int = 42
    simulation_horizon_years: int = 30


def get_settings() -> Settings:
    """Singleton-benzeri settings erişimi.

    Her çağrıda yeni bir Settings nesnesi oluşturur, böylece
    test sırasında environment override yapılabilir.
    """
    return Settings()
