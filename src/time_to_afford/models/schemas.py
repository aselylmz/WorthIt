"""
Pydantic veri şemaları.

API ve iç modüller arası veri aktarımında kullanılan
doğrulanmış veri yapıları.
"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class InvestmentType(str, Enum):
    """Desteklenen yatırım araçları."""

    GOLD = "gold"
    BIST = "bist"
    DEPOSIT = "deposit"  # Mevduat


class TargetType(str, Enum):
    """Hedef varlık türleri."""

    HOUSE = "house"
    CAR = "car"


class UserInput(BaseModel):
    """Kullanıcıdan alınan girdiler.

    API endpoint'inin beklediği request body şeması.
    """

    profession: str = Field(
        ...,
        min_length=1,
        description="Kullanıcının mesleği.",
    )
    initial_savings: float = Field(
        ...,
        ge=0,
        description="Mevcut birikim miktarı (TL).",
    )
    investment_type: InvestmentType = Field(
        ...,
        description="Birikimin bulunduğu yatırım aracı.",
    )
    target_type: TargetType = Field(
        ...,
        description="Hedef varlık türü (ev veya araba).",
    )
    target_price: float = Field(
        ...,
        gt=0,
        description="Hedef varlığın mevcut fiyatı (TL).",
    )
    monthly_saving: float = Field(
        ...,
        ge=0,
        description="Aylık düzenli tasarruf miktarı (TL).",
    )


class ScenarioResult(BaseModel):
    """Tek bir senaryo sonucu."""

    label: str = Field(..., description="Senaryo adı (ör: 'Temel').")
    years: int = Field(..., ge=0, description="Tahmini yıl.")
    months: int = Field(..., ge=0, lt=12, description="Tahmini ay (yıl üstü).")
    total_months: int = Field(..., ge=0, description="Toplam ay cinsinden süre.")


class SimulationResult(BaseModel):
    """Monte Carlo simülasyonu sonuç özeti."""

    median_time: ScenarioResult
    optimistic_time: ScenarioResult
    pessimistic_time: ScenarioResult

    probability_5y: float = Field(
        ..., ge=0, le=1, description="5 yıl içinde alma olasılığı."
    )
    probability_10y: float = Field(
        ..., ge=0, le=1, description="10 yıl içinde alma olasılığı."
    )
    probability_15y: float = Field(
        ..., ge=0, le=1, description="15 yıl içinde alma olasılığı."
    )

    num_simulations: int = Field(..., gt=0, description="Toplam simülasyon sayısı.")
    never_affordable_pct: float = Field(
        ...,
        ge=0,
        le=1,
        description="Simülasyon ufku içinde hedefe hiç ulaşamayan senaryoların oranı.",
    )

    disclaimer: str = Field(
        default=(
            "Bu sonuçlar istatistiksel bir tahmindir ve kesin bir finansal "
            "tavsiye veya garanti niteliği taşımaz. Gerçek sonuçlar ekonomik "
            "koşullara bağlı olarak farklılık gösterebilir."
        ),
        description="Yasal uyarı.",
    )
