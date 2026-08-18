"""Time to Afford — Simulation modülü."""

from time_to_afford.simulation.distributions import (
    LogNormalParams,
    NormalParams,
    sample_investment_return,
    sample_inflation,
    sample_salary_growth,
    sample_target_price_growth,
    cumulative_product,
)
from time_to_afford.simulation.monte_carlo import SimulationOutput, run_simulation

__all__ = [
    "LogNormalParams",
    "NormalParams",
    "sample_investment_return",
    "sample_inflation",
    "sample_salary_growth",
    "sample_target_price_growth",
    "cumulative_product",
    "SimulationOutput",
    "run_simulation",
]
