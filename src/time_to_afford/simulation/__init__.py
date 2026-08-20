"""Time to Afford — Simulation modülü."""

from time_to_afford.simulation.distributions import (
    DEFAULT_INFLATION_RHOS,
    LogNormalParams,
    NormalParams,
    cumulative_product,
    sample_common_factor_shocks,
    sample_correlated_variables,
    sample_inflation,
    sample_investment_return,
    sample_salary_growth,
    sample_target_price_growth,
)
from time_to_afford.simulation.monte_carlo import SimulationOutput, run_simulation
from time_to_afford.simulation.scenarios import (
    BASELINE,
    OPTIMISTIC,
    PESSIMISTIC,
    PRESETS,
    ScenarioParams,
    get_preset,
    run_scenario,
)

__all__ = [
    "LogNormalParams",
    "NormalParams",
    "sample_investment_return",
    "sample_inflation",
    "sample_salary_growth",
    "sample_target_price_growth",
    "sample_common_factor_shocks",
    "sample_correlated_variables",
    "DEFAULT_INFLATION_RHOS",
    "cumulative_product",
    "SimulationOutput",
    "run_simulation",
    "ScenarioParams",
    "BASELINE",
    "OPTIMISTIC",
    "PESSIMISTIC",
    "PRESETS",
    "run_scenario",
    "get_preset",
]
