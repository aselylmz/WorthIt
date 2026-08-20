"""
`data/processed/macro_monthly.parquet` üzerinden simülasyon dağılım
parametrelerini (mu, sigma) ve enflasyon ortak faktörü korelasyonlarını
(rho) yeniden hesaplar.

Bu script `simulation/distributions.py` içindeki sabitleri OTOMATİK
GÜNCELLEMEZ — çıktıyı insan gözden geçirsin diye yalnızca raporlar.
Yeni veri çekildikçe (`scripts/fetch_data.py`) kalibrasyonu tekrarlamak
için kullanın.

Kapsam dışı: `salary_growth_rate` (gerçek maaş verisi yok, parametrik
varsayım olarak kalır) ve `car_price` (mevcut `vehicle_price_proxy`
serisi — TP.TUFE1YI.T7 — gerçekte Yİ-ÜFE ailesine ait olduğu ve
ekonomik olarak anlamsız aylık sıçramalar içerdiği için GÜVENİLMEZ;
bkz. docs/data_sources.md § 2.5).

Kullanım:
    python scripts/calibrate_distributions.py
    python scripts/calibrate_distributions.py --start 2015-01-01
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from time_to_afford.config.settings import get_settings

DEFAULT_START = "2019-01-01"

# dağılım adı -> (macro_monthly.parquet'teki return kolonu)
# NOT: car_price / vehicle_proxy_return kasıtlı olarak dışarıda bırakıldı.
CALIBRATION_TARGETS = {
    "inflation": "cpi_return",
    "house_price": "house_price_return",
    "gold": "gold_return",
    "bist": "bist100_return",
    "deposit": "deposit_monthly_return",
}


def load_processed_dataset(processed_dir: Path | None = None) -> pd.DataFrame:
    """`macro_monthly.parquet`'i okur."""
    if processed_dir is None:
        processed_dir = get_settings().processed_data_dir
    path = Path(processed_dir) / "macro_monthly.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} bulunamadı. Önce `python scripts/fetch_data.py` çalıştırın."
        )
    return pd.read_parquet(path)


def compute_marginal_params(df: pd.DataFrame, start: str) -> dict[str, tuple[float, float]]:
    """Her hedef değişken için (mu, sigma) döndürür."""
    sub = df[df.index >= start]
    results = {}
    for name, col in CALIBRATION_TARGETS.items():
        vals = sub[col].dropna()
        results[name] = (float(vals.mean()), float(vals.std()), len(vals))
    return results


def compute_inflation_rhos(df: pd.DataFrame, start: str) -> dict[str, float]:
    """Her hedef değişkenin cpi_return ile korelasyonunu (rho) döndürür."""
    sub = df[df.index >= start]
    cpi = sub["cpi_return"]
    rhos = {}
    for name, col in CALIBRATION_TARGETS.items():
        if name == "inflation":
            continue
        paired = pd.concat([cpi, sub[col]], axis=1).dropna()
        rhos[name] = float(paired.iloc[:, 0].corr(paired.iloc[:, 1]))
    return rhos


def main(start: str) -> None:
    df = load_processed_dataset()
    end = df.index.max().date()

    marginals = compute_marginal_params(df, start)
    rhos = compute_inflation_rhos(df, start)

    print(f"Kalibrasyon penceresi: {start} -> {end}")
    print()
    print("=== Marjinal parametreler ===")
    for name, (mu, sigma, n) in marginals.items():
        print(f"  {name:12s} mu={mu:.4f}  sigma={sigma:.4f}  (n={n})")
    print()
    print("=== Enflasyon ortak faktörü korelasyonları (rho) ===")
    for name, rho in rhos.items():
        print(f"  {name:12s} rho={rho:.3f}")
    print()
    print("=== distributions.py'ye yapıştırılabilir Python literal ===")
    for name, (mu, sigma, _n) in marginals.items():
        cls = "LogNormalParams" if name in ("inflation", "house_price") else "NormalParams"
        print(f"# {name}: {cls}(mu={mu:.4f}, sigma={sigma:.4f})")
    print("DEFAULT_INFLATION_RHOS_CALIBRATED = {")
    for name, rho in rhos.items():
        print(f'    "{name}": {rho:.3f},')
    print("}")
    print()
    print(
        "NOT: 'salary_growth' ve 'car_price' bu raporda YOK — gerçek veri "
        "olmadığı / mevcut vehicle_price_proxy verisi güvenilmez olduğu için "
        "kasıtlı olarak dışarıda bırakıldı."
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--start",
        default=DEFAULT_START,
        help=f"Kalibrasyon penceresinin başlangıcı (YYYY-MM-DD). Varsayılan: {DEFAULT_START}",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(args.start)
