
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "outputs" / "data_audit" / "financial_sector_matches.csv"
OUTPUT = ROOT / "outputs" / "model_audit"
OUTPUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT)

target = "esg_risk_score"

features = [
    "market_cap",
    "pe_ratio",
    "pb_ratio",
    "eps",
    "dividend_yield"
]

numeric_columns = features + [
    "ebitda",
    "esg_risk_score",
    "public_esg_score"
]

for column in numeric_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

model_df = df.dropna(subset=[target]).copy()

model_df.replace(
    [np.inf, -np.inf],
    np.nan,
    inplace=True
)

model_df["log_market_cap"] = np.where(
    model_df["market_cap"] > 0,
    np.log10(model_df["market_cap"].clip(lower=1)),
    np.nan
)

print("\n=== PRÓBA BADAWCZA ===")
print("Wszystkie spółki finansowe:", len(df))
print("Spółki z oceną ESG:", len(model_df))
print("Unikalne tickery:", model_df["ticker_key"].nunique())

print("\n=== STATYSTYKI ZMIENNEJ DOCELOWEJ ===")
print(model_df[target].describe().round(3))

print("\n=== KOMPLETNOŚĆ PREDYKTORÓW ===")
print(
    model_df[features]
    .isna()
    .agg(["sum", "mean"])
    .to_string()
)

print("\n=== KOMPLETNE OBSERWACJE ===")
complete_mask = model_df[features].notna().all(axis=1)

print(
    "Spółki ze wszystkimi pięcioma predyktorami:",
    int(complete_mask.sum())
)

print("\n=== ROZKŁAD BRANŻ ===")
industry_stats = (
    model_df.groupby("financial_industry")[target]
    .agg(["count", "mean", "median", "std"])
    .sort_values("count", ascending=False)
)

print(industry_stats.round(2).to_string())

print("\n=== KORELACJE SPEARMANA ===")
correlations = (
    model_df[features + [target]]
    .corr(method="spearman")[target]
    .drop(target)
    .sort_values()
)

print(correlations.round(3).to_string())

model_df.to_csv(
    OUTPUT / "financial_esg_candidates.csv",
    index=False
)

industry_stats.to_csv(
    OUTPUT / "industry_statistics.csv"
)

correlations.to_csv(
    OUTPUT / "spearman_correlations.csv"
)

plt.figure(figsize=(9, 5))

plt.hist(
    model_df[target],
    bins=15,
    edgecolor="black"
)

plt.xlabel("ESG Risk Score")
plt.ylabel("Liczba przedsiębiorstw")
plt.title("Rozkład ryzyka ESG w sektorze finansowym")
plt.tight_layout()
plt.savefig(
    OUTPUT / "01_financial_esg_distribution.png",
    dpi=300
)
plt.close()

valid_cap = model_df.dropna(
    subset=["log_market_cap", target]
)

plt.figure(figsize=(9, 5))

plt.scatter(
    valid_cap["log_market_cap"],
    valid_cap[target],
    alpha=0.75
)

plt.xlabel("log10(kapitalizacja rynkowa)")
plt.ylabel("ESG Risk Score")
plt.title("Kapitalizacja a ocena ryzyka ESG")
plt.tight_layout()
plt.savefig(
    OUTPUT / "02_market_cap_vs_esg.png",
    dpi=300
)
plt.close()

valid_industries = industry_stats[
    industry_stats["count"] >= 3
].index.tolist()

boxplot_data = [
    model_df.loc[
        model_df["financial_industry"] == industry,
        target
    ].dropna().to_numpy()
    for industry in valid_industries
]

if boxplot_data:
    plt.figure(figsize=(11, 7))

    plt.boxplot(
        boxplot_data,
        tick_labels=valid_industries,
        vert=False
    )

    plt.xlabel("ESG Risk Score")
    plt.ylabel("Branża")
    plt.title("Zróżnicowanie ryzyka ESG w branżach finansowych")
    plt.tight_layout()
    plt.savefig(
        OUTPUT / "03_industry_risk_boxplot.png",
        dpi=300
    )
    plt.close()

print("\n=== ANALIZA ZAKOŃCZONA ===")
print("Wyniki:", OUTPUT.resolve())
