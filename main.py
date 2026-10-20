
from pathlib import Path

import kagglehub
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUTPUT = Path("outputs/eda")
OUTPUT.mkdir(parents=True, exist_ok=True)

dataset_path = Path(
    kagglehub.dataset_download(
        "pritish509/s-and-p-500-esg-risk-ratings"
    )
)

csv_files = list(dataset_path.rglob("*.csv"))

if not csv_files:
    raise FileNotFoundError("Nie znaleziono pliku CSV.")

df = pd.read_csv(csv_files[0])

score_columns = [
    "Total ESG Risk score",
    "Environment Risk Score",
    "Governance Risk Score",
    "Social Risk Score",
    "Controversy Score"
]

for column in score_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

df["Employees"] = pd.to_numeric(
    df["Full Time Employees"]
    .astype("string")
    .str.replace(",", "", regex=False)
    .str.replace(" ", "", regex=False),
    errors="coerce"
)

df["Sector"] = df["Sector"].fillna("Unknown")
df["Industry"] = df["Industry"].fillna("Unknown")

target = "Total ESG Risk score"

df_valid = df.dropna(subset=[target]).copy()

print("\n=== PODSTAWOWE INFORMACJE ===")
print(f"Wszystkie obserwacje: {len(df)}")
print(f"Obserwacje z oceną ESG: {len(df_valid)}")
print(f"Brakujące oceny: {df[target].isna().sum()}")
print(f"Liczba sektorów: {df_valid['Sector'].nunique()}")
print(f"Liczba branż: {df_valid['Industry'].nunique()}")

print("\n=== STATYSTYKI RYZYKA ESG ===")
print(df_valid[target].describe())

print("\n=== ROZKŁAD KATEGORII RYZYKA ===")
print(df_valid["ESG Risk Level"].value_counts())

df_valid["Components Sum"] = df_valid[
    [
        "Environment Risk Score",
        "Governance Risk Score",
        "Social Risk Score"
    ]
].sum(axis=1, min_count=3)

df_valid["Components Difference"] = (
    df_valid[target] - df_valid["Components Sum"]
)

print("\n=== KONTROLA WYCIEKU DANYCH ===")
print(
    df_valid["Components Difference"]
    .abs()
    .describe()
)

print("\n=== BRAKI DANYCH ===")
missing = (
    df.isna()
    .mean()
    .mul(100)
    .sort_values(ascending=False)
)

print(missing.round(2))

df_valid.to_csv(
    OUTPUT / "esg_clean.csv",
    index=False
)

df_valid.describe().to_csv(
    OUTPUT / "statistics.csv"
)

sector_stats = (
    df_valid.groupby("Sector")[target]
    .agg(["count", "mean", "median", "std"])
    .sort_values("mean")
)

sector_stats.to_csv(
    OUTPUT / "sector_statistics.csv"
)

plt.figure(figsize=(10, 6))

plt.hist(
    df_valid[target],
    bins=25,
    edgecolor="black",
    alpha=0.8
)

plt.axvline(
    df_valid[target].mean(),
    linestyle="--",
    linewidth=2,
    label=f"Średnia: {df_valid[target].mean():.2f}"
)

plt.axvline(
    df_valid[target].median(),
    linestyle=":",
    linewidth=2,
    label=f"Mediana: {df_valid[target].median():.2f}"
)

plt.xlabel("ESG Risk Score")
plt.ylabel("Liczba przedsiębiorstw")
plt.title("Rozkład wartości ESG Risk Score")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT / "01_esg_distribution.png", dpi=300)
plt.close()

risk_order = [
    "Negligible",
    "Low",
    "Medium",
    "High",
    "Severe"
]

risk_counts = (
    df_valid["ESG Risk Level"]
    .value_counts()
    .reindex(risk_order, fill_value=0)
)

plt.figure(figsize=(10, 6))

bars = plt.bar(
    risk_counts.index,
    risk_counts.values,
    edgecolor="black"
)

for bar in bars:
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 2,
        str(int(bar.get_height())),
        ha="center"
    )

plt.xlabel("Kategoria ryzyka ESG")
plt.ylabel("Liczba przedsiębiorstw")
plt.title("Rozkład kategorii ryzyka ESG")
plt.ylim(0, max(risk_counts.values) * 1.15)
plt.tight_layout()
plt.savefig(OUTPUT / "02_esg_risk_levels.png", dpi=300)
plt.close()

sectors = sector_stats[
    sector_stats["count"] >= 10
].sort_values("mean")

plt.figure(figsize=(11, 7))

plt.barh(
    sectors.index,
    sectors["mean"],
    edgecolor="black"
)

plt.xlabel("Średni ESG Risk Score")
plt.ylabel("Sektor")
plt.title("Średni poziom ryzyka ESG według sektorów")
plt.tight_layout()
plt.savefig(OUTPUT / "03_esg_by_sector.png", dpi=300)
plt.close()

selected_sectors = (
    df_valid["Sector"]
    .value_counts()
    .loc[lambda x: x >= 10]
    .index.tolist()
)

boxplot_data = [
    df_valid.loc[
        df_valid["Sector"] == sector,
        target
    ].values
    for sector in selected_sectors
]

if boxplot_data:
    plt.figure(
        figsize=(12, max(6, len(selected_sectors) * 0.55))
    )

    plt.boxplot(
        boxplot_data,
        tick_labels=selected_sectors,
        vert=False,
        patch_artist=True
    )

    plt.xlabel("ESG Risk Score")
    plt.ylabel("Sektor")
    plt.title("Zróżnicowanie ryzyka ESG między sektorami")
    plt.tight_layout()
    plt.savefig(OUTPUT / "04_esg_sector_boxplot.png", dpi=300)
    plt.close()

selected_columns = [
    "Total ESG Risk score",
    "Environment Risk Score",
    "Social Risk Score",
    "Governance Risk Score",
    "Controversy Score",
    "Employees",
    "Sector",
    "Industry"
]

missing_selected = (
    df[selected_columns]
    .isna()
    .mean()
    .mul(100)
    .sort_values()
)

plt.figure(figsize=(10, 6))
plt.barh(
    missing_selected.index,
    missing_selected.values,
    edgecolor="black"
)

plt.xlabel("Braki danych (%)")
plt.title("Kompletność wybranych zmiennych")
plt.tight_layout()
plt.savefig(OUTPUT / "05_missing_values.png", dpi=300)
plt.close()

print("\n=== ŚREDNI ESG RISK SCORE WEDŁUG SEKTORA ===")
print(sector_stats.round(2).to_string())

print("\nAnaliza EDA zakończona.")
print(f"Wyniki zapisano w: {OUTPUT.resolve()}")
