
from pathlib import Path

import kagglehub
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs" / "data_audit"
OUTPUT.mkdir(parents=True, exist_ok=True)


def find_csv(folder, filename):
    matches = [
        file for file in folder.rglob("*.csv")
        if file.name.casefold() == filename.casefold()
    ]

    if not matches:
        raise FileNotFoundError(
            f"Nie znaleziono {filename} w {folder}"
        )

    return matches[0]


def normalize_ticker(series):
    return (
        series.astype("string")
        .str.upper()
        .str.strip()
        .str.replace(r"[^A-Z0-9]", "", regex=True)
        .replace("", pd.NA)
    )


def unique_companies(df, name):
    df = df.dropna(subset=["ticker_key"]).copy()

    duplicate_mask = df["ticker_key"].duplicated(keep=False)

    if duplicate_mask.any():
        print(f"\nNiejednoznaczne tickery w {name}:")
        print(
            df.loc[duplicate_mask, "ticker_key"]
            .value_counts()
            .to_string()
        )
        df = df.loc[~duplicate_mask].copy()

    return df


financial_file = ROOT / "constituents-financials.csv"

if not financial_file.exists():
    raise FileNotFoundError(
        f"Nie znaleziono pliku: {financial_file}"
    )

sp500_path = Path(
    kagglehub.dataset_download(
        "pritish509/s-and-p-500-esg-risk-ratings"
    )
)

public_path = Path(
    kagglehub.dataset_download(
        "alistairking/public-company-esg-ratings-dataset"
    )
)

financial = pd.read_csv(financial_file, low_memory=False)

sp500 = pd.read_csv(
    find_csv(sp500_path, "SP 500 ESG Risk Ratings.csv"),
    low_memory=False
)

public = pd.read_csv(
    find_csv(public_path, "data.csv"),
    low_memory=False
)

for df in [financial, sp500, public]:
    df.columns = df.columns.str.strip()

financial["ticker_key"] = normalize_ticker(financial["Symbol"])
sp500["ticker_key"] = normalize_ticker(sp500["Symbol"])
public["ticker_key"] = normalize_ticker(public["ticker"])

sp500["Total ESG Risk score"] = pd.to_numeric(
    sp500["Total ESG Risk score"],
    errors="coerce"
)

public["total_score"] = pd.to_numeric(
    public["total_score"],
    errors="coerce"
)

financial = financial.rename(
    columns={
        "Symbol": "financial_ticker",
        "Name": "financial_name",
        "Sector": "financial_industry",
        "Market Cap": "market_cap",
        "EBITDA": "ebitda",
        "Price/Earnings": "pe_ratio",
        "Price/Book": "pb_ratio",
        "Earnings/Share": "eps",
        "Dividend Yield": "dividend_yield"
    }
)

sp500 = sp500.rename(
    columns={
        "Symbol": "sp500_ticker",
        "Name": "sp500_name",
        "Sector": "sp500_sector",
        "Industry": "sp500_industry",
        "Total ESG Risk score": "esg_risk_score",
        "ESG Risk Level": "esg_risk_level"
    }
)

public = public.rename(
    columns={
        "ticker": "public_ticker",
        "name": "public_name",
        "total_score": "public_esg_score",
        "total_grade": "public_esg_grade"
    }
)

financial_columns = [
    "ticker_key",
    "financial_ticker",
    "financial_name",
    "financial_industry",
    "market_cap",
    "ebitda",
    "pe_ratio",
    "pb_ratio",
    "eps",
    "dividend_yield"
]

sp500_columns = [
    "ticker_key",
    "sp500_ticker",
    "sp500_name",
    "sp500_sector",
    "sp500_industry",
    "esg_risk_score",
    "esg_risk_level"
]

public_columns = [
    "ticker_key",
    "public_ticker",
    "public_name",
    "public_esg_score",
    "public_esg_grade"
]

financial_data = unique_companies(
    financial[financial_columns],
    "Financial"
)

sp500_data = unique_companies(
    sp500[sp500_columns],
    "S&P 500 ESG"
)

public_data = unique_companies(
    public[public_columns],
    "Public ESG"
)

matches = (
    financial_data
    .merge(
        sp500_data,
        on="ticker_key",
        how="left",
        validate="one_to_one"
    )
    .merge(
        public_data,
        on="ticker_key",
        how="left",
        validate="one_to_one"
    )
)

matches["has_sp500_esg"] = (
    matches["esg_risk_score"].notna()
)

matches["has_public_esg"] = (
    matches["public_esg_score"].notna()
)

matches["has_both_esg"] = (
    matches["has_sp500_esg"]
    & matches["has_public_esg"]
)

matches["sector_key"] = (
    matches["sp500_sector"]
    .astype("string")
    .str.strip()
    .str.casefold()
)

financial_sector_names = {
    "financial services",
    "financials"
}

financials = matches[
    matches["sector_key"].isin(financial_sector_names)
].copy()

complete_financials = financials[
    financials["has_both_esg"]
].copy()

print("\n" + "=" * 70)
print("ŹRÓDŁA DANYCH")
print("=" * 70)

print("Finanse:", len(financial_data))
print("S&P 500 ESG:", len(sp500_data))
print("Public ESG:", len(public_data))

print("\n=== NAZWY SEKTORÓW Z BAZY ESG ===")
print(
    matches["sp500_sector"]
    .value_counts(dropna=False)
    .to_string()
)

print("\n" + "=" * 70)
print("DOPASOWANIE WSZYSTKICH SPÓŁEK")
print("=" * 70)

print(
    "Z oceną S&P ESG:",
    int(matches["has_sp500_esg"].sum())
)

print(
    "Z oceną Public ESG:",
    int(matches["has_public_esg"].sum())
)

print(
    "Z obiema ocenami:",
    int(matches["has_both_esg"].sum())
)

print("\n" + "=" * 70)
print("SEKTOR FINANSOWY")
print("=" * 70)

print("Wszystkie firmy finansowe:", len(financials))

print(
    "Z oceną S&P ESG:",
    int(financials["has_sp500_esg"].sum())
)

print(
    "Z oceną Public ESG:",
    int(financials["has_public_esg"].sum())
)

print(
    "Z obiema ocenami ESG:",
    len(complete_financials)
)

print("\n=== BRANŻE W SEKTORZE FINANSOWYM ===")

print(
    financials["financial_industry"]
    .value_counts()
    .to_string()
)

print("\n=== DOPASOWANE SPÓŁKI FINANSOWE ===")

print(
    complete_financials[
        [
            "financial_ticker",
            "financial_name",
            "financial_industry",
            "market_cap",
            "ebitda",
            "esg_risk_score",
            "public_esg_score",
            "public_esg_grade"
        ]
    ].to_string(index=False)
)

print("\n=== BRAKI DANYCH FINANSOWYCH ===")

features = [
    "market_cap",
    "ebitda",
    "pe_ratio",
    "pb_ratio",
    "eps",
    "dividend_yield"
]

print(
    complete_financials[features]
    .isna()
    .sum()
    .to_string()
)

matches.to_csv(
    OUTPUT / "financial_all_matches.csv",
    index=False
)

financials.to_csv(
    OUTPUT / "financial_sector_matches.csv",
    index=False
)

complete_financials.to_csv(
    OUTPUT / "financial_complete_matches.csv",
    index=False
)

print("\nAnaliza zakończona.")
print("Wyniki zapisano w:", OUTPUT.resolve())
