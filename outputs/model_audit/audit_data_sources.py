
from pathlib import Path
import hashlib

import kagglehub
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs" / "source_audit"
OUTPUT.mkdir(parents=True, exist_ok=True)

financial_file = ROOT / "constituents-financials.csv"

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


def find_file(folder, filename):
    files = [
        path for path in folder.rglob("*.csv")
        if path.name.casefold() == filename.casefold()
    ]

    if len(files) != 1:
        raise FileNotFoundError(
            f"Oczekiwano jednego pliku {filename}. "
            f"Znaleziono: {files}"
        )

    return files[0]


def file_hash(path):
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def read_dataset(path):
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip()

    return df


sp500_file = find_file(
    sp500_path,
    "SP 500 ESG Risk Ratings.csv"
)

public_file = find_file(
    public_path,
    "data.csv"
)

datasets = {
    "Financial Information": financial_file,
    "S&P 500 ESG Risk Ratings": sp500_file,
    "Public Company ESG Ratings": public_file
}

frames = {}
records = []

for name, path in datasets.items():
    df = read_dataset(path)
    frames[name] = df

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    print("Plik:", path)
    print("Liczba wierszy:", len(df))
    print("Liczba kolumn:", len(df.columns))
    print("SHA256:", file_hash(path))

    print("\nKOLUMNY")
    print(df.columns.tolist())

    date_columns = [
        column for column in df.columns
        if any(
            word in column.casefold()
            for word in [
                "date",
                "year",
                "period",
                "fiscal",
                "updated"
            ]
        )
    ]

    print("\nKOLUMNY DOTYCZĄCE DAT")
    print(date_columns)

    if date_columns:
        for column in date_columns:
            dates = pd.to_datetime(
                df[column],
                errors="coerce",
                utc=True
            )

            print(f"\n{column}")
            print("Najstarsza data:", dates.min())
            print("Najnowsza data:", dates.max())
            print("Liczba poprawnych dat:", dates.notna().sum())

            if dates.notna().any():
                print(
                    "Rozkład lat:\n",
                    dates.dt.year.value_counts()
                    .sort_index()
                    .to_string()
                )
    else:
        print("Brak kolumn określających datę obserwacji.")

    records.append({
        "dataset": name,
        "filename": path.name,
        "rows": len(df),
        "columns": len(df.columns),
        "date_columns": ", ".join(date_columns),
        "sha256": file_hash(path)
    })

financial = frames["Financial Information"]
sp500 = frames["S&P 500 ESG Risk Ratings"]
public = frames["Public Company ESG Ratings"]

print("\n" + "=" * 70)
print("KONTROLA DANYCH FINANSOWYCH")
print("=" * 70)

selected_tickers = [
    "MMM",
    "AOS",
    "JPM",
    "BAC",
    "MSFT"
]

financial_examples = financial[
    financial["Symbol"].isin(selected_tickers)
].copy()

financial_columns = [
    column for column in [
        "Symbol",
        "Name",
        "Sector",
        "Price",
        "Market Cap",
        "EBITDA",
        "Price/Earnings",
        "Price/Book"
    ]
    if column in financial_examples.columns
]

print(
    financial_examples[financial_columns]
    .to_string(index=False)
)

print("\n" + "=" * 70)
print("DATY PUBLIC ESG")
print("=" * 70)

if "last_processing_date" in public.columns:
    public_dates = pd.to_datetime(
        public["last_processing_date"],
        errors="coerce",
        utc=True
    )

    print("Minimum:", public_dates.min())
    print("Maksimum:", public_dates.max())

    print("\nLiczba obserwacji według roku:")
    print(
        public_dates.dt.year
        .value_counts(dropna=False)
        .sort_index()
        .to_string()
    )

print("\n" + "=" * 70)
print("KONTROLA PRÓBY BADAWCZEJ")
print("=" * 70)

sample_file = (
    ROOT
    / "outputs"
    / "data_audit"
    / "financial_sector_matches.csv"
)

if sample_file.exists():
    sample = pd.read_csv(sample_file)

    print("Wszystkie dopasowane firmy finansowe:", len(sample))

    for column in [
        "has_sp500_esg",
        "has_public_esg",
        "has_both_esg"
    ]:
        if column in sample.columns:
            values = (
                sample[column]
                .astype("string")
                .str.lower()
                .eq("true")
            )

            print(column, int(values.sum()))

    if "market_cap" in sample.columns:
        print("\nKapitalizacja rynkowa:")
        print(
            pd.to_numeric(
                sample["market_cap"],
                errors="coerce"
            ).describe().to_string()
        )

    if "esg_risk_score" in sample.columns:
        print("\nOceny ryzyka ESG:")
        print(
            pd.to_numeric(
                sample["esg_risk_score"],
                errors="coerce"
            ).describe().to_string()
        )
else:
    print("Brak pliku próby badawczej.")

metadata = pd.DataFrame(records)

metadata.to_csv(
    OUTPUT / "source_provenance.csv",
    index=False
)

financial_examples.to_csv(
    OUTPUT / "financial_source_examples.csv",
    index=False
)

print("\n" + "=" * 70)
print("AUDYT ZAKOŃCZONY")
print("=" * 70)
print("Wyniki:", OUTPUT.resolve())
print("Data modyfikacji pliku nie jest datą obserwacji.")
