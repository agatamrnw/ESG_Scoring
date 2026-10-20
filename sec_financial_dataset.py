
from pathlib import Path
import re
import time

import kagglehub
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs" / "sec_financial"
OUTPUT.mkdir(parents=True, exist_ok=True)

SAMPLE_FILE = (
    ROOT / "outputs" / "data_audit"
    / "financial_sector_matches.csv"
)

if not SAMPLE_FILE.exists():
    raise FileNotFoundError(SAMPLE_FILE)

sample = pd.read_csv(SAMPLE_FILE)

sample["esg_risk_score"] = pd.to_numeric(
    sample["esg_risk_score"], errors="coerce"
)

sample["public_esg_score"] = pd.to_numeric(
    sample["public_esg_score"], errors="coerce"
)

sample = sample.dropna(
    subset=["esg_risk_score", "public_esg_score"]
).copy()

path = Path(
    kagglehub.dataset_download(
        "alistairking/public-company-esg-ratings-dataset"
    )
)

files = sorted(path.rglob("data.csv"))

if not files:
    raise FileNotFoundError("Nie znaleziono data.csv")

public = pd.read_csv(
    files[0],
    dtype={"ticker": "string", "cik": "string"}
)

public["ticker_key"] = (
    public["ticker"].str.upper().str.strip()
    .str.replace(r"[^A-Z0-9]", "", regex=True)
)

public["processing_date"] = pd.to_datetime(
    public["last_processing_date"],
    format="%d-%m-%Y",
    errors="coerce"
)

public_info = public[
    ["ticker_key", "cik", "processing_date"]
].drop_duplicates(subset="ticker_key")

sample = sample.merge(
    public_info,
    on="ticker_key",
    how="left",
    validate="one_to_one"
)


def normalize_cik(value):
    if pd.isna(value):
        return None

    value = re.sub(r"\.0+$", "", str(value).strip())

    if not value.isdigit():
        return None

    return value.zfill(10)


sample["cik_clean"] = sample["cik"].apply(normalize_cik)

email = input("Podaj e-mail kontaktowy do SEC: ").strip()

if "@" not in email:
    raise ValueError("Niepoprawny adres e-mail")

session = requests.Session()
session.headers.update({
    "User-Agent": f"ESG Scoring Academic Research {email}",
    "Accept": "application/json"
})


def fetch_json(url):
    response = session.get(url, timeout=45)
    response.raise_for_status()
    return response.json()


def extract_fact(data, tags, year, cutoff, duration):
    facts = data.get("facts", {}).get("us-gaap", {})
    candidates = []

    for tag in tags:
        observations = (
            facts.get(tag, {})
            .get("units", {})
            .get("USD", [])
        )

        for item in observations:
            if item.get("form") not in ["10-K", "10-K/A"]:
                continue

            end = pd.to_datetime(
                item.get("end"), errors="coerce"
            )
            filed = pd.to_datetime(
                item.get("filed"), errors="coerce"
            )

            if pd.isna(end) or pd.isna(filed):
                continue

            if end.year != year or filed > cutoff:
                continue

            if duration:
                start = pd.to_datetime(
                    item.get("start"), errors="coerce"
                )

                if pd.isna(start):
                    continue

                if not 280 <= (end - start).days <= 380:
                    continue

            candidates.append({
                "value": item.get("val"),
                "tag": tag,
                "filed": filed,
                "period_end": end,
                "accession": item.get("accn")
            })

    if not candidates:
        return None

    candidates.sort(
        key=lambda x: (x["filed"], x["period_end"]),
        reverse=True
    )

    return candidates[0]


metrics = {
    "assets_2020": (["Assets"], 2020, False),
    "assets_2021": (["Assets"], 2021, False),
    "equity_2020": (
        ["StockholdersEquity"], 2020, False
    ),
    "equity_2021": (
        ["StockholdersEquity"], 2021, False
    ),
    "net_income_2021": (
        ["NetIncomeLoss", "ProfitLoss"], 2021, True
    )
}

results = []

print("\nLiczba przedsiębiorstw:", len(sample))

for i, (_, company) in enumerate(sample.iterrows(), 1):
    ticker = company["ticker_key"]
    cik = company["cik_clean"]
    cutoff = company["processing_date"]

    print(f"\n[{i}/{len(sample)}] {ticker}")

    row = {
        "ticker": ticker,
        "financial_name": company["financial_name"],
        "financial_industry": company["financial_industry"],
        "cik": cik,
        "esg_risk_score": company["esg_risk_score"],
        "public_esg_score": company["public_esg_score"],
        "processing_date": (
            cutoff.date() if pd.notna(cutoff) else None
        ),
        "status": "pending"
    }

    if cik is None or pd.isna(cutoff):
        row["status"] = "missing_cik_or_date"
        results.append(row)
        continue

    try:
        facts_url = (
            "https://data.sec.gov/api/xbrl/"
            f"companyfacts/CIK{cik}.json"
        )

        submissions_url = (
            f"https://data.sec.gov/submissions/CIK{cik}.json"
        )

        facts = fetch_json(facts_url)
        time.sleep(0.25)
        submissions = fetch_json(submissions_url)

        row["sec_registered_name"] = submissions.get("name")
        row["sec_companyfacts_name"] = facts.get("entityName")
        row["sec_tickers"] = ", ".join(
            submissions.get("tickers", [])
        )
        row["status"] = "OK"

        for metric, (tags, year, duration) in metrics.items():
            fact = extract_fact(
                facts, tags, year, cutoff, duration
            )

            row[f"{metric}_usd"] = (
                fact["value"] if fact else None
            )
            row[f"{metric}_filed"] = (
                fact["filed"].date() if fact else None
            )
            row[f"{metric}_period_end"] = (
                fact["period_end"].date() if fact else None
            )
            row[f"{metric}_tag"] = (
                fact["tag"] if fact else None
            )
            row[f"{metric}_accession"] = (
                fact["accession"] if fact else None
            )

        print("SEC:", row["sec_registered_name"])
        print(
            "Dostępne wskaźniki:",
            sum(
                pd.notna(row.get(f"{metric}_usd"))
                for metric in metrics
            ),
            "/ 5"
        )

    except (requests.RequestException, ValueError) as error:
        row["status"] = f"error: {error}"
        print(row["status"])

    results.append(row)
    time.sleep(0.35)

df = pd.DataFrame(results)

for metric in metrics:
    column = f"{metric}_usd"

    if column not in df.columns:
        df[column] = pd.NA

    df[column] = pd.to_numeric(
        df[column], errors="coerce"
    )


def average_ratio(numerator, first, second):
    denominator = (first + second) / 2

    return (
        (numerator / denominator * 100)
        .where(denominator > 0)
    )


df["roa_2021_pct"] = average_ratio(
    df["net_income_2021_usd"],
    df["assets_2020_usd"],
    df["assets_2021_usd"]
)

df["roe_2021_pct"] = average_ratio(
    df["net_income_2021_usd"],
    df["equity_2020_usd"],
    df["equity_2021_usd"]
)

df["equity_to_assets_2021"] = (
    df["equity_2021_usd"] / df["assets_2021_usd"]
)

df.to_csv(
    OUTPUT / "financial_esg_sec_2021.csv",
    index=False
)

print("\n=== STATUS POBIERANIA ===")
print(df["status"].value_counts().to_string())

print("\n=== KOMPLETNOŚĆ ZMIENNYCH ===")

columns = [
    "assets_2021_usd",
    "equity_2021_usd",
    "net_income_2021_usd",
    "roa_2021_pct",
    "roe_2021_pct",
    "equity_to_assets_2021"
]

print(df[columns].notna().sum().to_string())

print("\n=== WYNIKI ===")
print(
    df[
        [
            "ticker",
            "status",
            "roa_2021_pct",
            "roe_2021_pct",
            "esg_risk_score"
        ]
    ].round(3).to_string(index=False)
)

print("\nZapisano:")
print(OUTPUT / "financial_esg_sec_2021.csv")
