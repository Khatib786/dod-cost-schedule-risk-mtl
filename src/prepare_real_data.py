
# Raw FPDSData.csv is transaction-level: one base award and later modifications per contract.
# Link records using Agency ID + Office ID + PIID because PIID alone is not unique.
# For each contract, use the earliest Date Signed as the base record and the latest as the current record.
# Cost growth = (latest value - base value) / base value.
# Schedule slip = latest estimated completion date - base completion date.
# Some linkage imperfections exist in the real data and are documented as a methodology limitation.


import pandas as pd
import numpy as np

RAW_COLUMNS = {
    "agency": "Contracting Agency ID",
    "office": "Contracting Office ID",
    "piid": "PIID",
    "date_signed": "Date Signed",
    "effective_date": "Effective Date",
    "completion_date": "Completion Date",
    "ultimate_completion_date": "Est. Ultimate Completion Date",
    "total_value": "Base and All Options Value (Total Contract Value)",
    "fiscal_year": "Fiscal Year",
    "military_branch": "Major Command Name",
    "naics_code": "NAICS Code",
    "psc_code": "Product or Service Code",
    "contract_type": "Type of Contract",
    "extent_competed": "Extent Competed",
    "number_of_offers": "Number of Offers Received",
    "place_state": "Principal Place of Performance State Code",
}


def prepare(raw_path: str, out_path: str):
    print(f"Loading raw transaction file {raw_path} ...")
    df = pd.read_csv(raw_path, low_memory=False, usecols=list(RAW_COLUMNS.values()))
    df = df.rename(columns={v: k for k, v in RAW_COLUMNS.items()})
    print(f"  {len(df):,} raw transaction rows")

    df["date_signed"] = pd.to_datetime(df["date_signed"], errors="coerce")
    df["effective_date"] = pd.to_datetime(df["effective_date"], errors="coerce")
    df["completion_date"] = pd.to_datetime(df["completion_date"], errors="coerce")
    df["ultimate_completion_date"] = pd.to_datetime(df["ultimate_completion_date"], errors="coerce")
    df["total_value"] = pd.to_numeric(df["total_value"], errors="coerce")

    df["key"] = (df["agency"].astype(str) + "_" + df["office"].astype(str) + "_" + df["piid"].astype(str))
    df = df.dropna(subset=["date_signed", "total_value"])
    df = df.sort_values(["key", "date_signed"])

    print("Aggregating transactions into one row per project ...")
    first = df.groupby("key", as_index=False).first()
    last = df.groupby("key", as_index=False).last()

    proj = first[[
        "key", "effective_date", "completion_date", "fiscal_year",
        "military_branch", "naics_code", "psc_code", "contract_type",
        "extent_competed", "number_of_offers", "place_state",
    ]].copy()
    proj = proj.rename(columns={"total_value": "base_value"})
    proj["base_value"] = first["total_value"].values
    proj["final_value"] = last["total_value"].values
    proj["ultimate_completion_date"] = last["ultimate_completion_date"].values
    proj["n_transactions"] = df.groupby("key").size().values

    # --- targets ---
    proj["cost_growth_ratio"] = (proj["final_value"] - proj["base_value"]) / proj["base_value"].replace(0, np.nan)
    proj["schedule_slip_days"] = (proj["ultimate_completion_date"] - proj["completion_date"]).dt.days
    proj["planned_duration_days"] = (proj["completion_date"] - proj["effective_date"]).dt.days

    # drop nonsensical / corrupted rows (negative base value, absurd durations)
    proj = proj[proj["base_value"] > 0]
    proj = proj[(proj["planned_duration_days"] > 0) & (proj["planned_duration_days"] < 3650)]

    keep_cols = ["fiscal_year", "base_value", "cost_growth_ratio", "schedule_slip_days",
                 "planned_duration_days", "number_of_offers", "n_transactions",
                 "military_branch", "naics_code", "psc_code", "contract_type",
                 "extent_competed", "place_state"]
    proj = proj[keep_cols]
    proj.to_csv(out_path, index=False)
    print(f"Wrote {len(proj):,} project-level rows -> {out_path}")
    return proj


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--raw", required=True)
    p.add_argument("--out", default="projects_clean.csv")
    args = p.parse_args()
    prepare(args.raw, args.out)
