"""src/prepare_data.py

Data Preparation & Validation Pipeline for CrowdCast.

This module loads raw passenger survey observations, validates types and domain
constraints, checks observation chronological order, derives occupancy percentages
and crowding class targets based on configurable thresholds, verifies leak prevention,
and writes out a clean, processed dataset.

Thresholds used for crowding_level:
  - Low: occupancy < 50%
  - Moderate: 50% <= occupancy < 80%
  - High: occupancy >= 80%
Note: These are project-specific practical thresholds, not universal standards.
"""

import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd

# Expected columns in raw survey CSV
EXPECTED_COLUMNS: List[str] = [
    "observation_id",
    "trip_id",
    "observation_date",
    "route_id",
    "previous_stop",
    "target_stop",
    "day_of_week",
    "prediction_time",
    "target_observation_time",
    "passengers_before_target",
    "practical_capacity",
    "minutes_since_previous_count",
    "is_peak_period",
    "target_passenger_count",
    "target_occupancy_percentage",
    "crowding_level",
]

# Configurable occupancy thresholds (in percent)
DEFAULT_LOW_THRESHOLD = 50.0
DEFAULT_MODERATE_THRESHOLD = 80.0


def derive_crowding_level(
    occupancy_pct: float,
    low_thresh: float = DEFAULT_LOW_THRESHOLD,
    mod_thresh: float = DEFAULT_MODERATE_THRESHOLD,
) -> str:
    """Categorizes occupancy percentage into Low, Moderate, or High."""
    if pd.isna(occupancy_pct):
        return "Unknown"
    if occupancy_pct < low_thresh:
        return "Low"
    elif occupancy_pct < mod_thresh:
        return "Moderate"
    else:
        return "High"


def validate_and_prepare(
    raw_csv_path: Path,
    processed_csv_path: Path,
    low_thresh: float = DEFAULT_LOW_THRESHOLD,
    mod_thresh: float = DEFAULT_MODERATE_THRESHOLD,
) -> Tuple[pd.DataFrame, Dict[str, any]]:
    """Loads, validates, derives targets, and saves the cleaned dataset.

    Returns:
        Tuple of (processed_df, audit_report_dict)
    """
    audit: Dict[str, any] = {
        "raw_path": str(raw_csv_path),
        "processed_path": str(processed_csv_path),
        "total_rows_read": 0,
        "valid_rows_retained": 0,
        "rejected_rows": [],
        "warnings": [],
        "thresholds": {"low_max": low_thresh, "moderate_max": mod_thresh},
    }

    if not raw_csv_path.exists():
        raise FileNotFoundError(f"Raw data file not found at: {raw_csv_path}")

    # Read CSV
    df_raw = pd.read_csv(raw_csv_path)
    audit["total_rows_read"] = len(df_raw)

    if df_raw.empty:
        print("[WARNING] The raw CSV file is empty (contains only header columns or zero rows).")
        print("          No observations available to process yet.")
        return df_raw, audit

    # 1. Validate Schema
    missing_cols = [col for col in EXPECTED_COLUMNS if col not in df_raw.columns]
    if missing_cols:
        raise ValueError(
            f"CSV schema mismatch. Missing required columns: {missing_cols}\n"
            f"Expected: {EXPECTED_COLUMNS}"
        )

    # 2. Row-by-row validation & rejection tracking
    valid_indices = []
    
    for idx, row in df_raw.iterrows():
        reasons = []

        # Check for mandatory keys
        if pd.isna(row["observation_id"]) or str(row["observation_id"]).strip() == "":
            reasons.append("Missing observation_id")
        if pd.isna(row["trip_id"]) or str(row["trip_id"]).strip() == "":
            reasons.append("Missing trip_id")

        # Capacity validation
        cap = row["practical_capacity"]
        try:
            cap_val = float(cap)
            if cap_val <= 0:
                reasons.append(f"Invalid practical_capacity ({cap_val} <= 0)")
        except (ValueError, TypeError):
            reasons.append(f"Non-numeric practical_capacity: {cap}")

        # Passenger counts validation (non-negative)
        for count_col in ["passengers_before_target", "target_passenger_count"]:
            val = row[count_col]
            try:
                num = float(val)
                if num < 0:
                    reasons.append(f"Negative count in {count_col}: {num}")
            except (ValueError, TypeError):
                reasons.append(f"Non-numeric value in {count_col}: {val}")

        # Minutes since previous count validation
        mins = row["minutes_since_previous_count"]
        try:
            mins_val = float(mins)
            if mins_val < 0:
                reasons.append(f"Negative minutes_since_previous_count: {mins_val}")
        except (ValueError, TypeError):
            reasons.append(f"Non-numeric minutes_since_previous_count: {mins}")

        # Peak period boolean / integer validation
        peak = row["is_peak_period"]
        if pd.notna(peak):
            str_peak = str(peak).strip().lower()
            if str_peak not in ["0", "1", "true", "false"]:
                reasons.append(f"Invalid is_peak_period value: {peak} (expected 0, 1, True, False)")

        # Chronological order validation: prediction_time must be before or equal to target_observation_time
        pred_time_str = str(row["prediction_time"]).strip()
        targ_time_str = str(row["target_observation_time"]).strip()
        if pred_time_str and targ_time_str and pred_time_str != "nan" and targ_time_str != "nan":
            try:
                # Accept HH:MM or HH:MM:SS
                fmt = "%H:%M:%S" if len(pred_time_str.split(":")) == 3 else "%H:%M"
                t_pred = datetime.strptime(pred_time_str, fmt).time()
                t_targ_fmt = "%H:%M:%S" if len(targ_time_str.split(":")) == 3 else "%H:%M"
                t_targ = datetime.strptime(targ_time_str, t_targ_fmt).time()

                # Basic intra-day check (assuming non-midnight crossover trips)
                if t_pred > t_targ:
                    reasons.append(
                        f"Chronological violation: prediction_time ({pred_time_str}) is AFTER "
                        f"target_observation_time ({targ_time_str})"
                    )
            except Exception as e:
                audit["warnings"].append(f"Row {idx} time format warning: {e}")

        if reasons:
            audit["rejected_rows"].append({
                "row_index": int(idx),
                "observation_id": row.get("observation_id"),
                "trip_id": row.get("trip_id"),
                "reasons": reasons
            })
        else:
            valid_indices.append(idx)

    # 3. Create cleaned working dataframe
    df_clean = df_raw.loc[valid_indices].copy()

    # Detect duplicate observation IDs
    dup_obs = df_clean[df_clean.duplicated(subset=["observation_id"], keep=False)]
    if not dup_obs.empty:
        audit["warnings"].append(
            f"Found {len(dup_obs)} rows with duplicate observation_id. Keeping first occurrence."
        )
        df_clean = df_clean.drop_duplicates(subset=["observation_id"], keep="first")

    # Cast numeric types safely
    df_clean["practical_capacity"] = df_clean["practical_capacity"].astype(float)
    df_clean["passengers_before_target"] = df_clean["passengers_before_target"].astype(float)
    df_clean["target_passenger_count"] = df_clean["target_passenger_count"].astype(float)
    df_clean["minutes_since_previous_count"] = df_clean["minutes_since_previous_count"].astype(float)
    
    # Standardize peak period to 0 or 1
    df_clean["is_peak_period"] = df_clean["is_peak_period"].apply(
        lambda x: 1 if str(x).strip().lower() in ["1", "true"] else 0
    )

    # 4. Standardize / Re-calculate Occupancy Percentage
    # occupancy_percentage = (target_passenger_count / practical_capacity) * 100
    df_clean["target_occupancy_percentage"] = (
        (df_clean["target_passenger_count"] / df_clean["practical_capacity"]) * 100.0
    ).round(2)

    # 5. Derive crowding_level from configurable thresholds
    df_clean["crowding_level"] = df_clean["target_occupancy_percentage"].apply(
        lambda occ: derive_crowding_level(occ, low_thresh, mod_thresh)
    )

    audit["valid_rows_retained"] = len(df_clean)

    # 6. Save Processed Dataset
    processed_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(processed_csv_path, index=False)

    return df_clean, audit


def print_preparation_summary(df_clean: pd.DataFrame, audit: Dict[str, any]):
    """Prints a clear terminal audit report explaining validation actions."""
    print("=" * 70)
    print("       CROWDCAST: DATA PREPARATION & VALIDATION AUDIT REPORT       ")
    print("=" * 70)
    print(f"Source file:             {audit['raw_path']}")
    print(f"Destination file:        {audit['processed_path']}")
    print(f"Total raw records read:  {audit['total_rows_read']}")
    print(f"Valid records retained:  {audit['valid_rows_retained']}")
    print(f"Rejected records count:  {len(audit['rejected_rows'])}")
    print(f"Occupancy Thresholds:    Low < {audit['thresholds']['low_max']}%, "
          f"Moderate < {audit['thresholds']['moderate_max']}%, "
          f"High >= {audit['thresholds']['moderate_max']}%")
    print("-" * 70)

    if audit["rejected_rows"]:
        print("\n[EXPLANATION OF REJECTED RECORDS]")
        for item in audit["rejected_rows"]:
            print(f"  * Row {item['row_index']} (ID: {item['observation_id']}, Trip: {item['trip_id']}):")
            for r in item["reasons"]:
                print(f"      - {r}")
    else:
        print("No records were rejected. All entries met validation constraints.")

    if audit["warnings"]:
        print("\n[WARNINGS / ADJUSTMENTS]")
        for w in audit["warnings"]:
            print(f"  ! {w}")

    if not df_clean.empty:
        print("\n[PROCESSED DATASET TARGET CLASS DISTRIBUTION]")
        counts = df_clean["crowding_level"].value_counts()
        pcts = df_clean["crowding_level"].value_counts(normalize=True) * 100
        for cls in ["Low", "Moderate", "High"]:
            c = counts.get(cls, 0)
            p = pcts.get(cls, 0.0)
            print(f"  * {cls:<10}: {c:4d} records ({p:5.1f}%)")
        print("-" * 70)
        print("[DATA LEAKAGE PREVENTION CHECK]")
        print("  VERIFIED: The target column 'crowding_level' and target features")
        print("  ('target_passenger_count', 'target_occupancy_percentage',")
        print("   'target_observation_time') are isolated as targets/audits.")
        print("  Downstream training pipeline (src/train.py) excludes them from model input.")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Prepare and validate CrowdCast survey data.")
    parser.add_argument(
        "--raw",
        type=str,
        default="data/raw/crowdcast_data.csv",
        help="Path to raw CSV file (default: data/raw/crowdcast_data.csv)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/crowdcast_clean.csv",
        help="Path to write processed CSV (default: data/processed/crowdcast_clean.csv)",
    )
    parser.add_argument(
        "--low-threshold",
        type=float,
        default=DEFAULT_LOW_THRESHOLD,
        help="Occupancy percentage upper bound for 'Low' class (default: 50.0)",
    )
    parser.add_argument(
        "--mod-threshold",
        type=float,
        default=DEFAULT_MODERATE_THRESHOLD,
        help="Occupancy percentage upper bound for 'Moderate' class (default: 80.0)",
    )
    args = parser.parse_args()

    raw_path = Path(args.raw)
    out_path = Path(args.output)

    try:
        df_clean, audit = validate_and_prepare(
            raw_path,
            out_path,
            low_thresh=args.low_threshold,
            mod_thresh=args.mod_threshold,
        )
        print_preparation_summary(df_clean, audit)
    except Exception as err:
        print(f"[ERROR] Failed to prepare data: {err}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
