"""collect_data.py

Interactive Command-Line Tool for Survey Data Entry in CrowdCast.

This script helps an undergraduate student record real bus survey observations
one by one, validating entries in real-time, computing target occupancy,
and appending records safely to data/raw/crowdcast_data.csv.

Also includes an option (--demo-synthetic) strictly for software test verification,
which generates an explicit, clearly labeled testing set without pretending
to be real physical data.
"""

import argparse
from datetime import datetime
from pathlib import Path
import sys
import pandas as pd

RAW_DATA_PATH = Path("data/raw/crowdcast_data.csv")

EXPECTED_HEADER = [
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


def ensure_raw_file_initialized(path: Path):
    """Creates raw CSV with headers if it does not exist."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.stat().st_size == 0:
        df_empty = pd.DataFrame(columns=EXPECTED_HEADER)
        df_empty.to_csv(path, index=False)


def prompt_string(prompt_text: str, default: str = "") -> str:
    """Prompt for non-empty string."""
    while True:
        display = f"{prompt_text} [{default}]: " if default else f"{prompt_text}: "
        val = input(display).strip()
        if not val and default:
            return default
        if val:
            return val
        print("  [Error] Value cannot be empty. Please enter a valid string.")


def prompt_float(prompt_text: str, min_val: float = 0.0, default: float = None) -> float:
    """Prompt for a float >= min_val."""
    while True:
        display = f"{prompt_text} [{default}]: " if default is not None else f"{prompt_text}: "
        val = input(display).strip()
        if not val and default is not None:
            return default
        try:
            f = float(val)
            if f < min_val:
                print(f"  [Error] Value must be at least {min_val}.")
                continue
            return f
        except ValueError:
            print("  [Error] Please enter a valid numeric number.")


def prompt_time(prompt_text: str, default: str = "") -> str:
    """Prompt for HH:MM time string."""
    while True:
        display = f"{prompt_text} (HH:MM 24-hr) [{default}]: " if default else f"{prompt_text} (HH:MM 24-hr): "
        val = input(display).strip()
        if not val and default:
            val = default
        try:
            datetime.strptime(val, "%H:%M")
            return val
        except ValueError:
            print("  [Error] Invalid time format. Please use HH:MM (e.g. 08:30 or 17:45).")


def interactive_collection():
    """Runs interactive terminal prompt loop for student data collection."""
    print("=" * 70)
    print("    CROWDCAST: REAL-WORLD PASSENGER SURVEY COLLECTION ASSISTANT    ")
    print("=" * 70)
    print("Enter observation details below. To cancel at any time, press Ctrl+C.")
    print("-" * 70)

    ensure_raw_file_initialized(RAW_DATA_PATH)
    existing_df = pd.read_csv(RAW_DATA_PATH)
    next_id_num = len(existing_df) + 1
    default_obs_id = f"OBS_{next_id_num:04d}"

    obs_id = prompt_string("Observation ID", default=default_obs_id)
    trip_id = prompt_string("Trip ID (e.g. TRIP_20261026_01)")
    obs_date = prompt_string("Observation Date (YYYY-MM-DD)", default=datetime.today().strftime("%Y-%m-%d"))
    route_id = prompt_string("Route ID / Number (e.g. Route_12, 500A, BlueLine)")
    prev_stop = prompt_string("Previous Stop Name (where you made the initial count)")
    targ_stop = prompt_string("Target Stop Name (where bus will arrive)")
    day_week = prompt_string("Day of Week", default=datetime.today().strftime("%A"))

    print("\n--- Timestamps & Earlier Status (Features Known Before Target Stop) ---")
    pred_time = prompt_time("Prediction Time (time at previous stop)")
    prev_count = prompt_float("Passenger count before target stop", min_val=0.0)
    cap = prompt_float("Practical capacity of this bus (seated + practical standees)", min_val=1.0, default=60.0)
    mins_since = prompt_float("Estimated travel minutes from previous stop to target stop", min_val=0.0, default=5.0)
    peak_in = prompt_string("Is this peak period? (yes/no)", default="no").lower()
    is_peak = 1 if peak_in.startswith("y") else 0

    print("\n--- Actual Ground Truth (Observed When Bus Reached Target Stop) ---")
    targ_time = prompt_time("Target observation arrival time", default=pred_time)
    targ_count = prompt_float("Actual passenger count at target stop", min_val=0.0)

    # Calculate target occupancy & crowding level
    occupancy = round((targ_count / cap) * 100.0, 2)
    if occupancy < 50.0:
        level = "Low"
    elif occupancy < 80.0:
        level = "Moderate"
    else:
        level = "High"

    print("\n" + "-" * 70)
    print(f"Computed Occupancy: {occupancy}%")
    print(f"Derived Crowding Level: {level}")
    print("-" * 70)

    confirm = prompt_string("Save this observation to dataset? (yes/no)", default="yes").lower()
    if confirm.startswith("y"):
        new_row = {
            "observation_id": obs_id,
            "trip_id": trip_id,
            "observation_date": obs_date,
            "route_id": route_id,
            "previous_stop": prev_stop,
            "target_stop": targ_stop,
            "day_of_week": day_week,
            "prediction_time": pred_time,
            "target_observation_time": targ_time,
            "passengers_before_target": prev_count,
            "practical_capacity": cap,
            "minutes_since_previous_count": mins_since,
            "is_peak_period": is_peak,
            "target_passenger_count": targ_count,
            "target_occupancy_percentage": occupancy,
            "crowding_level": level,
        }
        df_new = pd.concat([existing_df, pd.DataFrame([new_row])], ignore_index=True)
        df_new.to_csv(RAW_DATA_PATH, index=False)
        print(f"[SUCCESS] Record saved to {RAW_DATA_PATH}. Total observations now: {len(df_new)}")
    else:
        print("[CANCELLED] Record not saved.")


def generate_testing_scaffold_csv(output_path: Path, count: int = 400):
    """Generates a clearly labeled synthetic dataset STRICTLY for software testing

    and pipeline verification when the student has not yet completed physical field collection.
    Notice: This synthetic data must NEVER be presented as real-world predictive evidence.
    """
    import numpy as np

    print(f"[TEST SCAFFOLD] Generating {count} synthetic test observations for code verification...")
    np.random.seed(42)

    routes = ["Route_101", "Route_202", "Route_303", "Route_404"]
    stops_by_route = {
        "Route_101": ["Central Station", "Market Square", "Tech Park", "University East", "Terminal 1"],
        "Route_202": ["West Suburb", "Civic Center", "Hospital Gate", "Metro Mall", "Airport North"],
        "Route_303": ["South Hill", "Garden Circle", "High School", "Railway Jn", "Harbor Port"],
        "Route_404": ["North Village", "Industrial Zone", "IT Hub", "City Center", "Lake Park"],
    }
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    rows = []
    obs_counter = 1
    trip_idx = 1

    while obs_counter <= count:
        route = np.random.choice(routes)
        stops = stops_by_route[route]
        trip_id = f"SYN_TRIP_{trip_idx:03d}"
        day = np.random.choice(days)
        is_weekend = 1 if day in ["Saturday", "Sunday"] else 0

        # Pick random start hour
        hour = np.random.choice([7, 8, 9, 12, 13, 16, 17, 18, 20])
        is_peak = 1 if (hour in [8, 9, 17, 18] and not is_weekend) else 0
        minute = np.random.randint(0, 45)

        # Base practical capacity (e.g. 50 seated + 15 standees = 65)
        capacity = float(np.random.choice([50.0, 60.0, 70.0]))

        # Running passengers on the bus (varied baseline so all classes are populated)
        crowd_scenario = np.random.choice(["light", "medium", "heavy"], p=[0.35, 0.35, 0.30])
        if crowd_scenario == "light":
            current_passengers = float(np.random.randint(5, int(capacity * 0.4)))
        elif crowd_scenario == "medium":
            current_passengers = float(np.random.randint(int(capacity * 0.4), int(capacity * 0.7)))
        else:
            current_passengers = float(np.random.randint(int(capacity * 0.65), int(capacity * 0.95)))

        for stop_i in range(len(stops) - 1):
            if obs_counter > count:
                break
            prev_stop = stops[stop_i]
            target_stop = stops[stop_i + 1]

            travel_mins = float(np.random.randint(4, 12))
            pred_time = f"{hour:02d}:{minute:02d}"

            # Advance clock for target arrival
            target_minute_total = minute + int(travel_mins)
            targ_hour = hour + (target_minute_total // 60)
            targ_min = target_minute_total % 60
            target_time = f"{targ_hour:02d}:{targ_min:02d}"
            minute = targ_min
            hour = targ_hour

            passengers_before = current_passengers

            # Passenger boarding / alighting dynamics at target stop
            if crowd_scenario == "heavy" or is_peak:
                boarders = np.random.poisson(lam=16.0)
                alighters = np.random.poisson(lam=4.0)
            elif crowd_scenario == "medium":
                boarders = np.random.poisson(lam=8.0)
                alighters = np.random.poisson(lam=6.0)
            else:
                boarders = np.random.poisson(lam=4.0)
                alighters = np.random.poisson(lam=6.0)

            target_count = max(0.0, passengers_before + boarders - alighters)
            # Maximum physical crush capacity ~1.15 * practical capacity
            target_count = min(target_count, capacity * 1.15)
            current_passengers = target_count

            occ_pct = round((target_count / capacity) * 100.0, 2)
            if occ_pct < 50.0:
                level = "Low"
            elif occ_pct < 80.0:
                level = "Moderate"
            else:
                level = "High"

            rows.append({
                "observation_id": f"SYN_OBS_{obs_counter:04d}",
                "trip_id": trip_id,
                "observation_date": f"2026-10-{(10 + (trip_idx % 15)):02d}",
                "route_id": route,
                "previous_stop": prev_stop,
                "target_stop": target_stop,
                "day_of_week": day,
                "prediction_time": pred_time,
                "target_observation_time": target_time,
                "passengers_before_target": passengers_before,
                "practical_capacity": capacity,
                "minutes_since_previous_count": travel_mins,
                "is_peak_period": is_peak,
                "target_passenger_count": target_count,
                "target_occupancy_percentage": occ_pct,
                "crowding_level": level,
            })
            obs_counter += 1
        trip_idx += 1

    df_syn = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_syn.to_csv(output_path, index=False)
    print(f"[SUCCESS] Test scaffold written to: {output_path}")
    print(f"          Total synthetic records: {len(df_syn)}")
    print(f"          Class balance: \n{df_syn['crowding_level'].value_counts()}")
    print("          NOTE: Synthetic data is solely for software execution testing.")


def main():
    parser = argparse.ArgumentParser(description="CrowdCast Data Collection Tool")
    parser.add_argument(
        "--demo-synthetic",
        action="store_true",
        help="Generate synthetic test scaffold dataset for code & pipeline verification only.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/raw/crowdcast_data.csv",
        help="Target output CSV file path.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=400,
        help="Number of test records to generate if --demo-synthetic is used (default: 400).",
    )
    args = parser.parse_args()

    if args.demo_synthetic:
        generate_testing_scaffold_csv(Path(args.output), count=args.count)
    else:
        interactive_collection()


if __name__ == "__main__":
    main()
