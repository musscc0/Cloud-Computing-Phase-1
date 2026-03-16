import json
import os
import traceback
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from google.cloud import pubsub_v1, storage

PROJECT_ID = os.environ["PROJECT_ID"]
TOPIC_ID = os.environ["TOPIC_ID"]
SUBSCRIPTION_ID = os.environ["SUBSCRIPTION_ID"]

WINDOW_FRAMES = int(os.getenv("WINDOW_FRAMES", "50"))
STRIDE_FRAMES = int(os.getenv("STRIDE_FRAMES", "10"))
PLOT_LIMIT = int(os.getenv("PLOT_LIMIT", "6"))

publisher = pubsub_v1.PublisherClient()
subscriber = pubsub_v1.SubscriberClient()
storage_client = storage.Client()

topic_path = publisher.topic_path(PROJECT_ID, TOPIC_ID)
subscription_path = subscriber.subscription_path(PROJECT_ID, SUBSCRIPTION_ID)

REQUIRED_COLS = [
    "Vehicle_ID", "Frame_ID", "Global_Time", "Local_X", "Local_Y",
    "v_Vel", "v_Acc", "Lane_ID", "Preceding", "Following"
]

OPTIONAL_COLS = ["Space_Headway", "Time_Headway"]


def download_blob(bucket_name, blob_name, local_path):
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.download_to_filename(local_path)


def upload_file(bucket_name, local_path, blob_name):
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(local_path)


def upload_text(bucket_name, blob_name, text):
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_string(text, content_type="application/json")


def standardize_columns(df):
    df.columns = [c.strip() for c in df.columns]
    return df


def preprocess(df, local_y_min=None, local_y_max=None):
    df = standardize_columns(df)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    numeric_cols = REQUIRED_COLS + [c for c in OPTIONAL_COLS if c in df.columns]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Vehicle_ID", "Global_Time", "Local_X", "Local_Y", "Lane_ID"])
    df = df.sort_values(["Vehicle_ID", "Global_Time"]).drop_duplicates(["Vehicle_ID", "Global_Time"])

    if local_y_min is not None:
        df = df[df["Local_Y"] >= local_y_min]
    if local_y_max is not None:
        df = df[df["Local_Y"] <= local_y_max]

    counts = df["Vehicle_ID"].value_counts()
    valid_ids = counts[counts >= WINDOW_FRAMES].index
    df = df[df["Vehicle_ID"].isin(valid_ids)].copy()

    lead_df = df[["Global_Time", "Vehicle_ID", "Lane_ID", "Local_Y", "v_Vel"]].rename(
        columns={
            "Vehicle_ID": "Preceding",
            "Lane_ID": "lead_lane",
            "Local_Y": "lead_local_y",
            "v_Vel": "lead_v_vel"
        }
    )
    df = df.merge(lead_df, on=["Global_Time", "Preceding"], how="left")

    if "Space_Headway" not in df.columns:
        df["Space_Headway"] = df["lead_local_y"] - df["Local_Y"]

    df["rel_speed"] = df["lead_v_vel"] - df["v_Vel"]
    return df


def contiguous_window(window):
    diffs = window["Global_Time"].diff().dropna()
    if len(diffs) == 0:
        return False
    return (diffs <= 100).all()


def nearby_density(window, all_df):
    counts = []
    for _, row in window.iterrows():
        frame = all_df[all_df["Global_Time"] == row["Global_Time"]]
        nearby = frame[
            (frame["Vehicle_ID"] != row["Vehicle_ID"]) &
            (frame["Lane_ID"].between(row["Lane_ID"] - 1, row["Lane_ID"] + 1)) &
            ((frame["Local_Y"] - row["Local_Y"]).abs() <= 100)
        ]
        counts.append(len(nearby))
    return float(np.mean(counts)) if counts else 0.0


def is_lane_change(window):
    start_lane = int(window["Lane_ID"].iloc[0])
    end_lane = int(window["Lane_ID"].iloc[-1])
    unique_lanes = window["Lane_ID"].dropna().astype(int).unique()

    if len(unique_lanes) < 2:
        return False
    if abs(end_lane - start_lane) != 1:
        return False
    lateral_shift = abs(window["Local_X"].iloc[-1] - window["Local_X"].iloc[0])
    return lateral_shift >= 8.0


def is_car_following(window):
    if window["Lane_ID"].nunique() != 1:
        return False

    lead_present_ratio = (window["Preceding"] > 0).mean()
    if lead_present_ratio < 0.9:
        return False

    gap_ok = window["Space_Headway"].between(15, 200, inclusive="both").mean() >= 0.9
    rel_speed_ok = window["rel_speed"].dropna().abs().mean() <= 10 if window["rel_speed"].dropna().size else False

    same_lane_ok = True
    if "lead_lane" in window.columns:
        tmp = window.dropna(subset=["lead_lane"])
        if len(tmp) > 0:
            same_lane_ok = (tmp["Lane_ID"] == tmp["lead_lane"]).mean() >= 0.9

    return gap_ok and rel_speed_ok and same_lane_ok


def is_congested(window, all_df):
    mean_speed = window["v_Vel"].mean()
    mean_density = nearby_density(window, all_df)

    acc = window["v_Acc"].fillna(0).to_numpy()
    sign_changes = np.sum(np.sign(acc[1:]) != np.sign(acc[:-1])) if len(acc) > 1 else 0

    return mean_speed < 20 and mean_density >= 4 and sign_changes >= 3


def get_label(window, all_df):
    if is_lane_change(window):
        return "lane_change"
    if is_congested(window, all_df):
        return "congested_traffic"
    if is_car_following(window):
        return "car_following"
    return None


def get_surrounding_rows(window, all_df):
    vehicle_ids = set()
    for _, row in window.iterrows():
        frame = all_df[all_df["Global_Time"] == row["Global_Time"]]
        nearby = frame[
            (frame["Vehicle_ID"] != row["Vehicle_ID"]) &
            (frame["Lane_ID"].between(row["Lane_ID"] - 1, row["Lane_ID"] + 1)) &
            ((frame["Local_Y"] - row["Local_Y"]).abs() <= 100)
        ]
        vehicle_ids.update(nearby["Vehicle_ID"].tolist())

    t0 = window["Global_Time"].min()
    t1 = window["Global_Time"].max()

    surrounding = all_df[
        (all_df["Vehicle_ID"].isin(vehicle_ids)) &
        (all_df["Global_Time"] >= t0) &
        (all_df["Global_Time"] <= t1)
    ].copy()
    ego = all_df[
        (all_df["Vehicle_ID"] == window["Vehicle_ID"].iloc[0]) &
        (all_df["Global_Time"] >= t0) &
        (all_df["Global_Time"] <= t1)
    ].copy()

    ego["is_ego"] = 1
    surrounding["is_ego"] = 0
    return pd.concat([ego, surrounding], ignore_index=True)


def plot_sample(sample_df, label, scenario_id, out_png):
    plt.figure(figsize=(8, 6))

    ego = sample_df[sample_df["is_ego"] == 1]
    others = sample_df[sample_df["is_ego"] == 0]

    for vid, g in others.groupby("Vehicle_ID"):
        plt.plot(g["Local_Y"], g["Local_X"], alpha=0.4)

    if not ego.empty:
        plt.plot(ego["Local_Y"], ego["Local_X"], linewidth=2)

    plt.xlabel("Local_Y")
    plt.ylabel("Local_X")
    plt.title(f"{scenario_id} | {label}")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def process_job(job):
    bucket_name = job["bucket"]
    object_name = job["object"]
    output_prefix = job.get("output_prefix", "runs/run1")
    local_y_min = job.get("local_y_min")
    local_y_max = job.get("local_y_max")

    workdir = Path("/tmp/ngsim")
    workdir.mkdir(parents=True, exist_ok=True)

    input_csv = workdir / "input.csv"
    cleaned_csv = workdir / "cleaned.csv"
    index_csv = workdir / "scenario_index.csv"

    download_blob(bucket_name, object_name, str(input_csv))
    raw_df = pd.read_csv(input_csv)
    df = preprocess(raw_df, local_y_min, local_y_max)
    df.to_csv(cleaned_csv, index=False)
    upload_file(bucket_name, str(cleaned_csv), f"{output_prefix}/intermediate/cleaned.csv")

    scenario_rows = []
    plots_created = 0

    for vehicle_id, group in df.groupby("Vehicle_ID"):
        group = group.sort_values("Global_Time").reset_index(drop=True)

        for start in range(0, len(group) - WINDOW_FRAMES + 1, STRIDE_FRAMES):
            window = group.iloc[start:start + WINDOW_FRAMES].copy()
            if not contiguous_window(window):
                continue

            label = get_label(window, df)
            if label is None:
                continue

            scenario_id = f"{label}_veh{int(vehicle_id)}_{int(window['Global_Time'].iloc[0])}"
            sample_df = get_surrounding_rows(window, df)
            sample_df["scenario_id"] = scenario_id
            sample_df["scenario_label"] = label

            sample_path = workdir / f"{scenario_id}.csv"
            sample_df.to_csv(sample_path, index=False)
            upload_file(bucket_name, str(sample_path), f"{output_prefix}/samples/{scenario_id}.csv")

            if plots_created < PLOT_LIMIT:
                plot_path = workdir / f"{scenario_id}.png"
                plot_sample(sample_df, label, scenario_id, plot_path)
                upload_file(bucket_name, str(plot_path), f"{output_prefix}/plots/{scenario_id}.png")
                plots_created += 1

            scenario_rows.append({
                "scenario_id": scenario_id,
                "scenario_label": label,
                "ego_vehicle_id": int(vehicle_id),
                "start_global_time": int(window["Global_Time"].iloc[0]),
                "end_global_time": int(window["Global_Time"].iloc[-1]),
                "lane_start": int(window["Lane_ID"].iloc[0]),
                "lane_end": int(window["Lane_ID"].iloc[-1]),
                "frames": len(window)
            })

    scenario_index = pd.DataFrame(scenario_rows)
    scenario_index.to_csv(index_csv, index=False)
    upload_file(bucket_name, str(index_csv), f"{output_prefix}/scenario_index.csv")

    report = {
        "status": "done",
        "input_object": object_name,
        "output_prefix": output_prefix,
        "cleaned_rows": int(len(df)),
        "scenario_count": int(len(scenario_rows)),
        "counts_by_label": scenario_index["scenario_label"].value_counts().to_dict() if len(scenario_index) else {}
    }
    upload_text(bucket_name, f"{output_prefix}/run_report.json", json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


def callback(message):
    try:
        job = json.loads(message.data.decode("utf-8"))
        print("Received job:", job)
        process_job(job)
        message.ack()
    except Exception:
        print("Job failed")
        print(traceback.format_exc())
        message.nack()


def main():
    print(f"Listening on {subscription_path}")
    streaming_pull_future = subscriber.subscribe(subscription_path, callback=callback)
    streaming_pull_future.result()


if __name__ == "__main__":
    main()