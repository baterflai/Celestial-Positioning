from __future__ import annotations

import json
from pathlib import Path

from complementary_filter import ComplementaryFilter, ComplementaryFilterConfig

NEGATE_MAG_Z = True

def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


def get_latest_mag(mag_samples, imu_timestamp, start_idx):
    """
    Returns the latest magnetometer sample whose timestamp
    is <= imu_timestamp.
    """
    idx = start_idx
    latest_mag = None

    while idx < len(mag_samples) and mag_samples[idx]["timestamp"] <= imu_timestamp:
        mz = float(mag_samples[idx]["mag_z"])
        if NEGATE_MAG_Z:
            mz = -mz

        latest_mag = (
            float(mag_samples[idx]["mag_x"]),
            float(mag_samples[idx]["mag_y"]),
            mz,
        )
        idx += 1

    return latest_mag, idx


def main():

    # imu_path = Path("imu_data_20260311_153709.json")
    # mag_path = Path("mag_data_20260311_153709.json")

    imu_path = Path("imu_data_with_cal.json")
    mag_path = Path("mag_data_with_cal.json")

    imu_samples = load_json(str(imu_path))
    mag_samples = load_json(str(mag_path))

    filt = ComplementaryFilter(
        ComplementaryFilterConfig(
            alpha_rp=0.99,
            alpha_yaw=0.99,
            use_mag_yaw=True,
        )
    )

    mag_idx = 0
    last_mag = None
    results = []

    for i, sample in enumerate(imu_samples):

        t = float(sample["timestamp"])

        # update magnetometer
        last_mag, mag_idx = get_latest_mag(mag_samples, t, mag_idx)

        accel = (
            float(sample["acc_x"]),
            float(sample["acc_y"]),
            float(sample["acc_z"]),
        )

        gyro = (
            float(sample["gyro_x"]),
            float(sample["gyro_y"]),
            float(sample["gyro_z"]),
        )

        roll, pitch, yaw = filt.update(
            timestamp=t,
            accel=accel,
            gyro=gyro,
            mag=last_mag,
        )

        # Only save every 10th timestep
        if i % 10 == 0:
            results.append(
                {
                    "timestamp": t,
                    "roll": roll,
                    "pitch": pitch,
                    "yaw": yaw,
                }
            )

    print("\nExample filtered samples:")
    for row in results[:10]:
        print(
            f"t={row['timestamp']:.6f}  "
            f"roll={row['roll']:.6f}  "
            f"pitch={row['pitch']:.6f}  "
            f"yaw={row['yaw']:.6f}"
        )

    # out_path = Path("complementary_filter_output.json")
    out_path = Path("complementary_filter_output_with_cal_99_mag_flip.json")

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved filtered results to: {out_path}")
    print(f"Total samples processed: {len(imu_samples)}")
    print(f"Samples saved (every 10th): {len(results)}")


if __name__ == "__main__":
    main()
