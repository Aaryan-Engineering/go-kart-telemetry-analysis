"""
spins.py — Detect laps where the kart likely spun (lost grip and rotated
on its own momentum), as distinct from a normal hard corner.

Why this heuristic:
In a normal hard corner, a high yaw rate (heading changing quickly) is
accompanied by high MEASURED grip force (lateral_g / combined_g), because
the tyres are loaded and it's that cornering force that's producing the
rotation. In a spin, once the tyres break loose the kart keeps rotating on
its own angular momentum while the accelerometer reads comparatively LOW
g, because the tyres are no longer generating much force. So: high yaw
rate + low measured g, sustained for more than a single noisy sample, is
the signature we look for. It's a heuristic (same spirit as the
threshold-based recommendation engine in analysis.py) — tune the
thresholds below if it's over- or under-triggering on a given dataset.

Requires 'combined_g' already present on each lap's DataFrame (i.e. run
this after derive.add_derived_channels_session).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _yaw_rate_deg_s(df: pd.DataFrame, dt: float, clip_deg_s: float = 1080.0) -> np.ndarray:
    """
    Heading rate of change in deg/s. Orientation is unwrapped first so a
    genuine 359 -> 1 deg wrap (crossing north) isn't seen as a huge jump;
    the clip afterwards is just a sanity ceiling (3 full rotations/sec) to
    stop a single corrupted GPS heading sample from blowing up the signal.
    """
    heading = df["Orientation"].to_numpy()
    heading_unwrapped = np.degrees(np.unwrap(np.radians(heading)))
    rate = np.gradient(heading_unwrapped) / dt
    return np.clip(rate, -clip_deg_s, clip_deg_s)


def detect_spin_windows(
    df: pd.DataFrame,
    yaw_rate_threshold_deg_s: float = 100.0,
    low_g_threshold: float = 0.45,
    min_speed_kmh: float = 8.0,
    min_duration_s: float = 0.3,
) -> pd.DataFrame:
    """
    Adds 'yaw_rate_deg_s' and boolean 'spin_flag' columns to a copy of df.
    spin_flag is True for samples inside a sustained window where yaw rate
    is high, measured combined-g is low, and the kart is still moving
    (below min_speed_kmh, GPS heading is too noisy to trust).
    """
    df = df.copy()
    dt = df["time_s"].diff().median()
    if not dt or np.isnan(dt) or dt <= 0 or "combined_g" not in df.columns:
        df["yaw_rate_deg_s"] = np.nan
        df["spin_flag"] = False
        return df

    yaw_rate = _yaw_rate_deg_s(df, dt)
    combined_g = df["combined_g"].to_numpy()
    speed = df["Speed GPS"].to_numpy()

    candidate = (
        (np.abs(yaw_rate) > yaw_rate_threshold_deg_s)
        & (combined_g < low_g_threshold)
        & (speed > min_speed_kmh)
    )

    # require a sustained run, not a single noisy sample, before calling it a spin
    min_samples = max(int(round(min_duration_s / dt)), 1)
    flag = np.zeros(len(df), dtype=bool)
    run_start = None
    for i, c in enumerate(candidate):
        if c and run_start is None:
            run_start = i
        elif not c and run_start is not None:
            if i - run_start >= min_samples:
                flag[run_start:i] = True
            run_start = None
    if run_start is not None and len(candidate) - run_start >= min_samples:
        flag[run_start:] = True

    df["yaw_rate_deg_s"] = yaw_rate
    df["spin_flag"] = flag
    return df


def detect_spin_laps(
    laps: dict[int, pd.DataFrame],
    yaw_rate_threshold_deg_s: float = 100.0,
    low_g_threshold: float = 0.45,
    min_speed_kmh: float = 8.0,
    min_duration_s: float = 0.3,
) -> tuple[dict[int, pd.DataFrame], dict[int, bool]]:
    """
    Run spin detection on every lap.

    Returns
    -------
    laps_flagged : dict[int, pd.DataFrame]
        Same laps, each with 'yaw_rate_deg_s' and 'spin_flag' columns added
        (so the flag rides along into the per-lap CSVs and the plots).
    spin_by_lap : dict[int, bool]
        True if lap `lap_num` contains at least one sustained spin window.
    """
    laps_flagged = {}
    spin_by_lap = {}
    for lap_num, df in laps.items():
        flagged = detect_spin_windows(
            df,
            yaw_rate_threshold_deg_s=yaw_rate_threshold_deg_s,
            low_g_threshold=low_g_threshold,
            min_speed_kmh=min_speed_kmh,
            min_duration_s=min_duration_s,
        )
        laps_flagged[lap_num] = flagged
        spin_by_lap[lap_num] = bool(flagged["spin_flag"].any())
    return laps_flagged, spin_by_lap
