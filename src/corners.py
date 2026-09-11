"""
corners.py — Automated corner detection and per-corner metrics.

Corners are detected from the reference line's curvature (rate of heading
change per metre), computed once from the reference lap's geometry, so
corner boundaries are consistent across every lap (they're properties of
the track, not of any individual lap's noisy yaw signal).

For each detected corner, per-lap metrics (entry/min/exit speed, peak
lateral G, brake/throttle points) are computed by slicing each lap's data
between the corner's start/end distance-along-reference.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _heading_deg(x, y):
    dx = np.gradient(x)
    dy = np.gradient(y)
    return np.degrees(np.arctan2(dy, dx))


def detect_corners(
    ref_line: pd.DataFrame,
    curvature_threshold_deg_per_10m: float = 8.0,
    min_corner_length_m: float = 5.0,
    smooth_window: int = 5,
    wrap_pad: int = 10,
) -> pd.DataFrame:
    """
    Detect corners from the reference line geometry.

    The track is a closed loop, so the array is cyclically padded before
    computing heading/curvature — otherwise the start/finish seam looks
    like a fake discontinuity and gets misdetected as a corner.

    Returns a DataFrame: corner_id, start_m, end_m, length_m, max_curvature.
    """
    x = ref_line["x_m"].to_numpy()
    y = ref_line["y_m"].to_numpy()
    s = ref_line["arc_length_m"].to_numpy()
    n = len(x)

    # cyclic pad
    x_p = np.concatenate([x[-wrap_pad:], x, x[:wrap_pad]])
    y_p = np.concatenate([y[-wrap_pad:], y, y[:wrap_pad]])

    heading_p = _heading_deg(x_p, y_p)
    heading_p_unwrapped = np.degrees(np.unwrap(np.radians(heading_p)))
    heading_smooth_p = pd.Series(heading_p_unwrapped).rolling(
        smooth_window, center=True, min_periods=1
    ).mean().to_numpy()

    # arc length also needs padding for correct ds at the seam
    track_length = s[-1] + (s[-1] - s[-2] if n > 1 else 0)
    s_p = np.concatenate([s[-wrap_pad:] - track_length, s, s[:wrap_pad] + track_length])

    ds_p = np.gradient(s_p)
    ds_p[ds_p == 0] = np.nan
    curvature_p = np.abs(np.gradient(heading_smooth_p) / ds_p) * 10  # deg per 10m

    # slice back to original range
    curvature = curvature_p[wrap_pad:wrap_pad + n]

    is_corner = curvature > curvature_threshold_deg_per_10m

    # group consecutive True runs into corners (circularly: a run touching
    # both the first and last index wraps around and is one corner)
    corners = []
    in_corner = False
    start_idx = None
    for i, flag in enumerate(is_corner):
        if flag and not in_corner:
            in_corner = True
            start_idx = i
        elif not flag and in_corner:
            in_corner = False
            end_idx = i
            length = s[end_idx] - s[start_idx]
            if length >= min_corner_length_m:
                corners.append((start_idx, end_idx))
    if in_corner:
        # runs to the end of the array — check if it wraps into corner[0]
        if corners and corners[0][0] == 0 and is_corner[0]:
            # merge wrap-around corner with the first one
            first = corners.pop(0)
            length = (track_length - s[start_idx]) + s[first[1]]
            if length >= min_corner_length_m:
                corners.append((start_idx, first[1]))
        else:
            length = s[-1] - s[start_idx]
            if length >= min_corner_length_m:
                corners.append((start_idx, len(s) - 1))

    rows = []
    for i, (i0, i1) in enumerate(corners, start=1):
        wraps = i1 < i0
        rows.append({
            "corner_id": i,
            "start_m": s[i0],
            "end_m": s[i1],
            "length_m": (track_length - s[i0] + s[i1]) if wraps else (s[i1] - s[i0]),
            "max_curvature": curvature[i0:i1 + 1].max() if not wraps else max(
                curvature[i0:].max(), curvature[:i1 + 1].max()
            ),
            "wraps_seam": wraps,
        })
    return pd.DataFrame(rows)


def corner_metrics_by_lap(
    laps: dict[int, pd.DataFrame], corners: pd.DataFrame,
    brake_search_margin_m: float = 15.0,
    accel_search_margin_m: float = 15.0,
    brake_g_threshold: float = 0.15,
    accel_proxy_threshold: float = 20.0,
) -> pd.DataFrame:
    """
    For every lap x corner, compute entry/min/exit speed, peak lateral/braking/
    combined G, and the brake/acceleration point locations.

    Brake point: last position, searching backward from the corner's min-speed
    point (with a margin before the corner start), where longitudinal_g first
    exceeds brake_g_threshold — i.e. where sustained braking begins.
    Accel point: first position after the min-speed point, searching forward
    (with a margin past corner end), where accel_proxy first exceeds
    accel_proxy_threshold — i.e. where meaningful acceleration begins.
    Both are approximate (threshold-based) and depend on the same proxy
    channels described in derive.py.
    """
    rows = []
    for lap_num, df in laps.items():
        df_sorted = df.sort_values("ref_distance_m").reset_index(drop=True)
        for _, corner in corners.iterrows():
            mask = (df["ref_distance_m"] >= corner["start_m"]) & (
                df["ref_distance_m"] <= corner["end_m"]
            )
            seg = df.loc[mask]
            if seg.empty:
                continue
            min_idx = seg["Speed GPS"].idxmin()
            min_speed_dist = seg.loc[min_idx, "ref_distance_m"]

            # brake point: search the window before the apex
            brake_window = df_sorted[
                (df_sorted["ref_distance_m"] >= corner["start_m"] - brake_search_margin_m)
                & (df_sorted["ref_distance_m"] <= min_speed_dist)
            ]
            brake_point = np.nan
            if "longitudinal_g" in brake_window and (brake_window["longitudinal_g"] > brake_g_threshold).any():
                brake_point = brake_window.loc[
                    brake_window["longitudinal_g"] > brake_g_threshold, "ref_distance_m"
                ].iloc[0]

            # accel point: search the window after the apex
            accel_window = df_sorted[
                (df_sorted["ref_distance_m"] >= min_speed_dist)
                & (df_sorted["ref_distance_m"] <= corner["end_m"] + accel_search_margin_m)
            ]
            accel_point = np.nan
            if "accel_proxy" in accel_window and (accel_window["accel_proxy"] > accel_proxy_threshold).any():
                accel_point = accel_window.loc[
                    accel_window["accel_proxy"] > accel_proxy_threshold, "ref_distance_m"
                ].iloc[0]

            rows.append({
                "lap": lap_num,
                "corner_id": int(corner["corner_id"]),
                "entry_speed_kmh": seg["Speed GPS"].iloc[0],
                "min_speed_kmh": seg["Speed GPS"].min(),
                "exit_speed_kmh": seg["Speed GPS"].iloc[-1],
                "min_speed_distance_m": min_speed_dist,
                "peak_lateral_g": seg["lateral_g"].abs().max() if "lateral_g" in seg else np.nan,
                "peak_braking_g": seg["longitudinal_g"].clip(lower=0).max() if "longitudinal_g" in seg else np.nan,
                "peak_combined_g": seg["combined_g"].max() if "combined_g" in seg else np.nan,
                "brake_point_m": brake_point,
                "accel_point_m": accel_point,
                "brake_point_offset_m": (corner["start_m"] - brake_point) if not np.isnan(brake_point) else np.nan,
                "accel_point_offset_m": (accel_point - corner["start_m"]) if not np.isnan(accel_point) else np.nan,
            })
    return pd.DataFrame(rows)
