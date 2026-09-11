"""
derive.py — Derived telemetry channels.

This dataset has no throttle or brake pedal sensors (normal for a kart
logger). Everything here is derived/proxy signal, clearly labelled as such.

Channels produced:
  longitudinal_g  : calibrated true longitudinal acceleration (g), positive
                     = braking, negative = accelerating. Calibrated by
                     regressing raw Gf.Y against GPS-speed-derived
                     acceleration, pooled across the whole session (see
                     calibrate_longitudinal_g). This corrects for the raw
                     channel's unknown absolute scale while keeping the
                     accelerometer's superior sample-to-sample precision
                     over differentiated GPS speed.
  lateral_g       : Gf.X re-centred on its session baseline (no independent
                     calibration source available; centring only).
  accel_proxy  : 0-100, derived from RPM rate-of-rise. On a fixed-gear
                     kart RPM tracks wheel speed ~1:1, so acceleration
                     phases show up directly as positive RPM slope.
                     NOTE: this reflects RPM rise, not throttle pedal
                     position — RPM can rise/fall from gearing, load, or
                     engine-braking effects too. Named "accel_proxy"
                     rather than "throttle_proxy" for this reason.
  brake_proxy     : 0-100, derived from longitudinal_g when it exceeds a
                     braking threshold (positive = decelerating).

These are estimates for analysis purposes, not measured pedal positions —
label plots accordingly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def calibrate_lateral_g(laps: dict[int, pd.DataFrame]) -> tuple[float, float]:
    """
    Fit lateral_g = m * Gf.X + b by regressing against yaw-rate-derived
    lateral acceleration (a_lat = v * yaw_rate), pooled across all laps.
    Returns (m, b).
    """
    xs, ys = [], []
    for df in laps.values():
        dt = df["time_s"].diff().median()
        speed_ms = df["Speed GPS"] / 3.6
        yaw_rate = np.radians(df["Orientation"].diff()) / dt
        yaw_rate = yaw_rate.clip(-5, 5)  # discard GPS heading wraparound spikes
        a_lat = (yaw_rate * speed_ms) / 9.81
        mask = a_lat.notna()
        xs.append(df.loc[mask, "Gf. X"])
        ys.append(a_lat[mask])
    x = pd.concat(xs)
    y = pd.concat(ys)
    m, b = np.polyfit(x, y, 1)
    return float(m), float(b)


def calibrate_longitudinal_g(laps: dict[int, pd.DataFrame]) -> tuple[float, float]:
    """
    Fit longitudinal_g = m * Gf.Y + b by regressing against GPS-speed-derived
    acceleration, pooled across all laps for a robust estimate.
    Returns (m, b).
    """
    xs, ys = [], []
    for df in laps.values():
        dt = df["time_s"].diff().median()
        speed_ms = df["Speed GPS"] / 3.6
        a_long = speed_ms.diff() / dt / 9.81
        mask = a_long.notna()
        xs.append(df.loc[mask, "Gf. Y"])
        ys.append(a_long[mask])
    x = pd.concat(xs)
    y = pd.concat(ys)
    m, b = np.polyfit(x, y, 1)
    return float(m), float(b)


def add_derived_channels(
    df: pd.DataFrame,
    g_calibration: tuple[float, float],
    lateral_g_calibration: tuple[float, float],
    accel_rpm_slope_norm: float | None = None,
    brake_g_threshold: float = 0.15,
    brake_g_norm: float = 1.5,
) -> pd.DataFrame:
    """Add longitudinal_g, lateral_g, accel_proxy, brake_proxy to a lap df."""
    df = df.copy()
    m, b = g_calibration
    lm, lb = lateral_g_calibration

    # sign convention: positive = braking (deceleration), matches m,b fit
    # where positive Gf.Y-driven output already corresponds to deceleration
    df["longitudinal_g"] = -(m * df["Gf. Y"] + b)  # flip so positive = braking
    df["lateral_g"] = lm * df["Gf. X"] + lb
    df["combined_g"] = np.hypot(df["longitudinal_g"], df["lateral_g"])

    dt = df["time_s"].diff().median()
    rpm_smooth = df["RPM"].rolling(5, center=True, min_periods=1).mean()
    rpm_slope = rpm_smooth.diff() / dt  # RPM per second, smoothed to avoid amplifying sample noise

    if accel_rpm_slope_norm is None:
        accel_rpm_slope_norm = rpm_slope.clip(lower=0).quantile(0.95) or 1.0

    df["accel_proxy"] = (
        (rpm_slope.clip(lower=0) / accel_rpm_slope_norm) * 100
    ).clip(0, 100)

    brake_signal = (df["longitudinal_g"] - brake_g_threshold).clip(lower=0)
    df["brake_proxy"] = ((brake_signal / brake_g_norm) * 100).clip(0, 100)

    return df


def add_derived_channels_session(
    laps: dict[int, pd.DataFrame]
) -> dict[int, pd.DataFrame]:
    """Calibrate once on the full session, then apply to every lap consistently."""
    g_cal = calibrate_longitudinal_g(laps)
    lat_g_cal = calibrate_lateral_g(laps)

    # normalize throttle proxy using a session-wide RPM slope reference so
    # laps are comparable to each other, not each self-normalized
    all_slopes = []
    for df in laps.values():
        dt = df["time_s"].diff().median()
        rpm_smooth = df["RPM"].rolling(5, center=True, min_periods=1).mean()
        all_slopes.append((rpm_smooth.diff() / dt).clip(lower=0))
    session_norm = pd.concat(all_slopes).quantile(0.95)

    return {
        lap_num: add_derived_channels(
            df, g_cal, lat_g_cal, accel_rpm_slope_norm=session_norm
        )
        for lap_num, df in laps.items()
    }
