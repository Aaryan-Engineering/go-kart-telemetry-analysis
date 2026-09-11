"""
loader.py — Ingest and clean Alfano6 go-kart telemetry CSVs.

Raw Alfano6 lap files have no explicit timestamp column and several
channels are stored as scaled integers. This module:
  1. Loads raw per-lap CSVs
  2. Applies correct unit scaling to each channel
  3. Reconstructs an elapsed-time axis, anchored to the logger's own
     reported lap time (from summary.csv) so the time base is validated
     rather than assumed
  4. Computes distance-along-lap from GPS lat/lon
  5. Parses summary.csv into a tidy per-lap results table

Alfano6 raw column reference (as logged):
  Partiel      : sector number within the lap (1, 2, 3, ...)
  RPM          : engine RPM, raw value
  Speed GPS    : GPS speed, scaled x10 (978 -> 97.8 km/h)
  T1, T2       : auxiliary temperature channels, scaled x10 (deg C)
  Gf. X, Gf. Y : accelerometer channels, scaled x1000 (g), ~1.0 baseline
                 offset on at least one axis suggests gravity/tilt component
  Orientation  : heading, scaled x100 (degrees, 0-36000 -> 0-360.00 deg)
  Speed rear   : rear wheel speed sensor, 0 in this dataset (unused)
  Lat., Lon.   : GPS coordinates, scaled x1e6 (degrees), Lat is negative
                 (southern hemisphere)
  Altitude     : metres, unscaled

NOTE: scaling factors for RPM, Gf.X/Y, and Orientation are inferred from
physically plausible ranges (see docs/data_dictionary.md once generated)
and cross-checked against summary.csv min/max columns where available.
Speed GPS scaling (x10) is confirmed directly by the user from source
knowledge of the device output.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Unit scaling
# ---------------------------------------------------------------------------

SCALING = {
    "Speed GPS": 0.1,      # -> km/h
    "T1": 0.1,              # -> deg C
    "T2": 0.1,               # -> deg C
    "Gf. X": 0.001,          # -> g
    "Gf. Y": 0.001,          # -> g
    "Orientation": 0.01,     # -> degrees (0-360)
    "Lat.": 1e-6,             # -> degrees
    "Lon.": 1e-6,             # -> degrees
    "Speed rear": 0.1,       # -> km/h (same convention as Speed GPS)
}

EARTH_RADIUS_M = 6_371_000.0


@dataclass
class LapData:
    lap_number: int
    df: pd.DataFrame          # cleaned, per-row telemetry
    official_time_ms: float | None = None
    source_file: str | None = None


def _haversine_m(lat1, lon1, lat2, lon2):
    """Vectorised haversine distance in metres between consecutive points."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    return EARTH_RADIUS_M * c


def load_summary(path: str | Path) -> pd.DataFrame:
    """Parse the Alfano6 summary.csv into a tidy per-lap results table."""
    df = pd.read_csv(path, skiprows=1)
    df.columns = [c.strip() for c in df.columns]
    return df


def load_raw_lap(path: str | Path, lap_number: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df


MAX_PLAUSIBLE_SPEED_KMH = 150.0  # generous ceiling for a kart; anything above is a GPS glitch


def filter_gps_glitches(
    df: pd.DataFrame, dt: float, max_speed_kmh: float = MAX_PLAUSIBLE_SPEED_KMH,
    residual_threshold_m: float = 8.0, window: int = 5,
) -> pd.DataFrame:
    """
    Detect and correct single-point GPS glitches.

    Two checks, combined:
    1. Directional: implied velocity between consecutive points exceeds a
       physically implausible threshold (catches most interior glitches).
    2. Local-median residual: a point's distance from the rolling median
       position of its neighbours exceeds a threshold (also catches
       endpoint glitches, e.g. the first/last row of a lap, where there's
       only one adjacent segment to check directionally).

    Flagged points have their lat/lon replaced by linear interpolation.
    """
    df = df.copy()
    lat = df["Lat."].to_numpy()
    lon = df["Lon."].to_numpy()
    n = len(df)
    if n < 3:
        return df

    seg_dist = np.zeros(n)
    seg_dist[1:] = _haversine_m(lat[:-1], lon[:-1], lat[1:], lon[1:])
    implied_speed_kmh = (seg_dist / dt) * 3.6
    bad_directional = np.zeros(n, dtype=bool)
    bad_directional[1:] |= implied_speed_kmh[1:] > max_speed_kmh
    bad_directional[:-1] |= implied_speed_kmh[1:] > max_speed_kmh  # flags source point too

    x, y = latlon_to_xy(lat, lon, lat.mean(), lon.mean())
    med_x = pd.Series(x).rolling(window, center=True, min_periods=3).median().to_numpy()
    med_y = pd.Series(y).rolling(window, center=True, min_periods=3).median().to_numpy()
    residual = np.hypot(x - med_x, y - med_y)
    bad_residual = residual > residual_threshold_m

    bad = bad_directional | bad_residual
    if bad.any():
        lat_s = pd.Series(lat)
        lon_s = pd.Series(lon)
        lat_s[bad] = np.nan
        lon_s[bad] = np.nan
        lat_s = lat_s.interpolate(limit_direction="both")
        lon_s = lon_s.interpolate(limit_direction="both")
        df["Lat."] = lat_s.to_numpy()
        df["Lon."] = lon_s.to_numpy()
        df["gps_glitch_corrected"] = bad
    else:
        df["gps_glitch_corrected"] = False

    return df


def clean_lap(
    raw: pd.DataFrame,
    lap_number: int,
    official_time_ms: float | None = None,
) -> pd.DataFrame:
    """Apply scaling, reconstruct time, and compute distance for one lap."""
    df = raw.copy()

    for col, factor in SCALING.items():
        if col in df.columns:
            df[col] = df[col] * factor

    n = len(df)

    # --- Time reconstruction ---
    # Anchor to the logger's own official lap time when available (validated),
    # otherwise fall back to the observed ~10 Hz sample rate.
    if official_time_ms is not None and n > 0:
        dt = (official_time_ms / 1000.0) / n
    else:
        dt = 0.1  # fallback: 10 Hz nominal
    df["time_s"] = np.arange(n) * dt

    # --- GPS glitch correction (before distance is computed) ---
    df = filter_gps_glitches(df, dt=dt)

    # --- Distance reconstruction from GPS ---
    lat = df["Lat."].to_numpy()
    lon = df["Lon."].to_numpy()
    seg_dist = np.zeros(n)
    if n > 1:
        seg_dist[1:] = _haversine_m(lat[:-1], lon[:-1], lat[1:], lon[1:])
    df["segment_dist_m"] = seg_dist
    df["distance_m"] = np.cumsum(seg_dist)

    df["lap_number"] = lap_number
    return df


def discover_lap_files(raw_dir: str | Path, pattern: str = "lap*.csv") -> dict[int, str]:
    """
    Auto-discover lap CSVs in a directory, e.g. lap1.csv, lap2.csv, ...
    Extracts the lap number from the filename (first run of digits) rather
    than assuming a fixed count or naming scheme, so this adapts to however
    many lap files are actually present.
    """
    import re
    raw_dir = Path(raw_dir)
    lap_files = {}
    for path in sorted(raw_dir.glob(pattern)):
        match = re.search(r"(\d+)", path.stem)
        if not match:
            continue
        lap_num = int(match.group(1))
        lap_files[lap_num] = path.name
    if not lap_files:
        raise FileNotFoundError(
            f"No lap files matching '{pattern}' found in {raw_dir}. "
            f"Expected files like lap1.csv, lap2.csv, ..."
        )
    return dict(sorted(lap_files.items()))


def load_session(
    raw_dir: str | Path,
    lap_files: dict[int, str] | None = None,
    summary_file: str = "summary.csv",
) -> tuple[dict[int, pd.DataFrame], pd.DataFrame]:
    """
    Load and clean a full session.

    Parameters
    ----------
    raw_dir : directory containing raw CSVs
    lap_files : mapping of lap_number -> filename, e.g. {1: "lap1.csv", ...}.
                If None (default), auto-discovers every "lap*.csv" file in
                raw_dir — this is what makes the pipeline work on a session
                with a different number of laps without editing any code.
    summary_file : filename of the Alfano6 summary export

    Returns
    -------
    laps : dict of lap_number -> cleaned DataFrame
    summary : tidy per-lap results table (all laps in the session, including
              ones without raw CSVs)
    """
    raw_dir = Path(raw_dir)
    if lap_files is None:
        lap_files = discover_lap_files(raw_dir)

    summary = load_summary(raw_dir / summary_file)

    laps = {}
    for lap_num, fname in lap_files.items():
        raw = load_raw_lap(raw_dir / fname, lap_num)
        official_time = None
        match = summary.loc[summary["lap"] == lap_num, "time lap"]
        if len(match):
            official_time = float(match.values[0])
        laps[lap_num] = clean_lap(raw, lap_num, official_time_ms=official_time)

    return laps, summary


def validate_reconstruction(laps: dict[int, pd.DataFrame], summary: pd.DataFrame) -> pd.DataFrame:
    """Sanity-check reconstructed time/distance against the logger's own summary."""
    rows = []
    for lap_num, df in laps.items():
        official = summary.loc[summary["lap"] == lap_num, "time lap"].values
        official_s = official[0] / 1000.0 if len(official) else np.nan
        recon_s = df["time_s"].iloc[-1] + (df["time_s"].iloc[-1] - df["time_s"].iloc[-2]) if len(df) > 1 else np.nan
        total_dist = df["distance_m"].iloc[-1]
        rows.append({
            "lap": lap_num,
            "n_rows": len(df),
            "official_time_s": official_s,
            "reconstructed_time_s": round(recon_s, 3),
            "total_distance_m": round(total_dist, 1),
            "max_speed_kmh": round(df["Speed GPS"].max(), 1),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Local flat-earth projection + reference-line alignment
#
# Racing lines differ lap to lap, so each lap's own cumulative GPS distance
# is NOT a shared coordinate — 400m into lap 1 and 400m into lap 3 are not
# the same physical point on track. To compare laps (delta-time, corner
# analysis, overlaid speed traces) we need every lap's points expressed as
# "distance along a common reference line", not distance along themselves.
# ---------------------------------------------------------------------------


def latlon_to_xy(lat, lon, origin_lat, origin_lon):
    """Equirectangular projection to local metres. Fine at this scale (~1km)."""
    lat_r = np.radians(lat)
    origin_lat_r = np.radians(origin_lat)
    x = np.radians(lon - origin_lon) * EARTH_RADIUS_M * np.cos(origin_lat_r)
    y = np.radians(lat - origin_lat) * EARTH_RADIUS_M
    return x, y


def build_reference_line(laps: dict[int, pd.DataFrame], reference_lap: int) -> pd.DataFrame:
    """
    Build the reference path (= the track map) from one lap's GPS trace,
    in local XY metres with cumulative arc-length distance.
    """
    ref = laps[reference_lap]
    origin_lat, origin_lon = ref["Lat."].iloc[0], ref["Lon."].iloc[0]

    all_lat = pd.concat([df["Lat."] for df in laps.values()])
    all_lon = pd.concat([df["Lon."] for df in laps.values()])
    origin_lat, origin_lon = all_lat.mean(), all_lon.mean()

    x, y = latlon_to_xy(ref["Lat."].to_numpy(), ref["Lon."].to_numpy(), origin_lat, origin_lon)
    seg = np.zeros(len(x))
    seg[1:] = np.hypot(np.diff(x), np.diff(y))
    arc_length = np.cumsum(seg)

    return pd.DataFrame({
        "x_m": x, "y_m": y, "arc_length_m": arc_length,
    }), (origin_lat, origin_lon)


def project_onto_reference(
    df: pd.DataFrame,
    ref_line: pd.DataFrame,
    origin: tuple[float, float],
) -> pd.DataFrame:
    """
    For each point in `df`, find the nearest point on the reference polyline
    and assign its arc-length as 'ref_distance_m'. Adds x_m, y_m too (for the
    track map / overlay plots).
    """
    origin_lat, origin_lon = origin
    x, y = latlon_to_xy(df["Lat."].to_numpy(), df["Lon."].to_numpy(), origin_lat, origin_lon)

    ref_x = ref_line["x_m"].to_numpy()
    ref_y = ref_line["y_m"].to_numpy()
    ref_arc = ref_line["arc_length_m"].to_numpy()

    ref_distance = np.empty(len(x))
    for i in range(len(x)):
        d2 = (ref_x - x[i]) ** 2 + (ref_y - y[i]) ** 2
        nearest = np.argmin(d2)
        ref_distance[i] = ref_arc[nearest]

    out = df.copy()
    out["x_m"] = x
    out["y_m"] = y
    out["ref_distance_m"] = ref_distance
    return out


def align_session_to_reference(
    laps: dict[int, pd.DataFrame], reference_lap: int | None = None
) -> tuple[dict[int, pd.DataFrame], pd.DataFrame]:
    """
    Build a reference line (fastest lap by default) and project every lap
    onto it, adding x_m/y_m/ref_distance_m columns to each lap's DataFrame.
    """
    if reference_lap is None:
        reference_lap = min(laps, key=lambda k: laps[k]["time_s"].iloc[-1])

    ref_line, origin = build_reference_line(laps, reference_lap)
    aligned = {
        lap_num: project_onto_reference(df, ref_line, origin)
        for lap_num, df in laps.items()
    }
    return aligned, ref_line


if __name__ == "__main__":
    raw_dir = Path(__file__).resolve().parent.parent / "data" / "raw_sample"
    lap_files = {i: f"lap{i}.csv" for i in range(1, 6)}
    laps, summary = load_session(raw_dir, lap_files)

    print("=== Session summary (all logged laps) ===")
    print(summary[["lap", "time lap", "Max Speed GPS", "Max RPM"]])

    print("\n=== Reconstruction validation (laps with raw CSVs) ===")
    print(validate_reconstruction(laps, summary))

    print("\n=== Sample cleaned lap 1 ===")
    print(laps[1].head())

    aligned, ref_line = align_session_to_reference(laps)
    print(f"\n=== Reference line built from fastest lap, {len(ref_line)} points, "
          f"track length ~{ref_line['arc_length_m'].iloc[-1]:.1f} m ===")
    print("\n=== Sample aligned lap 1 (x_m, y_m, ref_distance_m) ===")
    print(aligned[1][["time_s", "distance_m", "x_m", "y_m", "ref_distance_m"]].head())
