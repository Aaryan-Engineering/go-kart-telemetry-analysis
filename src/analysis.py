"""
analysis.py — Driver consistency metrics and best theoretical lap.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def lap_time_table(summary: pd.DataFrame, laps_available: list[int]) -> pd.DataFrame:
    """Per-lap time and sector times, in seconds (source data is in milliseconds)."""
    df = summary[["lap", "time lap", "time partiel 1", "time partiel 2", "time partiel 3"]].copy()
    df = df.rename(columns={
        "time lap": "lap_time_s",
        "time partiel 1": "sector_1_time_s",
        "time partiel 2": "sector_2_time_s",
        "time partiel 3": "sector_3_time_s",
    })
    for col in ["lap_time_s", "sector_1_time_s", "sector_2_time_s", "sector_3_time_s"]:
        df[col] = df[col] / 1000.0
    df["has_raw_data"] = df["lap"].isin(laps_available)
    return df


def best_theoretical_lap(summary: pd.DataFrame) -> dict:
    """Best sector 1 + best sector 2 + best sector 3 time, stitched together. All values in seconds."""
    s1 = summary["time partiel 1"].min() / 1000.0
    s2 = summary["time partiel 2"].min() / 1000.0
    s3 = summary["time partiel 3"].min() / 1000.0
    best_lap_actual = summary["time lap"].min() / 1000.0
    best_lap_row = summary.loc[summary["time lap"].idxmin()]
    return {
        "best_sector_1_s": s1,
        "best_sector_2_s": s2,
        "best_sector_3_s": s3,
        "theoretical_best_s": s1 + s2 + s3,
        "best_actual_lap_s": best_lap_actual,
        "best_actual_lap_number": int(best_lap_row["lap"]),
        "best_actual_lap_sector_1_s": best_lap_row["time partiel 1"] / 1000.0,
        "best_actual_lap_sector_2_s": best_lap_row["time partiel 2"] / 1000.0,
        "best_actual_lap_sector_3_s": best_lap_row["time partiel 3"] / 1000.0,
        "gap_s": best_lap_actual - (s1 + s2 + s3),
    }


# ---------------------------------------------------------------------------
# Lap exclusion for corner-comparison analysis
#
# Corner-comparison metrics (consistency ranking, corner time-loss, the
# "worst lap through this corner" recommendation engine) are only
# meaningful when comparing laps that were genuinely representative
# attempts at the corner. Two categories of lap break that assumption and
# are excluded from these specific comparisons (but NOT from the raw plots
# — they're still drawn, just colour-coded so they aren't mistaken for a
# normal lap):
#   1. Out-lap(s) — cold tyres/brakes, not a real attempt, conventionally
#      the first lap of a session.
#   2. Spin lap(s) — see spins.py. A lap can be fastest through a corner's
#      entry/mid-corner and still be dragged down badly by a spin
#      elsewhere, which would otherwise wrongly flag it as "the best lap"
#      or, worse, make an unrelated corner look like the worst offender.
# ---------------------------------------------------------------------------

DEFAULT_OUT_LAPS = (1,)


def resolve_excluded_laps(
    laps_available: list[int],
    spin_by_lap: dict[int, bool] | None = None,
    out_laps: tuple[int, ...] = DEFAULT_OUT_LAPS,
) -> list[int]:
    """
    Laps to exclude from corner-comparison analysis: the conventional
    out-lap(s) (only if actually present in this session) plus any lap
    with a detected spin.
    """
    excluded = {lap for lap in out_laps if lap in laps_available}
    if spin_by_lap:
        excluded |= {lap for lap, is_spin in spin_by_lap.items() if is_spin}
    return sorted(excluded)


def filter_excluded_laps(df: pd.DataFrame, excluded_laps, lap_col: str = "lap") -> pd.DataFrame:
    """
    Drop rows belonging to excluded laps from a per-lap metrics table,
    without touching the underlying per-lap telemetry used elsewhere (e.g.
    the raw plots, where excluded laps are still shown, just colour-coded).
    """
    excluded_laps = set(excluded_laps or [])
    if not excluded_laps or lap_col not in df.columns:
        return df
    return df[~df[lap_col].isin(excluded_laps)].reset_index(drop=True)


def racing_line_consistency(aligned: dict[int, pd.DataFrame], corners: pd.DataFrame) -> pd.DataFrame:
    """
    For each corner, spatial consistency of the (x_m, y_m) position at the
    min-speed point across laps.

    NOTE: an earlier version used std-dev of radial distance from the
    centroid as the headline metric. That's misleading — two apex points on
    opposite sides of the centroid at the same radius give a falsely low
    value despite being spatially very inconsistent. This version reports
    std-dev in X and Y separately (captures directional spread) plus RMS and
    max distance from centroid (captures overall magnitude), which together
    are a more defensible description of apex repeatability.

    Pass only the laps you want counted (e.g. excluding out-laps/spins) —
    this function doesn't filter anything itself.
    """
    rows = []
    for _, c in corners.iterrows():
        xs, ys = [], []
        for lap_num, df in aligned.items():
            mask = (df["ref_distance_m"] >= c["start_m"]) & (df["ref_distance_m"] <= c["end_m"])
            seg = df.loc[mask]
            if seg.empty:
                continue
            min_idx = seg["Speed GPS"].idxmin()
            xs.append(seg.loc[min_idx, "x_m"])
            ys.append(seg.loc[min_idx, "y_m"])
        if len(xs) < 2:
            continue
        xs, ys = np.array(xs), np.array(ys)
        centroid = (xs.mean(), ys.mean())
        dist = np.hypot(xs - centroid[0], ys - centroid[1])
        rows.append({
            "corner_id": int(c["corner_id"]),
            "apex_std_x_m": xs.std(),
            "apex_std_y_m": ys.std(),
            "apex_rms_m": np.sqrt(np.mean(dist ** 2)),
            "apex_max_m": dist.max(),
        })
    return pd.DataFrame(rows)


def corner_apex_positions(aligned: dict[int, pd.DataFrame], corners: pd.DataFrame) -> pd.DataFrame:
    """Per-lap, per-corner apex (min-speed point) position and deviation from that
    corner's across-lap centroid. Used by the recommendation engine.

    Pass only the laps you want counted (e.g. excluding out-laps/spins) —
    this function doesn't filter anything itself.
    """
    rows = []
    for _, c in corners.iterrows():
        corner_id = int(c["corner_id"])
        pts = {}
        for lap_num, df in aligned.items():
            mask = (df["ref_distance_m"] >= c["start_m"]) & (df["ref_distance_m"] <= c["end_m"])
            seg = df.loc[mask]
            if seg.empty:
                continue
            min_idx = seg["Speed GPS"].idxmin()
            pts[lap_num] = (seg.loc[min_idx, "x_m"], seg.loc[min_idx, "y_m"])
        if len(pts) < 2:
            continue
        cx = np.mean([p[0] for p in pts.values()])
        cy = np.mean([p[1] for p in pts.values()])
        for lap_num, (x, y) in pts.items():
            rows.append({
                "corner_id": corner_id, "lap": lap_num,
                "apex_x_m": x, "apex_y_m": y,
                "apex_deviation_m": np.hypot(x - cx, y - cy),
            })
    return pd.DataFrame(rows)


def corner_time_loss(delta_df: pd.DataFrame, corners: pd.DataFrame) -> pd.DataFrame:
    """
    Time gained/lost specifically WITHIN each corner (not cumulative from lap start).

    Computed as delta(exit) - delta(entry) for each lap, using the same
    reference lap `delta_df` was built against. Because delta-time is itself
    a cumulative difference vs the reference, this subtraction isolates the
    marginal time lost/gained in that specific track segment, independent of
    what happened earlier in the lap.
    """
    rows = []
    dist = delta_df["distance_m"].to_numpy()
    for _, c in corners.iterrows():
        corner_id = int(c["corner_id"])
        for col in delta_df.columns:
            if col == "distance_m":
                continue
            lap_num = int(col.split("_")[1])
            vals = delta_df[col].to_numpy()
            entry_delta = np.interp(c["start_m"], dist, vals)
            exit_delta = np.interp(c["end_m"], dist, vals)
            rows.append({
                "corner_id": corner_id, "lap": lap_num,
                "time_loss_s": exit_delta - entry_delta,
            })
    return pd.DataFrame(rows)


def generate_recommendations(
    corner_metrics: pd.DataFrame,
    time_loss: pd.DataFrame,
    apex_positions: pd.DataFrame,
    corners: pd.DataFrame,
    top_n: int = 3,
) -> list[dict]:
    """
    Heuristic, rule-based improvement suggestions per corner. Compares the
    lap that lost the most time in a corner against the lap that lost the
    least (i.e. was fastest through it) and reports what differed, in
    Observation -> Evidence -> Diagnosis -> Proposed Change -> Expected
    Effect form.

    These are auto-generated from simple threshold rules on the computed
    metrics, not a learned or validated diagnostic model — treat them as a
    starting point for review, not a definitive verdict.

    Pass corner_metrics/time_loss/apex_positions that have already had
    out-laps and spin laps filtered out (see filter_excluded_laps) — this
    function doesn't filter anything itself, and a spin or out-lap showing
    up as "the worst lap" here would produce a misleading diagnosis.
    """
    merged = time_loss.merge(corner_metrics, on=["corner_id", "lap"], how="left")
    merged = merged.merge(apex_positions, on=["corner_id", "lap"], how="left")

    if merged.empty:
        return []

    worst_per_corner = merged.loc[merged.groupby("corner_id")["time_loss_s"].idxmax()]
    worst_per_corner = worst_per_corner.sort_values("time_loss_s", ascending=False).head(top_n)

    recommendations = []
    for _, worst in worst_per_corner.iterrows():
        corner_id = worst["corner_id"]
        corner_rows = merged[merged["corner_id"] == corner_id]
        best = corner_rows.loc[corner_rows["time_loss_s"].idxmin()]

        if worst["lap"] == best["lap"] or worst["time_loss_s"] <= 0:
            continue  # no meaningful loss detected for this corner

        min_speed_diff = worst["min_speed_kmh"] - best["min_speed_kmh"]
        exit_speed_diff = worst["exit_speed_kmh"] - best["exit_speed_kmh"]
        entry_speed_diff = worst["entry_speed_kmh"] - best["entry_speed_kmh"]
        apex_dev = worst["apex_deviation_m"]
        apex_dev_val = float(apex_dev) if apex_dev is not None and not np.isnan(apex_dev) else None

        # simple rule cascade — checked in order of "most likely root cause
        # upstream" so a big entry-speed deficit isn't overridden by a smaller,
        # more local symptom further down the cascade. Each branch carries a
        # matching proposed change, not just a diagnosis label.
        if entry_speed_diff < -3:
            diagnosis = ("Entry speed itself was down — the loss likely originates from the "
                         "approach (previous corner/straight), not this corner in isolation.")
            proposed_change = (f"Review the exit of the corner/straight before Corner {int(corner_id)} "
                                f"on this lap — carrying {abs(entry_speed_diff):.1f} km/h more into the "
                                f"entry here would resolve most of this loss without changing anything "
                                f"about this corner itself.")
        elif abs(entry_speed_diff) < 3 and min_speed_diff < -3:
            diagnosis = ("Similar entry speed but a lower minimum speed — likely over-slowing "
                         "mid-corner rather than a braking problem.")
            proposed_change = (f"Brake later or less aggressively, or delay turn-in slightly, to carry "
                                f"~{abs(min_speed_diff):.1f} km/h more through the apex; check for "
                                f"early/heavy trail-braking.")
        elif exit_speed_diff < -3 and min_speed_diff >= -3:
            diagnosis = ("Minimum speed was comparable but exit speed was down — likely late or "
                         "hesitant throttle application getting the kart rotated and back on power.")
            proposed_change = (f"Get back on the power earlier/more decisively on exit; target "
                                f"~{abs(exit_speed_diff):.1f} km/h more at the exit point without "
                                f"compromising the apex speed already achieved.")
        elif apex_dev_val is not None and apex_dev_val > 2.5:
            diagnosis = (f"Apex position deviated {apex_dev_val:.1f}m from this corner's typical line "
                        f"— likely an inconsistent racing line (wide entry or late apex) rather than a "
                        f"braking/throttle issue.")
            proposed_change = (f"Tighten the racing line to within ~1m of the reference apex position "
                                f"used on the best lap through this corner.")
        else:
            diagnosis = ("Loss is spread across entry, minimum, and exit speed without one dominant "
                        "cause — worth reviewing on video.")
            proposed_change = "Review the full speed trace and onboard video for this corner specifically."

        recommendations.append({
            "corner_id": int(corner_id),
            "worst_lap": int(worst["lap"]),
            "best_lap": int(best["lap"]),
            "time_loss_s": float(worst["time_loss_s"]),
            "min_speed_diff_kmh": float(min_speed_diff),
            "exit_speed_diff_kmh": float(exit_speed_diff),
            "entry_speed_diff_kmh": float(entry_speed_diff),
            "apex_deviation_m": apex_dev_val,
            "observation": (f"Lap {int(worst['lap'])} lost {worst['time_loss_s']:.3f}s in Corner "
                            f"{int(corner_id)} relative to Lap {int(best['lap'])}, its best lap through "
                            f"this corner."),
            "evidence": (f"Entry {entry_speed_diff:+.1f} km/h, minimum {min_speed_diff:+.1f} km/h, "
                        f"exit {exit_speed_diff:+.1f} km/h"
                        + (f", apex deviation {apex_dev_val:.1f}m" if apex_dev_val is not None else "")
                        + " vs. the best lap through this corner."),
            "diagnosis": diagnosis,
            "proposed_change": proposed_change,
            "expected_effect_s": float(worst["time_loss_s"]),
            "issue": diagnosis,  # kept for backward compatibility with existing dashboard code
        })
    return recommendations


def corner_loss_summary(time_loss: pd.DataFrame, corners: pd.DataFrame) -> pd.DataFrame:
    """
    Per-corner worst-case time loss (the single largest loss any lap showed
    in that corner) plus the lap it occurred on. Summing the positive values
    gives a "total identifiable corner loss" — an estimate of how much lap
    time is sitting in specific, attributable corner mistakes across the
    whole session (not a claim that fixing all of them simultaneously in one
    lap is achievable, since they may come from different laps).

    Pass a time_loss table that's already had out-laps/spin laps filtered
    out — otherwise an out-lap or spin will dominate every corner's "worst
    lap" and drown out genuine technique losses.
    """
    if time_loss.empty:
        return pd.DataFrame(columns=["corner_id", "worst_lap", "time_loss_s"])
    worst = time_loss.loc[time_loss.groupby("corner_id")["time_loss_s"].idxmax()]
    worst = worst.rename(columns={"lap": "worst_lap"}).sort_values("time_loss_s", ascending=False)
    return worst[["corner_id", "worst_lap", "time_loss_s"]].reset_index(drop=True)


def consistency_summary(
    corner_metrics: pd.DataFrame, racing_line: pd.DataFrame
) -> pd.DataFrame:
    """Merge corner-speed variability with racing-line variability, per corner.

    Pass a corner_metrics table that's already had out-laps/spin laps
    filtered out — otherwise a cold out-lap or a spin's crashed min-speed
    inflates every corner's variability and makes normal corners look far
    less consistent than the driving actually was.
    """
    speed_consistency = corner_metrics.groupby("corner_id").agg(
        min_speed_mean=("min_speed_kmh", "mean"),
        min_speed_std=("min_speed_kmh", "std"),
    ).reset_index()
    out = speed_consistency.merge(racing_line, on="corner_id", how="left")
    out["min_speed_cv_pct"] = (out["min_speed_std"] / out["min_speed_mean"]) * 100
    return out.sort_values("min_speed_cv_pct", ascending=False)


def driver_summary(lap_times: pd.DataFrame, theoretical: dict, consistency: pd.DataFrame,
                    corner_metrics: pd.DataFrame) -> dict:
    """Headline numbers for a dashboard summary panel — one shared source so
    every surface (CLI print, Streamlit, HTML dashboard) reports the same figures."""
    least_consistent = consistency.iloc[0]
    return {
        "best_lap_s": lap_times["lap_time_s"].min(),
        "best_lap_number": int(lap_times.loc[lap_times["lap_time_s"].idxmin(), "lap"]),
        "theoretical_best_s": theoretical["theoretical_best_s"],
        "potential_gain_s": theoretical["gap_s"],
        "least_consistent_corner": int(least_consistent["corner_id"]),
        "least_consistent_cv_pct": float(least_consistent["min_speed_cv_pct"]),
        "peak_lateral_g": float(corner_metrics["peak_lateral_g"].max()),
        "peak_braking_g": float(corner_metrics["peak_braking_g"].max()),
        "peak_combined_g": float(corner_metrics["peak_combined_g"].max()),
    }


def detect_likely_outliers(lap_times: pd.DataFrame, threshold_pct: float = 3.0) -> pd.DataFrame:
    """
    Flag laps whose time is more than `threshold_pct`% slower than the
    median of the other laps with raw data — a simple, data-driven way to
    surface laps that are worth a manual look, on top of (not instead of)
    the explicit out-lap/spin exclusion applied elsewhere in the pipeline.
    """
    with_data = lap_times[lap_times["has_raw_data"]].copy()
    if len(with_data) < 2:
        with_data["pct_slower_than_median"] = 0.0
        with_data["likely_outlier"] = False
        return with_data
    median_time = with_data["lap_time_s"].median()
    with_data["pct_slower_than_median"] = (with_data["lap_time_s"] - median_time) / median_time * 100
    with_data["likely_outlier"] = with_data["pct_slower_than_median"] > threshold_pct
    return with_data


def generate_engineering_summary(
    lap_times: pd.DataFrame,
    theoretical: dict,
    consistency: pd.DataFrame,
    recommendations: list[dict],
    loss_summary: pd.DataFrame,
    summary_stats: dict,
    n_corners: int,
    excluded_laps: list[int] | None = None,
    spin_by_lap: dict[int, bool] | None = None,
) -> str:
    """
    Fully data-driven engineering summary in markdown — regenerated from
    scratch on every run, so it describes whatever session was actually
    loaded rather than a fixed, hand-written narrative. Intentionally more
    "structured report" than "hand-authored prose"; the methodology notes at
    the end describe the pipeline itself (invariant across datasets), not
    this session's specific numbers.
    """
    excluded_laps = excluded_laps or []
    spin_by_lap = spin_by_lap or {}

    lines = ["# Engineering Summary (auto-generated)\n"]
    lines.append(f"Laps with full telemetry: {lap_times['has_raw_data'].sum()} of {len(lap_times)} "
                 f"in the session. {n_corners} corners automatically detected.\n")

    lines.append(f"\n## Pace\n")
    lines.append(f"- Best lap: Lap {summary_stats['best_lap_number']} — {summary_stats['best_lap_s']:.3f}s\n")
    lines.append(f"- Theoretical best (best sectors stitched): {summary_stats['theoretical_best_s']:.3f}s "
                 f"({summary_stats['potential_gain_s']:.3f}s potential gain)\n")

    lines.append("\n## Laps excluded from corner-comparison analysis\n")
    if excluded_laps:
        for lap in excluded_laps:
            reason = ("a spin was detected (sustained high yaw-rate with low measured grip)"
                      if spin_by_lap.get(lap) else "conventional out-lap (cold tyres/brakes)")
            lines.append(
                f"- **Lap {lap}**: {reason}. Excluded from corner consistency, corner time-loss, and the "
                f"recommendation engine (so it can't be mistaken for the 'best' or 'worst' lap through a "
                f"corner) — but its telemetry is still shown on the raw plots"
                + (", coloured black, " if spin_by_lap.get(lap) else " ")
                + "so it isn't hidden from view.\n"
            )
    else:
        lines.append("- None — no out-lap or spin detected in this session.\n")

    outliers = detect_likely_outliers(lap_times)
    flagged = outliers[outliers["likely_outlier"] & ~outliers["lap"].isin(excluded_laps)]
    if len(flagged):
        lines.append("\n## Other slow laps worth a manual look (not already excluded above)\n")
        for _, row in flagged.iterrows():
            lines.append(f"- Lap {int(row['lap'])}: {row['lap_time_s']:.3f}s, "
                         f"{row['pct_slower_than_median']:.1f}% slower than the session median "
                         f"({outliers['lap_time_s'].median():.3f}s).\n")

    lines.append(f"\n## Grip envelope\n")
    lines.append(f"- Peak lateral: {summary_stats['peak_lateral_g']:.2f}g\n")
    lines.append(f"- Peak braking: {summary_stats['peak_braking_g']:.2f}g\n")
    lines.append(f"- Peak combined: {summary_stats['peak_combined_g']:.2f}g\n")

    lines.append(f"\n## Corner consistency (top 3 least consistent by minimum-speed CV%)\n")
    lines.append("Computed excluding out-lap(s) and any spin lap(s) — see exclusions above.\n\n")
    lines.append("| Corner | Mean min speed (km/h) | CV% | Apex std-X (m) | Apex std-Y (m) | Apex RMS (m) |\n")
    lines.append("|---|---|---|---|---|---|\n")
    for _, row in consistency.head(3).iterrows():
        lines.append(f"| {int(row['corner_id'])} | {row['min_speed_mean']:.1f} | {row['min_speed_cv_pct']:.1f}% | "
                     f"{row['apex_std_x_m']:.2f} | {row['apex_std_y_m']:.2f} | {row['apex_rms_m']:.2f} |\n")

    total_loss = loss_summary["time_loss_s"].clip(lower=0).sum() if len(loss_summary) else 0.0
    lines.append(f"\n## Corner-by-corner time loss\n")
    lines.append(f"**Total identifiable corner loss across the session: {total_loss:.3f}s** "
                 f"(sum of the single worst-lap loss in each corner, excluding out-lap(s)/spin lap(s) — "
                 f"not all recoverable in one lap simultaneously, since the worst instances may come from "
                 f"different laps).\n\n")
    lines.append("| Corner | Worst lap | Time lost (s) |\n|---|---|---|\n")
    for _, row in loss_summary.head(5).iterrows():
        lines.append(f"| T{int(row['corner_id'])} | L{int(row['worst_lap'])} | "
                     f"{row['time_loss_s']:+.3f} |\n")

    if recommendations:
        lines.append(f"\n## Top improvement opportunities\n")
        for r in recommendations:
            lines.append(f"\n### Corner {r['corner_id']} — up to {r['expected_effect_s']:.3f}s available\n")
            lines.append(f"- **Observation:** {r['observation']}\n")
            lines.append(f"- **Evidence:** {r['evidence']}\n")
            lines.append(f"- **Diagnosis:** {r['diagnosis']}\n")
            lines.append(f"- **Proposed change:** {r['proposed_change']}\n")
            lines.append(f"- **Expected effect:** up to {r['expected_effect_s']:.3f}s if corrected to "
                         f"match Lap {r['best_lap']}'s pace through this corner\n")

    lines.append("\n## Methodology notes (describe the pipeline, not this session's data)\n")
    lines.append(
        "- Accel/brake are derived proxies (RPM slope; calibrated longitudinal G), not measured pedal data.\n"
        "- G-force channels are calibrated per-session by regressing against independent GPS-derived "
        "acceleration, not a fixed constant, so this recalibrates automatically for different sessions.\n"
        "- Corner detection is automatic from track curvature and should be visually spot-checked against "
        "the corner-validation track map for a new track shape.\n"
        "- Spin detection flags sustained windows of high yaw-rate combined with low measured grip "
        "(a spinning kart keeps rotating on momentum after the tyres let go) — a heuristic, tune the "
        "thresholds in spins.py if it over- or under-triggers on a given dataset.\n"
        "- Recommendations are threshold-based heuristics on entry/apex/exit speed samples, not a "
        "validated diagnostic model — treat as a starting point for review, not a final verdict.\n"
        "- Reference line is built from a single lap (the fastest, excluding out-lap(s)/spin lap(s)), "
        "not a smoothed multi-lap centerline.\n"
    )
    return "".join(lines)
