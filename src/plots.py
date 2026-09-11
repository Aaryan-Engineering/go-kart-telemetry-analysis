"""
plots.py — Telemetry visualisations.

All cross-lap plots use `ref_distance_m` (distance along the shared
reference line, see loader.align_session_to_reference) as the x-axis, not
each lap's own distance — otherwise corners don't line up between laps
with different racing lines.

Laps with a detected spin (see spins.py) are drawn in black across every
cross-lap plot, so a lap that was fast in places but wrecked by a spin
elsewhere isn't mistaken for a clean, representative lap. Pass `spin_laps`
(an iterable/set of lap numbers) to any plotting function below to enable
this; it's optional and defaults to no laps flagged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SPIN_COLOR = "black"
_PALETTE = plt.cm.tab10.colors


def get_lap_colors(lap_numbers, spin_laps=None) -> dict:
    """
    Consistent lap -> colour mapping shared by every plot in this module.
    Spin laps are always black regardless of palette position, so they
    read the same way on every chart.
    """
    spin_laps = set(spin_laps or [])
    return {
        lap: (SPIN_COLOR if lap in spin_laps else _PALETTE[i % len(_PALETTE)])
        for i, lap in enumerate(sorted(lap_numbers))
    }


def _lap_label(lap_num, spin_laps=None) -> str:
    return f"Lap {lap_num} (SPIN)" if spin_laps and lap_num in spin_laps else f"Lap {lap_num}"


def plot_lap_times(lap_times: pd.DataFrame, ax=None, spin_laps=None):
    """Bar chart of every lap's time, in seconds. Laps without raw telemetry are shown lighter.
    A lap with a detected spin is shown in black regardless of whether it was fastest."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    spin_laps = set(spin_laps or [])
    fastest_idx = lap_times["lap_time_s"].idxmin()
    colors = []
    for i, (lap, has) in enumerate(zip(lap_times["lap"], lap_times["has_raw_data"])):
        if lap in spin_laps:
            colors.append(SPIN_COLOR)
        elif i == fastest_idx:
            colors.append("tab:green")
        elif has:
            colors.append("tab:blue")
        else:
            colors.append("lightgrey")
    ax.bar(lap_times["lap"], lap_times["lap_time_s"], color=colors)
    ax.set_xlabel("Lap"); ax.set_ylabel("Lap time (s)")
    title = "Lap times across the session (green = fastest, grey = no raw telemetry"
    title += ", black = spin detected)" if spin_laps else ")"
    ax.set_title(title)
    ax.set_xticks(lap_times["lap"])
    ymin = lap_times["lap_time_s"].min() - 0.5
    ax.set_ylim(bottom=ymin)
    return ax


def plot_theoretical_vs_best(theoretical: dict, ax=None):
    """Stacked-bar comparison: best-sector theoretical lap vs the best actual lap, sector by sector."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))

    labels = ["Theoretical best\n(best sectors stitched)",
              f"Best actual lap\n(Lap {theoretical['best_actual_lap_number']})"]
    s1 = [theoretical["best_sector_1_s"], theoretical["best_actual_lap_sector_1_s"]]
    s2 = [theoretical["best_sector_2_s"], theoretical["best_actual_lap_sector_2_s"]]
    s3 = [theoretical["best_sector_3_s"], theoretical["best_actual_lap_sector_3_s"]]

    ax.bar(labels, s1, label="Sector 1", color="tab:blue")
    ax.bar(labels, s2, bottom=s1, label="Sector 2", color="tab:orange")
    bottom_s3 = [a + b for a, b in zip(s1, s2)]
    ax.bar(labels, s3, bottom=bottom_s3, label="Sector 3", color="tab:green")

    totals = [theoretical["theoretical_best_s"], theoretical["best_actual_lap_s"]]
    for i, total in enumerate(totals):
        ax.text(i, total + 0.15, f"{total:.3f}s", ha="center", fontweight="bold")

    ax.set_ylabel("Time (s)")
    ax.set_title(f"Theoretical best vs. best actual lap\n"
                 f"(gap: {theoretical['gap_s']:.3f}s)")
    ax.legend()
    ax.set_ylim(0, max(totals) + 1)
    return ax


def plot_corner_consistency(consistency: pd.DataFrame, ax=None):
    """Bar chart ranking corners by minimum-speed variability (coefficient of variation).
    Computed upstream excluding out-laps/spin laps — see analysis.consistency_summary."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 5))
    ordered = consistency.sort_values("min_speed_cv_pct", ascending=False)
    colors = plt.cm.Reds(np.linspace(0.8, 0.3, len(ordered)))
    ax.bar(ordered["corner_id"].astype(str), ordered["min_speed_cv_pct"], color=colors)
    ax.set_xlabel("Corner"); ax.set_ylabel("Min-speed variability across laps (CV %)")
    ax.set_title("Corner consistency ranking (higher = less consistent, biggest opportunity)")
    return ax


def plot_track_map_with_corners(ref_line: pd.DataFrame, corners: pd.DataFrame, ax=None):
    """
    Track map with detected corners numbered at their midpoint, for visual
    validation of the automated corner-detection algorithm against the
    actual GPS trace.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 9))
    ax.plot(ref_line["x_m"], ref_line["y_m"], color="black", lw=1.5, alpha=0.7)

    x = ref_line["x_m"].to_numpy()
    y = ref_line["y_m"].to_numpy()
    s = ref_line["arc_length_m"].to_numpy()

    for _, c in corners.iterrows():
        mask = (s >= c["start_m"]) & (s <= c["end_m"])
        if not mask.any():
            continue
        ax.plot(x[mask], y[mask], color="tab:red", lw=3, alpha=0.8)
        mid_idx = np.where(mask)[0][len(np.where(mask)[0]) // 2]
        ax.annotate(f"T{int(c['corner_id'])}", (x[mid_idx], y[mid_idx]),
                    fontsize=11, fontweight="bold", color="white",
                    ha="center", va="center",
                    bbox=dict(boxstyle="circle", facecolor="tab:red", edgecolor="none"))

    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
    ax.set_title("Detected corners (for visual validation against the GPS track)")
    ax.set_aspect("equal", adjustable="datalim")
    return ax


def plot_track_map(aligned: dict[int, pd.DataFrame], ax=None, title="Track map", spin_laps=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))
    colors = get_lap_colors(aligned.keys(), spin_laps)
    for lap_num, df in aligned.items():
        ax.plot(df["x_m"], df["y_m"], label=_lap_label(lap_num, spin_laps),
                color=colors[lap_num], alpha=0.85 if lap_num not in (spin_laps or []) else 1.0,
                lw=2.2 if spin_laps and lap_num in spin_laps else 1.5)
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
    ax.set_title(title); ax.set_aspect("equal"); ax.legend(fontsize=8)
    return ax


def plot_telemetry_stack(df: pd.DataFrame, lap_number: int, corners: pd.DataFrame | None = None,
                          is_spin: bool = False):
    """Speed / RPM / acceleration proxy / brake proxy vs aligned distance, one lap."""
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    line_color = SPIN_COLOR if is_spin else "tab:blue"
    axes[0].plot(df["ref_distance_m"], df["Speed GPS"], color=line_color)
    axes[0].set_ylabel("Speed\n(km/h)")

    axes[1].plot(df["ref_distance_m"], df["RPM"], color="tab:purple" if not is_spin else SPIN_COLOR)
    axes[1].set_ylabel("RPM")

    axes[2].fill_between(df["ref_distance_m"], df["accel_proxy"], color="tab:green", alpha=0.7)
    axes[2].set_ylabel("Accel\nproxy (%)")
    axes[2].set_ylim(0, 105)

    axes[3].fill_between(df["ref_distance_m"], df["brake_proxy"], color="tab:red", alpha=0.7)
    axes[3].set_ylabel("Brake\nproxy (%)")
    axes[3].set_ylim(0, 105)
    axes[3].set_xlabel("Distance along reference line (m)")

    if "spin_flag" in df.columns and df["spin_flag"].any():
        for ax in axes:
            for _, seg in df[df["spin_flag"]].groupby((~df["spin_flag"]).cumsum()):
                if seg["spin_flag"].any():
                    ax.axvspan(seg["ref_distance_m"].min(), seg["ref_distance_m"].max(),
                               color=SPIN_COLOR, alpha=0.15)

    if corners is not None:
        for ax in axes:
            for _, c in corners.iterrows():
                ax.axvspan(c["start_m"], c["end_m"], color="grey", alpha=0.12)

    title = f"Lap {lap_number} — telemetry stack (accel/brake are derived proxies, not measured)"
    if is_spin:
        title = f"⚠ SPIN DETECTED — {title}"
    fig.suptitle(title, color=SPIN_COLOR if is_spin else "black")
    fig.tight_layout()
    return fig


def plot_gg_diagram(laps: dict[int, pd.DataFrame], ax=None, spin_laps=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))
    colors = get_lap_colors(laps.keys(), spin_laps)
    for lap_num, df in laps.items():
        ax.scatter(df["lateral_g"], df["longitudinal_g"], s=4, alpha=0.4,
                   label=_lap_label(lap_num, spin_laps), color=colors[lap_num])
    ax.axhline(0, color="grey", lw=0.5); ax.axvline(0, color="grey", lw=0.5)
    ax.set_xlabel("Lateral G"); ax.set_ylabel("Longitudinal G (+ = braking, - = accelerating)")
    ax.set_title("G-G diagram — grip envelope usage")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(fontsize=8, markerscale=3)
    return ax


def plot_lap_comparison_speed(aligned: dict[int, pd.DataFrame], corners: pd.DataFrame | None = None,
                               ax=None, spin_laps=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 5))
    colors = get_lap_colors(aligned.keys(), spin_laps)
    for lap_num, df in aligned.items():
        is_spin = bool(spin_laps and lap_num in spin_laps)
        ax.plot(df["ref_distance_m"], df["Speed GPS"], label=_lap_label(lap_num, spin_laps),
                color=colors[lap_num], alpha=1.0 if is_spin else 0.8,
                lw=2.2 if is_spin else 1.5, zorder=5 if is_spin else 2)
    if corners is not None:
        for _, c in corners.iterrows():
            ax.axvspan(c["start_m"], c["end_m"], color="grey", alpha=0.1)
    ax.set_xlabel("Distance along reference line (m)"); ax.set_ylabel("Speed (km/h)")
    ax.set_title("Speed comparison across laps" + (" (black = spin detected)" if spin_laps else ""))
    ax.legend(fontsize=8)
    return ax


def compute_delta_time(
    aligned: dict[int, pd.DataFrame], reference_lap: int, distance_grid: np.ndarray | None = None
) -> pd.DataFrame:
    """
    Time delta of every lap vs a reference lap, evaluated on a common
    distance grid. Positive = slower than reference at that point on track.
    """
    ref = aligned[reference_lap].sort_values("ref_distance_m")
    ref_dist = ref["ref_distance_m"].to_numpy()
    ref_time = ref["time_s"].to_numpy()

    if distance_grid is None:
        distance_grid = np.linspace(ref_dist.min(), ref_dist.max(), 500)

    ref_time_grid = np.interp(distance_grid, ref_dist, ref_time)

    out = {"distance_m": distance_grid}
    for lap_num, df in aligned.items():
        d = df.sort_values("ref_distance_m")
        dist = d["ref_distance_m"].to_numpy()
        time = d["time_s"].to_numpy()
        # dedupe repeated distance values (can occur at track boundary projection)
        dist_u, idx = np.unique(dist, return_index=True)
        time_u = time[idx]
        lap_time_grid = np.interp(distance_grid, dist_u, time_u)
        out[f"lap_{lap_num}_delta_s"] = lap_time_grid - ref_time_grid

    return pd.DataFrame(out)


def plot_delta_time(delta_df: pd.DataFrame, reference_lap: int, ax=None, spin_laps=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 5))
    lap_numbers = [int(col.split("_")[1]) for col in delta_df.columns if col != "distance_m"]
    colors = get_lap_colors(lap_numbers, spin_laps)
    for col in delta_df.columns:
        if col == "distance_m" or col == f"lap_{reference_lap}_delta_s":
            continue
        lap_num = int(col.split("_")[1])
        ax.plot(delta_df["distance_m"], delta_df[col], label=_lap_label(lap_num, spin_laps),
                color=colors[lap_num])
    ax.axhline(0, color="black", lw=1)
    ax.set_xlabel("Distance along reference line (m)")
    ax.set_ylabel(f"Time delta vs Lap {reference_lap} (s)\n+ = slower")
    ax.set_title(f"Delta-time vs Lap {reference_lap} (fastest lap)")
    ax.legend(fontsize=8)
    return ax
