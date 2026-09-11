"""
app.py — Interactive dashboard for the go-kart telemetry analysis.

Run with (from the project root):
    streamlit run dashboard/app.py

(This file lives in dashboard/, one level below the project root. It
locates data/ and src/ relative to its own file location, not the current
working directory, so it works whether you run it from the project root
or from inside dashboard/.)
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # dashboard/app.py -> project root
sys.path.insert(0, str(ROOT / "src"))

import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

from loader import load_session, align_session_to_reference, discover_lap_files
from derive import add_derived_channels_session
from spins import detect_spin_laps
from corners import detect_corners, corner_metrics_by_lap
from analysis import (
    lap_time_table, best_theoretical_lap, racing_line_consistency, consistency_summary,
    corner_apex_positions, corner_time_loss, generate_recommendations, driver_summary,
    corner_loss_summary, resolve_excluded_laps, filter_excluded_laps,
)
from plots import (
    plot_track_map, plot_track_map_with_corners, plot_telemetry_stack, plot_gg_diagram,
    plot_lap_comparison_speed, compute_delta_time, plot_delta_time,
    plot_lap_times, plot_theoretical_vs_best, plot_corner_consistency,
)

RAW_DIR = ROOT / "data" / "raw_sample"

# The first lap of a session is conventionally an out-lap (cold tyres/brakes,
# not a genuine attempt) and is excluded from corner-comparison analysis
# below. Change/extend this if a given session's out-lap isn't lap 1.
OUT_LAPS = (1,)

st.set_page_config(page_title="Go-Kart Telemetry Dashboard", layout="wide")


@st.cache_data
def run_pipeline():
    lap_files = discover_lap_files(RAW_DIR)
    laps, summary = load_session(RAW_DIR, lap_files)
    laps = add_derived_channels_session(laps)
    aligned, ref_line = align_session_to_reference(laps)
    aligned, spin_by_lap = detect_spin_laps(aligned)
    excluded_laps = resolve_excluded_laps(list(aligned.keys()), spin_by_lap, out_laps=OUT_LAPS)
    comparison_laps = {k: v for k, v in aligned.items() if k not in excluded_laps}

    corners = detect_corners(ref_line, min_corner_length_m=12)
    corner_metrics = corner_metrics_by_lap(aligned, corners)
    corner_metrics_cmp = filter_excluded_laps(corner_metrics, excluded_laps)

    racing_line = racing_line_consistency(comparison_laps, corners)
    consistency = consistency_summary(corner_metrics_cmp, racing_line)
    theoretical = best_theoretical_lap(summary)
    lap_times = lap_time_table(summary, laps_available=list(lap_files.keys()))

    candidate_fastest = [k for k in aligned if k not in excluded_laps] or list(aligned.keys())
    fastest_lap = min(candidate_fastest, key=lambda k: aligned[k]["time_s"].iloc[-1])

    delta_df = compute_delta_time(aligned, reference_lap=fastest_lap)
    time_loss = corner_time_loss(delta_df, corners)
    time_loss_cmp = filter_excluded_laps(time_loss, excluded_laps)
    apex_positions = corner_apex_positions(comparison_laps, corners)
    recommendations = generate_recommendations(corner_metrics_cmp, time_loss_cmp, apex_positions, corners, top_n=5)
    loss_summary = corner_loss_summary(time_loss_cmp, corners)
    summary_stats = driver_summary(lap_times, theoretical, consistency, corner_metrics_cmp)
    return {
        "aligned": aligned, "ref_line": ref_line, "summary": summary,
        "corners": corners, "corner_metrics": corner_metrics,
        "consistency": consistency, "theoretical": theoretical,
        "lap_times": lap_times, "fastest_lap": fastest_lap, "time_loss": time_loss,
        "apex_positions": apex_positions, "recommendations": recommendations,
        "summary_stats": summary_stats, "loss_summary": loss_summary,
        "spin_by_lap": spin_by_lap, "excluded_laps": excluded_laps,
    }


data = run_pipeline()
aligned = data["aligned"]
ref_line = data["ref_line"]
corners = data["corners"]
lap_numbers = sorted(aligned.keys())
fastest_lap = data["fastest_lap"]
spin_by_lap = data["spin_by_lap"]
spin_laps = {lap for lap, is_spin in spin_by_lap.items() if is_spin}
excluded_laps = data["excluded_laps"]

st.title("🏁 Go-Kart Telemetry Dashboard")
st.caption("Go Kart Club of Victoria session — Alfano6 logger. Acceleration/brake are derived proxies, not measured pedal data.")

if excluded_laps:
    reason_bits = []
    for lap in excluded_laps:
        reason = "spin detected" if lap in spin_laps else "out-lap"
        reason_bits.append(f"Lap {lap} ({reason})")
    st.info(
        f"Excluded from corner-comparison analysis (consistency, time-loss, recommendations): "
        f"{', '.join(reason_bits)}. Still shown on the raw plots"
        + (" — spin laps in **black**." if spin_laps else ".")
    )

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
st.sidebar.header("Controls")
selected_laps = st.sidebar.multiselect(
    "Laps to include in comparison plots", lap_numbers, default=lap_numbers
)
reference_lap = st.sidebar.selectbox(
    "Reference lap (delta-time baseline)", lap_numbers,
    index=lap_numbers.index(fastest_lap),
)
single_lap = st.sidebar.selectbox(
    "Lap for telemetry stack", lap_numbers, index=lap_numbers.index(fastest_lap),
    format_func=lambda l: f"Lap {l} (SPIN)" if l in spin_laps else f"Lap {l}",
)
corner_numbers = sorted(corners["corner_id"].astype(int).tolist())
selected_corner = st.sidebar.selectbox("Corner for detail view", corner_numbers)
show_corners = st.sidebar.checkbox("Shade detected corners on plots", value=True)

if not selected_laps:
    st.warning("Select at least one lap in the sidebar.")
    st.stop()

selected_aligned = {k: v for k, v in aligned.items() if k in selected_laps}
corners_arg = corners if show_corners else None

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_overview, tab_track, tab_telemetry, tab_compare, tab_delta, tab_corner, tab_gg, tab_consistency, tab_recs, tab_conclusions = st.tabs(
    ["Overview", "Track Map", "Telemetry Stack", "Lap Comparison",
     "Delta-Time", "Corner Detail", "G-G Diagram", "Consistency", "Recommendations", "Conclusions"]
)

with tab_overview:
    st.subheader("Driver summary")
    stats = data["summary_stats"]
    total_identifiable_loss = float(data["loss_summary"]["time_loss_s"].clip(lower=0).sum()) if len(data["loss_summary"]) else 0.0
    scol1, scol2, scol3, scol4, scol5 = st.columns(5)
    scol1.metric("Best lap", f"Lap {stats['best_lap_number']}", f"{stats['best_lap_s']:.3f}s")
    scol2.metric("Theoretical best", f"{stats['theoretical_best_s']:.3f}s",
                 f"-{stats['potential_gain_s']:.3f}s potential")
    scol3.metric("Total identifiable corner loss", f"{total_identifiable_loss:.3f}s")
    scol4.metric("Least consistent corner", f"T{stats['least_consistent_corner']}",
                 f"{stats['least_consistent_cv_pct']:.1f}% CV")
    scol5.metric("Peak combined G", f"{stats['peak_combined_g']:.2f}g",
                 f"lat {stats['peak_lateral_g']:.2f}g / brake {stats['peak_braking_g']:.2f}g")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Lap times")
        fig, ax = plt.subplots(figsize=(8, 5))
        plot_lap_times(data["lap_times"], ax=ax, spin_laps=spin_laps)
        st.pyplot(fig); plt.close(fig)
    with col2:
        st.subheader("Theoretical vs. best actual lap")
        fig, ax = plt.subplots(figsize=(6, 5))
        plot_theoretical_vs_best(data["theoretical"], ax=ax)
        st.pyplot(fig); plt.close(fig)

    st.subheader("Session lap/sector times (seconds)")
    st.dataframe(data["lap_times"].round(3), use_container_width=True)

    st.subheader("Corner-by-corner time loss")
    st.caption("Single worst-lap loss per corner, excluding out-lap(s)/spin lap(s). Sum is an "
               "upper-bound estimate — the worst instances may come from different laps, so not "
               "all recoverable simultaneously.")
    st.dataframe(data["loss_summary"].round(3), use_container_width=True, hide_index=True)

with tab_track:
    st.subheader("Track map")
    fig, ax = plt.subplots(figsize=(8, 8))
    plot_track_map(selected_aligned, ax=ax, spin_laps=spin_laps)
    st.pyplot(fig); plt.close(fig)

    st.subheader("Corner validation")
    st.caption("Automated corner detection, checked visually against the actual GPS trace.")
    fig, ax = plt.subplots(figsize=(8, 8))
    plot_track_map_with_corners(ref_line, corners, ax=ax)
    st.pyplot(fig); plt.close(fig)

with tab_telemetry:
    title = f"Telemetry stack — Lap {single_lap}"
    if single_lap in spin_laps:
        title += " ⚠ SPIN DETECTED"
    st.subheader(title)
    fig = plot_telemetry_stack(aligned[single_lap], single_lap, corners=corners_arg,
                                is_spin=single_lap in spin_laps)
    st.pyplot(fig); plt.close(fig)

with tab_compare:
    st.subheader("Speed comparison across laps")
    fig, ax = plt.subplots(figsize=(12, 5))
    plot_lap_comparison_speed(selected_aligned, corners=corners_arg, ax=ax, spin_laps=spin_laps)
    st.pyplot(fig); plt.close(fig)

with tab_delta:
    st.subheader(f"Delta-time vs Lap {reference_lap}")
    delta_df = compute_delta_time({k: v for k, v in aligned.items() if k in selected_laps or k == reference_lap},
                                   reference_lap=reference_lap)
    fig, ax = plt.subplots(figsize=(12, 5))
    plot_delta_time(delta_df, reference_lap=reference_lap, ax=ax, spin_laps=spin_laps)
    st.pyplot(fig); plt.close(fig)

with tab_corner:
    corner_row = corners[corners["corner_id"] == selected_corner].iloc[0]
    margin = max((corner_row["end_m"] - corner_row["start_m"]) * 0.5, 15)
    st.subheader(f"Corner {selected_corner} detail")

    fig, ax = plt.subplots(figsize=(10, 5))
    from plots import get_lap_colors, _lap_label
    colors = get_lap_colors(aligned.keys(), spin_laps)
    for lap_num, df in aligned.items():
        ax.plot(df["ref_distance_m"], df["Speed GPS"], label=_lap_label(lap_num, spin_laps),
                color=colors[lap_num], alpha=0.85)
    ax.axvspan(corner_row["start_m"], corner_row["end_m"], color="grey", alpha=0.15)
    ax.set_xlim(corner_row["start_m"] - margin, corner_row["end_m"] + margin)
    ax.set_xlabel("Distance along reference line (m)"); ax.set_ylabel("Speed (km/h)")
    ax.set_title(f"Speed through Corner {selected_corner} (all laps, black = spin)")
    ax.legend(fontsize=8)
    st.pyplot(fig); plt.close(fig)

    st.subheader("Per-lap metrics for this corner")
    st.caption("Includes out-lap/spin laps for reference (marked below), even though they're "
               "excluded from the consistency/recommendation calculations.")
    cm = data["corner_metrics"]
    tl = data["time_loss"]
    apex = data["apex_positions"]
    merged = tl.merge(cm, on=["corner_id", "lap"], how="left").merge(
        apex[["corner_id", "lap", "apex_deviation_m"]], on=["corner_id", "lap"], how="left"
    )
    detail = merged[merged["corner_id"] == selected_corner].sort_values("lap")[
        ["lap", "entry_speed_kmh", "min_speed_kmh", "exit_speed_kmh",
         "brake_point_offset_m", "accel_point_offset_m", "apex_deviation_m",
         "peak_lateral_g", "peak_braking_g", "time_loss_s"]
    ].round(2)
    detail["excluded_from_comparison"] = detail["lap"].isin(excluded_laps)
    st.dataframe(detail, use_container_width=True, hide_index=True)
    st.caption("Brake/accel point offset: metres before/after the corner's start where "
               "braking/acceleration was detected. Time loss: seconds gained (-) or lost (+) "
               "specifically within this corner vs the fastest lap.")

with tab_gg:
    st.subheader("G-G diagram")
    fig, ax = plt.subplots(figsize=(9, 5))
    plot_gg_diagram(selected_aligned, ax=ax, spin_laps=spin_laps)
    st.pyplot(fig); plt.close(fig)

with tab_consistency:
    st.subheader("Corner consistency ranking")
    st.caption("Excludes out-lap(s)/spin lap(s) from the underlying per-lap metrics.")
    fig, ax = plt.subplots(figsize=(9, 5))
    plot_corner_consistency(data["consistency"], ax=ax)
    st.pyplot(fig); plt.close(fig)
    st.subheader("Per-corner detail")
    st.dataframe(data["consistency"].round(2), use_container_width=True)

with tab_recs:
    st.subheader("Automatic improvement recommendations")
    st.caption("Auto-generated from threshold rules on the computed metrics — a starting point "
               "for review, not a validated diagnosis. Each entry compares the lap that lost the "
               "most time in that corner against the lap that was fastest through it (out-lap(s) "
               "and spin lap(s) excluded from this comparison).")
    recs = data["recommendations"]
    if not recs:
        st.info("No significant corner-specific time losses detected.")
    for r in recs:
        with st.container(border=True):
            st.markdown(f"**Corner {r['corner_id']} — up to {r['expected_effect_s']:.3f}s available**")
            st.write(f"**Observation:** {r['observation']}")
            st.write(f"**Evidence:** {r['evidence']}")
            st.write(f"**Diagnosis:** {r['diagnosis']}")
            st.write(f"**Proposed change:** {r['proposed_change']}")
            st.write(f"**Expected effect:** up to {r['expected_effect_s']:.3f}s if corrected to "
                     f"match Lap {r['best_lap']}'s pace through this corner")

with tab_conclusions:
    conclusions_path = ROOT / "outputs" / "09_engineering_conclusions.md"
    if conclusions_path.exists():
        st.markdown(conclusions_path.read_text())
    else:
        st.info("Run `python src/main.py` first to generate the engineering conclusions.")
