"""
main.py — Full pipeline: raw CSVs in, cleaned data + every plot out.

Usage:
    python src/main.py
"""

from pathlib import Path

import pandas as pd

from loader import load_session, align_session_to_reference, discover_lap_files
from derive import add_derived_channels_session
from spins import detect_spin_laps
from corners import detect_corners, corner_metrics_by_lap
from analysis import (
    lap_time_table, best_theoretical_lap, racing_line_consistency, consistency_summary,
    corner_apex_positions, corner_time_loss, generate_recommendations, driver_summary,
    corner_loss_summary, generate_engineering_summary, resolve_excluded_laps, filter_excluded_laps,
)
from plots import (
    plot_track_map, plot_track_map_with_corners, plot_telemetry_stack, plot_gg_diagram,
    plot_lap_comparison_speed, compute_delta_time, plot_delta_time,
    plot_lap_times, plot_theoretical_vs_best, plot_corner_consistency,
)

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw_sample"
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUTS_DIR = ROOT / "outputs"

# The first lap of a session is conventionally an out-lap (cold tyres/brakes,
# not a genuine attempt) and is excluded from corner-comparison analysis
# below. Change/extend this if a given session's out-lap isn't lap 1.
OUT_LAPS = (1,)


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    lap_files = discover_lap_files(RAW_DIR)

    print("Loading and cleaning session...")
    laps, summary = load_session(RAW_DIR, lap_files)
    laps = add_derived_channels_session(laps)
    aligned, ref_line = align_session_to_reference(laps)

    print("Detecting spins...")
    aligned, spin_by_lap = detect_spin_laps(aligned)
    spin_laps = {lap for lap, is_spin in spin_by_lap.items() if is_spin}
    if spin_laps:
        print(f"  -> spin detected on lap(s): {sorted(spin_laps)} (shown in black on plots)")
    else:
        print("  -> no spins detected")

    excluded_laps = resolve_excluded_laps(list(aligned.keys()), spin_by_lap, out_laps=OUT_LAPS)
    comparison_laps = {k: v for k, v in aligned.items() if k not in excluded_laps}
    print(f"  -> laps excluded from corner-comparison analysis: {excluded_laps or 'none'}")

    for lap_num, df in aligned.items():
        df.to_csv(PROCESSED_DIR / f"lap{lap_num}_cleaned.csv", index=False)
    ref_line.to_csv(PROCESSED_DIR / "reference_line.csv", index=False)
    print(f"  -> cleaned data written to {PROCESSED_DIR}")

    print("Detecting corners...")
    corners = detect_corners(ref_line, min_corner_length_m=12)
    corners.to_csv(PROCESSED_DIR / "corners.csv", index=False)
    # full per-lap corner metrics (all laps, including excluded ones) for detail views
    corner_metrics = corner_metrics_by_lap(aligned, corners)
    corner_metrics.to_csv(PROCESSED_DIR / "corner_metrics_by_lap.csv", index=False)
    # filtered version (out-laps/spins removed) used for anything that judges "best/worst lap"
    corner_metrics_cmp = filter_excluded_laps(corner_metrics, excluded_laps)
    print(f"  -> {len(corners)} corners detected")

    fig, ax = plt.subplots(figsize=(9, 9))
    plot_track_map_with_corners(ref_line, corners, ax=ax)
    fig.tight_layout(); fig.savefig(OUTPUTS_DIR / "01b_track_map_corners_validated.png", dpi=140); plt.close(fig)

    print("Generating plots...")

    fig, ax = plt.subplots(figsize=(8, 8))
    plot_track_map(aligned, ax=ax, spin_laps=spin_laps)
    fig.tight_layout(); fig.savefig(OUTPUTS_DIR / "01_track_map.png", dpi=140); plt.close(fig)

    candidate_fastest = [k for k in aligned if k not in excluded_laps] or list(aligned.keys())
    fastest_lap = min(candidate_fastest, key=lambda k: aligned[k]["time_s"].iloc[-1])
    fig = plot_telemetry_stack(aligned[fastest_lap], fastest_lap, corners=corners,
                                is_spin=fastest_lap in spin_laps)
    fig.savefig(OUTPUTS_DIR / "02_telemetry_stack_fastest_lap.png", dpi=140); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    plot_gg_diagram(aligned, ax=ax, spin_laps=spin_laps)
    fig.tight_layout(); fig.savefig(OUTPUTS_DIR / "03_gg_diagram.png", dpi=140); plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    plot_lap_comparison_speed(aligned, corners=corners, ax=ax, spin_laps=spin_laps)
    fig.tight_layout(); fig.savefig(OUTPUTS_DIR / "04_lap_comparison_speed.png", dpi=140); plt.close(fig)

    delta_df = compute_delta_time(aligned, reference_lap=fastest_lap)
    delta_df.to_csv(PROCESSED_DIR / "delta_time.csv", index=False)
    fig, ax = plt.subplots(figsize=(12, 5))
    plot_delta_time(delta_df, reference_lap=fastest_lap, ax=ax, spin_laps=spin_laps)
    fig.tight_layout(); fig.savefig(OUTPUTS_DIR / "05_delta_time.png", dpi=140); plt.close(fig)

    print(f"  -> plots written to {OUTPUTS_DIR}")

    print("Computing consistency metrics + theoretical best lap...")
    # racing line / consistency / recommendations all use comparison_laps
    # (out-lap(s) and spin lap(s) excluded) so they aren't skewed by a cold
    # out-lap or a lap that crashed mid-corner due to a spin.
    racing_line = racing_line_consistency(comparison_laps, corners)
    consistency = consistency_summary(corner_metrics_cmp, racing_line)
    consistency.to_csv(PROCESSED_DIR / "consistency_by_corner.csv", index=False)

    theoretical = best_theoretical_lap(summary)
    lap_times = lap_time_table(summary, laps_available=list(lap_files.keys()))
    lap_times.to_csv(PROCESSED_DIR / "lap_time_table.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    plot_lap_times(lap_times, ax=ax, spin_laps=spin_laps)
    fig.tight_layout(); fig.savefig(OUTPUTS_DIR / "06_lap_times.png", dpi=140); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    plot_theoretical_vs_best(theoretical, ax=ax)
    fig.tight_layout(); fig.savefig(OUTPUTS_DIR / "07_theoretical_vs_best.png", dpi=140); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    plot_corner_consistency(consistency, ax=ax)
    fig.tight_layout(); fig.savefig(OUTPUTS_DIR / "08_corner_consistency.png", dpi=140); plt.close(fig)

    print("Computing corner-level time loss and generating recommendations...")
    time_loss = corner_time_loss(delta_df, corners)
    time_loss.to_csv(PROCESSED_DIR / "corner_time_loss.csv", index=False)
    time_loss_cmp = filter_excluded_laps(time_loss, excluded_laps)
    apex_positions = corner_apex_positions(comparison_laps, corners)
    recommendations = generate_recommendations(corner_metrics_cmp, time_loss_cmp, apex_positions, corners, top_n=5)
    loss_summary = corner_loss_summary(time_loss_cmp, corners)
    loss_summary.to_csv(PROCESSED_DIR / "corner_loss_summary.csv", index=False)
    total_identifiable_loss = loss_summary["time_loss_s"].clip(lower=0).sum() if len(loss_summary) else 0.0

    rec_lines = ["# Automatic Improvement Recommendations\n",
                 "Auto-generated from threshold rules on the computed metrics — a starting "
                 "point for review, not a validated diagnosis. Each entry compares the lap "
                 "that lost the most time in that corner against the lap that was fastest "
                 "through it (out-lap(s) and spin lap(s) excluded from this comparison — "
                 f"excluded: {excluded_laps or 'none'}).\n",
                 f"\n**Total identifiable corner loss across the session: "
                 f"{total_identifiable_loss:.3f}s** (sum of the single worst-lap loss per "
                 f"corner — see caveat in the engineering summary).\n"]
    for r in recommendations:
        rec_lines.append(
            f"\n## Corner {r['corner_id']} — up to {r['expected_effect_s']:.3f}s available\n"
            f"- **Observation:** {r['observation']}\n"
            f"- **Evidence:** {r['evidence']}\n"
            f"- **Diagnosis:** {r['diagnosis']}\n"
            f"- **Proposed change:** {r['proposed_change']}\n"
            f"- **Expected effect:** up to {r['expected_effect_s']:.3f}s if corrected to match "
            f"Lap {r['best_lap']}'s pace through this corner\n"
        )
    (OUTPUTS_DIR / "10_recommendations.md").write_text("".join(rec_lines))
    print(f"  -> {len(recommendations)} recommendations written to outputs/10_recommendations.md")
    print(f"  -> total identifiable corner loss: {total_identifiable_loss:.3f}s")

    print("\n=== Session summary (all times in seconds) ===")
    print(lap_times.round(3).to_string(index=False))
    print(f"\nFastest lap with raw data (excluding out-lap/spin): Lap {fastest_lap} "
          f"({summary.loc[summary['lap']==fastest_lap, 'time lap'].values[0]/1000:.3f}s)")
    print(f"Theoretical best lap (best sectors stitched): "
          f"{theoretical['theoretical_best_s']:.3f}s "
          f"(actual best: Lap {theoretical['best_actual_lap_number']}, "
          f"{theoretical['best_actual_lap_s']:.3f}s, "
          f"gap: {theoretical['gap_s']:.3f}s)")
    print("\nLeast consistent corners (highest speed variability, excluding out-lap/spins):")
    print(consistency.head(3)[["corner_id", "min_speed_mean", "min_speed_std", "min_speed_cv_pct"]]
          .round(2).to_string(index=False))

    summary_stats = driver_summary(lap_times, theoretical, consistency, corner_metrics_cmp)
    print("\n=== Driver summary ===")
    for k, v in summary_stats.items():
        print(f"  {k}: {v}")

    print("\nWriting auto-generated engineering summary...")
    engineering_summary_md = generate_engineering_summary(
        lap_times, theoretical, consistency, recommendations, loss_summary, summary_stats,
        len(corners), excluded_laps=excluded_laps, spin_by_lap=spin_by_lap,
    )
    (OUTPUTS_DIR / "09_engineering_conclusions.md").write_text(engineering_summary_md)
    print(f"  -> outputs/09_engineering_conclusions.md regenerated from this session's data "
          f"(this file is no longer hand-written — it updates automatically on every run)")

    return {
        "laps": aligned, "summary": summary, "corners": corners,
        "corner_metrics": corner_metrics, "consistency": consistency,
        "theoretical": theoretical, "delta_df": delta_df,
        "spin_by_lap": spin_by_lap, "excluded_laps": excluded_laps,
    }


if __name__ == "__main__":
    main()
