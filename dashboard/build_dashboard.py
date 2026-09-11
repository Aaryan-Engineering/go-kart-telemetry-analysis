"""
build_dashboard.py — Generates a single, self-contained, interactive HTML
dashboard. No server, no Streamlit, no install required to VIEW it — anyone
can just double-click the resulting outputs/dashboard.html and open it in
any browser.

Run with (from the project root):
    python dashboard/build_dashboard.py

This is a normal Python script — no special launch command needed, and no
install needed for anyone you send the HTML file to. You (the person
generating it) need `plotly` installed: pip install plotly

(This file lives in dashboard/, one level below the project root. It
locates data/, src/, and outputs/ relative to its own file location, not
the current working directory, so it works whether you run it from the
project root or from inside dashboard/.)
"""

import sys
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # dashboard/build_dashboard.py -> project root
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from loader import load_session, align_session_to_reference, discover_lap_files
from derive import add_derived_channels_session
from spins import detect_spin_laps
from corners import detect_corners, corner_metrics_by_lap
from analysis import (
    lap_time_table, best_theoretical_lap, racing_line_consistency, consistency_summary,
    corner_apex_positions, corner_time_loss, generate_recommendations, driver_summary,
    corner_loss_summary, resolve_excluded_laps, filter_excluded_laps,
)
from plots import compute_delta_time

RAW_DIR = ROOT / "data" / "raw_sample"
OUTPUT_HTML = ROOT / "outputs" / "dashboard.html"
CONCLUSIONS_MD = ROOT / "outputs" / "09_engineering_conclusions.md"

# The first lap of a session is conventionally an out-lap (cold tyres/brakes,
# not a genuine attempt) and is excluded from corner-comparison analysis
# below. Change/extend this if a given session's out-lap isn't lap 1.
OUT_LAPS = (1,)

# Cycling palette rather than a fixed lap->color dict, so this scales to any
# number of laps instead of silently running out of colors past lap 5.
PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
           "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
SPIN_COLOR = "#000000"


def lap_color(lap_num: int, lap_numbers: list, spin_laps: set | None = None) -> str:
    if spin_laps and lap_num in spin_laps:
        return SPIN_COLOR
    return PALETTE[lap_numbers.index(lap_num) % len(PALETTE)]


def lap_label(lap_num: int, spin_laps: set | None = None) -> str:
    return f"Lap {lap_num} (SPIN)" if spin_laps and lap_num in spin_laps else f"Lap {lap_num}"


def run_pipeline():
    lap_files = discover_lap_files(RAW_DIR)
    laps, summary = load_session(RAW_DIR, lap_files)
    laps = add_derived_channels_session(laps)
    aligned, ref_line = align_session_to_reference(laps)
    aligned, spin_by_lap = detect_spin_laps(aligned)
    spin_laps = {lap for lap, is_spin in spin_by_lap.items() if is_spin}
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
    return dict(aligned=aligned, ref_line=ref_line, summary=summary, corners=corners,
                corner_metrics=corner_metrics, consistency=consistency,
                theoretical=theoretical, lap_times=lap_times, fastest_lap=fastest_lap,
                delta_df=delta_df, time_loss=time_loss, apex_positions=apex_positions,
                recommendations=recommendations, summary_stats=summary_stats,
                loss_summary=loss_summary, spin_by_lap=spin_by_lap, spin_laps=spin_laps,
                excluded_laps=excluded_laps)


# ---------------------------------------------------------------------------
# Figure builders — each returns a plotly go.Figure
# ---------------------------------------------------------------------------

def fig_lap_times(lap_times: pd.DataFrame, spin_laps: set | None = None) -> go.Figure:
    spin_laps = spin_laps or set()
    colors = []
    for lap, t, has in zip(lap_times["lap"], lap_times["lap_time_s"], lap_times["has_raw_data"]):
        if lap in spin_laps:
            colors.append(SPIN_COLOR)
        elif t == lap_times["lap_time_s"].min():
            colors.append("#2ca02c")
        elif has:
            colors.append("#1f77b4")
        else:
            colors.append("#d3d3d3")
    fig = go.Figure(go.Bar(
        x=lap_times["lap"], y=lap_times["lap_time_s"], marker_color=colors,
        text=[f"{t:.2f}s" for t in lap_times["lap_time_s"]], textposition="outside",
    ))
    title = "Lap times across the session (green = fastest, grey = no raw telemetry"
    title += ", black = spin detected)" if spin_laps else ")"
    fig.update_layout(title=title,
                       xaxis_title="Lap", yaxis_title="Lap time (s)",
                       yaxis_range=[lap_times["lap_time_s"].min() - 0.5, lap_times["lap_time_s"].max() + 1])
    return fig


def fig_theoretical_vs_best(theoretical: dict) -> go.Figure:
    labels = ["Theoretical best<br>(best sectors stitched)", f"Best actual lap<br>(Lap {theoretical['best_actual_lap_number']})"]
    s1 = [theoretical["best_sector_1_s"], theoretical["best_actual_lap_sector_1_s"]]
    s2 = [theoretical["best_sector_2_s"], theoretical["best_actual_lap_sector_2_s"]]
    s3 = [theoretical["best_sector_3_s"], theoretical["best_actual_lap_sector_3_s"]]
    fig = go.Figure()
    fig.add_bar(x=labels, y=s1, name="Sector 1", marker_color="#1f77b4")
    fig.add_bar(x=labels, y=s2, name="Sector 2", marker_color="#ff7f0e")
    fig.add_bar(x=labels, y=s3, name="Sector 3", marker_color="#2ca02c")
    totals = [theoretical["theoretical_best_s"], theoretical["best_actual_lap_s"]]
    for label, total in zip(labels, totals):
        fig.add_annotation(x=label, y=total + 0.3, text=f"<b>{total:.3f}s</b>", showarrow=False)
    fig.update_layout(barmode="stack", title=f"Theoretical best vs. best actual lap (gap: {theoretical['gap_s']:.3f}s)",
                       yaxis_title="Time (s)")
    return fig


def fig_track_map_with_corners(ref_line: pd.DataFrame, corners: pd.DataFrame) -> go.Figure:
    """Annotated track map for visually validating the automated corner detection."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ref_line["x_m"], y=ref_line["y_m"], mode="lines",
                              line=dict(color="black", width=1.5), name="Track", showlegend=False))
    x = ref_line["x_m"].to_numpy(); y = ref_line["y_m"].to_numpy(); s = ref_line["arc_length_m"].to_numpy()
    for _, c in corners.iterrows():
        mask = (s >= c["start_m"]) & (s <= c["end_m"])
        if not mask.any():
            continue
        idx = np.where(mask)[0]
        fig.add_trace(go.Scatter(x=x[idx], y=y[idx], mode="lines", line=dict(color="#d62728", width=5),
                                  showlegend=False, hoverinfo="skip"))
        mid = idx[len(idx) // 2]
        fig.add_annotation(x=x[mid], y=y[mid], text=f"<b>T{int(c['corner_id'])}</b>",
                            showarrow=False, font=dict(color="white", size=12),
                            bgcolor="#d62728", borderpad=6, bordercolor="#d62728", borderwidth=1)
    fig.update_layout(title="Detected corners — visual validation against the GPS track",
                       xaxis_title="X (m)", yaxis_title="Y (m)",
                       yaxis=dict(scaleanchor="x", scaleratio=1))
    return fig


def fig_track_map(aligned: dict, spin_laps: set | None = None) -> go.Figure:
    fig = go.Figure()
    lap_numbers = sorted(aligned.keys())
    for lap_num, df in aligned.items():
        is_spin = bool(spin_laps and lap_num in spin_laps)
        fig.add_trace(go.Scatter(x=df["x_m"], y=df["y_m"], mode="lines", name=lap_label(lap_num, spin_laps),
                                  line=dict(color=lap_color(lap_num, lap_numbers, spin_laps),
                                            width=3 if is_spin else 2)))
    fig.update_layout(title="Track map (click legend entries to toggle laps; black = spin detected)",
                       xaxis_title="X (m)", yaxis_title="Y (m)",
                       yaxis=dict(scaleanchor="x", scaleratio=1))
    return fig


def fig_lap_comparison(aligned: dict, corners: pd.DataFrame, spin_laps: set | None = None) -> go.Figure:
    fig = go.Figure()
    lap_numbers = sorted(aligned.keys())
    for lap_num, df in aligned.items():
        is_spin = bool(spin_laps and lap_num in spin_laps)
        fig.add_trace(go.Scatter(x=df["ref_distance_m"], y=df["Speed GPS"], mode="lines",
                                  name=lap_label(lap_num, spin_laps),
                                  line=dict(color=lap_color(lap_num, lap_numbers, spin_laps),
                                            width=3 if is_spin else 2)))
    for _, c in corners.iterrows():
        fig.add_vrect(x0=c["start_m"], x1=c["end_m"], fillcolor="grey", opacity=0.1, line_width=0)
    fig.update_layout(title="Speed comparison across laps (shaded = detected corners; black = spin detected)",
                       xaxis_title="Distance along reference line (m)", yaxis_title="Speed (km/h)")
    return fig


def fig_delta_time(aligned: dict, candidate_refs: list, spin_laps: set | None = None) -> go.Figure:
    """Dropdown lets the viewer pick the reference lap; all options precomputed, toggled via visibility."""
    fig = go.Figure()
    lap_numbers = sorted(aligned.keys())
    trace_groups = []  # list of (ref_lap, [trace_indices])
    for ref_lap in candidate_refs:
        delta_df = compute_delta_time(aligned, reference_lap=ref_lap)
        start_idx = len(fig.data)
        for lap_num in aligned:
            if lap_num == ref_lap:
                continue
            col = f"lap_{lap_num}_delta_s"
            fig.add_trace(go.Scatter(x=delta_df["distance_m"], y=delta_df[col], mode="lines",
                                      name=lap_label(lap_num, spin_laps),
                                      line=dict(color=lap_color(lap_num, lap_numbers, spin_laps)),
                                      visible=(ref_lap == candidate_refs[0])))
        trace_groups.append((ref_lap, list(range(start_idx, len(fig.data)))))

    total_traces = len(fig.data)
    buttons = []
    for ref_lap, idxs in trace_groups:
        visible = [False] * total_traces
        for i in idxs:
            visible[i] = True
        buttons.append(dict(label=f"Ref: Lap {ref_lap}", method="update",
                             args=[{"visible": visible},
                                   {"title": f"Delta-time vs Lap {ref_lap}"}]))

    fig.add_hline(y=0, line_color="black", line_width=1)
    fig.update_layout(
        title=f"Delta-time vs Lap {candidate_refs[0]}",
        xaxis_title="Distance along reference line (m)", yaxis_title="Time delta (s), + = slower",
        updatemenus=[dict(buttons=buttons, direction="down", x=1.0, xanchor="right", y=1.15, yanchor="top")],
    )
    return fig


def fig_gg_diagram(aligned: dict, spin_laps: set | None = None) -> go.Figure:
    fig = go.Figure()
    lap_numbers = sorted(aligned.keys())
    for lap_num, df in aligned.items():
        fig.add_trace(go.Scattergl(x=df["lateral_g"], y=df["longitudinal_g"], mode="markers",
                                    name=lap_label(lap_num, spin_laps),
                                    marker=dict(size=4, opacity=0.5, color=lap_color(lap_num, lap_numbers, spin_laps))))
    fig.add_hline(y=0, line_color="lightgrey"); fig.add_vline(x=0, line_color="lightgrey")
    fig.update_layout(title="G-G diagram — grip envelope usage",
                       xaxis_title="Lateral G", yaxis_title="Longitudinal G (+ = braking, - = accelerating)",
                       yaxis=dict(scaleanchor="x", scaleratio=1))
    return fig


def fig_corner_consistency(consistency: pd.DataFrame) -> go.Figure:
    ordered = consistency.sort_values("min_speed_cv_pct", ascending=False)
    n = len(ordered)
    reds = [f"rgb({int(200 - 100*i/max(n-1,1))},{int(40 + 60*i/max(n-1,1))},{int(40 + 60*i/max(n-1,1))})" for i in range(n)]
    fig = go.Figure(go.Bar(x=[str(c) for c in ordered["corner_id"]], y=ordered["min_speed_cv_pct"], marker_color=reds))
    fig.update_layout(title="Corner consistency ranking (higher = less consistent, biggest opportunity; "
                             "out-lap/spin laps excluded)",
                       xaxis_title="Corner", yaxis_title="Min-speed variability across laps (CV %)")
    return fig


def fig_telemetry_stack(aligned: dict, corners: pd.DataFrame, lap_numbers: list,
                         spin_laps: set | None = None) -> go.Figure:
    """Dropdown lets the viewer pick which lap's telemetry stack to view."""
    spin_laps = spin_laps or set()
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                         subplot_titles=("Speed (km/h)", "RPM", "Acceleration proxy (%)", "Brake proxy (%)"))
    traces_per_lap = 4
    for i, lap_num in enumerate(lap_numbers):
        df = aligned[lap_num]
        visible = (i == 0)
        speed_color = SPIN_COLOR if lap_num in spin_laps else "#1f77b4"
        rpm_color = SPIN_COLOR if lap_num in spin_laps else "#9467bd"
        fig.add_trace(go.Scatter(x=df["ref_distance_m"], y=df["Speed GPS"], mode="lines",
                                  line=dict(color=speed_color), showlegend=False, visible=visible), row=1, col=1)
        fig.add_trace(go.Scatter(x=df["ref_distance_m"], y=df["RPM"], mode="lines",
                                  line=dict(color=rpm_color), showlegend=False, visible=visible), row=2, col=1)
        fig.add_trace(go.Scatter(x=df["ref_distance_m"], y=df["accel_proxy"], mode="lines",
                                  fill="tozeroy", line=dict(color="#2ca02c"), showlegend=False, visible=visible), row=3, col=1)
        fig.add_trace(go.Scatter(x=df["ref_distance_m"], y=df["brake_proxy"], mode="lines",
                                  fill="tozeroy", line=dict(color="#d62728"), showlegend=False, visible=visible), row=4, col=1)

    total_traces = len(fig.data)
    buttons = []
    for i, lap_num in enumerate(lap_numbers):
        visible = [False] * total_traces
        for j in range(traces_per_lap):
            visible[i * traces_per_lap + j] = True
        label = lap_label(lap_num, spin_laps)
        title = f"Telemetry stack — {label}" + (" ⚠ SPIN DETECTED" if lap_num in spin_laps else "")
        buttons.append(dict(label=label, method="update",
                             args=[{"visible": visible}, {"title": title}]))

    for _, c in corners.iterrows():
        for r in range(1, 5):
            fig.add_vrect(x0=c["start_m"], x1=c["end_m"], fillcolor="grey", opacity=0.08, line_width=0, row=r, col=1)

    fig.update_xaxes(title_text="Distance along reference line (m)", row=4, col=1)
    first_title = f"Telemetry stack — {lap_label(lap_numbers[0], spin_laps)} " \
                  f"(accel/brake are derived proxies, not measured)"
    fig.update_layout(height=800, title=first_title,
                       updatemenus=[dict(buttons=buttons, direction="down", x=1.0, xanchor="right", y=1.12, yanchor="top")])
    return fig


def fig_corner_detail(aligned: dict, corners: pd.DataFrame, corner_metrics: pd.DataFrame,
                       time_loss: pd.DataFrame, apex_positions: pd.DataFrame, lap_numbers: list,
                       spin_laps: set | None = None, excluded_laps: list | None = None) -> go.Figure:
    """
    Corner selector: zooms the speed comparison to the selected corner (with
    a margin either side) and shows a metrics table for every lap through it.
    """
    excluded_laps = set(excluded_laps or [])
    fig = make_subplots(rows=2, cols=1, row_heights=[0.55, 0.45], vertical_spacing=0.08,
                         specs=[[{"type": "scatter"}], [{"type": "table"}]],
                         subplot_titles=("Speed through corner (all laps; black = spin)", "Per-lap corner metrics"))

    for lap_num in lap_numbers:
        df = aligned[lap_num]
        fig.add_trace(go.Scatter(x=df["ref_distance_m"], y=df["Speed GPS"], mode="lines",
                                  name=lap_label(lap_num, spin_laps),
                                  line=dict(color=lap_color(lap_num, lap_numbers, spin_laps)),
                                  legendgroup=f"lap{lap_num}"), row=1, col=1)

    merged = time_loss.merge(corner_metrics, on=["corner_id", "lap"], how="left")
    merged = merged.merge(apex_positions[["corner_id", "lap", "apex_deviation_m"]], on=["corner_id", "lap"], how="left")

    table_trace_start = len(fig.data)
    corner_list = corners["corner_id"].astype(int).tolist()
    for c_id in corner_list:
        sub = merged[merged["corner_id"] == c_id].sort_values("lap")
        lap_display = [f"{int(l)}*" if int(l) in excluded_laps else str(int(l)) for l in sub["lap"]]
        header = ["Lap (* excluded)", "Entry (km/h)", "Min (km/h)", "Exit (km/h)", "Brake pt offset (m)",
                  "Accel pt offset (m)", "Apex dev (m)", "Time loss (s)"]
        cells = [
            lap_display, sub["entry_speed_kmh"].round(1), sub["min_speed_kmh"].round(1),
            sub["exit_speed_kmh"].round(1), sub["brake_point_offset_m"].round(1),
            sub["accel_point_offset_m"].round(1), sub["apex_deviation_m"].round(1),
            sub["time_loss_s"].round(3),
        ]
        fig.add_trace(go.Table(header=dict(values=header, fill_color="#333", font=dict(color="white", size=11)),
                                cells=dict(values=cells, height=24),
                                visible=(c_id == corner_list[0])), row=2, col=1)

    total_traces = len(fig.data)
    n_speed_traces = table_trace_start
    buttons = []
    for i, c_id in enumerate(corner_list):
        row = corners[corners["corner_id"] == c_id].iloc[0]
        margin = max((row["end_m"] - row["start_m"]) * 0.5, 15)
        visible = [True] * n_speed_traces + [False] * len(corner_list)
        visible[n_speed_traces + i] = True
        buttons.append(dict(label=f"Corner {c_id}", method="update",
                             args=[{"visible": visible},
                                   {"title": f"Corner {c_id} detail",
                                    "xaxis.range": [row["start_m"] - margin, row["end_m"] + margin]}]))

    fig.update_layout(height=750, title=f"Corner {corner_list[0]} detail",
                       updatemenus=[dict(buttons=buttons, direction="down", x=1.0, xanchor="right", y=1.12, yanchor="top")])
    fig.update_xaxes(title_text="Distance along reference line (m)", row=1, col=1,
                      range=[corners.iloc[0]["start_m"] - 15, corners.iloc[0]["end_m"] + 15])
    fig.update_yaxes(title_text="Speed (km/h)", row=1, col=1)
    return fig


# ---------------------------------------------------------------------------
# HTML assembly
# ---------------------------------------------------------------------------

def figs_to_html_block(figs_with_titles):
    """Combine multiple figures into HTML divs. Only the first embeds the plotly.js
    library (inline, so the whole file works fully offline) — the rest reuse it."""
    parts = []
    for i, (title, fig) in enumerate(figs_with_titles):
        include_js = "inline" if i == 0 else False
        html = fig.to_html(full_html=False, include_plotlyjs=include_js, div_id=f"fig{i}")
        parts.append(f'<section class="card"><h2>{title}</h2>{html}</section>')
    return "\n".join(parts)


def build_conclusions_html():
    if not CONCLUSIONS_MD.exists():
        return "<p><em>Run <code>python src/main.py</code> first to generate the conclusions.</em></p>"
    text = CONCLUSIONS_MD.read_text()
    try:
        import markdown
        return markdown.markdown(text, extensions=["tables"])
    except ImportError:
        # graceful fallback with no extra dependency: minimal manual rendering
        html_lines = []
        for line in text.split("\n"):
            if line.startswith("## "):
                html_lines.append(f"<h3>{line[3:]}</h3>")
            elif line.startswith("# "):
                html_lines.append(f"<h2>{line[2:]}</h2>")
            elif line.strip() == "":
                html_lines.append("<br>")
            else:
                html_lines.append(f"<p>{line}</p>")
        return "\n".join(html_lines)


def build_driver_summary_html(stats: dict, total_identifiable_loss: float) -> str:
    rows = [
        ("Best lap", f"Lap {stats['best_lap_number']} — {stats['best_lap_s']:.3f}s"),
        ("Theoretical best lap", f"{stats['theoretical_best_s']:.3f}s"),
        ("Potential gain", f"{stats['potential_gain_s']:.3f}s"),
        ("Total identifiable corner loss", f"{total_identifiable_loss:.3f}s"),
        ("Least consistent corner", f"Corner {stats['least_consistent_corner']} "
                                     f"({stats['least_consistent_cv_pct']:.1f}% CV)"),
        ("Peak lateral G", f"{stats['peak_lateral_g']:.2f} g"),
        ("Peak braking G", f"{stats['peak_braking_g']:.2f} g"),
        ("Peak combined G", f"{stats['peak_combined_g']:.2f} g"),
    ]
    cells = "".join(f"<tr><td>{label}</td><td><b>{value}</b></td></tr>" for label, value in rows)
    return f'<table class="summary-table">{cells}</table>'


def build_exclusions_html(excluded_laps: list, spin_by_lap: dict) -> str:
    if not excluded_laps:
        return "<p>None — no out-lap or spin detected in this session.</p>"
    items = []
    for lap in excluded_laps:
        reason = "spin detected (sustained high yaw-rate with low measured grip)" if spin_by_lap.get(lap) else "conventional out-lap (cold tyres/brakes)"
        items.append(f"<li><b>Lap {lap}</b>: {reason}. Excluded from corner consistency, corner "
                      f"time-loss, and the recommendation engine — still shown on the raw plots"
                      f"{' (in black)' if spin_by_lap.get(lap) else ''}.</li>")
    return f"<ul>{''.join(items)}</ul>"


def build_corner_loss_table_html(loss_summary: pd.DataFrame, total_identifiable_loss: float) -> str:
    rows = "".join(
        f"<tr><td>T{int(r['corner_id'])}</td><td>L{int(r['worst_lap'])}</td>"
        f"<td>{r['time_loss_s']:+.3f}</td></tr>"
        for _, r in loss_summary.iterrows()
    )
    return (
        f'<p><b>Total identifiable corner loss across the session: {total_identifiable_loss:.3f}s</b> '
        f"(sum of the single worst-lap loss in each corner, excluding out-lap(s)/spin lap(s) — not all "
        f"recoverable in one lap simultaneously, since the worst instances may come from different laps).</p>"
        f'<table class="loss-table"><tr><th>Corner</th><th>Worst lap</th><th>Time lost (s)</th></tr>'
        f"{rows}</table>"
    )


def build_recommendations_html(recommendations: list) -> str:
    if not recommendations:
        return "<p><em>No significant corner-specific time losses detected.</em></p>"
    parts = ['<p style="color:#555;font-size:14px">Auto-generated from threshold rules on the computed '
             "metrics — a starting point for review, not a validated diagnosis. Each entry compares the "
             "lap that lost the most time in that corner against the lap that was fastest through it "
             "(out-lap(s) and spin lap(s) excluded from this comparison).</p>"]
    for r in recommendations:
        parts.append(
            f'<div class="rec-card">'
            f"<h3>Corner {r['corner_id']} — up to {r['expected_effect_s']:.3f}s available</h3>"
            f"<p><b>Observation:</b> {r['observation']}</p>"
            f"<p><b>Evidence:</b> {r['evidence']}</p>"
            f"<p><b>Diagnosis:</b> {r['diagnosis']}</p>"
            f"<p><b>Proposed change:</b> {r['proposed_change']}</p>"
            f"<p><b>Expected effect:</b> up to {r['expected_effect_s']:.3f}s if corrected to match "
            f"Lap {r['best_lap']}'s pace through this corner</p>"
            f"</div>"
        )
    return "\n".join(parts)


def main():
    print("Running pipeline...")
    data = run_pipeline()
    aligned = data["aligned"]
    lap_numbers = sorted(aligned.keys())
    fastest_lap = data["fastest_lap"]
    spin_laps = data["spin_laps"]
    excluded_laps = data["excluded_laps"]
    ordered_refs = [fastest_lap] + [l for l in lap_numbers if l != fastest_lap]

    if spin_laps:
        print(f"  -> spin detected on lap(s): {sorted(spin_laps)}")
    print(f"  -> laps excluded from corner-comparison analysis: {excluded_laps or 'none'}")

    print("Building figures...")
    figs = [
        ("Overview — Lap Times", fig_lap_times(data["lap_times"], spin_laps)),
        ("Overview — Theoretical vs. Best Actual Lap", fig_theoretical_vs_best(data["theoretical"])),
        ("Track Map", fig_track_map(aligned, spin_laps)),
        ("Track Map — Corner Validation", fig_track_map_with_corners(data["ref_line"], data["corners"])),
        ("Telemetry Stack (use dropdown to pick lap)", fig_telemetry_stack(aligned, data["corners"], lap_numbers, spin_laps)),
        ("Lap Comparison — Speed", fig_lap_comparison(aligned, data["corners"], spin_laps)),
        ("Delta-Time (use dropdown to pick reference lap)", fig_delta_time(aligned, ordered_refs, spin_laps)),
        ("Corner Detail (use dropdown to pick corner)", fig_corner_detail(
            aligned, data["corners"], data["corner_metrics"], data["time_loss"],
            data["apex_positions"], lap_numbers, spin_laps, excluded_laps)),
        ("G-G Diagram", fig_gg_diagram(aligned, spin_laps)),
        ("Corner Consistency Ranking", fig_corner_consistency(data["consistency"])),
    ]

    body = figs_to_html_block(figs)
    conclusions_html = build_conclusions_html()
    total_identifiable_loss = float(data["loss_summary"]["time_loss_s"].clip(lower=0).sum()) if len(data["loss_summary"]) else 0.0
    summary_html = build_driver_summary_html(data["summary_stats"], total_identifiable_loss)
    exclusions_html = build_exclusions_html(excluded_laps, data["spin_by_lap"])
    loss_table_html = build_corner_loss_table_html(data["loss_summary"], total_identifiable_loss)
    recommendations_html = build_recommendations_html(data["recommendations"])

    page = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Go-Kart Telemetry Dashboard</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; background: #f7f7f9; margin: 0; padding: 0 0 60px 0; }}
  header {{ background: #111; color: white; padding: 24px 40px; }}
  header p {{ color: #ccc; margin: 4px 0 0 0; }}
  .card {{ background: white; margin: 24px auto; padding: 20px 24px; max-width: 1100px;
           border-radius: 8px; box-shadow: 0 1px 4px rgba(0,0,0,0.1); }}
  .card h2 {{ margin-top: 0; font-size: 18px; color: #222; }}
  .conclusions {{ max-width: 1100px; margin: 24px auto; padding: 20px 24px; background: white;
                  border-radius: 8px; box-shadow: 0 1px 4px rgba(0,0,0,0.1); }}
  .conclusions h2, .conclusions h3 {{ color: #222; }}
  .conclusions table {{ border-collapse: collapse; width: 100%; }}
  .conclusions th, .conclusions td {{ border: 1px solid #ddd; padding: 6px 10px; text-align: left; }}
  .note {{ max-width: 1100px; margin: 16px auto; color: #555; font-size: 14px; padding: 0 24px; }}
  .summary-table {{ width: 100%; border-collapse: collapse; }}
  .summary-table td {{ padding: 10px 14px; border-bottom: 1px solid #eee; font-size: 15px; }}
  .summary-table td:first-child {{ color: #555; width: 45%; }}
  .rec-card {{ border-left: 4px solid #d62728; background: #fafafa; padding: 10px 16px; margin: 14px 0; border-radius: 4px; }}
  .rec-card h3 {{ margin: 0 0 6px 0; font-size: 15px; }}
  .rec-card p {{ margin: 4px 0; font-size: 14px; }}
  .loss-table {{ border-collapse: collapse; width: 100%; margin-top: 10px; }}
  .loss-table th, .loss-table td {{ border: 1px solid #ddd; padding: 6px 12px; text-align: left; font-size: 14px; }}
  .loss-table th {{ background: #333; color: white; }}
</style>
</head>
<body>
<header>
  <h1>🏁 Go-Kart Telemetry Dashboard</h1>
  <p>Go Kart Club of Victoria session — Alfano6 logger. Acceleration/brake are derived proxies, not measured pedal data.</p>
</header>
<div class="note">Click legend entries to toggle laps on/off. Use the dropdown menus (top-right of the relevant charts) to switch laps/corners/reference. Laps drawn in <b>black</b> had a spin detected. This file is fully self-contained — no server, no install needed to view it.</div>
<section class="card"><h2>Driver Summary</h2>{summary_html}</section>
<div class="conclusions">
<h2>Laps Excluded From Corner-Comparison Analysis</h2>
{exclusions_html}
</div>
{body}
<div class="conclusions">
<h2>Corner-by-Corner Time Loss</h2>
{loss_table_html}
</div>
<div class="conclusions">
<h2>Automatic Improvement Recommendations</h2>
{recommendations_html}
</div>
<div class="conclusions">
<h2>Engineering Conclusions</h2>
{conclusions_html}
</div>
</body>
</html>
"""

    OUTPUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_HTML.write_text(page)
    print(f"Dashboard written to {OUTPUT_HTML}")

    try:
        webbrowser.open(f"file://{OUTPUT_HTML.resolve()}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
