"""Make an animated replay of one route: the track drawn on a map coloured by
speed, next to its speed and elevation over time. Used for the README demo.

Needs matplotlib on top of gpx_vis:

    pip install matplotlib
    python example/demo_animation.py                                  # example ride -> media/demo.gif
    python example/demo_animation.py path/to/tracks "Morning Ride" out.gif

Also writes a still of the last frame next to the GIF (same name, .png).
Privacy: the picture shows the whole route; avoid publishing one that starts
or ends at your home.
"""
import io
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.ticker  # noqa: E402,F401
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from PIL import Image  # noqa: E402

from gpx_vis import Track  # noqa: E402

HERE = Path(__file__).resolve().parent

# light chart theme: surface, ink and one hue per measure (as in the map popups)
SURFACE = "#fcfcfb"
MAP_BG = "#f0efec"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
CONTEXT = "#c3c2b7"
SPEED = "#2a78d6"
ELEVATION = "#eb6834"

N_FRAMES = 90
FRAME_MS = 70
HOLD_MS = 2500
SIZE_PX = (880, 495)
DPI = 110


def fmt_duration(seconds):
    minutes = int(round(seconds / 60))
    hours, minutes = divmod(minutes, 60)
    return f"{hours} h {minutes:02d} min" if hours else f"{minutes} min"


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    data = Path(argv[0]) if len(argv) > 0 else HERE / "data"
    route = argv[1] if len(argv) > 1 else "cyclingBoston"
    out = Path(argv[2]) if len(argv) > 2 else HERE.parent / "media" / "demo.gif"

    track = Track(data)
    splits = track.idx_trksplit()
    names = [track.name[start] for start, _ in splits]
    if route not in names:
        raise SystemExit(f"No route named {route!r}. Routes: {names}")
    start, end = splits[names.index(route)]
    stats = track.summary.iloc[names.index(route)]

    # route data
    lat, lon = track.y[start:end], track.x[start:end]
    speed = track.speed[start:end]
    elevation = pd.Series(track.z[start:end]).interpolate(limit_direction="both").to_numpy()
    times = pd.to_datetime(pd.Series(track.t[start:end], dtype=object), utc=True)
    seconds = (times - times.iloc[0]).dt.total_seconds().to_numpy()
    distance = np.cumsum(track._step_km[start:end])
    tnum = mdates.date2num(times.dt.tz_convert(None).to_numpy())

    # flat map coordinates: longitude shrunk by cos(latitude)
    coslat = np.cos(np.radians(np.mean(track.y)))
    mx, my = lon * coslat, lat
    vmin, vmax = np.floor(np.percentile(speed, 2)), np.ceil(np.percentile(speed, 98))
    norm = plt.Normalize(vmin, vmax)
    points = np.c_[mx, my]
    segments = np.stack([points[:-1], points[1:]], axis=1)
    segment_speed = (speed[:-1] + speed[1:]) / 2

    fig = plt.figure(figsize=(SIZE_PX[0] / DPI, SIZE_PX[1] / DPI), dpi=DPI, facecolor=SURFACE)
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": CONTEXT, "axes.labelcolor": INK_2,
                         "xtick.color": MUTED, "ytick.color": MUTED})
    ax_map = fig.add_axes([0.03, 0.10, 0.44, 0.74])
    ax_speed = fig.add_axes([0.58, 0.47, 0.39, 0.37])
    ax_elev = fig.add_axes([0.58, 0.12, 0.39, 0.27], sharex=ax_speed)
    ax_cbar = fig.add_axes([0.06, 0.055, 0.18, 0.018])

    # --- static layers ---------------------------------------------------------
    ax_map.set_facecolor(MAP_BG)
    for s, e in splits:  # every route, faint, for context
        ax_map.plot(track.x[s:e] * coslat, track.y[s:e], color=CONTEXT, lw=1.2, solid_capstyle="round", zorder=1)
    pad = 0.12 * max(np.ptp(mx), np.ptp(my))
    ax_map.set_xlim(mx.min() - pad, mx.max() + pad)
    ax_map.set_ylim(my.min() - pad, my.max() + pad)
    ax_map.set_aspect("equal")
    ax_map.set_xticks([])
    ax_map.set_yticks([])
    for spine in ax_map.spines.values():
        spine.set_visible(False)
    # scale bar in the emptier bottom corner
    km_deg = 1 / 111.195
    bar = 5 if np.ptp(my) > 5 * km_deg else 1
    left_empty = np.mean(mx[my < np.percentile(my, 25)]) > np.mean(mx)
    xlo, xhi = ax_map.get_xlim()
    x0 = xlo + 0.06 * (xhi - xlo) if left_empty else xhi - 0.06 * (xhi - xlo) - bar * km_deg
    y0 = ax_map.get_ylim()[0] + 0.04 * np.diff(ax_map.get_ylim())[0]
    ax_map.plot([x0, x0 + bar * km_deg], [y0, y0], color=INK_2, lw=1.5, solid_capstyle="butt")
    ax_map.text(x0 + bar * km_deg / 2, y0 + 0.012 * np.diff(ax_map.get_ylim())[0], f"{bar} km",
                color=INK_2, ha="center", va="bottom", fontsize=7)
    ax_map.scatter([mx[0]], [my[0]], s=40, color="#f0f921", edgecolor=INK, linewidth=1, zorder=5)

    sm = plt.cm.ScalarMappable(norm=norm, cmap="plasma")
    cbar = fig.colorbar(sm, cax=ax_cbar, orientation="horizontal")
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=7, length=0, pad=2)
    cbar.ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
    ax_cbar.set_title("Speed (km/h)", fontsize=7, color=INK_2, loc="left", pad=2)

    for ax, values, color, label in ((ax_speed, speed, SPEED, "Speed (km/h)"),
                                     (ax_elev, elevation, ELEVATION, "Elevation (m)")):
        ax.set_facecolor(SURFACE)
        ax.plot(tnum, values, color=color, lw=1.2, alpha=0.25)  # whole ride, faint
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.spines["left"].set_color(GRID)
        ax.set_ylabel(label, fontsize=8)
        ax.tick_params(length=0)
    ax_speed.set_ylim(0, max(1.0, np.nanmax(speed) * 1.15))
    elev_pad = max(10.0, 0.1 * np.ptp(elevation))
    ax_elev.set_ylim(elevation.min() - elev_pad, elevation.max() + elev_pad)
    ax_elev.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax_elev.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=6))
    ax_elev.set_xlabel("Time (UTC)", fontsize=8)
    plt.setp(ax_speed.get_xticklabels(), visible=False)
    ax_speed.set_xlim(tnum[0], tnum[-1])

    fig.text(0.03, 0.93, route, fontsize=13, fontweight="bold", color=INK)
    fig.text(0.03, 0.885,
             f"{stats['distance_km']:.1f} km  ·  {fmt_duration(stats['moving_time'].total_seconds())} moving  ·  "
             f"average {stats['moving_speed_kmh']:.1f} km/h",
             fontsize=8.5, color=INK_2)
    fig.text(0.97, 0.02, "made with gpx_vis", fontsize=7, color=MUTED, ha="right")

    # --- moving layers ---------------------------------------------------------
    casing = LineCollection([], colors="#1a1a19", linewidths=4.2, alpha=0.45, capstyle="round", zorder=2)
    trail = LineCollection([], cmap="plasma", norm=norm, linewidths=2.4, capstyle="round", zorder=3)
    ax_map.add_collection(casing)
    ax_map.add_collection(trail)
    dot = ax_map.scatter([], [], s=60, color=INK, edgecolor="white", linewidth=1.8, zorder=6)
    played = [ax_speed.plot([], [], color=SPEED, lw=2)[0], ax_elev.plot([], [], color=ELEVATION, lw=2)[0]]
    fill = [None]
    cursors = [ax.axvline(tnum[0], color=MUTED, lw=0.8) for ax in (ax_speed, ax_elev)]
    dots = [ax_speed.scatter([], [], s=36, color=SPEED, edgecolor="white", linewidth=1.5, zorder=5),
            ax_elev.scatter([], [], s=36, color=ELEVATION, edgecolor="white", linewidth=1.5, zorder=5)]
    label_box = dict(boxstyle="round,pad=0.25", fc=SURFACE, ec="none", alpha=0.9)
    labels = [ax.annotate("", (0, 0), xytext=(6, 6), textcoords="offset points", fontsize=8, color=INK,
                          va="bottom", bbox=label_box, zorder=6) for ax in (ax_speed, ax_elev)]
    finish = ax_map.scatter([], [], s=46, marker="s", color="#72b026", edgecolor="white", linewidth=1.5, zorder=7)
    readout = fig.text(0.97, 0.93, "", fontsize=9, color=INK_2, ha="right")

    frames = []
    for k in range(N_FRAMES):
        tcut = seconds[-1] * k / (N_FRAMES - 1)
        i = int(np.searchsorted(seconds, tcut, side="right"))  # points [0, i) are played
        i = max(2, min(i, len(seconds)))
        casing.set_segments(segments[:i - 1])
        trail.set_segments(segments[:i - 1])
        trail.set_array(segment_speed[:i - 1])
        dot.set_offsets([[mx[i - 1], my[i - 1]]])
        played[0].set_data(tnum[:i], speed[:i])
        played[1].set_data(tnum[:i], elevation[:i])
        if fill[0] is not None:
            fill[0].remove()
        fill[0] = ax_elev.fill_between(tnum[:i], elevation[:i], ax_elev.get_ylim()[0], color=ELEVATION, alpha=0.1, lw=0)
        for cursor in cursors:
            cursor.set_xdata([tnum[i - 1]])
        dots[0].set_offsets([[tnum[i - 1], speed[i - 1]]])
        dots[1].set_offsets([[tnum[i - 1], elevation[i - 1]]])
        # direct labels beside the cursor dots (flip to the left near the right edge)
        right_side = (i - 1) < 0.8 * len(tnum)
        for label, value, unit, ax in ((labels[0], speed[i - 1], "km/h", ax_speed),
                                       (labels[1], elevation[i - 1], "m", ax_elev)):
            label.set_text(f"{value:.0f} {unit}")
            label.xy = (tnum[i - 1], value)
            label.set_position((6, 6) if right_side else (-6, 6))
            label.set_ha("left" if right_side else "right")
        if k == N_FRAMES - 1:
            finish.set_offsets([[mx[-1], my[-1]]])
            dot.set_offsets(np.empty((0, 2)))
        readout.set_text(f"{times.iloc[i - 1]:%H:%M} UTC   {distance[i - 1]:.1f} km")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=DPI, facecolor=SURFACE)
        frames.append(Image.open(buf).convert("RGB"))

    out.parent.mkdir(parents=True, exist_ok=True)
    frames[-1].save(out.with_suffix(".png"))
    # one shared palette from the last frame keeps colours stable and the file small
    palette = frames[-1].quantize(colors=160, method=Image.Quantize.MEDIANCUT)
    gif_frames = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
    durations = [FRAME_MS] * (len(gif_frames) - 1) + [HOLD_MS]
    gif_frames[0].save(out, save_all=True, append_images=gif_frames[1:], duration=durations,
                       loop=0, optimize=True, disposal=1)
    print(f"Saved {out} ({out.stat().st_size / 1e6:.1f} MB) and {out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
