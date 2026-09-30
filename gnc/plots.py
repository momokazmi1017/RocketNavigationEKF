"""Shared plot style (same palette as projects 1 and 2)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
CRITICAL = "#d03b3b"
SHADE = "#efeee8"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 10,
    "text.color": INK, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
    "lines.linewidth": 2.0, "legend.frameon": False, "legend.labelcolor": INK_2,
})


def limit_line(ax, y, text, above=True):
    ax.axhline(y, color=CRITICAL, lw=1.0, ls="--")
    ax.annotate(text, (1.0, y), xycoords=("axes fraction", "data"), xytext=(-4, 4 if above else -12),
                textcoords="offset points", ha="right", fontsize=9, color=INK_2)


def shade_spans(ax, spans, label=None):
    """Grey bands for time spans (GPS outage, barometer lockout); label on the first."""
    for i, (a, b) in enumerate(spans):
        ax.axvspan(a, b, color=SHADE, lw=0, zorder=0)
        if label and i == 0:
            ax.annotate(label, ((a + b) / 2, 1.0), xycoords=("data", "axes fraction"), xytext=(0, -11),
                        textcoords="offset points", ha="center", fontsize=8, color=MUTED)


def spans_from_times(times, gap):
    """Group sorted sample times into contiguous spans (a new span starts after a gap > `gap`)."""
    if len(times) == 0:
        return []
    spans, start, prev = [], times[0], times[0]
    for t in times[1:]:
        if t - prev > gap:
            spans.append((start, prev))
            start = t
        prev = t
    spans.append((start, prev))
    return spans
