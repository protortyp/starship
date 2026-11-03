#!/usr/bin/env python3
"""Generate the progress figure embedded in the single-salto training PR."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "assets" / "salto-training-progress.svg"
RANK_EXPERIMENT = ROOT / "logs" / "salto_stage78_dense_tail_v1"
LINE = re.compile(r"eval landed\s+(?P<landed>\d+\.\d+).*stage\s+(?P<stage>\d+)")
PROMOTED_DENSE = "promoted dense tail (58.75 to 47.5 m/s)"


@dataclass(frozen=True)
class Trace:
    label: str
    path: Path
    stage: int
    color: str
    skip: int = 0


TRACES = (
    Trace("3D distance only", ROOT / "out/salto-stage60-pad-shaping-v1.log", 60, "#9ca3af"),
    Trace("lateral pad shaping", ROOT / "out/salto-stage60-lateral-pad-v1.log", 60, "#10b981"),
    Trace("first stage-62 midpoint", ROOT / "out/salto-stage62-midpoint-v1.log", 62, "#f59e0b"),
    Trace("refined stage-62 bridge", ROOT / "out/salto-stage62-refined-v1.log", 62, "#2563eb"),
    Trace(
        "2x lateral-speed shaping",
        ROOT / "out/salto-stage62-lateral-speed-v1.log",
        62,
        "#dc2626",
    ),
    Trace(
        "20-step persistent count",
        ROOT / "out/salto-stage62-hybrid-hold20-v1.log",
        62,
        "#db2777",
    ),
    Trace(
        "phase-coherent landing adapter",
        ROOT / "out/salto-stage62-postloop-phase-v1.log",
        62,
        "#0891b2",
    ),
    Trace(
        "62.5 m/s long braking bridge",
        ROOT / "out/salto-stage74-bridges-v1.log",
        76,
        "#7c3aed",
        1,
    ),
    Trace(
        "60 m/s braking boundary",
        ROOT / "out/salto-stage74-bridges-v1.log",
        77,
        "#ea580c",
        1,
    ),
    Trace(
        PROMOTED_DENSE,
        ROOT / "out/salto-stage78-dense-tail-v1.log",
        78,
        "#92400e",
    ),
    Trace(
        PROMOTED_DENSE,
        ROOT / "out/salto-stage78-dense-tail-v1.log",
        79,
        "#92400e",
        1,
    ),
    Trace(
        PROMOTED_DENSE,
        ROOT / "out/salto-stage78-dense-tail-v1.log",
        80,
        "#92400e",
        1,
    ),
    Trace(
        PROMOTED_DENSE,
        ROOT / "out/salto-stage78-dense-tail-v1.log",
        81,
        "#92400e",
        1,
    ),
    Trace(
        PROMOTED_DENSE,
        ROOT / "out/salto-stage78-dense-tail-v1.log",
        82,
        "#92400e",
        1,
    ),
    Trace(
        PROMOTED_DENSE,
        ROOT / "out/salto-stage78-dense-tail-v1.log",
        83,
        "#92400e",
        1,
    ),
    Trace(
        PROMOTED_DENSE,
        ROOT / "out/salto-stage78-dense-tail-v1.log",
        84,
        "#92400e",
        1,
    ),
    Trace(
        PROMOTED_DENSE,
        ROOT / "out/salto-stage78-dense-tail-v1.log",
        85,
        "#92400e",
        1,
    ),
    Trace(
        PROMOTED_DENSE,
        ROOT / "out/salto-stage78-dense-tail-v1.log",
        86,
        "#92400e",
        1,
    ),
    Trace(
        PROMOTED_DENSE,
        ROOT / "out/salto-stage78-dense-tail-v1.log",
        87,
        "#92400e",
        1,
    ),
    Trace(
        "46.25 m/s dense tail (live)",
        ROOT / "out/salto-stage78-dense-tail-v1.log",
        88,
        "#0f766e",
        1,
    ),
)


def rolling(values: list[float], window: int = 25) -> list[float]:
    result: list[float] = []
    total = 0.0
    for index, value in enumerate(values):
        total += value
        if index >= window:
            total -= values[index - window]
        result.append(total / min(index + 1, window))
    return result


def read_landing(trace: Trace) -> list[float]:
    values: list[float] = []
    for line in trace.path.read_text().splitlines():
        match = LINE.search(line)
        if match and int(match["stage"]) == trace.stage:
            values.append(float(match["landed"]))
    return values[trace.skip :]


def polyline(
    values: list[float],
    *,
    left: float,
    top: float,
    width: float,
    height: float,
    x_max: float,
    y_min: float,
    y_max: float,
    color: str,
    x_stride: float = 1.0,
) -> str:
    if len(values) < 2:
        return ""
    points = []
    for index, value in enumerate(values):
        x = left + index * x_stride / x_max * width
        y = top + (y_max - value) / (y_max - y_min) * height
        points.append(f"{x:.1f},{y:.1f}")
    return (
        f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" '
        'stroke-width="2.4" stroke-linejoin="round" stroke-linecap="round"/>'
    )


def text(x: float, y: float, value: str, *, size: int = 13, anchor: str = "start") -> str:
    return (
        f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" '
        f'fill="#374151">{value}</text>'
    )


def landing_panel() -> list[str]:
    left, top, width, height = 72, 92, 742, 270
    y_min, y_max = 0.25, 0.55
    series = [(trace, rolling(read_landing(trace))) for trace in TRACES]
    longest = max(len(values) for _, values in series)
    x_max = max(500, ((longest + 499) // 500) * 500)
    lines = [text(left, 55, "Landing success at the hard braking boundary", size=20)]
    for tick in (0.3, 0.4, 0.45, 0.5):
        y = top + (y_max - tick) / (y_max - y_min) * height
        dash = ' stroke-dasharray="5 5"' if tick == 0.45 else ""
        color = "#dc2626" if tick == 0.45 else "#e5e7eb"
        lines.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{left + width}" y2="{y:.1f}" '
            f'stroke="{color}"{dash}/>'
        )
        lines.append(text(left - 10, y + 4, f"{tick:.0%}", anchor="end"))
    for tick in range(0, x_max + 1, 500):
        x = left + tick / x_max * width
        lines.append(
            f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + height}" stroke="#f3f4f6"/>'
        )
        lines.append(text(x, top + height + 23, str(tick), anchor="middle"))
    lines.append(text(left + width / 2, top + height + 48, "evaluation iteration", anchor="middle"))
    gate_y = top + (y_max - 0.45) / (y_max - y_min) * height
    lines.append(text(left + 5, gate_y - 6, "promotion gate", size=12))
    for trace, values in series:
        lines.append(
            polyline(
                values,
                left=left,
                top=top,
                width=width,
                height=height,
                x_max=x_max,
                y_min=y_min,
                y_max=y_max,
                color=trace.color,
            )
        )
    legend = list(dict.fromkeys((trace.label, trace.color) for trace in TRACES))
    legend_x = 535
    lines.append(
        f'<rect x="{legend_x - 10}" y="30" width="290" height="{10 + 20 * len(legend)}" '
        'fill="white" opacity="0.9"/>'
    )
    for index, (label, color) in enumerate(legend):
        y = 43 + 20 * index
        lines.append(
            f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 22}" y2="{y}" '
            f'stroke="{color}" stroke-width="3"/>'
        )
        lines.append(text(legend_x + 29, y + 4, label, size=12))
    return lines


def latest_event_file() -> Path:
    candidates = list(RANK_EXPERIMENT.glob("*/*tfevents*"))
    return max(candidates, key=lambda path: path.stat().st_mtime)


def rank_panel() -> list[str]:
    left, top, width, height = 72, 465, 742, 225
    accumulator = EventAccumulator(str(latest_event_file())).Reload()
    tags = (
        ("plasticity/actor/features/effective_rank", "actor rank", "#2563eb"),
        ("plasticity/critic/features/effective_rank", "critic rank", "#7c3aed"),
    )
    series = [
        (label, color, [event.value for event in accumulator.Scalars(tag)])
        for tag, label, color in tags
    ]
    # Plasticity diagnostics are emitted every ten actor updates, so each scalar sample spans ten
    # updates rather than one. Keep the axis in actual training updates and inside the viewport.
    diagnostics_interval = 10
    x_max = max(500, (max(len(values) for _, _, values in series) - 1) * diagnostics_interval)
    lines = [text(left, 430, "Policy plasticity during the live dense tail", size=20)]
    for tick in (0, 32, 64, 96, 128):
        y = top + (128 - tick) / 128 * height
        lines.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{left + width}" y2="{y:.1f}" stroke="#e5e7eb"/>'
        )
        lines.append(text(left - 10, y + 4, str(tick), anchor="end"))
    tick_step = max(100, (x_max // 400) * 100)
    for tick in range(0, x_max + 1, tick_step):
        x = left + tick / max(x_max, 1) * width
        lines.append(text(x, top + height + 23, str(tick), anchor="middle"))
    lines.append(text(left + width / 2, top + height + 48, "training update", anchor="middle"))
    for index, (label, color, values) in enumerate(series):
        lines.append(
            polyline(
                values,
                left=left,
                top=top,
                width=width,
                height=height,
                x_max=max(x_max, 1),
                y_min=0,
                y_max=128,
                color=color,
                x_stride=diagnostics_interval,
            )
        )
        legend_x = 650
        legend_y = 430 + index * 20
        lines.append(
            f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 22}" '
            f'y2="{legend_y}" stroke="{color}" stroke-width="3"/>'
        )
        lines.append(text(legend_x + 29, legend_y + 4, label, size=12))
    return lines


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    elements = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="880" height="760" viewBox="0 0 880 760">',
        '<rect width="880" height="760" fill="white"/>',
        '<g font-family="ui-sans-serif, system-ui, sans-serif">',
        *landing_panel(),
        *rank_panel(),
        "</g>",
        "</svg>",
    ]
    OUTPUT.write_text("\n".join(elements) + "\n")


if __name__ == "__main__":
    main()
