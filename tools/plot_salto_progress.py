#!/usr/bin/env python3
"""Generate the progress figure embedded in the single-salto training PR."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "assets" / "salto-training-progress.svg"
RANK_EXPERIMENT = ROOT / "logs" / "salto_stage92_count_hold_v1"
SAFE_DECEL_EXPERIMENT = ROOT / "logs" / "salto_stage94_safe_decel_v1"
CRITIC_EXTENSION_EXPERIMENT = ROOT / "logs" / "salto_stage94_critic_extension_v1"
WARMSTART_EXPERIMENT = ROOT / "logs" / "salto_stage94_throttle_warmstart_v1"
RESTART_EXPERIMENT = ROOT / "logs" / "salto_stage97_best_restart_v1"
MIDPOINT_EXPERIMENT = ROOT / "logs" / "salto_stage97_midpoint_v1"
STAGE99_MIDPOINT_EXPERIMENT = ROOT / "logs" / "salto_stage99_midpoint_v1"
STAGE99_SPLIT_EXPERIMENT = ROOT / "logs" / "salto_stage99_split_v2"
STAGE99_TOUCHDOWN_EXPERIMENT = ROOT / "logs" / "salto_stage99_touchdown_v1"
STAGE99_TOUCHDOWN1000_EXPERIMENT = ROOT / "logs" / "salto_stage99_touchdown1000_v1"
STAGE99_SPEED20_EXPERIMENT = ROOT / "logs" / "salto_stage99_speed20_v1"
STAGE99_SAFE1_EXPERIMENT = ROOT / "logs" / "salto_stage99_safe1_v1"
STAGE99_SAFE05_EXPERIMENT = ROOT / "logs" / "salto_stage99_safe05_v1"
STAGE99_VERTICAL_REWARD_EXPERIMENT = ROOT / "logs" / "salto_stage99_vertical_reward_v1"
STAGE99_VERTICAL_SAFE1_EXPERIMENT = ROOT / "logs" / "salto_stage99_vertical_safe1_v1"
STAGE99_VERTICAL_SAFE1_FAST_EXPERIMENT = ROOT / "logs" / "salto_stage99_vertical_safe1_fast_v1"
STAGE99_VERTICAL_SAFE1_PAD01_EXPERIMENT = ROOT / "logs" / "salto_stage99_vertical_safe1_pad01_v1"
STAGE99_VERTICAL_SAFE1_PAD02_EXPERIMENT = ROOT / "logs" / "salto_stage99_vertical_safe1_pad02_v1"
STAGE99_VERTICAL_SAFE1_PAD500_EXPERIMENT = ROOT / "logs" / "salto_stage99_vertical_safe1_pad500_v1"
STAGE99_VERTICAL_SAFE1_PAD500_CENTER01_EXPERIMENT = (
    ROOT / "logs" / "salto_stage99_vertical_safe1_pad500_center01_v1"
)
STAGE99_TOUCHDOWN_COUNT_EXPERIMENT = ROOT / "logs" / "salto_stage99_touchdown_count_v1"
STAGE99_TOUCHDOWN_COUNT_FAST_EXPERIMENT = ROOT / "logs" / "salto_stage99_touchdown_count_fast_v1"
STAGE99_TOUCHDOWN_COUNT_V90_EXPERIMENT = ROOT / "logs" / "salto_stage99_touchdown_count_v90_v1"
STAGE99_TOUCHDOWN_COUNT_V90_FAST_EXPERIMENT = (
    ROOT / "logs" / "salto_stage99_touchdown_count_v90_fast_v1"
)
STAGE99_TOUCHDOWN_COUNT_V90_HSPEED5_EXPERIMENT = (
    ROOT / "logs" / "salto_stage99_touchdown_count_v90_hspeed5_v2"
)
STAGE166_RECOVERY_EXPERIMENT = ROOT / "logs" / "salto_stage166_touchdown_count_v90_recover_v1"
STAGE167_ACTOR_RESET_EXPERIMENT = (
    ROOT / "logs" / "salto_stage167_touchdown_count_v90_actor_reset_v1"
)
STAGE167_MIDPOINT_EXPERIMENT = ROOT / "logs" / "salto_stage167_midpoint_v1"
STAGE99_VERTICAL_TOUCHDOWN1000_EXPERIMENT = (
    ROOT / "logs" / "salto_stage99_vertical_touchdown1000_v1"
)
LINE = re.compile(r"eval landed\s+(?P<landed>\d+\.\d+).*stage\s+(?P<stage>\d+)")
PROMOTED_DENSE = "promoted dense tail (58.75 to 42.5 m/s)"
COUNT_COHERENT = "count-coherent promotions (stages 92-93)"


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
        PROMOTED_DENSE,
        ROOT / "out/salto-stage78-dense-tail-v1.log",
        88,
        "#92400e",
        1,
    ),
    Trace(
        PROMOTED_DENSE,
        ROOT / "out/salto-stage78-dense-tail-v1.log",
        89,
        "#92400e",
        1,
    ),
    Trace(
        PROMOTED_DENSE,
        ROOT / "out/salto-stage78-dense-tail-v1.log",
        90,
        "#92400e",
        1,
    ),
    Trace(
        PROMOTED_DENSE,
        ROOT / "out/salto-stage78-dense-tail-v1.log",
        91,
        "#92400e",
        1,
    ),
    Trace(
        "stage 92, std 0.10 collapse",
        ROOT / "out/salto-stage78-dense-tail-v1.log",
        92,
        "#dc2626",
        1,
    ),
    Trace(
        "stage 92, lower learning rate",
        ROOT / "out/salto-stage92-slow-v1.log",
        92,
        "#f59e0b",
    ),
    Trace(
        "stage 92, std 0.03 old decoder",
        ROOT / "out/salto-stage92-lowstd-v1.log",
        92,
        "#0f766e",
    ),
    Trace(
        COUNT_COHERENT,
        ROOT / "out/salto-stage92-count-hold-v1.log",
        92,
        "#059669",
    ),
    Trace(
        COUNT_COHERENT,
        ROOT / "out/salto-stage92-count-hold-v1.log",
        93,
        "#059669",
        1,
    ),
    Trace(
        "long control: stages 94-95 (retired)",
        ROOT / "out/salto-stage92-count-hold-v1.log",
        94,
        "#0f766e",
        1,
    ),
    Trace(
        "long control: stages 94-95 (retired)",
        ROOT / "out/salto-stage92-count-hold-v1.log",
        95,
        "#0f766e",
    ),
    Trace(
        "2 m/s² one-engine braking A/B",
        ROOT / "out/salto-stage94-safe-decel-v1.log",
        94,
        "#7c3aed",
        1,
    ),
    Trace(
        "corrected braking + throttle warm start (live)",
        ROOT / "out/salto-stage94-throttle-warmstart-v1.log",
        95,
        "#db2777",
    ),
    Trace(
        "corrected braking + throttle warm start (live)",
        ROOT / "out/salto-stage94-throttle-warmstart-v1.log",
        96,
        "#db2777",
    ),
    Trace(
        "corrected braking, original stage-97 window",
        ROOT / "out/salto-stage94-throttle-warmstart-v1.log",
        97,
        "#db2777",
    ),
    Trace(
        "protected stage-97 best restart (live)",
        ROOT / "out/salto-stage97-best-restart-v1.log",
        97,
        "#111827",
    ),
    Trace(
        "35.625 m/s protected midpoint (passed)",
        ROOT / "out/salto-stage97-midpoint-v1.log",
        97,
        "#0284c7",
    ),
    Trace(
        "original 35 m/s boundary (passed)",
        ROOT / "out/salto-stage97-midpoint-v1.log",
        98,
        "#0369a1",
    ),
    Trace(
        "first 30 m/s braking step (live)",
        ROOT / "out/salto-stage97-midpoint-v1.log",
        99,
        "#0e7490",
    ),
    Trace(
        "34.583 m/s protected midpoint (live)",
        ROOT / "out/salto-stage99-midpoint-v1.log",
        99,
        "#7c2d12",
    ),
    Trace(
        "vertical-only bridge, fresh critic (retired)",
        ROOT / "out/salto-stage99-split-v2.log",
        99,
        "#7e22ce",
    ),
    Trace(
        "500 m touchdown specialist (retired)",
        ROOT / "out/salto-stage99-touchdown-v1.log",
        99,
        "#be123c",
    ),
    Trace(
        "1,000 m legacy-objective specialist (retired)",
        ROOT / "out/salto-stage99-touchdown1000-v1.log",
        99,
        "#c026d3",
    ),
    Trace(
        "2x vertical-speed shaping (retired)",
        ROOT / "out/salto-stage99-speed20-v1.log",
        99,
        "#4d7c0f",
    ),
    Trace(
        "1 m/s² legacy objective (retired)",
        ROOT / "out/salto-stage99-safe1-v1.log",
        99,
        "#6d28d9",
    ),
    Trace(
        "0.5 m/s² critic warmup (superseded)",
        ROOT / "out/salto-stage99-safe05-v1.log",
        99,
        "#0f766e",
    ),
    Trace(
        "vertical-only 2 m/s² objective (retired)",
        ROOT / "out/salto-stage99-vertical-reward-v1.log",
        99,
        "#0891b2",
    ),
    Trace(
        "vertical-only + 1 m/s² profile (retired)",
        ROOT / "out/salto-stage99-vertical-safe1-v1.log",
        99,
        "#047857",
    ),
    Trace(
        "vertical-only + 1,000 m specialist (retired)",
        ROOT / "out/salto-stage99-vertical-touchdown1000-v1.log",
        99,
        "#0e7490",
    ),
    Trace(
        "vertical-only + 1 m/s², 4x actor rate (retired)",
        ROOT / "out/salto-stage99-vertical-safe1-fast-v1.log",
        99,
        "#b45309",
    ),
    Trace(
        "vertical-only + 1 m/s² + 2x pad shaping (retired)",
        ROOT / "out/salto-stage99-vertical-safe1-pad01-v1.log",
        99,
        "#15803d",
    ),
    Trace(
        "vertical-only + 1 m/s² + 4x pad shaping (retired)",
        ROOT / "out/salto-stage99-vertical-safe1-pad02-v1.log",
        99,
        "#166534",
    ),
    Trace(
        "500 m gate to pad re-tightening (retired)",
        ROOT / "out/salto-stage99-vertical-safe1-pad500-v1.log",
        99,
        "#1d4ed8",
    ),
    Trace(
        "500 m gate + 2x centering to re-tightening (retired)",
        ROOT / "out/salto-stage99-vertical-safe1-pad500-center01-v1.log",
        99,
        "#7c3aed",
    ),
    Trace(
        "altitude-gated engine-count bridge (live)",
        ROOT / "out/salto-stage99-touchdown-count-v1.log",
        99,
        "#dc2626",
    ),
    Trace(
        "altitude-gated count, 4k fast A/B (live)",
        ROOT / "out/salto-stage99-touchdown-count-fast-v1.log",
        99,
        "#ea580c",
    ),
    Trace(
        "altitude-gated count, 90 m/s 1k (retired)",
        ROOT / "out/salto-stage99-touchdown-count-v90-v1.log",
        99,
        "#9333ea",
    ),
    Trace(
        "altitude-gated count, 90 m/s 4k (live)",
        ROOT / "out/salto-stage99-touchdown-count-v90-fast-v1.log",
        99,
        "#c026d3",
    ),
    Trace(
        "90 m/s + 5x horizontal shaping (retired warmup)",
        ROOT / "out/salto-stage99-touchdown-count-v90-hspeed5-v2.log",
        99,
        "#4f46e5",
    ),
    Trace(
        "protected 1e-6 recovery (retired)",
        ROOT / "out/salto-stage166-touchdown-count-v90-recover-v1.log",
        166,
        "#16a34a",
    ),
    Trace(
        "protected 1e-6 recovery (retired)",
        ROOT / "out/salto-stage166-touchdown-count-v90-recover-v1.log",
        167,
        "#16a34a",
    ),
    Trace(
        "actor-moment reset at 5e-7 (retired)",
        ROOT / "out/salto-stage167-touchdown-count-v90-actor-reset-v1.log",
        167,
        "#0d9488",
    ),
    Trace(
        "22.5 m/s measured midpoint (live)",
        ROOT / "out/salto-stage167-midpoint-v1.log",
        167,
        "#0f766e",
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
    lines = [text(left, 55, "Hard-boundary landing success", size=20)]
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
    legend_x, legend_top = 450, 43
    legend_rows = (len(legend) + 1) // 2
    lines.append(
        f'<rect x="425" y="30" width="440" height="{10 + 14 * legend_rows}" fill="white"/>'
    )
    for index, (label, color) in enumerate(legend):
        column, row = divmod(index, legend_rows)
        legend_item_x = legend_x + column * 205
        y = legend_top + 14 * row
        lines.append(
            f'<line x1="{legend_item_x}" y1="{y}" x2="{legend_item_x + 22}" y2="{y}" '
            f'stroke="{color}" stroke-width="3"/>'
        )
        lines.append(text(legend_item_x + 29, y + 4, label, size=8))
    return lines


def latest_event_file(experiment: Path) -> Path:
    candidates = list(experiment.glob("*/*tfevents*"))
    return max(candidates, key=lambda path: path.stat().st_mtime)


def rank_panel() -> list[str]:
    left, top, width, height = 72, 625, 742, 225
    control = EventAccumulator(str(latest_event_file(RANK_EXPERIMENT))).Reload()
    safe_decel = EventAccumulator(str(latest_event_file(SAFE_DECEL_EXPERIMENT))).Reload()
    critic_extension = EventAccumulator(
        str(latest_event_file(CRITIC_EXTENSION_EXPERIMENT))
    ).Reload()
    warmstart = EventAccumulator(str(latest_event_file(WARMSTART_EXPERIMENT))).Reload()
    restart = EventAccumulator(str(latest_event_file(RESTART_EXPERIMENT))).Reload()
    midpoint = EventAccumulator(str(latest_event_file(MIDPOINT_EXPERIMENT))).Reload()
    stage99_midpoint = EventAccumulator(
        str(latest_event_file(STAGE99_MIDPOINT_EXPERIMENT))
    ).Reload()
    stage99_split = EventAccumulator(str(latest_event_file(STAGE99_SPLIT_EXPERIMENT))).Reload()
    stage99_touchdown = EventAccumulator(
        str(latest_event_file(STAGE99_TOUCHDOWN_EXPERIMENT))
    ).Reload()
    stage99_touchdown1000 = EventAccumulator(
        str(latest_event_file(STAGE99_TOUCHDOWN1000_EXPERIMENT))
    ).Reload()
    stage99_speed20 = EventAccumulator(str(latest_event_file(STAGE99_SPEED20_EXPERIMENT))).Reload()
    stage99_safe1 = EventAccumulator(str(latest_event_file(STAGE99_SAFE1_EXPERIMENT))).Reload()
    stage99_safe05 = EventAccumulator(str(latest_event_file(STAGE99_SAFE05_EXPERIMENT))).Reload()
    stage99_vertical_reward = EventAccumulator(
        str(latest_event_file(STAGE99_VERTICAL_REWARD_EXPERIMENT))
    ).Reload()
    stage99_vertical_safe1 = EventAccumulator(
        str(latest_event_file(STAGE99_VERTICAL_SAFE1_EXPERIMENT))
    ).Reload()
    stage99_vertical_safe1_fast = EventAccumulator(
        str(latest_event_file(STAGE99_VERTICAL_SAFE1_FAST_EXPERIMENT))
    ).Reload()
    stage99_vertical_safe1_pad01 = EventAccumulator(
        str(latest_event_file(STAGE99_VERTICAL_SAFE1_PAD01_EXPERIMENT))
    ).Reload()
    stage99_vertical_safe1_pad02 = EventAccumulator(
        str(latest_event_file(STAGE99_VERTICAL_SAFE1_PAD02_EXPERIMENT))
    ).Reload()
    stage99_vertical_safe1_pad500 = EventAccumulator(
        str(latest_event_file(STAGE99_VERTICAL_SAFE1_PAD500_EXPERIMENT))
    ).Reload()
    stage99_vertical_safe1_pad500_center01 = EventAccumulator(
        str(latest_event_file(STAGE99_VERTICAL_SAFE1_PAD500_CENTER01_EXPERIMENT))
    ).Reload()
    stage99_touchdown_count = EventAccumulator(
        str(latest_event_file(STAGE99_TOUCHDOWN_COUNT_EXPERIMENT))
    ).Reload()
    stage99_touchdown_count_fast = EventAccumulator(
        str(latest_event_file(STAGE99_TOUCHDOWN_COUNT_FAST_EXPERIMENT))
    ).Reload()
    stage99_touchdown_count_v90 = EventAccumulator(
        str(latest_event_file(STAGE99_TOUCHDOWN_COUNT_V90_EXPERIMENT))
    ).Reload()
    stage99_touchdown_count_v90_fast = EventAccumulator(
        str(latest_event_file(STAGE99_TOUCHDOWN_COUNT_V90_FAST_EXPERIMENT))
    ).Reload()
    stage99_touchdown_count_v90_hspeed5 = EventAccumulator(
        str(latest_event_file(STAGE99_TOUCHDOWN_COUNT_V90_HSPEED5_EXPERIMENT))
    ).Reload()
    stage99_vertical_touchdown1000 = EventAccumulator(
        str(latest_event_file(STAGE99_VERTICAL_TOUCHDOWN1000_EXPERIMENT))
    ).Reload()
    stage166_recovery = EventAccumulator(
        str(latest_event_file(STAGE166_RECOVERY_EXPERIMENT))
    ).Reload()
    stage167_actor_reset = EventAccumulator(
        str(latest_event_file(STAGE167_ACTOR_RESET_EXPERIMENT))
    ).Reload()
    stage167_midpoint = EventAccumulator(
        str(latest_event_file(STAGE167_MIDPOINT_EXPERIMENT))
    ).Reload()
    tags = (
        (control, "plasticity/actor/features/effective_rank", "control actor rank", "#2563eb"),
        (control, "plasticity/critic/features/effective_rank", "control critic rank", "#7c3aed"),
        (
            safe_decel,
            "plasticity/critic/features/effective_rank",
            "2 m/s² critic warmup rank",
            "#db2777",
        ),
        (
            critic_extension,
            "plasticity/critic/features/effective_rank",
            "critic extension rank",
            "#ea580c",
        ),
        (
            warmstart,
            "plasticity/actor/features/effective_rank",
            "warm-start actor rank",
            "#059669",
        ),
        (
            warmstart,
            "plasticity/critic/features/effective_rank",
            "warm-start critic rank",
            "#dc2626",
        ),
        (
            restart,
            "plasticity/actor/features/effective_rank",
            "best-restart actor rank",
            "#111827",
        ),
        (
            restart,
            "plasticity/critic/features/effective_rank",
            "best-restart critic rank",
            "#78716c",
        ),
        (
            midpoint,
            "plasticity/actor/features/effective_rank",
            "midpoint actor rank",
            "#0284c7",
        ),
        (
            midpoint,
            "plasticity/critic/features/effective_rank",
            "midpoint critic rank",
            "#0e7490",
        ),
        (
            stage99_midpoint,
            "plasticity/actor/features/effective_rank",
            "stage-99 midpoint actor rank",
            "#7c2d12",
        ),
        (
            stage99_midpoint,
            "plasticity/critic/features/effective_rank",
            "stage-99 midpoint critic rank",
            "#a16207",
        ),
        (
            stage99_split,
            "plasticity/critic/features/effective_rank",
            "vertical-only fresh critic rank",
            "#7e22ce",
        ),
        (
            stage99_touchdown,
            "plasticity/critic/features/effective_rank",
            "500 m specialist critic rank",
            "#be123c",
        ),
        (
            stage99_touchdown1000,
            "plasticity/critic/features/effective_rank",
            "1,000 m specialist critic rank",
            "#c026d3",
        ),
        (
            stage99_speed20,
            "plasticity/critic/features/effective_rank",
            "2x-speed-shaping critic rank",
            "#4d7c0f",
        ),
        (
            stage99_safe1,
            "plasticity/critic/features/effective_rank",
            "1 m/s² braking critic rank",
            "#6d28d9",
        ),
        (
            stage99_safe05,
            "plasticity/critic/features/effective_rank",
            "0.5 m/s² braking critic rank",
            "#0f766e",
        ),
        (
            stage99_vertical_reward,
            "plasticity/critic/features/effective_rank",
            "vertical-only reward critic rank",
            "#0891b2",
        ),
        (
            stage99_vertical_safe1,
            "plasticity/critic/features/effective_rank",
            "vertical-only + 1 m/s² critic rank",
            "#047857",
        ),
        (
            stage99_vertical_safe1_fast,
            "plasticity/critic/features/effective_rank",
            "vertical-only + 1 m/s² fast critic rank",
            "#b45309",
        ),
        (
            stage99_vertical_safe1_pad01,
            "plasticity/critic/features/effective_rank",
            "vertical-only + 2x pad critic rank",
            "#15803d",
        ),
        (
            stage99_vertical_safe1_pad02,
            "plasticity/critic/features/effective_rank",
            "vertical-only + 4x pad critic rank",
            "#166534",
        ),
        (
            stage99_vertical_safe1_pad500,
            "plasticity/critic/features/effective_rank",
            "500 m pad-gate critic rank",
            "#1d4ed8",
        ),
        (
            stage99_vertical_safe1_pad500_center01,
            "plasticity/critic/features/effective_rank",
            "500 m + 2x centering critic rank",
            "#7c3aed",
        ),
        (
            stage99_touchdown_count,
            "plasticity/critic/features/effective_rank",
            "altitude-gated count critic rank",
            "#dc2626",
        ),
        (
            stage99_touchdown_count_fast,
            "plasticity/critic/features/effective_rank",
            "altitude-gated 4k critic rank",
            "#ea580c",
        ),
        (
            stage99_touchdown_count_v90,
            "plasticity/critic/features/effective_rank",
            "altitude-gated 90 m/s critic rank",
            "#9333ea",
        ),
        (
            stage99_touchdown_count_v90_fast,
            "plasticity/critic/features/effective_rank",
            "altitude-gated 90 m/s 4k critic rank",
            "#c026d3",
        ),
        (
            stage99_touchdown_count_v90_hspeed5,
            "plasticity/critic/features/effective_rank",
            "90 m/s + 5x horizontal critic rank",
            "#4f46e5",
        ),
        (
            stage99_vertical_touchdown1000,
            "plasticity/critic/features/effective_rank",
            "vertical-only + 1,000 m critic rank",
            "#0e7490",
        ),
        (
            stage166_recovery,
            "plasticity/actor/features/effective_rank",
            "protected 1e-6 recovery actor rank",
            "#16a34a",
        ),
        (
            stage166_recovery,
            "plasticity/critic/features/effective_rank",
            "protected 1e-6 recovery critic rank",
            "#65a30d",
        ),
        (
            stage167_actor_reset,
            "plasticity/actor/features/effective_rank",
            "actor-reset 5e-7 actor rank",
            "#0d9488",
        ),
        (
            stage167_actor_reset,
            "plasticity/critic/features/effective_rank",
            "actor-reset 5e-7 critic rank",
            "#14b8a6",
        ),
        (
            stage167_midpoint,
            "plasticity/actor/features/effective_rank",
            "22.5 m/s midpoint actor rank",
            "#0f766e",
        ),
        (
            stage167_midpoint,
            "plasticity/critic/features/effective_rank",
            "22.5 m/s midpoint critic rank",
            "#115e59",
        ),
    )
    series = [
        (label, color, [event.value for event in accumulator.Scalars(tag)])
        for accumulator, tag, label, color in tags
    ]
    # Plasticity diagnostics are emitted every ten actor updates, so each scalar sample spans ten
    # updates rather than one. Keep the axis in actual training updates and inside the viewport.
    diagnostics_interval = 10
    x_max = max(500, (max(len(values) for _, _, values in series) - 1) * diagnostics_interval)
    lines = [text(left, 590, "Protected braking-policy plasticity comparisons", size=20)]
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
    legend_x, legend_top = 455, 615
    legend_rows = (len(series) + 1) // 2
    for _, color, values in series:
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
    lines.append(
        f'<rect x="{legend_x - 10}" y="{legend_top - 17}" width="420" '
        f'height="{10 + 15 * legend_rows}" fill="white"/>'
    )
    for index, (label, color, _) in enumerate(series):
        column, row = divmod(index, legend_rows)
        legend_item_x = legend_x + column * 205
        legend_y = legend_top + row * 15
        lines.append(
            f'<line x1="{legend_item_x}" y1="{legend_y}" x2="{legend_item_x + 22}" '
            f'y2="{legend_y}" stroke="{color}" stroke-width="3"/>'
        )
        lines.append(text(legend_item_x + 29, legend_y + 4, label, size=8))
    return lines


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    elements = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="880" height="930" viewBox="0 0 880 930">',
        '<rect width="880" height="930" fill="white"/>',
        '<g font-family="ui-sans-serif, system-ui, sans-serif">',
        *landing_panel(),
        *rank_panel(),
        "</g>",
        "</svg>",
    ]
    OUTPUT.write_text("\n".join(elements) + "\n")


if __name__ == "__main__":
    main()
