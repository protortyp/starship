#!/usr/bin/env python3
"""Generate the focused landing-expert progress figure embedded in the salto PR."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "assets" / "salto-landing-expert-progress.svg"
EVAL = re.compile(
    r"it (?P<iteration>\d+)/\d+.*eval landed (?P<landed>\d+\.\d+)"
    r"(?:\s+target landed (?P<target>\d+\.\d+))?.*stage (?P<stage>\d+)"
)


@dataclass(frozen=True)
class Evaluation:
    iteration: int
    landed: float
    target: float | None
    stage: int


def evaluations(path: Path) -> list[Evaluation]:
    result = []
    for line in path.read_text().splitlines():
        match = EVAL.search(line)
        if match:
            result.append(
                Evaluation(
                    iteration=int(match["iteration"]),
                    landed=float(match["landed"]),
                    target=float(match["target"]) if match["target"] is not None else None,
                    stage=int(match["stage"]),
                )
            )
    return result


def text(x: float, y: float, value: str, *, size: int = 13, anchor: str = "start") -> str:
    return (
        f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" '
        f'fill="#374151">{value}</text>'
    )


def polyline(points: list[tuple[float, float]], color: str, *, dashed: bool = False) -> str:
    dash = ' stroke-dasharray="7 5"' if dashed else ""
    coords = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    return (
        f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="3"'
        f' stroke-linejoin="round" stroke-linecap="round"{dash}/>'
    )


def main() -> None:
    controls = (
        (
            "no-fuel strict control",
            ROOT / "out" / "salto-landing-direct-center-nofuel-2k-v10.log",
            "#94a3b8",
        ),
        (
            "0.05 pad shaping, strict",
            ROOT / "out" / "salto-landing-direct-padshape-2k-v11.log",
            "#64748b",
        ),
        (
            "0.10 pad shaping, strict",
            ROOT / "out" / "salto-landing-direct-padshape10-2k-v12.log",
            "#334155",
        ),
    )
    live = evaluations(ROOT / "out" / "salto-landing-direct-gatecurriculum-2k-v13.log")
    series = [(label, evaluations(path), color) for label, path, color in controls]
    left, width = 78, 760
    success_top, success_height = 100, 300
    stage_top, stage_height = 520, 160
    x_max = max(
        1500,
        max(values[-1].iteration - values[0].iteration for _, values, _ in series if values),
    )

    elements = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="920" height="760" viewBox="0 0 920 760">',
        '<rect width="920" height="760" fill="white"/>',
        '<g font-family="ui-sans-serif, system-ui, sans-serif">',
        text(left, 48, "Landing-expert training progress", size=24),
        text(left, 76, "Deterministic 128-episode evaluations every 25 PPO updates", size=13),
    ]

    for rate in (0, 0.25, 0.5, 0.75, 0.9, 1.0):
        y = success_top + (1 - rate) * success_height
        color = "#dc2626" if rate == 0.9 else "#e5e7eb"
        dash = ' stroke-dasharray="6 5"' if rate == 0.9 else ""
        elements.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{left + width}" y2="{y:.1f}" '
            f'stroke="{color}"{dash}/>'
        )
        elements.append(text(left - 10, y + 4, f"{rate:.0%}", anchor="end"))
    elements.append(
        text(left + width - 8, success_top + 26, "90% promotion gate", size=12, anchor="end")
    )

    def success_points(values: list[Evaluation], target: bool = False) -> list[tuple[float, float]]:
        start = values[0].iteration
        return [
            (
                left + (value.iteration - start) / x_max * width,
                success_top
                + (1 - (value.target if target and value.target is not None else value.landed))
                * success_height,
            )
            for value in values
            if not target or value.target is not None
        ]

    for _label, values, color in series:
        points = success_points(values)
        if len(points) > 1:
            elements.append(polyline(points, color))
    if len(live) > 1:
        elements.append(polyline(success_points(live), "#16a34a"))
        elements.append(polyline(success_points(live, target=True), "#f97316", dashed=True))

    legend = [
        *((label, color, False) for label, _, color in series),
        ("live curriculum gate", "#16a34a", False),
        ("live strict target", "#f97316", True),
    ]
    for index, (label, color, dashed) in enumerate(legend):
        x = 470 if index >= 3 else 100
        y = 124 + (index - 3 if index >= 3 else index) * 22
        dash = ' stroke-dasharray="7 5"' if dashed else ""
        elements.append(
            f'<line x1="{x}" y1="{y}" x2="{x + 28}" y2="{y}" stroke="{color}" '
            f'stroke-width="3"{dash}/>'
        )
        elements.append(text(x + 36, y + 4, label, size=12))

    for update in range(0, x_max + 1, 500):
        x = left + update / x_max * width
        elements.append(
            f'<line x1="{x:.1f}" y1="{success_top}" x2="{x:.1f}" '
            f'y2="{success_top + success_height}" stroke="#f3f4f6"/>'
        )
        elements.append(text(x, success_top + success_height + 24, str(update), anchor="middle"))
    elements.append(
        text(left + width / 2, success_top + success_height + 50, "PPO update", anchor="middle")
    )

    elements.append(text(left, 486, "Live curriculum stage", size=20))
    for stage in (0, 6, 12, 18, 24):
        y = stage_top + (24 - stage) / 24 * stage_height
        elements.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{left + width}" y2="{y:.1f}" stroke="#e5e7eb"/>'
        )
        elements.append(text(left - 10, y + 4, str(stage), anchor="end"))
    if len(live) > 1:
        start = live[0].iteration
        stage_points = [
            (
                left + (value.iteration - start) / x_max * width,
                stage_top + (24 - value.stage) / 24 * stage_height,
            )
            for value in live
        ]
        elements.append(polyline(stage_points, "#2563eb"))
        last = live[-1]
        elements.append(
            text(
                stage_points[-1][0] + 8,
                stage_points[-1][1] - 8,
                f"stage {last.stage}/24",
                size=12,
            )
        )
    elements.extend(["</g>", "</svg>"])
    OUTPUT.write_text("\n".join(elements) + "\n")


if __name__ == "__main__":
    main()
