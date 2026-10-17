"""Render the small landing GIF in the README from a trained policy.

    uv run python tools/readme_gif.py <policy.pt> [--out docs/landing.gif]

One rocket from the 3 km belly-flop start, drawn in a few flat colors at low resolution so the
file stays small. The camera zooms in as the rocket descends, and the rocket is drawn larger than
scale while it is high up, otherwise it would be a single pixel.
"""

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from starship.control.policy import PolicyController
from starship.render.geometry import hull_outline, plumes
from starship.sim import FLYING, SimConfig, Simulator

WIDTH, HEIGHT = 320, 200
PALETTE = {"sky": (255, 255, 255), "ground": (200, 200, 200), "body": (40, 40, 48)}
FLAME = (255, 140, 0)
MIN_ROCKET_PX = 22


def render(pos, theta, action, rocket, alt) -> Image.Image:
    view_h = float(np.clip(1.4 * alt + 80.0, 160.0, 3600.0))
    scale = HEIGHT / view_h
    cx, bottom = 0.5 * pos[0], -0.1 * view_h

    def px(points):
        p = np.asarray(points)
        return [
            (WIDTH / 2 + (x - cx) * scale, HEIGHT - (y - bottom) * scale)
            for x, y in p.reshape(-1, 2)
        ]

    img = Image.new("RGB", (WIDTH, HEIGHT), PALETTE["sky"])
    draw = ImageDraw.Draw(img)
    ground_y = HEIGHT + bottom * scale
    draw.rectangle([0, ground_y, WIDTH, HEIGHT], fill=PALETTE["ground"])
    pad = px([(-20.0, 0.0), (20.0, 0.0)])
    draw.line(pad, fill=PALETTE["body"], width=2)

    # Enlarge the rocket about its center while it is too small to see.
    grow = max(1.0, MIN_ROCKET_PX / (rocket.length * scale))
    center = np.asarray(pos)

    def enlarged(points):
        return center + (np.asarray(points) - center) * grow

    for tri in plumes(pos, theta, action, rocket):
        draw.polygon(px(enlarged(tri)), fill=FLAME)
    hull = hull_outline(center[None], np.array([theta]), rocket)[0]
    draw.polygon(px(enlarged(hull)), fill=PALETTE["body"])
    return img


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("policy")
    parser.add_argument("--out", default="docs/landing.gif")
    parser.add_argument("--x", type=float, default=-200.0, help="spawn offset from the pad (m)")
    parser.add_argument("--every", type=int, default=4, help="policy steps per GIF frame")
    args = parser.parse_args()

    config = SimConfig()
    config = replace(config, spawn=replace(config.spawn, x=args.x))
    sim = Simulator(1, config, device="cpu", seed=0)
    controller = PolicyController(sim, args.policy)
    frames, step = [], 0
    while True:
        s = sim.snapshot()
        flying = s.status[0] == FLYING
        action = int(controller(s)[0]) if flying else 0
        if step % args.every == 0 or not flying:
            frames.append(render(s.pos[0], float(s.theta[0]), action, config.rocket, s.pos[0, 1]))
        if not flying:
            break
        sim.step(np.array([action], dtype=np.int32))
        step += 1
    frames += [frames[-1]] * 15
    palette = Image.new("P", (1, 1))
    colors = [*PALETTE.values(), FLAME]
    palette.putpalette([c for rgb in colors for c in rgb] + [0] * (768 - 3 * len(colors)))
    frames = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        args.out, save_all=True, append_images=frames[1:], duration=80, loop=0, optimize=True
    )
    size = Path(args.out).stat().st_size / 1024
    print(
        f"{args.out}: {len(frames)} frames, {size:.0f} KB, status {int(s.status[0])} (1 = landed)"
    )


if __name__ == "__main__":
    main()
