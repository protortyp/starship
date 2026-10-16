"""Interactive viewer: steps the simulator in real time and draws one tracked env.

Run `uv run starship-view` to fly the rocket with the keyboard, or pass `--controller heuristic`
to watch the scripted baseline. `--headless` renders offscreen and saves PNG frames instead.
"""

import argparse
import os
from pathlib import Path

import numpy as np

from starship.control import HeuristicController, IdleController, RandomController
from starship.sim import FLYING, LEFT, MAIN, RIGHT, Simulator, StateSnapshot

FPS = 60
SPEEDS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
AUTO_RESET_DELAY = 3.0


class KeyboardController:
    """Applies the pressed thrusters to every env. Keys are mapped by rotation direction."""

    def __call__(self, s: StateSnapshot) -> np.ndarray:
        import pygame

        keys = pygame.key.get_pressed()
        action = 0
        if keys[pygame.K_w] or keys[pygame.K_UP]:
            action |= MAIN
        # Rotating counter-clockwise (nose left) needs the right-side thruster, and vice versa.
        if keys[pygame.K_a] or keys[pygame.K_LEFT]:
            action |= RIGHT
        if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            action |= LEFT
        return np.full(len(s.status), action, dtype=np.int32)


def make_controller(name: str, sim: Simulator):
    return {
        "keyboard": KeyboardController,
        "heuristic": lambda: HeuristicController(sim.config),
        "random": RandomController,
        "idle": IdleController,
    }[name]()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--controller", choices=("keyboard", "heuristic", "random", "idle"), default="keyboard"
    )
    parser.add_argument("--envs", type=int, default=1, help="parallel envs (others drawn faint)")
    parser.add_argument("--device", default=None, help="Warp device, e.g. cpu or cuda:0")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--headless", action="store_true", help="render offscreen, save PNGs")
    parser.add_argument("--frames", type=int, default=600, help="headless: frames to render")
    parser.add_argument("--save-every", type=int, default=60, help="headless: save period")
    parser.add_argument("--out", type=Path, default=Path("out/frames"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
    import pygame

    from starship.render.viewer import Viewer

    pygame.init()
    screen = pygame.display.set_mode((args.width, args.height))
    pygame.display.set_caption("starship")
    clock = pygame.time.Clock()

    sim = Simulator(args.envs, device=args.device)
    controller = make_controller(args.controller, sim)
    viewer = Viewer(sim.config, args.width, args.height)
    policy_dt = sim.config.policy_dt

    snapshot = sim.snapshot()
    actions = np.zeros(args.envs, dtype=np.int32)
    env = 0
    speed = args.speed
    paused = False
    accumulator = 0.0
    finished_for = 0.0
    if args.headless:
        args.out.mkdir(parents=True, exist_ok=True)

    frame = 0
    running = True
    while running:
        dt = 1.0 / FPS if args.headless else clock.tick(FPS) / 1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEWHEEL:
                viewer.zoom(1.15**event.y)
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_r:
                    sim.reset()
                    snapshot = sim.snapshot()
                    accumulator = finished_for = 0.0
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_TAB:
                    env = (env + 1) % args.envs
                elif event.key == pygame.K_RIGHTBRACKET:
                    speed = next((v for v in SPEEDS if v > speed), speed)
                elif event.key == pygame.K_LEFTBRACKET:
                    speed = next((v for v in reversed(SPEEDS) if v < speed), speed)
                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    viewer.zoom(1.25)
                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    viewer.zoom(0.8)

        if not paused:
            accumulator += dt * speed
            while accumulator >= policy_dt:
                actions = controller(snapshot)
                sim.step(actions)
                snapshot = sim.snapshot()
                accumulator -= policy_dt
            if (snapshot.status == FLYING).any():
                finished_for = 0.0
            else:
                finished_for += dt
                if finished_for > AUTO_RESET_DELAY:
                    sim.reset()
                    snapshot = sim.snapshot()
                    finished_for = 0.0

        viewer.draw(screen, snapshot, env, actions, args.controller, speed, paused)
        pygame.display.flip()

        frame += 1
        if args.headless:
            if frame % args.save_every == 0:
                pygame.image.save(screen, args.out / f"frame_{frame:05d}.png")
            if frame >= args.frames:
                running = False

    pygame.quit()


if __name__ == "__main__":
    main()
