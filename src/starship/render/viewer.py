"""Pygame drawing of one tracked rocket, faint "ghosts" of the other envs, and a telemetry HUD."""

import math

import numpy as np
import pygame

from starship.sim import FLYING, LEFT, MAIN, RIGHT, STATUS_NAMES, SimConfig, StateSnapshot

SKY_LOW = np.array([38, 62, 96])
SKY_HIGH = np.array([8, 12, 28])
GRID = (255, 255, 255, 22)
GRID_LABEL = (150, 165, 190)
GROUND = (44, 40, 36)
GROUND_EDGE = (90, 82, 70)
PAD = (120, 124, 130)
PAD_MARK = (240, 200, 60)
HULL = (205, 210, 218)
HULL_EDGE = (120, 126, 136)
GHOST = (205, 210, 218, 45)
FLAME_OUTER = (255, 140, 40)
FLAME_INNER = (255, 235, 150)
TEXT = (230, 234, 240)
TEXT_DIM = (150, 160, 175)
PANEL = (0, 0, 0, 140)
STATUS_COLORS = {
    "landed": (90, 220, 120),
    "crashed": (240, 90, 80),
    "out_of_bounds": (240, 170, 60),
    "timeout": (240, 170, 60),
}

GRID_SPACINGS = (1, 2, 5, 10, 25, 50, 100, 250, 500, 1000, 2500)
MAX_GHOSTS = 256
SKY_FADE_ALTITUDE = 8000.0


class Viewer:
    def __init__(self, config: SimConfig, width: int = 1280, height: int = 800, scale: float = 6.0):
        self.config = config
        self.width = width
        self.height = height
        self.scale = scale
        self.ground_margin = 90
        self.camera = np.zeros(2)
        self.font = pygame.font.SysFont("menlo,dejavusansmono,monospace", 15)
        self.font_big = pygame.font.SysFont("menlo,dejavusansmono,monospace", 40, bold=True)
        self.overlay = pygame.Surface((width, height), pygame.SRCALPHA)
        self._flicker = np.random.default_rng(0)

    def zoom(self, factor: float) -> None:
        self.scale = float(np.clip(self.scale * factor, 0.05, 40.0))

    def to_screen(self, p) -> tuple[float, float]:
        return (
            float(self.width / 2 + (p[0] - self.camera[0]) * self.scale),
            float(self.height / 2 - (p[1] - self.camera[1]) * self.scale),
        )

    def _update_camera(self, pos: np.ndarray) -> None:
        # Follow the rocket, but never lift the ground above its bottom margin.
        min_cam_y = (self.height / 2 - self.ground_margin) / self.scale
        self.camera = np.array([pos[0], max(pos[1], min_cam_y)])

    def draw(
        self,
        surface: pygame.Surface,
        s: StateSnapshot,
        env: int,
        actions: np.ndarray,
        controller_name: str,
        speed: float,
        paused: bool,
    ) -> None:
        self._update_camera(s.pos[env])
        self.overlay.fill((0, 0, 0, 0))
        self._draw_sky(surface)
        self._draw_grid(surface)
        self._draw_ground(surface)
        self._draw_ghosts(s, env)
        surface.blit(self.overlay, (0, 0))
        self._draw_flames(surface, s, env, int(actions[env]))
        self._draw_rocket(surface, s.pos[env], float(s.theta[env]))
        self._draw_hud(surface, s, env, int(actions[env]), controller_name, speed, paused)
        self._draw_altimeter(surface, float(s.pos[env, 1]))
        if s.status[env] != FLYING:
            self._draw_banner(surface, s, env)

    def _draw_sky(self, surface: pygame.Surface) -> None:
        t = min(max(self.camera[1] / SKY_FADE_ALTITUDE, 0.0), 1.0)
        surface.fill(tuple(int(c) for c in SKY_LOW + (SKY_HIGH - SKY_LOW) * t))

    def _draw_grid(self, surface: pygame.Surface) -> None:
        spacing = next((g for g in GRID_SPACINGS if g * self.scale >= 60), GRID_SPACINGS[-1])
        half_w = self.width / 2 / self.scale
        half_h = self.height / 2 / self.scale
        grid = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        x0 = math.floor((self.camera[0] - half_w) / spacing) * spacing
        for x in np.arange(x0, self.camera[0] + half_w + spacing, spacing):
            sx, _ = self.to_screen((x, 0))
            pygame.draw.line(grid, GRID, (sx, 0), (sx, self.height))
        y0 = max(0.0, math.floor((self.camera[1] - half_h) / spacing) * spacing)
        for y in np.arange(y0, self.camera[1] + half_h + spacing, spacing):
            _, sy = self.to_screen((0, y))
            pygame.draw.line(grid, GRID, (0, sy), (self.width, sy))
            # Labels sit just left of the altimeter so they never collide with the HUD panel.
            label = self.font.render(f"{y:.0f} m", True, GRID_LABEL)
            grid.blit(label, (self.width - 100 - label.get_width(), sy - label.get_height() - 2))
        surface.blit(grid, (0, 0))

    def _draw_ground(self, surface: pygame.Surface) -> None:
        _, sy = self.to_screen((0, 0))
        if sy > self.height:
            return
        pygame.draw.rect(surface, GROUND, (0, sy, self.width, self.height - sy))
        pygame.draw.line(surface, GROUND_EDGE, (0, sy), (self.width, sy), 2)
        half = self.config.landing.pad_half_width
        left, _ = self.to_screen((-half, 0))
        right, _ = self.to_screen((half, 0))
        thickness = max(3, int(1.5 * self.scale))
        pygame.draw.rect(surface, PAD, (left, sy, right - left, thickness))
        for x in (left, right):
            pygame.draw.rect(surface, PAD_MARK, (x - 2, sy - 6, 4, 6 + thickness))
        center, _ = self.to_screen((0, 0))
        pygame.draw.line(surface, PAD_MARK, (center, sy), (center, sy + thickness), 2)

    def _hull_points(self, pos, theta: float) -> list[tuple[float, float]]:
        r = self.config.rocket
        half_l, half_d = 0.5 * r.length, 0.5 * r.diameter
        nose = 0.25 * r.length
        a = np.array([-math.sin(theta), math.cos(theta)])
        n = np.array([math.cos(theta), math.sin(theta)])
        # Outline in body coordinates (along axis, along normal), tail first, with a tapered nose.
        body = [
            (-half_l, -half_d),
            (half_l - nose, -half_d),
            (half_l - 0.45 * nose, -0.8 * half_d),
            (half_l - 0.1 * nose, -0.4 * half_d),
            (half_l, 0.0),
            (half_l - 0.1 * nose, 0.4 * half_d),
            (half_l - 0.45 * nose, 0.8 * half_d),
            (half_l - nose, half_d),
            (-half_l, half_d),
        ]
        return [self.to_screen(pos + u * a + v * n) for u, v in body]

    def _draw_rocket(self, surface: pygame.Surface, pos, theta: float) -> None:
        points = self._hull_points(pos, theta)
        pygame.draw.polygon(surface, HULL, points)
        pygame.draw.aalines(surface, HULL_EDGE, True, points)

    def _draw_ghosts(self, s: StateSnapshot, env: int) -> None:
        others = [i for i in range(len(s.status)) if i != env][:MAX_GHOSTS]
        for i in others:
            pygame.draw.polygon(self.overlay, GHOST, self._hull_points(s.pos[i], float(s.theta[i])))

    def _flame(self, surface, base, direction, half_width, length) -> None:
        side = np.array([-direction[1], direction[0]])
        flicker = 0.85 + 0.3 * self._flicker.random()
        for color, w, ln in ((FLAME_OUTER, 1.0, 1.0), (FLAME_INNER, 0.5, 0.6)):
            tip = base + direction * length * ln * flicker
            points = [base + side * half_width * w, tip, base - side * half_width * w]
            pygame.draw.polygon(surface, color, [self.to_screen(p) for p in points])

    def _draw_flames(self, surface: pygame.Surface, s: StateSnapshot, env: int, action: int):
        if s.status[env] != FLYING:
            return
        r = self.config.rocket
        pos, theta = s.pos[env], float(s.theta[env])
        a = np.array([-math.sin(theta), math.cos(theta)])
        n = np.array([math.cos(theta), math.sin(theta)])
        if action & MAIN:
            self._flame(surface, pos - 0.5 * r.length * a, -a, 0.35 * r.diameter, 0.6 * r.length)
        nozzle = pos + (r.side_arm - 0.3 * r.length) * a
        # The left thruster pushes the rocket toward +n, so its exhaust leaves on the -n side.
        if action & LEFT:
            self._flame(surface, nozzle - 0.5 * r.diameter * n, -n, 1.2, 0.35 * r.length)
        if action & RIGHT:
            self._flame(surface, nozzle + 0.5 * r.diameter * n, n, 1.2, 0.35 * r.length)

    def _panel(self, surface: pygame.Surface, rect) -> None:
        panel = pygame.Surface(rect[2:], pygame.SRCALPHA)
        panel.fill(PANEL)
        surface.blit(panel, rect[:2])

    def _draw_hud(self, surface, s, env, action, controller_name, speed, paused) -> None:
        r = self.config.rocket
        x, y = s.pos[env]
        vx, vy = s.vel[env]
        theta = float(s.theta[env])
        bottom = y - 0.5 * r.length * abs(math.cos(theta)) - 0.5 * r.diameter * abs(math.sin(theta))
        status = STATUS_NAMES[int(s.status[env])]
        flying = int((s.status == FLYING).sum())
        lines = [
            (f"env {env} / {len(s.status)}   ({flying} flying)", TEXT),
            (f"time       {s.steps[env] * self.config.policy_dt:7.1f} s", TEXT),
            (f"altitude   {bottom:7.1f} m", TEXT),
            (f"offset x   {x:7.1f} m", TEXT),
            (f"vx         {vx:7.1f} m/s", TEXT),
            (f"vy         {vy:7.1f} m/s", TEXT),
            (f"tilt       {math.degrees(theta):7.1f} deg", TEXT),
            (f"spin       {math.degrees(s.omega[env]):7.1f} deg/s", TEXT),
            (f"status     {status}", STATUS_COLORS.get(status, TEXT)),
            (f"controller {controller_name}", TEXT_DIM),
            (f"speed      {speed:g}x{'  PAUSED' if paused else ''}", TEXT_DIM),
        ]
        line_h = self.font.get_linesize()
        self._panel(surface, (12, 12, 300, 18 + line_h * (len(lines) + 2)))
        for i, (text, color) in enumerate(lines):
            surface.blit(self.font.render(text, True, color), (24, 20 + i * line_h))

        thruster_y = 20 + len(lines) * line_h + 8
        for i, (label, bit) in enumerate((("LEFT", LEFT), ("MAIN", MAIN), ("RIGHT", RIGHT))):
            on = bool(action & bit) and s.status[env] == FLYING
            box = pygame.Rect(24 + i * 92, thruster_y, 84, line_h + 6)
            pygame.draw.rect(surface, FLAME_OUTER if on else (60, 64, 72), box, border_radius=4)
            text = self.font.render(label, True, (20, 20, 20) if on else TEXT_DIM)
            surface.blit(text, text.get_rect(center=box.center))

        help_text = (
            "W/Up main  A/Left rotate left  D/Right rotate right  |  R reset  Space pause  "
            "Tab next env  [ ] speed  scroll/+- zoom  Esc quit"
        )
        rendered = self.font.render(help_text, True, TEXT_DIM)
        self._panel(surface, (0, self.height - 28, self.width, 28))
        surface.blit(rendered, (12, self.height - 22))

    def _draw_altimeter(self, surface: pygame.Surface, y: float) -> None:
        top_alt = max(self.config.spawn.y * 1.1, y)
        x, top, bottom = self.width - 40, 40, self.height - 80
        self._panel(surface, (x - 52, top - 24, 80, bottom - top + 48))
        pygame.draw.line(surface, TEXT_DIM, (x, top), (x, bottom), 2)
        for alt in np.linspace(0, top_alt, 5):
            sy = bottom - (bottom - top) * alt / top_alt
            pygame.draw.line(surface, TEXT_DIM, (x - 5, sy), (x + 5, sy))
            label = self.font.render(f"{alt / 1000:.1f}k", True, TEXT_DIM)
            surface.blit(label, (x - 46, sy - label.get_height() / 2))
        sy = bottom - (bottom - top) * min(max(y, 0.0), top_alt) / top_alt
        pygame.draw.polygon(surface, PAD_MARK, [(x + 4, sy), (x + 14, sy - 6), (x + 14, sy + 6)])

    def _draw_banner(self, surface: pygame.Surface, s: StateSnapshot, env: int) -> None:
        status = STATUS_NAMES[int(s.status[env])]
        color = STATUS_COLORS.get(status, TEXT)
        title = self.font_big.render(status.replace("_", " ").upper(), True, color)
        vx, vy = s.vel[env]
        detail = self.font.render(
            f"vy {vy:.1f} m/s   vx {vx:.1f} m/s   tilt {math.degrees(s.theta[env]):.1f} deg   "
            f"offset {s.pos[env, 0]:.1f} m",
            True,
            TEXT,
        )
        cx, cy = self.width / 2, self.height * 0.22
        width = max(title.get_width(), detail.get_width()) + 48
        self._panel(surface, (cx - width / 2, cy - 40, width, 96))
        surface.blit(title, title.get_rect(center=(cx, cy)))
        surface.blit(detail, detail.get_rect(center=(cx, cy + 36)))
