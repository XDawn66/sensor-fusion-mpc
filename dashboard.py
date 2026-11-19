"""Pygame presentation layer. No estimation or control algorithms live here."""

import math
from functools import lru_cache

import numpy as np
import pygame

BG = (9, 15, 25)
PANEL = (15, 24, 38)
EDGE = (35, 49, 67)
TEXT = (230, 239, 249)
MUTED = (130, 152, 177)
CYAN = (71, 220, 239)
GREEN = (75, 227, 162)
AMBER = (246, 176, 79)
PURPLE = (183, 148, 255)
BLUE = (90, 151, 255)


@lru_cache(maxsize=12)
def font(size):
    return pygame.font.SysFont("segoeui", size)


def text(surface, value, xy, size=17, color=TEXT):
    surface.blit(font(size).render(str(value), True, color), xy)


def panel(surface, rect):
    pygame.draw.rect(surface, PANEL, rect, border_radius=14)
    pygame.draw.rect(surface, EDGE, rect, 1, border_radius=14)


def chart(surface, rect, values, colors, maximum=None):
    pygame.draw.rect(surface, BG, rect, border_radius=6)
    finite = [float(v) for series in values for v in series if np.isfinite(v)]
    top = maximum if maximum is not None else max([0.5] + finite) * 1.15
    for fraction in (0.25, 0.5, 0.75):
        y = rect.bottom - int(rect.height * fraction)
        pygame.draw.line(surface, EDGE, (rect.left, y), (rect.right, y))
    for series, color in zip(values, colors):
        segment = []
        for i, v in enumerate(series):
            if np.isfinite(v):
                segment.append((rect.left + int(i / 599 * (rect.width - 1)),
                                rect.bottom - 1 - int(min(v / top, 1) * (rect.height - 2))))
            else:
                if len(segment) > 1:
                    pygame.draw.lines(surface, color, False, segment, 2)
                segment = []
        if len(segment) > 1:
            pygame.draw.lines(surface, color, False, segment, 2)
    return top


def render(screen, sim, paused, plot, pixel, reference_state, dt, scale):
    screen.fill(BG)
    text(screen, "AUTONOMY / LAB", (26, 15), 15, CYAN)
    text(screen, "MPC + Sensor Fusion" if sim.mode == "mpc" else "Sensor Fusion", (24, 37), 30)
    text(screen, "Combine imperfect sensor readings, then plan the robot's next moves.", (26, 80), 17, MUTED)
    badge = pygame.Rect(1195, 34, 218, 40)
    pygame.draw.rect(screen, (24, 53, 54), badge, border_radius=20)
    text(screen, "PAUSED" if paused else "LIVE SIMULATION", (1222, 42), 17, GREEN)

    # Compact metric strip with values computed from the actual simulation.
    latest = sim.metrics[-1] if sim.metrics else (0, np.nan, 0)
    speed = np.linalg.norm(sim.truth[2:])
    metrics = [("MODE", "MPC + SENSOR FUSION" if sim.mode == "mpc" else "SENSOR FUSION"),
               ("SIMULATION", f"{sim.tick * dt:06.1f} s"),
               ("ROBOT SPEED", f"{speed:.2f} m/s"),
               ("DISTANCE FROM TARGET", f"{latest[0]:.2f} m" if sim.mode == "mpc" else "PRESET ROUTE"),
               ("PLANNING TIME", f"{sim.solve_ms:.1f} ms" if sim.solve_ms is not None else "--")]
    for i, (title, value) in enumerate(metrics):
        x = 26 + i * 281
        text(screen, title, (x, 112), 12, MUTED)
        text(screen, value, (x, 131), 19, CYAN if i == 0 else TEXT)

    panel(screen, plot)
    screen.set_clip(plot.inflate(-2, -2))
    for x in range(-14, 15):
        for y in range(-9, 10):
            pygame.draw.circle(screen, (33, 47, 65), pixel((x, y)), 1)
    pygame.draw.line(screen, (28, 42, 59), pixel((-20, 0)), pixel((20, 0)))
    pygame.draw.line(screen, (28, 42, 59), pixel((0, -20)), pixel((0, 20)))

    reference = [pixel(reference_state(t)[:2]) for t in np.linspace(0, 2 * math.pi / 0.22, 240)]
    pygame.draw.lines(screen, (22, 36, 51), True, reference, 28)
    pygame.draw.lines(screen, (48, 65, 84), True, reference, 1)
    for i in range(0, len(reference) - 1, 6):
        pygame.draw.line(screen, (106, 127, 149), reference[i], reference[i + 2], 2)

    for trail, color, width in [(sim.raw_trail, AMBER, 1), (sim.true_trail, TEXT, 2),
                                (sim.fused_trail, GREEN, 2)]:
        if len(trail) > 1:
            pygame.draw.lines(screen, color, False, [pixel(p) for p in trail], width)
    gps = list(sim.gps_trail)[-80:]
    for i, p in enumerate(gps):
        strength = (i + 1) / max(len(gps), 1)
        color = tuple(int(35 + (c - 35) * strength) for c in BLUE)
        pygame.draw.circle(screen, color, pixel(p), 2)
    if sim.gps is not None:
        px, py = pixel(sim.gps)
        pygame.draw.line(screen, BLUE, (px - 6, py), (px + 6, py), 2)
        pygame.draw.line(screen, BLUE, (px, py - 6), (px, py + 6), 2)
    pygame.draw.circle(screen, AMBER, pixel(sim.raw_position), 6, 2)

    # Show the controller's real rollout only; never fabricate a prediction.
    rollout = sim.controller.predicted_states
    if sim.mode == "mpc" and rollout is not None:
        rollout = np.asarray(rollout, dtype=float)
        if rollout.shape != (sim.controller.horizon + 1, 4) or not np.isfinite(rollout).all():
            raise ValueError("predicted_states must be finite with shape (horizon+1, 4)")
        points = [pixel(p[:2]) for p in rollout]
        if len(points) > 1:
            pygame.draw.lines(screen, PURPLE, False, points, 3)
        for point in points:
            pygame.draw.circle(screen, PURPLE, point, 4, 1)

    # Positional covariance ellipse: 95% joint confidence for a Gaussian in 2D.
    if sim.fused_trail:
        estimate = sim.fused_trail[-1]
        covariance = getattr(sim.filter, "P", None)
        if covariance is not None:
            covariance = np.asarray(covariance)[:2, :2]
            if covariance.shape == (2, 2) and np.isfinite(covariance).all():
                eigenvalues, vectors = np.linalg.eigh((covariance + covariance.T) / 2)
                if np.all(eigenvalues >= 0):
                    ellipse = [pixel(estimate + vectors @ (np.sqrt(5.991 * eigenvalues) *
                               np.array([math.cos(a), math.sin(a)])))
                               for a in np.linspace(0, 2 * math.pi, 64)]
                    pygame.draw.lines(screen, (34, 101, 83), True, ellipse, 1)
        pygame.draw.circle(screen, GREEN, pixel(estimate), 13, 2)

    if sim.mode == "mpc":
        target = pixel(reference_state(sim.tick * dt)[:2])
        pygame.draw.circle(screen, PURPLE, target, 16, 2)
        pygame.draw.circle(screen, PURPLE, target, 3)
    # The primary car follows the sensor-based estimate, not simulator truth.
    # This is the current corrected estimate; future MPC rollout stays purple.
    estimated_state = np.asarray(sim.filter.state, dtype=float)
    center = np.array(pixel(estimated_state[:2]), dtype=float)
    angle = math.atan2(-estimated_state[3], estimated_state[2])
    forward = np.array([math.cos(angle), math.sin(angle)])
    side = np.array([-forward[1], forward[0]])
    def polygon(points, color, width=0):
        transformed = [center + forward * a + side * b for a, b in points]
        pygame.draw.polygon(screen, color, transformed, width)

    pygame.draw.circle(screen, (24, 41, 56), center.astype(int), 27)
    for a in (-12, 10):
        for b in (-12, 9):
            polygon([(a-4, b), (a+4, b), (a+4, b+3), (a-4, b+3)], (4, 8, 14))
    body = [(22, -7), (17, -11), (-18, -11), (-22, -7),
            (-22, 7), (-18, 11), (17, 11), (22, 7)]
    polygon(body, GREEN)
    polygon(body, (145, 255, 206), 1)
    polygon([(10,-8), (5,-9), (5,9), (10,8)], (28, 66, 85))
    polygon([(-11,-8), (-15,-7), (-15,7), (-11,8)], (28, 66, 85))
    polygon([(3,-7), (-9,-7), (-9,7), (3,7)], (172, 191, 210))
    for b in (-7, 4):
        polygon([(19,b), (22,b), (22,b+3), (19,b+3)], CYAN)
        polygon([(-22,b), (-20,b), (-20,b+3), (-22,b+3)], (255, 100, 90))
    # Small comparison marker remains visible even when the positions overlap.
    actual = pixel(sim.truth[:2])
    pygame.draw.circle(screen, BG, actual, 6)
    pygame.draw.circle(screen, TEXT, actual, 4)
    screen.set_clip(None)
    text(screen, "01 / WORLD VIEW", (44, 181), 13, MUTED)
    text(screen, "Gray loop: planned route" if sim.mode == "mpc" else "Robot follows a preset route", (44, 203), 17)
    if not plot.collidepoint(pixel(estimated_state[:2])):
        text(screen, "Estimated vehicle outside view - R to reset", (300, 730), 19, AMBER)
    pygame.draw.line(screen, MUTED, (60, 780), (60 + 2 * scale, 780), 2)
    text(screen, "2 m", (62, 754), 13, MUTED)
    for x, y, label, color in [
        (190, 762, "Actual position (dot)", TEXT),
        (455, 762, "Estimated car / GPS corrected", GREEN),
        (750, 762, "GPS readings (noisy)", BLUE),
        (190, 790, "Without GPS correction (drifts)", AMBER),
        (455, 790, "Target / predicted motion", PURPLE),
        (750, 790, "Planned route", (106, 127, 149)),
    ]:
        pygame.draw.circle(screen, color, (x, y), 4)
        text(screen, label, (x + 12, y - 11), 13, MUTED)

    panel(screen, pygame.Rect(1040, 164, 376, 180))
    text(screen, "02 / HOW ACCURATE IS IT?", (1060, 182), 13, MUTED)
    fused = latest[1]
    text(screen, f"{fused:.2f} m" if np.isfinite(fused) else "PENDING", (1060, 204), 36, GREEN)
    text(screen, "With GPS correction: position error", (1060, 254), 15, MUTED)
    text(screen, f"Without GPS correction: {latest[2]:.2f} m", (1060, 281), 17, AMBER)
    text(screen, "GPS: 5 updates/s  |  speed: 60 updates/s", (1060, 311), 15, MUTED)

    panel(screen, pygame.Rect(1040, 358, 376, 220))
    text(screen, "03 / POSITION ERROR OVER TIME", (1060, 375), 13, MUTED)
    data = np.asarray(sim.metrics) if sim.metrics else np.empty((0, 3))
    top = chart(screen, pygame.Rect(1060, 412, 336, 112),
                [data[:, 1], data[:, 2]], [GREEN, AMBER])
    text(screen, f"0 - {top:.1f} m  /  last 10 s", (1060, 534), 14, MUTED)

    panel(screen, pygame.Rect(1040, 592, 376, 222))
    backend = "OPTIMIZATION" if sim.controller_backend == "optimization" else "SEQUENCE SEARCH"
    text(screen, "04 / " + backend, (1060, 609), 13, MUTED)
    active = sim.mode == "mpc"
    text(screen, "MPC: replans 10 times/s" if active else "PRESET MOTION", (1060, 634), 22, PURPLE)
    text(screen, f"Looks ahead {sim.controller.horizon * sim.controller.dt:.1f} s / {sim.controller.horizon} steps", (1060, 668), 16, MUTED)
    for i, name in enumerate(("X", "Y")):
        y = 706 + i * 30
        text(screen, f"{name}  {sim.control[i]:+5.2f}", (1060, y - 4), 16)
        bar = pygame.Rect(1178, y, 216, 10)
        pygame.draw.rect(screen, EDGE, bar, border_radius=5)
        fraction = np.clip(sim.control[i] / max(sim.controller.max_acceleration, 1e-9), -1, 1)
        if fraction:
            left = bar.centerx if fraction > 0 else bar.centerx + int(fraction * 108)
            pygame.draw.rect(screen, PURPLE, (left, y, max(1, int(abs(fraction) * 108)), 10), border_radius=4)
        pygame.draw.line(screen, MUTED, (bar.centerx, y - 2), (bar.centerx, y + 12))
    status = "Acceleration above in m/s^2 | prediction shown" if rollout is not None and active else "Acceleration in m/s^2 | prediction unavailable"
    if active and "pending" in sim.controller_status.lower():
        status = getattr(sim.controller, "status", "Controller pending / coasting")
    text(screen, status if active else "Press M to enable MPC mode", (1060, 777), 14, MUTED)

    text(screen, "SPACE  pause     R  reset     N  step     M  mode     F11  fullscreen     S  screenshot", (26, 838), 16, MUTED)
    text(screen, "READ SENSORS  >  ESTIMATE POSITION  >  PLAN MOVEMENT  >  MOVE ROBOT", (26, 870), 12, CYAN)
    text(screen, "Position error: distance between estimate and actual robot", (1050, 870), 12, MUTED)
