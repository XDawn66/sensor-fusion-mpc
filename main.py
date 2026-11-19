"""Pygame sensor-fusion and model predictive control simulation."""

import argparse
import math
from time import perf_counter
from collections import deque

import numpy as np
import pygame

from fusion import SensorFusion
from mpc import MPCController
from mpc_optimization import OptimizationMPCController

DT = 1 / 60  # Fixed simulation timestep, independent of rendering speed.
GPS_EVERY = 12  # 5 Hz GPS; velocity measurements arrive at 60 Hz.
GPS_STD = 0.8  # meters, per axis
VELOCITY_STD = 0.35  # meters/second, per axis
VELOCITY_BIAS = np.array([0.07, -0.04])  # makes dead-reckoning drift visible
WIDTH, HEIGHT = 1440, 900
SCALE = 40  # pixels per meter
PLOT = pygame.Rect(24, 164, 1000, 650)
ORIGIN = PLOT.center  # Center the world below the status panel.
CONTROL_EVERY = 6  # MPC solves at 10 Hz, plant and sensors run at 60 Hz.
BLACK, BLUE, ORANGE, GREEN = (30, 35, 44), (65, 125, 220), (220, 130, 35), (30, 160, 110)


def true_state(t):
    """Analytic curved path and its exact velocity; only the simulator uses this."""
    angle = 0.22 * t
    return np.array([9 * math.cos(angle), 6 * math.sin(angle),
                     -9 * 0.22 * math.sin(angle), 6 * 0.22 * math.cos(angle)])


class Simulation:
    def __init__(self, seed=7, mode="demo", controller_backend="search"):
        self.mode = mode
        self.controller_backend = controller_backend
        self.rng = np.random.default_rng(seed)
        self.tick = 0
        self.truth = true_state(0)
        controller_class = OptimizationMPCController if controller_backend == "optimization" else MPCController
        self.controller = controller_class(dt=CONTROL_EVERY * DT)
        self.control = np.zeros(2)
        self.controller_status = "MPC pending: zero acceleration (coasting)"
        self.solve_ms = None
        self.metrics = deque(maxlen=600)  # 10 seconds, independent of render FPS
        # Known starting position, shared by dead reckoning and your filter.
        self.raw_position = self.truth[:2].copy()
        self.filter = SensorFusion(self.raw_position.copy())
        self.gps = None
        self.velocity = np.zeros(2)
        self.true_trail = deque([self.truth[:2].copy()], maxlen=2400)
        self.raw_trail = deque([self.raw_position.copy()], maxlen=2400)
        self.fused_trail = deque(maxlen=2400)
        self.gps_trail = deque(maxlen=200)

    def step(self):
        if self.mode == "mpc":
            if self.tick % CONTROL_EVERY == 0:
                reference = np.array([
                    true_state(self.tick * DT + k * self.controller.dt)
                    for k in range(self.controller.horizon + 1)
                ])
                started = perf_counter()
                command = self.controller.solve(self.filter.state.copy(), reference)
                self.solve_ms = (perf_counter() - started) * 1000
                if command is None:
                    self.control = np.zeros(2)
                    self.controller_status = "MPC pending: zero acceleration (coasting)"
                else:
                    command = np.asarray(command, dtype=float)
                    if command.shape != (2,) or not np.isfinite(command).all():
                        raise ValueError("MPC solve must return finite [ax, ay] or None")
                    self.control = np.clip(command, -self.controller.max_acceleration,
                                           self.controller.max_acceleration)
                    self.controller_status = "MPC active"
            # Double-integrator plant, with constant acceleration over each step.
            self.truth[:2] += self.truth[2:] * DT + 0.5 * self.control * DT**2
            self.truth[2:] += self.control * DT
        self.tick += 1
        if self.mode == "demo":
            self.truth = true_state(self.tick * DT)
        self.velocity = self.truth[2:] + self.rng.normal(0, VELOCITY_STD, 2) + VELOCITY_BIAS
        self.raw_position += self.velocity * DT

        # Your filter receives time and noisy measurements, never ground truth.
        self.filter.predict(DT)
        if self.tick % GPS_EVERY == 0:
            self.gps = self.truth[:2] + self.rng.normal(0, GPS_STD, 2)
            self.gps_trail.append(self.gps.copy())
            self.filter.update_gps(self.gps.copy())
        self.filter.update_velocity(self.velocity.copy())

        self.true_trail.append(self.truth[:2].copy())
        self.raw_trail.append(self.raw_position.copy())
        estimate = self.filter.position
        if estimate is not None:
            self.fused_trail.append(np.asarray(estimate, dtype=float).copy())
        self.metrics.append((
            np.linalg.norm(self.truth[:2] - true_state(self.tick * DT)[:2]),
            np.linalg.norm(estimate - self.truth[:2]) if estimate is not None else np.nan,
            np.linalg.norm(self.raw_position - self.truth[:2]),
        ))


def pixel(position):
    return (round(ORIGIN[0] + position[0] * SCALE),
            round(ORIGIN[1] - position[1] * SCALE))


def draw(screen, font, small, sim, paused):
    from dashboard import render
    render(screen, sim, paused, PLOT, pixel, true_state, DT, SCALE)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=0, help="Exit after N frames (0 runs interactively).")
    parser.add_argument("--mode", choices=["demo", "mpc"], default="demo")
    parser.add_argument("--controller", choices=["search", "optimization"], default="search")
    args = parser.parse_args()
    pygame.display.init()
    pygame.font.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.SCALED | pygame.RESIZABLE)
    pygame.display.set_caption("Autonomy Lab | State Estimation & MPC")
    font = pygame.font.Font(None, 36)
    small = pygame.font.Font(None, 24)
    clock = pygame.time.Clock()
    sim, paused, running = Simulation(mode=args.mode, controller_backend=args.controller), False, True
    accumulator, frames = 0.0, 0
    try:
        while running:
            elapsed = min(clock.tick(60) / 1000, 0.1)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_F11:
                        pygame.display.toggle_fullscreen()
                    elif event.key == pygame.K_s:
                        from pathlib import Path
                        from datetime import datetime
                        directory = Path(__file__).parent / "screenshots"
                        directory.mkdir(exist_ok=True)
                        pygame.image.save(screen, str(directory / (datetime.now().strftime("autonomy-%Y%m%d-%H%M%S-%f") + ".png")))
                    elif event.key == pygame.K_SPACE:
                        paused = not paused
                        accumulator = 0.0
                        elapsed = 0.0
                    elif event.key == pygame.K_r:
                        sim = Simulation(mode=sim.mode, controller_backend=sim.controller_backend)
                        accumulator = 0.0
                        elapsed = 0.0
                    elif event.key == pygame.K_m:
                        sim = Simulation(mode="mpc" if sim.mode == "demo" else "demo",
                                         controller_backend=sim.controller_backend)
                        accumulator = 0.0
                        elapsed = 0.0
                    elif event.key == pygame.K_n and paused:
                        sim.step()
            if not paused:
                accumulator += elapsed
                while accumulator >= DT:
                    sim.step()
                    accumulator -= DT
            draw(screen, font, small, sim, paused)
            pygame.display.flip()
            frames += 1
            if args.frames > 0 and frames >= args.frames:
                running = False
    finally:
        pygame.quit()


if __name__ == "__main__":
    main()
