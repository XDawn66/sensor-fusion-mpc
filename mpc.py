"""Discrete sequence-search model predictive control. State [x,y,vx,vy]; control [ax,ay], SI units."""

import numpy as np
from itertools import product


class MPCController:
    def __init__(self, dt=0.1, horizon=4, max_acceleration=2.0):
        self.dt = dt
        self.horizon = horizon
        self.max_acceleration = max_acceleration  # per-axis bound, m/s^2
        self.predicted_states = None

        self.accel_values = [-2.0, 0.0, 2.0]

    def solve(self, estimated_state, reference_states):
        """Select the first acceleration from the lowest-cost discrete sequence.

        estimated_state: (4,), [x,y,vx,vy] from sensor fusion.
        reference_states: (horizon+1,4), spaced by controller dt.
        The simulator holds the first command until the next solve.
        """
        best_control = self.get_best_control(estimated_state, reference_states)
        return best_control

    def motion_model(self, state, control):

        """
        state = [x, y, vx, vy]
        control = [ax, ay]
        Uses forward-Euler position integration.
        """

        x = state[0]
        y = state[1]
        vx = state[2]
        vy = state[3]

        ax = control[0]
        ay = control[1]

        dx = x + vx * self.dt
        dy = y + vy * self.dt

        dvx = vx + ax * self.dt
        dvy = vy + ay * self.dt
   

        return [dx,dy,dvx,dvy]

    def calculate_cost(self, state, target, control):

        current_x = state[0]
        current_y = state[1]

        pos_error = (target[0] - current_x) ** 2 + (target[1] - current_y) ** 2

        vx_error = (target[2] - state[2]) ** 2
        vy_error = (target[3] - state[3]) ** 2

        vel_error = vx_error + vy_error

        control_effort = control[0] ** 2 + control[1] ** 2

        cost = pos_error + vel_error + 0.1 * control_effort

        return cost

    def generate_control_sequences(self):

        controls = []

        for ax in self.accel_values:
            for ay in self.accel_values:
                controls.append([ax, ay])

        sequences = list(product(controls, repeat = self.horizon))

        return sequences

    def evaluate_sequence(self, state, reference_states, control_sequence):

        predicted_states = state.copy()

        total_cost = 0.0

        for k, control in enumerate(control_sequence):
            next_state = self.motion_model(predicted_states, control)
            target = reference_states[k + 1]
            current_cost = self.calculate_cost(next_state, target, control)

            total_cost += current_cost
            predicted_states = next_state

        return total_cost

    def get_best_control(self, state, target):

        control_sequences = self.generate_control_sequences()

        best_cost = float("inf")
        best_sequence = None

        for seq in control_sequences:

            current_cost = self.evaluate_sequence(state, target, seq)

            if current_cost < best_cost:

                best_cost = current_cost
                best_sequence = seq

        return best_sequence[0]
