"""Continuous optimization-based model predictive control.

State [x,y,vx,vy], control [ax,ay], SI units. Shooting formulation:
optimize N continuous controls and roll out N+1 states. No search enumeration.
"""

import numpy as np
from scipy.optimize import minimize

class OptimizationMPCController:
    def __init__(self, dt=0.1, horizon=20, max_acceleration=2.0):
        self.dt = dt
        self.horizon = horizon
        self.max_acceleration = max_acceleration
        self.predicted_states = None
        self.previous_controls = None
        self.status = "Ready"
        # State tracking, control effort, and terminal cost weights.
        self.tracking = 1
        self.control_effort = 0.5
        self.terminal_weight = 2

    def motion_model(self, state, control):
        """Predict the next state (4,) using the controller timestep.

        Uses the simulator's constant-acceleration double integrator, including
        acceleration's contribution to position, at controller dt (0.1 s).
        """
        dx = state[0] + state[2] * self.dt + 0.5 * control[0] * self.dt **2
        dy = state[1] + state[3] * self.dt + 0.5 * control[1] * self.dt **2
        dvx = state[2] + control[0] * self.dt
        dvy = state[3] + control[1] * self.dt

        return [dx,dy,dvx,dvy]

    def rollout(self, initial_state, controls):
        """Roll out a control sequence from the initial state."""
        states = np.empty((self.horizon + 1, 4))
        states[0] = initial_state
        for k, control in enumerate(controls):
            states[k + 1] = self.motion_model(states[k], control)
        return states

    def objective(self, flat_controls, initial_state, reference_states):
        """Evaluate tracking, control effort, and terminal cost for one sequence.

        Controls have shape (N,2). Each predicted state is compared with the
        reference at the same future timestep; the final state has extra weight.
        """

        # One two-axis acceleration command per horizon step.
        flat_controls = np.reshape(flat_controls, (self.horizon, 2))
        rolled_states = self.rollout(initial_state, flat_controls)

        total_cost = 0.0

        for k, control in enumerate(flat_controls):

            rolled_state = rolled_states[k+1]
            cur_ref_state = reference_states[k+1]

            x_error = (cur_ref_state[0] - rolled_state[0]) ** 2
            y_error = (cur_ref_state[1] - rolled_state[1]) ** 2
            vx_error = (cur_ref_state[2] - rolled_state[2]) ** 2
            vy_error = (cur_ref_state[3] - rolled_state[3]) ** 2

            total_error = x_error + y_error + vx_error + vy_error

            control_effort = control[0] ** 2 + control[1] ** 2

            cost = self.tracking * total_error + self.control_effort * control_effort 

            total_cost += cost

        final_state = rolled_states[-1]
        final_ref = reference_states[-1]

        terminal_error = np.sum((final_ref - final_state) ** 2)

        total_cost += self.terminal_weight * terminal_error

        return total_cost

    def control_bounds(self):
        """Return per-axis acceleration bounds for all horizon steps.

        Order: [ax_0, ay_0, ax_1, ay_1, ...]. Bounds are symmetric around zero.
        """

        bounds = []

        for _ in range(self.horizon*2):
            bounds.append((-self.max_acceleration, self.max_acceleration))
        
        return bounds

    def optimize(self, initial_guess, initial_state, reference_states, bounds):
        """Minimize the horizon cost using bounded SLSQP optimization.

        Returns a solver result containing success, controls, and a message.
        Iterations are capped at 100 with a cost tolerance of 1e-4.
        """
        result = minimize(fun = self.objective, x0 = initial_guess, args=(initial_state,reference_states),
                          method="SLSQP", bounds=bounds, options={"maxiter":100, "ftol": 1e-4})

        return result

    def solve(self, estimated_state, reference_states):
        """Solve from the estimated state and apply only the first optimal control."""
        self.predicted_states = None  # Clear stale predictions on every solve.
        state = np.asarray(estimated_state, dtype=float)
        reference = np.asarray(reference_states, dtype=float)
        if state.shape != (4,) or reference.shape != (self.horizon + 1, 4):
            raise ValueError("Expected state (4,) and reference (horizon+1, 4)")
        if not np.isfinite(state).all() or not np.isfinite(reference).all():
            raise ValueError("MPC inputs must be finite")
        guess = np.zeros((self.horizon, 2))
        if self.previous_controls is not None:
            guess[:-1] = self.previous_controls[1:]
            guess[-1] = self.previous_controls[-1]
        try:
            self.motion_model(state, guess[0])
            self.objective(guess.ravel(), state, reference)
            bounds = self.control_bounds()
            result = self.optimize(guess.ravel(), state, reference, bounds)
        except NotImplementedError as error:
            self.status = str(error)
            return None

        # Reject failed or malformed solver output before publishing predictions.
        if result.success != True:
            self.status = result.message
            self.previous_controls = None
            return None

        if result.x.shape != (2 * self.horizon,):
            self.status = "incorrect optimizer shape"
            self.previous_controls = None
            return None

        if not np.isfinite(result.x).all():
            self.status = "Optimizer returned non-finite controls"
            self.previous_controls = None
            return None
   
        optimize_controls = np.reshape(result.x, (self.horizon, 2))
        self.predicted_states = self.rollout(state, optimize_controls)
        self.previous_controls = optimize_controls.copy()

        self.status = "Optimization successful"
        return optimize_controls[0].copy()
