# Optimization-based MPC

The continuous controller in `mpc_optimization.py` uses SciPy SLSQP to optimize
accelerations for tracking an elliptical route. The discrete sequence-search
controller remains available in `mpc.py` for comparison.

## Run

```powershell
python -m pip install -r requirements-optimization.txt
python main.py --mode mpc --controller optimization
```

Use `--controller search` for sequence search. Reset and mode switching preserve
the selected controller.

## How it works

1. Predict motion with a constant-acceleration double-integrator model.
2. Score state tracking error, acceleration effort, and terminal error.
3. Bound acceleration at every horizon step.
4. Optimize the continuous control sequence using SLSQP.
5. Apply only the first command and replan from the next sensor-based estimate.

The controller uses a shooting formulation: candidate controls are rolled out to
produce predicted states. Position includes the `0.5 * acceleration * dt^2`
contribution. The previous solution is shifted forward as the next initial guess.
The optimized trajectory is drawn in purple on the dashboard.

| Value | Meaning |
|---|---|
| State | `[x, y, vx, vy]`, shape (4,) |
| Control sequence | `[ax, ay]` per step, shape (N, 2) |
| Solver variables | Flattened controls, shape (2*N,) |
| Reference / prediction | Shape (N+1, 4), including the current timestep |
| Default horizon | 20 steps / 2 seconds |
| Controller rate | 10 Hz |
| Plant / velocity sensor | 60 Hz |
| GPS | 5 Hz |
| Acceleration limit | +/-2 m/s^2 per axis |

Failed or malformed solver results produce no command, and the simulator applies
zero acceleration (coasting). Solver bounds constrain accelerations; the plant
also clips applied commands. Coasting does not stop the vehicle.

The controller receives estimated state, never ground truth. Ground truth is
used for simulation and error measurement. The filter currently predicts constant
velocity without the applied acceleration input. The car icon represents a 2D
double integrator rather than steering-based vehicle dynamics.

The dashboard reports actual solve duration. The control period is 100 ms;
meeting that budget depends on hardware and solver convergence.
