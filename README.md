# Sensor Fusion & Model Predictive Control

A Python/Pygame robotics simulation that combines noisy sensor readings to estimate a vehicle's position, then uses Model Predictive Control (MPC) to follow an elliptical route. Built progressively to understand estimation, prediction, and closed-loop control.

```text
Noisy GPS + velocity sensor -> Kalman filter -> estimated state
                                                     |
                                                     v
                         vehicle motion <- acceleration <- MPC
```

## Project evolution

### Version 1 - Sensor simulation

Simulated a vehicle moving along a curved path, with noisy GPS and velocity measurements. Integrating velocity alone produced a position estimate that gradually drifted, demonstrating why sensor correction is needed.

### Version 2 - Kalman filter sensor fusion

Implemented prediction and separate GPS/velocity updates for state `[x, y, vx, vy]`. Combined the two sensors to correct drift, then tuned process noise to reduce abrupt GPS corrections. The implemented model is linear, so this is a standard Kalman filter rather than an EKF.

### Version 3 - Sequence-search MPC

Replaced scripted motion with an acceleration-controlled vehicle. Implemented a controller that enumerates discrete acceleration sequences, predicts their motion, and selects the sequence with the lowest tracking and control-effort cost. It applies only the first command, then replans using the latest estimated state. This version uses a four-step horizon.

### Version 4 - Presentation dashboard

Upgraded the visualization with a car icon, full reference track, sensor trails, uncertainty ellipse, error history, acceleration bars, and measured planning time. The green car shows the estimated position; a white dot shows the actual position for comparison. Added fullscreen and screenshot controls.

### Version 5 - Optimization-based MPC

Replaced discrete enumeration with continuous acceleration optimization using SciPy's SLSQP solver. Implemented a double-integrator prediction model, tracking/control-effort costs, terminal cost, and acceleration bounds. The controller plans 20 steps (2 seconds) ahead and uses the previous solution as a warm start. Its predicted trajectory appears in purple.

## Run

```powershell
python -m pip install -r requirements-optimization.txt

# Sensor fusion with scripted motion
python main.py

# Sensor fusion + sequence-search MPC
python main.py --mode mpc --controller search

# Sensor fusion + optimization-based MPC
python main.py --mode mpc --controller optimization
```

The simulation and velocity sensor run at 60 Hz, GPS at 5 Hz, and MPC replans at 10 Hz. Controls are acceleration `[ax, ay]`, limited to +/-2 m/s^2 per axis. The car icon is a visual representation; the motion model is a 2D double integrator, not a steering-based vehicle model.

## Reading the display

| Color | Meaning |
|---|---|
| Green car | Estimated position with GPS correction |
| White dot/line | Actual position and path |
| Orange | Position estimated without GPS correction; accumulates drift |
| Blue | Noisy GPS readings |
| Gray loop | Planned route |
| Purple | Moving target and controller's predicted trajectory, when supplied |

**Controls:** Space: pause/resume · R: reset · N: step while paused · M: switch fusion/MPC mode · F11: fullscreen · S: save screenshot · Esc: quit. Mode switching resets the simulation and preserves the selected controller.

## Code

- `main.py`: simulation, sensors, timing, and application controls.
- `fusion.py`: Kalman filter prediction and measurement updates.
- `mpc.py`: discrete sequence-search controller.
- `mpc_optimization.py`: continuous optimization-based controller.
- `dashboard.py`: visualization and live metrics.

The controller receives estimated state, never actual state. Actual state is used for simulation and evaluation only. The filter currently uses a constant-velocity prediction without the applied acceleration input. Acceleration bounds are enforced by the solver and commands are clipped by the simulator.
