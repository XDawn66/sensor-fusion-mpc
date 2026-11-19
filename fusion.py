"""Linear Kalman filter combining GPS and velocity measurements. All units are meters/seconds."""

import numpy as np


class SensorFusion:
    def __init__(self, initial_position):
        # State ordering: [x, y, vx, vy].
        self.state = np.array([
            initial_position[0],
            initial_position[1],
            0.0,
            0.0
        ], dtype=float)

        # uncertainty
        self.P = np.eye(4)

        # process noise
        self.Q = np.diag([0.0001, 0.0001, 0.01, 0.01])

        # GPS measurement noise
        self.R_gps = np.eye(2) * 0.8**2

        # velocity measurement noise
        self.R_velocity = np.eye(2) * 0.35**2

        self.H_gps = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])

        self.H_velocity = np.array([
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])


    def predict(self, dt):
        """propagate state and covariance by dt seconds."""

        # build motion model F
        self.F = np.array([[1,0,dt,0], 
                           [0,1,0,dt],
                           [0,0,1,0],
                           [0,0,0,1]],
                           )
    
        # predict state
        self.state = self.F @ self.state

        # predict covariance

        self.P = self.F @ self.P @ self.F.T + self.Q

    def update_gps(self, measurement):
        """update with a NumPy vector [x, y] in meters."""

        z = np.asarray(measurement, dtype=float)

        y = z - self.H_gps @ self.state

        S = self.H_gps @ self.P @ self.H_gps.T + self.R_gps

        K = self.P @ self.H_gps.T @ np.linalg.inv(S)

        self.state = self.state + K @ y

        self.P = (np.eye(4) - K @ self.H_gps) @ self.P

    def update_velocity(self, measurement):
        """update with a NumPy vector [vx, vy] in meters/second."""
        z = np.asarray(measurement, dtype=float)

        y = z - self.H_velocity @ self.state

        S = self.H_velocity @ self.P @ self.H_velocity.T + self.R_velocity

        K = self.P @ self.H_velocity.T @ np.linalg.inv(S)

        self.state = self.state + K @ y

        self.P = (np.eye(4) - K @ self.H_velocity) @ self.P


    @property
    def position(self):
        """Return the current corrected position estimate."""
        return self.state[:2].copy()
