from abc import ABC, abstractmethod

import numpy as np


class Vehicle(ABC):
    """Base class for ground-truth vehicle motion models.

    Subclasses own a state vector and advance it by one dt given
    some control input. What "control input" means (v/delta,
    v/omega, etc.) is left to the subclass.
    """

    def __init__(self, dt, state):
        self.dt = dt
        self.state = np.array(state, dtype=float)
        self.prev_state = self.state.copy()

    @abstractmethod
    def step(self, *controls):
        """Advance state by one dt.

        Returns (prev_state, curr_state), both as copies.
        """
        raise NotImplementedError


class BicycleVehicle(Vehicle):
    """Kinematic bicycle model. State: (x, y, theta). Control: (v, delta)."""

    def __init__(self, wheelbase, dt, state=(0.0, 0.0, 0.0)):
        super().__init__(dt, state)
        self.wheelbase = wheelbase

    def step(self, v, delta):
        prev_state = self.state.copy()
        x, y, theta = self.state
        x += v * np.cos(theta) * self.dt
        y += v * np.sin(theta) * self.dt
        theta += (v / self.wheelbase) * np.tan(delta) * self.dt
        self.state = np.array([x, y, theta])
        self.prev_state = prev_state
        return prev_state, self.state.copy()


class ImperfectBicycleVehicle(BicycleVehicle):
    """Kinematic bicycle model with noise added to the true state update."""

    def __init__(self, wheelbase, dt, state=(0.0, 0.0, 0.0), noise_std=(0.005, 0.005, 0.001)):
        super().__init__(wheelbase, dt, state)
        self.noise_std = noise_std

    def step(self, v, delta):
        prev_state, curr_state = super().step(v, delta)
        noise = np.random.normal(0.0, self.noise_std)
        self.state = curr_state + noise
        return prev_state, self.state.copy()


def relative_pose_local_frame(prev_state, curr_state):
    """Relative motion (dx, dy, dtheta) expressed in prev_state's local frame."""
    dx_global = curr_state[0] - prev_state[0]
    dy_global = curr_state[1] - prev_state[1]
    dtheta = curr_state[2] - prev_state[2]

    c, s = np.cos(prev_state[2]), np.sin(prev_state[2])
    dx_local = c * dx_global + s * dy_global
    dy_local = -s * dx_global + c * dy_global
    return dx_local, dy_local, dtheta