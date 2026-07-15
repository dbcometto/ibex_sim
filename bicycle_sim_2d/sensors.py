"""Defines sensors that produce noisy relative-pose measurements."""

from abc import ABC, abstractmethod

import numpy as np

from vehicle import relative_pose_local_frame


class Sensor(ABC):
    """Base class for anything that produces a noisy measurement of the
    vehicle's motion.

    measure() takes the same three arguments for every sensor:
      vehicle: the ground-truth Vehicle instance, post-step (gives
        access to vehicle.prev_state, vehicle.state, vehicle.dt, and
        model-specific attributes like vehicle.wheelbase where needed).
      v, delta: the control input that was applied for this step.
    Sensors that don't need one of these (e.g. LidarOdometrySensor
    ignores v/delta) just accept and ignore it, so any sensor can be
    called the same way in a loop.
    """

    @abstractmethod
    def measure(self, vehicle, v, delta):
        raise NotImplementedError


class LidarOdometrySensor(Sensor):
    """Simulates KISS-ICP style LiDAR odometry: registers each new true state
    against its own running pose estimate, with noise on the per-step
    relative motion. Unlike a single noisy relative measurement, error
    accumulates in self.global_pose over time (drift), matching how
    scan-to-local-map registration actually behaves.

    measure() returns the sensor's own global pose estimate (x, y, theta),
    not a clean relative delta.
    """

    def __init__(self, noise_std=(0.02, 0.02, 0.01), init_state=(0.0, 0.0, 0.0)):
        self.noise_std = noise_std
        self.global_pose = np.array(init_state, dtype=float)

    def measure(self, vehicle, v, delta):
        dx, dy, dtheta = relative_pose_local_frame(vehicle.prev_state, vehicle.state)
        dx += np.random.normal(0, self.noise_std[0])
        dy += np.random.normal(0, self.noise_std[1])
        dtheta += np.random.normal(0, self.noise_std[2])

        c, s = np.cos(self.global_pose[2]), np.sin(self.global_pose[2])
        self.global_pose[0] += c * dx - s * dy
        self.global_pose[1] += s * dx + c * dy
        self.global_pose[2] += dtheta

        return self.global_pose.copy()


class ImuSensor(Sensor):
    """Simulates a yaw-rate gyro + body-frame accelerometer.

    Uses the standard two-term IMU noise model: a slowly-drifting bias
    (random walk) plus additive white noise, on both gyro and accel.
    Two IMUs of differing quality (e.g. consumer vs. tactical-grade) are
    just two instances of this class with different noise/bias params.

    NOTE: true omega/accel are derived analytically from the bicycle
    model (v, delta, vehicle.wheelbase), assuming constant v over the
    step (so longitudinal accel is always 0 here). This ties ImuSensor
    to vehicles that expose a `.wheelbase` attribute — it will break on
    a Vehicle subclass with a different motion model.
    """

    def __init__(self, gyro_noise_std=0.01, accel_noise_std=0.05,
                 gyro_bias_walk_std=0.0005, accel_bias_walk_std=0.001,
                 init_gyro_bias=0.0, init_accel_bias=(0.0, 0.0)):
        self.gyro_noise_std = gyro_noise_std
        self.accel_noise_std = accel_noise_std
        self.gyro_bias_walk_std = gyro_bias_walk_std
        self.accel_bias_walk_std = accel_bias_walk_std

        self.gyro_bias = init_gyro_bias
        self.accel_bias = np.array(init_accel_bias, dtype=float)

    def measure(self, vehicle, v, delta):
        omega_true = (v / vehicle.wheelbase) * np.tan(delta)
        accel_true = np.array([0.0, v * omega_true])  # [longitudinal, lateral/centripetal]

        self.gyro_bias += np.random.normal(0, self.gyro_bias_walk_std)
        self.accel_bias += np.random.normal(0, self.accel_bias_walk_std, size=2)

        omega_meas = omega_true + self.gyro_bias + np.random.normal(0, self.gyro_noise_std)
        accel_meas = accel_true + self.accel_bias \
            + np.random.normal(0, self.accel_noise_std, size=2)

        return omega_meas, accel_meas
