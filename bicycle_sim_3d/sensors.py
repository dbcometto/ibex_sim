"""Defines sensors that produce noisy measurements of the vehicle's motion."""

from abc import ABC, abstractmethod

import numpy as np

from vehicle import relative_pose_local_frame, rotation_body_to_world

GRAVITY = 9.81


class Sensor(ABC):
    """Base class for anything that produces a noisy measurement of the
    vehicle's motion.

    measure() takes the same three arguments for every sensor:
      vehicle: the ground-truth Vehicle instance, post-step (gives
        access to vehicle.prev_state, vehicle.state, vehicle.dt, and
        model-specific attributes like vehicle.wheelbase where needed).
        State is now 6D: (x, y, z, roll, pitch, yaw).
      v, delta: the control input that was applied for this step.
    Sensors that don't need one of these just accept and ignore it, so
    any sensor can be called the same way in a loop.
    """

    @abstractmethod
    def measure(self, vehicle, v, delta):
        raise NotImplementedError


class LidarOdometrySensor(Sensor):
    """Simulates KISS-ICP style LiDAR odometry: registers each new true
    state against its own running 6D pose estimate, with noise on the
    per-step relative motion. Error accumulates in self.global_pose over
    time (drift), matching how scan-to-local-map registration behaves.

    measure() returns the sensor's own global pose estimate
    (x, y, z, roll, pitch, yaw), not a clean relative delta.
    """

    def __init__(self, noise_std=(0.02, 0.02, 0.02, 0.005, 0.005, 0.01),
                 init_state=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)):
        self.noise_std = noise_std
        self.global_pose = np.array(init_state, dtype=float)

    def measure(self, vehicle, v, delta):
        dx, dy, dz, droll, dpitch, dyaw = relative_pose_local_frame(
            vehicle.prev_state, vehicle.state
        )
        dx += np.random.normal(0, self.noise_std[0])
        dy += np.random.normal(0, self.noise_std[1])
        dz += np.random.normal(0, self.noise_std[2])
        droll += np.random.normal(0, self.noise_std[3])
        dpitch += np.random.normal(0, self.noise_std[4])
        dyaw += np.random.normal(0, self.noise_std[5])

        yaw = self.global_pose[5]
        c, s = np.cos(yaw), np.sin(yaw)
        self.global_pose[0] += c * dx - s * dy
        self.global_pose[1] += s * dx + c * dy
        self.global_pose[2] += dz
        self.global_pose[3] += droll
        self.global_pose[4] += dpitch
        self.global_pose[5] += dyaw

        return self.global_pose.copy()


class ImuSensor(Sensor):
    """Simulates a 3-axis gyro + 3-axis body-frame accelerometer.

    Gyro: angular rates (roll_rate, pitch_rate, yaw_rate) via finite
    difference of the true attitude over dt. This reads the vehicle's
    own state history directly rather than re-deriving rates
    analytically from (v, delta, wheelbase) -- an improvement over the
    2D version, since it no longer needs to assume a specific motion
    model, only that the vehicle exposes prev_state/state.

    Accelerometer: body-frame specific force = gravity rotated into the
    body frame (rotation_body_to_world(...).T @ [0,0,g]) plus a rough
    kinematic term (longitudinal ~0 assuming constant v, lateral ~
    centripetal v*yaw_rate). Vertical kinematic acceleration is NOT
    modeled (would need a second finite difference / more state history
    than is available here) -- so climbing/descending terrain shows up
    correctly via the gravity term's rotation, but any actual vertical
    acceleration from a change in climb rate is invisible to this
    sensor. That's a real gap, not just a caveat, if vertical dynamics
    matter to you.

    Uses the standard two-term IMU noise model: a slowly-drifting bias
    (random walk) plus additive white noise, isotropic across each
    sensor's 3 axes. Two IMUs of differing quality (e.g. consumer vs.
    tactical-grade) are just two instances with different noise/bias
    params.
    """

    def __init__(self, gyro_noise_std=0.01, accel_noise_std=0.05,
                 gyro_bias_walk_std=0.0005, accel_bias_walk_std=0.001,
                 init_gyro_bias=(0.0, 0.0, 0.0), init_accel_bias=(0.0, 0.0, 0.0)):
        self.gyro_noise_std = gyro_noise_std
        self.accel_noise_std = accel_noise_std
        self.gyro_bias_walk_std = gyro_bias_walk_std
        self.accel_bias_walk_std = accel_bias_walk_std

        self.gyro_bias = np.array(init_gyro_bias, dtype=float)
        self.accel_bias = np.array(init_accel_bias, dtype=float)

    def measure(self, vehicle, v, delta):
        prev_state, curr_state = vehicle.prev_state, vehicle.state
        dt = vehicle.dt

        roll, pitch, yaw = curr_state[3], curr_state[4], curr_state[5]
        roll_rate = (curr_state[3] - prev_state[3]) / dt
        pitch_rate = (curr_state[4] - prev_state[4]) / dt
        yaw_rate = (curr_state[5] - prev_state[5]) / dt
        omega_true = np.array([roll_rate, pitch_rate, yaw_rate])

        R = rotation_body_to_world(roll, pitch, yaw)
        gravity_reaction = R.T @ np.array([0.0, 0.0, GRAVITY])
        kinematic = np.array([0.0, v * yaw_rate, 0.0])  # vertical kinematic accel not modeled
        accel_true = gravity_reaction + kinematic

        self.gyro_bias += np.random.normal(0, self.gyro_bias_walk_std, size=3)
        self.accel_bias += np.random.normal(0, self.accel_bias_walk_std, size=3)

        omega_meas = omega_true + self.gyro_bias + np.random.normal(0, self.gyro_noise_std, size=3)
        accel_meas = accel_true + self.accel_bias + np.random.normal(0, self.accel_noise_std, size=3)

        return omega_meas, accel_meas