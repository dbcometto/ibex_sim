"""Async-rate versions of sensors.py's sensor models.

The original sensors read vehicle.prev_state / vehicle.state, which
only reflects the single most recent physics tick. Once the vehicle is
advanced many fine ticks between sensor samples (VehicleRunner), each
sensor instead needs to remember ITS OWN last-sampled state/time and
compute relative motion (or elapsed dt) since then -- otherwise most of
the true motion between sensor samples would be silently dropped.

Noise/bias/drift models are unchanged from sensors.py; only the
state-tracking is reworked.
"""
import numpy as np

from vehicle import relative_pose_local_frame, rotation_body_to_world

GRAVITY = 9.81


class LidarOdometrySensorAsync:
    """Same running-global-pose drift model as LidarOdometrySensor, but
    the per-call relative motion is computed against this sensor's own
    last sample, not the vehicle's last physics tick.
    """

    def __init__(self, noise_std, init_state):
        self.noise_std = noise_std
        self.global_pose = np.array(init_state, dtype=float)
        self.last_true_state = np.array(init_state, dtype=float)

    def measure(self, vehicle_runner):
        curr_state = vehicle_runner.vehicle.state
        dx, dy, dz, droll, dpitch, dyaw = relative_pose_local_frame(
            self.last_true_state, curr_state
        )
        self.last_true_state = curr_state.copy()

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


class ImuSensorAsync:
    """Same two-term (bias random walk + white noise) IMU model as
    ImuSensor, but gyro rate is a finite difference over the ACTUAL
    elapsed time since this sensor's own last call (dt = now - last_t),
    not a fixed vehicle.dt. Bias random walk is scaled by sqrt(dt) for
    the same reason -- a fixed-dt scaling would be wrong once dt varies
    call to call.
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
        self.last_state = None
        self.last_t = None

    def measure(self, vehicle_runner, v):
        curr_state = vehicle_runner.vehicle.state
        t = vehicle_runner.t

        if self.last_state is None:
            self.last_state = curr_state.copy()
            self.last_t = t
            return None  # no valid dt yet -- first call only

        dt = t - self.last_t
        if dt <= 0:
            return None  # guard against a repeated/degenerate timestamp

        roll, pitch, yaw = curr_state[3], curr_state[4], curr_state[5]
        roll_rate = (curr_state[3] - self.last_state[3]) / dt
        pitch_rate = (curr_state[4] - self.last_state[4]) / dt
        yaw_rate = (curr_state[5] - self.last_state[5]) / dt
        omega_true = np.array([roll_rate, pitch_rate, yaw_rate])

        R = rotation_body_to_world(roll, pitch, yaw)
        gravity_reaction = R.T @ np.array([0.0, 0.0, GRAVITY])
        kinematic = np.array([0.0, v * yaw_rate, 0.0])  # vertical kinematic accel not modeled
        accel_true = gravity_reaction + kinematic

        self.gyro_bias += np.random.normal(0, self.gyro_bias_walk_std * np.sqrt(dt), size=3)
        self.accel_bias += np.random.normal(0, self.accel_bias_walk_std * np.sqrt(dt), size=3)

        omega_meas = omega_true + self.gyro_bias + np.random.normal(0, self.gyro_noise_std, size=3)
        accel_meas = accel_true + self.accel_bias + np.random.normal(0, self.accel_noise_std, size=3)

        self.last_state = curr_state.copy()
        self.last_t = t
        return omega_meas, accel_meas, dt


class GpsSensorAsync:
    """Stationary-noise absolute position measurement. Rate/dropout are
    now entirely the scheduler's job (SensorStream period/dropout_prob)
    -- this class no longer owns a frequency/interval counter the way
    the original GpsSensor did.
    """

    def __init__(self, noise_std=(1.5, 1.5, 3.0)):
        self.noise_std = noise_std

    def measure(self, vehicle_runner):
        true_pos = vehicle_runner.vehicle.state[0:3]
        return true_pos + np.random.normal(0, self.noise_std, size=3)