"""Defines estimators that turn sensor measurements into a state estimate."""

from abc import ABC, abstractmethod

import numpy as np
import gtsam

GRAVITY = 9.81  # duplicated from sensors.py's constant of the same name -- kept
                # local rather than importing sensors here, to avoid coupling
                # the estimator module to the sensor module for one constant.


class Estimator(ABC):
    """Base class for anything that maintains a running state estimate."""

    def __init__(self, init_state=(0.0, 0.0, 0.0)):
        self.state = np.array(init_state, dtype=float)

    @abstractmethod
    def update(self, *args, **kwargs):
        """Consume a new measurement and return the updated state estimate."""
        raise NotImplementedError


class DeadReckoner(Estimator):
    """Open-loop strapdown integration of gyro + accelerometer — drifts unbounded, quadratically in position, by design.

    NOTE: left untouched -- still 2D-only (state is (x, y, theta)), not
    ported to the 6D (x, y, z, roll, pitch, yaw) convention. Per "forget
    about DR," this class is not currently usable in the 3D pipeline.
    """

    def __init__(self, dt, init_state=(0.0, 0.0, 0.0), init_velocity=(0.0, 0.0)):
        super().__init__(init_state)
        self.dt = dt
        self.velocity = np.array(init_velocity, dtype=float)  # body-frame (vx, vy)

    def update(self, measurements, v, delta):
        omega_meas, accel_meas = measurements["imu"]

        x, y, theta = self.state
        vx, vy = self.velocity
        ax, ay = accel_meas

        theta_new = theta + omega_meas * self.dt

        vx_new = vx + ax * self.dt
        vy_new = vy + ay * self.dt

        c, s = np.cos(theta), np.sin(theta)
        x_new = x + (c * vx - s * vy) * self.dt
        y_new = y + (s * vx + c * vy) * self.dt

        self.state = np.array([x_new, y_new, theta_new])
        self.velocity = np.array([vx_new, vy_new])
        return self.state.copy()


def _state_to_pose3(state):
    """Our convention (x, y, z, roll, pitch, yaw) -> gtsam.Pose3.

    UNVERIFIED: gtsam.Rot3.Ypr(yaw, pitch, roll) is documented/recalled
    to build R = Rz(yaw) @ Ry(pitch) @ Rx(roll), matching our own
    rotation_body_to_world -- but I could not run gtsam in this sandbox
    to confirm that against our tested sign conventions. Verify with
    the round-trip script before trusting this.
    """
    x, y, z, roll, pitch, yaw = state
    return gtsam.Pose3(gtsam.Rot3.Ypr(yaw, pitch, roll), gtsam.Point3(x, y, z))


def _pose3_to_state(pose):
    """gtsam.Pose3 -> our convention (x, y, z, roll, pitch, yaw)."""
    rot = pose.rotation()
    return np.array([pose.x(), pose.y(), pose.z(), rot.roll(), rot.pitch(), rot.yaw()])


def _sigmas_xyzrpy_to_gtsam(std_xyzrpy):
    """Reorder our (x, y, z, roll, pitch, yaw) sigma convention into
    gtsam's Pose3 tangent order, which is rotation-first:
    (roll, pitch, yaw, x, y, z). Getting this backwards doesn't crash --
    it silently swaps which physical quantity each sigma constrains.
    """
    x, y, z, roll, pitch, yaw = std_xyzrpy
    return np.array([roll, pitch, yaw, x, y, z])


class GraphReckonerLM(Estimator):
    """Factor-graph estimator, batch-optimized from scratch every keyframe
    for transparency. Now 3D: Pose3 nodes, (x, y, z, roll, pitch, yaw)
    state in/out.

    The dynamics factor only has an opinion on (dx, dy, dyaw) -- it
    predicts (dz, droll, dpitch) = 0 with deliberately loose noise on
    those three axes, since a real estimator has no way to predict
    terrain-driven z/roll/pitch changes from (v, delta) alone. This
    means the lidar prior is doing effectively all the work on those
    three dimensions; dynamics is only constraining planar motion.
    """

    def __init__(self, wheelbase, dt, init_state=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                 prior_noise_std=0.001,
                 dyn_noise_std=(0.005, 0.005, 1.0, 1.0, 1.0, 0.001),
                 lidar_noise_std=(0.05, 0.05, 0.05, 0.02, 0.02, 0.02),
                 enable_lidar=True,
                 # enable_IMU_1=True, enable_IMU_2=True
                 ):
        """
        dyn_noise_std, lidar_noise_std: given in our own
            (x, y, z, roll, pitch, yaw) order for readability --
            reordered internally to gtsam's rotation-first convention.
            dyn_noise_std's z/roll/pitch entries (1.0, 1.0, 1.0 here) are
            deliberately large/loose ("dynamics has no opinion here"),
            not a tuned value -- adjust only if you want dynamics to
            partially fight the lidar prior on those axes.
        """
        super().__init__(init_state)

        self.wheelbase = wheelbase
        self.dt = dt
        self.enable_lidar = enable_lidar
        # self.enable_IMU_1 = enable_IMU_1
        # self.enable_IMU_2 = enable_IMU_2

        prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([prior_noise_std] * 6))
        self.dyn_noise = gtsam.noiseModel.Diagonal.Sigmas(_sigmas_xyzrpy_to_gtsam(dyn_noise_std))
        self.lidar_noise_std = lidar_noise_std

        self.graph = gtsam.NonlinearFactorGraph()
        self.initial = gtsam.Values()
        self.index = 0

        key0 = gtsam.symbol('x', 0)
        pose0 = _state_to_pose3(init_state)
        self.graph.add(gtsam.PriorFactorPose3(key0, pose0, prior_noise))
        self.initial.insert(key0, pose0)

        self.result = gtsam.LevenbergMarquardtOptimizer(self.graph, self.initial).optimize()
        self.state = _pose3_to_state(self.result.atPose3(key0))

    def update(self, measurements, v, delta):
        prev_key = gtsam.symbol('x', self.index)
        self.index += 1
        curr_key = gtsam.symbol('x', self.index)

        dx, dy, dz, droll, dpitch, dyaw = self._predict_dynamics(v, delta)
        dyn_pose = gtsam.Pose3(gtsam.Rot3.Ypr(dyaw, dpitch, droll), gtsam.Point3(dx, dy, dz))
        self.graph.add(gtsam.BetweenFactorPose3(prev_key, curr_key, dyn_pose, self.dyn_noise))

        if self.enable_lidar:
            scaled_std = np.array(self.lidar_noise_std) * np.sqrt(self.index)
            odom_noise = gtsam.noiseModel.Diagonal.Sigmas(_sigmas_xyzrpy_to_gtsam(scaled_std))
            lidar_pose = _state_to_pose3(measurements["lidar_odom"])
            self.graph.add(gtsam.PriorFactorPose3(curr_key, lidar_pose, odom_noise))

        prev_pose = self.result.atPose3(prev_key)
        self.initial.insert(curr_key, prev_pose.compose(dyn_pose))

        self.result = gtsam.LevenbergMarquardtOptimizer(self.graph, self.initial).optimize()
        self.state = _pose3_to_state(self.result.atPose3(curr_key))

        return self.state.copy()

    def _predict_dynamics(self, v, delta):
        """Bicycle-model relative motion in the local frame. Only
        (dx, dy, dyaw) come from the motion model; (dz, droll, dpitch)
        are 0 -- see class docstring.
        """
        dx = v * self.dt
        dy = 0.0
        dz = 0.0
        droll = 0.0
        dpitch = 0.0
        dyaw = (v / self.wheelbase) * np.tan(delta) * self.dt
        return dx, dy, dz, droll, dpitch, dyaw


class GraphReckonerISAM(Estimator):
    """Same factor graph as GraphReckonerLM (Pose3 nodes, dynamics +
    unary lidar prior), but solved incrementally with ISAM2 instead of
    re-optimizing from scratch every keyframe. Faster for long runs;
    less transparent, since ISAM2 keeps its own internal state and
    self.graph/self.initial only ever hold the *newest* factors, not
    the full history.

    NOTE: no IMU support -- the earlier hand-rolled gravity-compensated
    strapdown IMU factor was removed from this class in favor of
    GraphReckonerPIM, which uses gtsam's real ImuFactor instead. This
    class is dynamics + lidar only now, same feature set as
    GraphReckonerLM (still duplicating its __init__/_predict_dynamics
    almost exactly, aside from the ISAM2 solve step).
    """

    def __init__(self, wheelbase, dt, init_state=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                 prior_noise_std=0.001,
                 dyn_noise_std=(0.005, 0.005, 1.0, 1.0, 1.0, 0.001),
                 lidar_noise_std=(0.05, 0.05, 0.05, 0.02, 0.02, 0.02),
                 enable_lidar=True):
        super().__init__(init_state)

        self.wheelbase = wheelbase
        self.dt = dt
        self.enable_lidar = enable_lidar

        prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([prior_noise_std] * 6))
        self.dyn_noise = gtsam.noiseModel.Diagonal.Sigmas(_sigmas_xyzrpy_to_gtsam(dyn_noise_std))
        self.lidar_noise_std = lidar_noise_std

        self.graph = gtsam.NonlinearFactorGraph()
        self.initial = gtsam.Values()
        self.isam = gtsam.ISAM2()
        self.index = 0

        key0 = gtsam.symbol('x', 0)
        pose0 = _state_to_pose3(init_state)
        self.graph.add(gtsam.PriorFactorPose3(key0, pose0, prior_noise))
        self.initial.insert(key0, pose0)

        self.isam.update(self.graph, self.initial)
        self.graph = gtsam.NonlinearFactorGraph()
        self.initial = gtsam.Values()

        self.state = _pose3_to_state(self.isam.calculateEstimate().atPose3(key0))

    def update(self, measurements, v, delta):
        prev_key = gtsam.symbol('x', self.index)
        self.index += 1
        curr_key = gtsam.symbol('x', self.index)

        dx, dy, dz, droll, dpitch, dyaw = self._predict_dynamics(v, delta)
        dyn_pose = gtsam.Pose3(gtsam.Rot3.Ypr(dyaw, dpitch, droll), gtsam.Point3(dx, dy, dz))
        self.graph.add(gtsam.BetweenFactorPose3(prev_key, curr_key, dyn_pose, self.dyn_noise))

        if self.enable_lidar:
            scaled_std = np.array(self.lidar_noise_std) * np.sqrt(self.index)
            odom_noise = gtsam.noiseModel.Diagonal.Sigmas(_sigmas_xyzrpy_to_gtsam(scaled_std))
            lidar_pose = _state_to_pose3(measurements["lidar_odom"])
            self.graph.add(gtsam.PriorFactorPose3(curr_key, lidar_pose, odom_noise))

        prev_pose = self.isam.calculateEstimate().atPose3(prev_key)
        self.initial.insert(curr_key, prev_pose.compose(dyn_pose))

        self.isam.update(self.graph, self.initial)
        self.graph = gtsam.NonlinearFactorGraph()
        self.initial = gtsam.Values()

        self.state = _pose3_to_state(self.isam.calculateEstimate().atPose3(curr_key))
        return self.state.copy()

    def _predict_dynamics(self, v, delta):
        """Identical to GraphReckonerLM._predict_dynamics -- see there."""
        dx = v * self.dt
        dy = 0.0
        dz = 0.0
        droll = 0.0
        dpitch = 0.0
        dyaw = (v / self.wheelbase) * np.tan(delta) * self.dt
        return dx, dy, dz, droll, dpitch, dyaw


class GraphReckonerPIM(Estimator):
    """Like GraphReckonerISAM, but IMU factors use gtsam's real
    PreintegratedImuMeasurements + ImuFactor instead of the hand-rolled
    strapdown integration. Gravity is handled by gtsam natively via
    PreintegrationParams.MakeSharedU(gravity) -- NOT by manually
    subtracting a body-frame gravity estimate the way
    GraphReckonerISAM's _integrate_imu_factor did. This avoids that
    method's orientation-estimate feedback loop, since gtsam defers
    gravity's effect to the factor's error function against the actual
    graph pose, not a running estimate we maintain ourselves.

    This adds two new kinds of graph variable beyond Pose3:
      - velocity V(i): one per keyframe, shared across all IMUs (there's
        only one true body velocity, however many IMUs observe it).
      - bias B_<imu>(i): one chain PER IMU, since each IMU's bias drifts
        independently. Consecutive biases are linked by a
        BetweenFactorConstantBias with small noise, letting bias
        random-walk -- matching ImuSensor's own bias model.

    NOTE on bias key symbols: each IMU needs its own single-character
    gtsam symbol prefix, currently assigned automatically as 'b', 'c',
    'd', ... in imu_configs dict order. This is a crude placeholder,
    not a robust scheme -- it will silently collide with any other
    single-letter symbol prefix used elsewhere ('x' and 'v' are already
    taken by pose/velocity), and offers no protection against typos or
    reordering imu_configs between runs. Fine for 2-3 IMUs; revisit if
    you add more or reuse prefixes elsewhere.

    UNVERIFIED: I have no gtsam installation in my sandbox and could
    not run any part of this -- not the ImuFactor construction, not the
    bias-chain wiring, not gravity handling itself. Given how much new
    surface area this adds (two new variable types, a new factor type,
    per-IMU state), treat this as a first draft to debug against real
    gtsam, not working code. Test on a trivial stationary case first
    (zero velocity, accel = pure gravity) and confirm the estimated
    pose barely moves, before trusting it on the real pipeline.
    """

    def __init__(self, wheelbase, dt, init_state=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                 prior_noise_std=0.001,
                 dyn_noise_std=(0.005, 0.005, 1.0, 1.0, 1.0, 0.001),
                 lidar_noise_std=(0.05, 0.05, 0.05, 0.02, 0.02, 0.02),
                 enable_lidar=True,
                 imu_configs=None, init_velocity=(0.0, 0.0, 0.0), gravity=GRAVITY,
                 init_velocity_noise_std=1.0):
        """
        imu_configs: dict mapping measurement-dict key (e.g.
            "imu_ouster") -> dict with keys 'gyro_noise_std',
            'accel_noise_std', 'bias_walk_std' (isotropic across the
            bias's 6 dims -- a simplification vs. ImuSensor's separate
            gyro/accel bias walk stds), and optionally
            'integration_noise_std' (default 1e-4, gtsam's own numerical-
            integration uncertainty term, not a physical sensor
            property). None/empty disables IMU factors entirely.
        init_velocity: WORLD-frame velocity at t=0, not body-frame --
            gtsam's ImuFactor/PIM define velocity in the world/nav frame.
            Get this wrong (e.g. leaving it at (0,0,0) while the vehicle
            actually starts moving) and every early ImuFactor residual
            fights a bad prior from step one.
        init_velocity_noise_std: loosened from an earlier hardcoded 0.1
            to 1.0 by default -- if you're not confident in
            init_velocity, a loose prior lets the graph correct itself
            instead of anchoring hard to a possibly-wrong guess.
        """
        super().__init__(init_state)

        self.wheelbase = wheelbase
        self.dt = dt
        self.enable_lidar = enable_lidar
        self.gravity = gravity

        prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([prior_noise_std] * 6))
        self.dyn_noise = gtsam.noiseModel.Diagonal.Sigmas(_sigmas_xyzrpy_to_gtsam(dyn_noise_std))
        self.lidar_noise_std = lidar_noise_std

        imu_configs = imu_configs or {}
        self.imu_names = list(imu_configs.keys())
        self.imu_pim = {}
        self.imu_bias_prefix = {}
        self.imu_bias_walk_std = {}
        for i, name in enumerate(self.imu_names):
            cfg = imu_configs[name]
            params = gtsam.PreintegrationParams.MakeSharedU(gravity)
            params.setGyroscopeCovariance(np.eye(3) * cfg.get('gyro_noise_std', 0.01) ** 2)
            params.setAccelerometerCovariance(np.eye(3) * cfg.get('accel_noise_std', 0.05) ** 2)
            params.setIntegrationCovariance(np.eye(3) * cfg.get('integration_noise_std', 1e-4) ** 2)

            self.imu_pim[name] = gtsam.PreintegratedImuMeasurements(params, gtsam.imuBias.ConstantBias())
            self.imu_bias_prefix[name] = chr(ord('b') + i)  # see class docstring caveat
            self.imu_bias_walk_std[name] = cfg.get('bias_walk_std', 0.001)

        self.graph = gtsam.NonlinearFactorGraph()
        self.initial = gtsam.Values()
        self.isam = gtsam.ISAM2()
        self.index = 0

        key0 = gtsam.symbol('x', 0)
        vel0_key = gtsam.symbol('v', 0)
        pose0 = _state_to_pose3(init_state)
        vel0 = np.array(init_velocity, dtype=float)

        self.graph.add(gtsam.PriorFactorPose3(key0, pose0, prior_noise))
        self.initial.insert(key0, pose0)

        vel_prior_noise = gtsam.noiseModel.Isotropic.Sigma(3, init_velocity_noise_std)
        self.graph.add(gtsam.PriorFactorVector(vel0_key, vel0, vel_prior_noise))
        self.initial.insert(vel0_key, vel0)

        bias_prior_noise = gtsam.noiseModel.Isotropic.Sigma(6, 0.1)
        for name in self.imu_names:
            bkey = gtsam.symbol(self.imu_bias_prefix[name], 0)
            self.graph.add(gtsam.PriorFactorConstantBias(bkey, gtsam.imuBias.ConstantBias(), bias_prior_noise))
            self.initial.insert(bkey, gtsam.imuBias.ConstantBias())

        self.isam.update(self.graph, self.initial)
        self.graph = gtsam.NonlinearFactorGraph()
        self.initial = gtsam.Values()

        est = self.isam.calculateEstimate()
        self.state = _pose3_to_state(est.atPose3(key0))
        self.velocity = est.atVector(vel0_key)

    def update(self, measurements, v, delta):
        prev_key = gtsam.symbol('x', self.index)
        prev_vel_key = gtsam.symbol('v', self.index)
        self.index += 1
        curr_key = gtsam.symbol('x', self.index)
        curr_vel_key = gtsam.symbol('v', self.index)

        dx, dy, dz, droll, dpitch, dyaw = self._predict_dynamics(v, delta)
        dyn_pose = gtsam.Pose3(gtsam.Rot3.Ypr(dyaw, dpitch, droll), gtsam.Point3(dx, dy, dz))
        self.graph.add(gtsam.BetweenFactorPose3(prev_key, curr_key, dyn_pose, self.dyn_noise))

        est = self.isam.calculateEstimate()

        for name in self.imu_names:
            omega_meas, accel_meas = measurements[name]
            pim = self.imu_pim[name]
            pim.integrateMeasurement(accel_meas, omega_meas, self.dt)

            prev_bias_key = gtsam.symbol(self.imu_bias_prefix[name], self.index - 1)
            curr_bias_key = gtsam.symbol(self.imu_bias_prefix[name], self.index)

            self.graph.add(gtsam.ImuFactor(prev_key, prev_vel_key, curr_key, curr_vel_key, prev_bias_key, pim))

            bias_noise = gtsam.noiseModel.Isotropic.Sigma(
                6, self.imu_bias_walk_std[name] * np.sqrt(self.dt)
            )
            self.graph.add(gtsam.BetweenFactorConstantBias(
                prev_bias_key, curr_bias_key, gtsam.imuBias.ConstantBias(), bias_noise
            ))

            prev_bias_est = est.atConstantBias(prev_bias_key)
            self.initial.insert(curr_bias_key, prev_bias_est)
            pim.resetIntegrationAndSetBias(prev_bias_est)

        if self.enable_lidar:
            scaled_std = np.array(self.lidar_noise_std) * np.sqrt(self.index)
            odom_noise = gtsam.noiseModel.Diagonal.Sigmas(_sigmas_xyzrpy_to_gtsam(scaled_std))
            lidar_pose = _state_to_pose3(measurements["lidar_odom"])
            self.graph.add(gtsam.PriorFactorPose3(curr_key, lidar_pose, odom_noise))

        prev_pose = est.atPose3(prev_key)
        prev_vel = est.atVector(prev_vel_key)
        self.initial.insert(curr_key, prev_pose.compose(dyn_pose))

        # Velocity initial guess: naively reusing prev_vel unchanged is
        # only right if the direction of motion is constant. Since this
        # vehicle can be turning (nonzero yaw rate), rotate prev_vel by
        # the incremental yaw change (dyaw) about the world z-axis
        # instead -- still an approximation (pitch/roll's effect on
        # velocity direction is ignored), but tracks curved motion much
        # better than carrying the stale vector forward unrotated.
        c, s = np.cos(dyaw), np.sin(dyaw)
        vel_guess = np.array([
            c * prev_vel[0] - s * prev_vel[1],
            s * prev_vel[0] + c * prev_vel[1],
            prev_vel[2],
        ])
        self.initial.insert(curr_vel_key, vel_guess)

        self.isam.update(self.graph, self.initial)
        self.graph = gtsam.NonlinearFactorGraph()
        self.initial = gtsam.Values()

        est = self.isam.calculateEstimate()
        self.state = _pose3_to_state(est.atPose3(curr_key))
        self.velocity = est.atVector(curr_vel_key)
        return self.state.copy()

    def _predict_dynamics(self, v, delta):
        """Identical to GraphReckonerLM._predict_dynamics -- see there."""
        dx = v * self.dt
        dy = 0.0
        dz = 0.0
        droll = 0.0
        dpitch = 0.0
        dyaw = (v / self.wheelbase) * np.tan(delta) * self.dt
        return dx, dy, dz, droll, dpitch, dyaw