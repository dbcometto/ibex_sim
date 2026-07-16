"""Defines estimators that turn sensor measurements into a state estimate."""

from abc import ABC, abstractmethod

import numpy as np
import gtsam
import gtsam_unstable

from factors import make_nhc_factor, make_rate_tie_factor, make_rate_cv_factor, make_lidar_drift_factor

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



#==========# Original Estimators #==========#

# Deprecated
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

# Deprecated
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















#==========# Better Estimator #==========#

def _bias_noise_sigmas(gyro_bias_walk_std, accel_bias_walk_std, dt):
    """Build the 6 sigmas for a BetweenFactorConstantBias's random-walk
    noise, from SEPARATE gyro (rad/s) and accel (m/s^2) bias-walk
    stds -- not one shared isotropic value across both.
 
    ORDER UNVERIFIED: assumes gtsam.imuBias.ConstantBias's tangent
    vector is (accel bias x,y,z, gyro bias x,y,z) -- accel first --
    matching the ConstantBias(accBias, gyroBias) constructor argument
    order. I could not run gtsam in this sandbox to confirm this is
    really how noiseModel.Diagonal.Sigmas indexes against it. Verify
    with the round-trip script before trusting this, same as we did for
    Rot3.Ypr/Pose3 earlier.
 
    Each std is scaled by sqrt(dt), consistent with modeling bias as a
    random walk (variance grows linearly with time, so std grows as
    sqrt(time)) -- same scaling the old isotropic version already used.
    """
    accel_sigma = accel_bias_walk_std * np.sqrt(dt)
    gyro_sigma = gyro_bias_walk_std * np.sqrt(dt)
    return np.array([accel_sigma] * 3 + [gyro_sigma] * 3)

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
 
    NOTE on ISAM2's relinearizeSkip: gtsam's ISAM2 defaults to
    relinearizeSkip=10 -- it only fully relinearizes around the current
    estimate every 10th update() call, accumulating cheaper linear
    corrections in between. ImuFactor (rotation/bias/velocity coupling)
    is sensitive enough to a stale linearization point that this
    produces a visible periodic drift-then-snap-back pattern once every
    10 keyframes (once per second at dt=0.1s) on a curving trajectory.
    Set to 1 here (relinearize every step) to remove that artifact --
    more computation per step, but no periodic wiggle.
 
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
                imu_configs=None, init_velocity=(0.0, 0.0, 0.0), gravity=GRAVITY,
                init_velocity_noise_std=1.0,
                nhc_noise_std=(0.001, 0.001),
                enable_lidar=True, enable_IMUs=True, enable_NHC = True, enable_rate=True, enable_gps=True, 
                rate_prior_std=(1.0, 1.0),
                rate_process_noise_std=(0.5, 0.5),
                rate_tie_noise_std=(0.01, 0.01),
                gps_noise_std=(1.5, 1.5, 3.0),
                lag_seconds=5.0):
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
        lag_seconds: how much history (in seconds) the smoother keeps
            before marginalizing older keyframes out. Smaller = bounded
            graph size / faster, at the cost of less smoothing benefit
            from older observations. No principled default -- 5.0 is a
            guess, not derived from anything about this vehicle or sensors.
        nhc_noise_std: (lateral, vertical) tolerance in body-frame m/s for
            the non-holonomic constraint -- how much real sideslip/bounce
            to permit. 0.05 is a guess, not measured from anything.
        rate_prior_std: prior on r0 = (roll_dot, pitch_dot) at t=0 -- loose,
            we don't know the true initial rate.
        rate_process_noise_std: how much r itself is allowed to change per
            step (rad/s per sqrt(dt)) -- smaller = smoother/more constant-rate
            assumption, larger = more agile tracking of fast-changing rates.
            A guess, not measured.
        rate_tie_noise_std: how tightly observed pose rotation must match
            r*dt -- this is closer to a kinematic identity than a noisy
            sensor, so it should be fairly tight. Also a guess.
        """
        # Init state (x y z roll pitch yaw)
        super().__init__(init_state)

        # Config
        self.wheelbase = wheelbase
        self.dt = dt
        self.gravity = gravity
        
        self.enable_lidar = enable_lidar
        self.enable_IMUs = enable_IMUs
        self.enable_NHC = enable_NHC
        self.enable_rate = enable_rate
        self.enable_gps = enable_gps
        self.uses_velocity = self.enable_IMUs or self.enable_NHC or self.enable_rate

        # Noise models (non-imu)
        prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([prior_noise_std] * 6))
        self.dyn_noise = gtsam.noiseModel.Diagonal.Sigmas(_sigmas_xyzrpy_to_gtsam(dyn_noise_std))
        self.lidar_noise_std = lidar_noise_std
        self.nhc_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array(nhc_noise_std))
        self.gps_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array(gps_noise_std))
        

        # IMU setup
        if self.enable_IMUs:
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
                self.imu_bias_walk_std[name] = (
                    cfg.get('gyro_bias_walk_std', 0.0005),
                    cfg.get('accel_bias_walk_std', 0.001),
                )

        # Graph setup
        self.graph = gtsam.NonlinearFactorGraph()
        self.initial = gtsam.Values()
        self.timestamps = gtsam_unstable.FixedLagSmootherKeyTimestampMap()
        isam_params = gtsam.ISAM2Params()
        isam_params.relinearizeSkip = 1  # gtsam default is 10 -- see class docstring
        self.smoother = gtsam_unstable.IncrementalFixedLagSmoother(lag_seconds, isam_params)
        self.index = 0

        # Graph initialization
        key0 = gtsam.symbol('x', 0)
        pose0 = _state_to_pose3(init_state)

        self.graph.add(gtsam.PriorFactorPose3(key0, pose0, prior_noise))
        self.initial.insert(key0, pose0)
        self.timestamps.insert((key0, 0.0))

        if self.uses_velocity:
            vel0_key = gtsam.symbol('v', 0)
            vel0 = np.array(init_velocity, dtype=float)
            vel_prior_noise = gtsam.noiseModel.Isotropic.Sigma(3, init_velocity_noise_std)
            self.graph.add(gtsam.PriorFactorVector(vel0_key, vel0, vel_prior_noise))
            self.initial.insert(vel0_key, vel0)
            self.timestamps.insert((vel0_key, 0.0))

        if self.enable_IMUs:
            bias_prior_noise = gtsam.noiseModel.Isotropic.Sigma(6, 0.1)
            for name in self.imu_names:
                bkey = gtsam.symbol(self.imu_bias_prefix[name], 0)
                self.graph.add(gtsam.PriorFactorConstantBias(bkey, gtsam.imuBias.ConstantBias(), bias_prior_noise))
                self.initial.insert(bkey, gtsam.imuBias.ConstantBias())
                self.timestamps.insert((bkey, 0.0))

        if self.enable_rate:
            self.rate_prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array(rate_prior_std))
            self.rate_cv_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array(rate_process_noise_std) * np.sqrt(dt))
            self.rate_tie_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array(rate_tie_noise_std))

            r0_key = gtsam.symbol('r', 0)
            r0 = np.zeros(2)
            self.graph.add(gtsam.PriorFactorVector(r0_key, r0, self.rate_prior_noise))
            self.initial.insert(r0_key, r0)
            self.timestamps.insert((r0_key, 0.0))

        # Initial push into smoother and reset graph
        self.smoother.update(self.graph, self.initial, self.timestamps)
        self.graph = gtsam.NonlinearFactorGraph()
        self.initial = gtsam.Values()
        self.timestamps = gtsam_unstable.FixedLagSmootherKeyTimestampMap()

        # Initial solve
        est = self.smoother.calculateEstimate()
        self.state = _pose3_to_state(est.atPose3(key0))
        if self.uses_velocity:
            self.velocity = est.atVector(vel0_key)



    def update(self, measurements, v, delta):
        # Setup
        prev_key = gtsam.symbol('x', self.index)
        if self.uses_velocity:
            prev_vel_key = gtsam.symbol('v', self.index)
        self.index += 1
        curr_key = gtsam.symbol('x', self.index)
        if self.uses_velocity:
            curr_vel_key = gtsam.symbol('v', self.index)
        t = self.index * self.dt

        # Connect dynamics factor
        dx, dy, dz, droll, dpitch, dyaw = self._predict_dynamics(v, delta)
        dyn_pose = gtsam.Pose3(gtsam.Rot3.Ypr(dyaw, dpitch, droll), gtsam.Point3(dx, dy, dz))
        self.graph.add(gtsam.BetweenFactorPose3(prev_key, curr_key, dyn_pose, self.dyn_noise))

        # Solve with just dynamics
        est = self.smoother.calculateEstimate()

        # Connect lidar odom factor
        if self.enable_lidar:
            scaled_std = np.array(self.lidar_noise_std) * np.sqrt(self.index)
            odom_noise = gtsam.noiseModel.Diagonal.Sigmas(_sigmas_xyzrpy_to_gtsam(scaled_std))
            lidar_pose = _state_to_pose3(measurements["lidar_odom"])
            self.graph.add(gtsam.PriorFactorPose3(curr_key, lidar_pose, odom_noise))

        # Connect IMU factors
        if self.enable_IMUs:
            for name in self.imu_names:
                omega_meas, accel_meas = measurements[name]
                pim = self.imu_pim[name]
                pim.integrateMeasurement(accel_meas, omega_meas, self.dt)

                prev_bias_key = gtsam.symbol(self.imu_bias_prefix[name], self.index - 1)
                curr_bias_key = gtsam.symbol(self.imu_bias_prefix[name], self.index)

                self.graph.add(gtsam.ImuFactor(prev_key, prev_vel_key, curr_key, curr_vel_key, prev_bias_key, pim))

                gyro_walk_std, accel_walk_std = self.imu_bias_walk_std[name]
                bias_noise = gtsam.noiseModel.Diagonal.Sigmas(
                    _bias_noise_sigmas(gyro_walk_std, accel_walk_std, self.dt)
                )
                self.graph.add(gtsam.BetweenFactorConstantBias(
                    prev_bias_key, curr_bias_key, gtsam.imuBias.ConstantBias(), bias_noise
                ))

                prev_bias_est = est.atConstantBias(prev_bias_key)
                self.initial.insert(curr_bias_key, prev_bias_est)
                self.timestamps.insert((curr_bias_key, t))
                pim.resetIntegrationAndSetBias(prev_bias_est)

        # Add NHC factor
        if self.enable_NHC:
            self.graph.add(make_nhc_factor(curr_key, curr_vel_key, self.nhc_noise))

        # Add angular rate factors
        if self.enable_rate:
            r_prev_key = gtsam.symbol('r', self.index - 1)
            r_curr_key = gtsam.symbol('r', self.index)

            self.graph.add(make_rate_tie_factor(prev_key, curr_key, r_prev_key, self.dt, self.rate_tie_noise))
            self.graph.add(make_rate_cv_factor(r_prev_key, r_curr_key, self.rate_cv_noise))

            r_prev_est = est.atVector(r_prev_key)
            self.initial.insert(r_curr_key, r_prev_est)
            self.timestamps.insert((r_curr_key, t))

        # Add gps factor (position only)
        if self.enable_gps:
            gps_pos = measurements["gps"]
            if gps_pos is not None:
                self.graph.add(gtsam.GPSFactor(curr_key, gtsam.Point3(*gps_pos), self.gps_noise))

        
        # Add initial pose guess
        prev_pose = est.atPose3(prev_key)

        curr_pose_guess = prev_pose.compose(dyn_pose)
        self.initial.insert(curr_key, curr_pose_guess)
        self.timestamps.insert((curr_key, t))

        # Velocity initial guess (assume constant in body frame)
        if self.uses_velocity:
            prev_vel = est.atVector(prev_vel_key)
            R_prev = prev_pose.rotation().matrix()
            R_curr = curr_pose_guess.rotation().matrix()
            v_body = R_prev.T @ prev_vel
            vel_guess = R_curr @ v_body
            self.initial.insert(curr_vel_key, vel_guess)
            self.timestamps.insert((curr_vel_key, t))

        # Push into smoother
        self.smoother.update(self.graph, self.initial, self.timestamps)
        self.graph = gtsam.NonlinearFactorGraph()
        self.initial = gtsam.Values()
        self.timestamps = gtsam_unstable.FixedLagSmootherKeyTimestampMap()

        # Solve
        est = self.smoother.calculateEstimate()
        self.state = _pose3_to_state(est.atPose3(curr_key))
        if self.uses_velocity:
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
    

class GraphReckoner(Estimator):
    """The full graph reckoner, reworked to model lidar drift.
    Like GraphReckonerISAM, but IMU factors use gtsam's real
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
 
    NOTE on ISAM2's relinearizeSkip: gtsam's ISAM2 defaults to
    relinearizeSkip=10 -- it only fully relinearizes around the current
    estimate every 10th update() call, accumulating cheaper linear
    corrections in between. ImuFactor (rotation/bias/velocity coupling)
    is sensitive enough to a stale linearization point that this
    produces a visible periodic drift-then-snap-back pattern once every
    10 keyframes (once per second at dt=0.1s) on a curving trajectory.
    Set to 1 here (relinearize every step) to remove that artifact --
    more computation per step, but no periodic wiggle.
 
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
                lidar_drift_prior_std=(0.01, 0.01, 0.01, 0.005, 0.005, 0.005),
                lidar_drift_process_noise_std=(0.01, 0.01, 0.01, 0.005, 0.005, 0.005),
                imu_configs=None, init_velocity=(0.0, 0.0, 0.0), gravity=GRAVITY,
                init_velocity_noise_std=1.0,
                nhc_noise_std=(0.001, 0.001),
                enable_lidar=True, enable_IMUs=True, enable_NHC = True, enable_rate=True, enable_gps=True, 
                rate_prior_std=(1.0, 1.0),
                rate_process_noise_std=(0.5, 0.5),
                rate_tie_noise_std=(0.01, 0.01),
                gps_noise_std=(1.5, 1.5, 3.0),
                lag_seconds=5.0):
        """
        lidar_drift_prior_std: prior on the drift variable at t=0 -- tight,
            since a freshly-initialized lidar odometry hasn't drifted yet.
        lidar_drift_process_noise_std: how fast the LATENT drift itself is
            allowed to wander per sqrt(dt) -- this is legitimately sqrt(dt)-
            scaled, same as IMU bias, because it's modeling a real random walk
            of the drift state itself (not compounding an already-compounded
            reading, which was the old bug).
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
        lag_seconds: how much history (in seconds) the smoother keeps
            before marginalizing older keyframes out. Smaller = bounded
            graph size / faster, at the cost of less smoothing benefit
            from older observations. No principled default -- 5.0 is a
            guess, not derived from anything about this vehicle or sensors.
        nhc_noise_std: (lateral, vertical) tolerance in body-frame m/s for
            the non-holonomic constraint -- how much real sideslip/bounce
            to permit. 0.05 is a guess, not measured from anything.
        rate_prior_std: prior on r0 = (roll_dot, pitch_dot) at t=0 -- loose,
            we don't know the true initial rate.
        rate_process_noise_std: how much r itself is allowed to change per
            step (rad/s per sqrt(dt)) -- smaller = smoother/more constant-rate
            assumption, larger = more agile tracking of fast-changing rates.
            A guess, not measured.
        rate_tie_noise_std: how tightly observed pose rotation must match
            r*dt -- this is closer to a kinematic identity than a noisy
            sensor, so it should be fairly tight. Also a guess.
        """
        # Init state (x y z roll pitch yaw)
        super().__init__(init_state)

        # Config
        self.wheelbase = wheelbase
        self.dt = dt
        self.gravity = gravity
        
        self.enable_lidar = enable_lidar
        self.enable_IMUs = enable_IMUs
        self.enable_NHC = enable_NHC
        self.enable_rate = enable_rate
        self.enable_gps = enable_gps
        self.uses_velocity = self.enable_IMUs or self.enable_NHC or self.enable_rate

        # Noise models (non-Lidar, non-imu)
        prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([prior_noise_std] * 6))
        self.dyn_noise = gtsam.noiseModel.Diagonal.Sigmas(_sigmas_xyzrpy_to_gtsam(dyn_noise_std))
        self.nhc_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array(nhc_noise_std))
        self.gps_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array(gps_noise_std))

        # Lidar Noise
        self.lidar_noise_std = lidar_noise_std
        self.lidar_measurement_noise = gtsam.noiseModel.Diagonal.Sigmas(_sigmas_xyzrpy_to_gtsam(lidar_noise_std))
        self.lidar_drift_prior_noise = gtsam.noiseModel.Diagonal.Sigmas(_sigmas_xyzrpy_to_gtsam(lidar_drift_prior_std))
        self.lidar_drift_process_noise = gtsam.noiseModel.Diagonal.Sigmas(_sigmas_xyzrpy_to_gtsam(lidar_drift_process_noise_std) * np.sqrt(dt))        

        # IMU setup
        if self.enable_IMUs:
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
                self.imu_bias_walk_std[name] = (
                    cfg.get('gyro_bias_walk_std', 0.0005),
                    cfg.get('accel_bias_walk_std', 0.001),
                )

        # Graph setup
        self.graph = gtsam.NonlinearFactorGraph()
        self.initial = gtsam.Values()
        self.timestamps = gtsam_unstable.FixedLagSmootherKeyTimestampMap()
        isam_params = gtsam.ISAM2Params()
        isam_params.relinearizeSkip = 1  # gtsam default is 10 -- see class docstring
        self.smoother = gtsam_unstable.IncrementalFixedLagSmoother(lag_seconds, isam_params)
        self.index = 0

        # Graph initialization
        key0 = gtsam.symbol('x', 0)
        pose0 = _state_to_pose3(init_state)

        self.graph.add(gtsam.PriorFactorPose3(key0, pose0, prior_noise))
        self.initial.insert(key0, pose0)
        self.timestamps.insert((key0, 0.0))

        if self.enable_lidar:
            d0_key = gtsam.symbol('d', 0)
            d0 = gtsam.Pose3()  # identity -- ASSUMED default constructor gives identity, worth a quick print(gtsam.Pose3()) check
            self.graph.add(gtsam.PriorFactorPose3(d0_key, d0, self.lidar_drift_prior_noise))
            self.initial.insert(d0_key, d0)
            self.timestamps.insert((d0_key, 0.0))

        if self.uses_velocity:
            vel0_key = gtsam.symbol('v', 0)
            vel0 = np.array(init_velocity, dtype=float)
            vel_prior_noise = gtsam.noiseModel.Isotropic.Sigma(3, init_velocity_noise_std)
            self.graph.add(gtsam.PriorFactorVector(vel0_key, vel0, vel_prior_noise))
            self.initial.insert(vel0_key, vel0)
            self.timestamps.insert((vel0_key, 0.0))

        if self.enable_IMUs:
            bias_prior_noise = gtsam.noiseModel.Isotropic.Sigma(6, 0.1)
            for name in self.imu_names:
                bkey = gtsam.symbol(self.imu_bias_prefix[name], 0)
                self.graph.add(gtsam.PriorFactorConstantBias(bkey, gtsam.imuBias.ConstantBias(), bias_prior_noise))
                self.initial.insert(bkey, gtsam.imuBias.ConstantBias())
                self.timestamps.insert((bkey, 0.0))

        if self.enable_rate:
            self.rate_prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array(rate_prior_std))
            self.rate_cv_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array(rate_process_noise_std) * np.sqrt(dt))
            self.rate_tie_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array(rate_tie_noise_std))

            r0_key = gtsam.symbol('r', 0)
            r0 = np.zeros(2)
            self.graph.add(gtsam.PriorFactorVector(r0_key, r0, self.rate_prior_noise))
            self.initial.insert(r0_key, r0)
            self.timestamps.insert((r0_key, 0.0))

        # Initial push into smoother and reset graph
        self.smoother.update(self.graph, self.initial, self.timestamps)
        self.graph = gtsam.NonlinearFactorGraph()
        self.initial = gtsam.Values()
        self.timestamps = gtsam_unstable.FixedLagSmootherKeyTimestampMap()

        # Initial solve
        est = self.smoother.calculateEstimate()
        self.state = _pose3_to_state(est.atPose3(key0))
        if self.uses_velocity:
            self.velocity = est.atVector(vel0_key)



    def update(self, measurements, v, delta):
        # Setup
        prev_key = gtsam.symbol('x', self.index)
        if self.uses_velocity:
            prev_vel_key = gtsam.symbol('v', self.index)
        self.index += 1
        curr_key = gtsam.symbol('x', self.index)
        if self.uses_velocity:
            curr_vel_key = gtsam.symbol('v', self.index)
        t = self.index * self.dt

        # Connect dynamics factor
        dx, dy, dz, droll, dpitch, dyaw = self._predict_dynamics(v, delta)
        dyn_pose = gtsam.Pose3(gtsam.Rot3.Ypr(dyaw, dpitch, droll), gtsam.Point3(dx, dy, dz))
        self.graph.add(gtsam.BetweenFactorPose3(prev_key, curr_key, dyn_pose, self.dyn_noise))

        # Solve with just dynamics
        est = self.smoother.calculateEstimate()

        # Connect lidar odom factor
        if self.enable_lidar:
            lidar_pose = _state_to_pose3(measurements["lidar_odom"])
            drift_prev_key = gtsam.symbol('d', self.index - 1)
            drift_curr_key = gtsam.symbol('d', self.index)

            self.graph.add(make_lidar_drift_factor(curr_key, drift_curr_key, lidar_pose, self.lidar_measurement_noise))
            self.graph.add(gtsam.BetweenFactorPose3(drift_prev_key, drift_curr_key, gtsam.Pose3(), self.lidar_drift_process_noise))

            prev_drift_est = est.atPose3(drift_prev_key)
            self.initial.insert(drift_curr_key, prev_drift_est)  # guess: drift barely changes step to step
            self.timestamps.insert((drift_curr_key, t))

        # Connect IMU factors
        if self.enable_IMUs:
            for name in self.imu_names:
                omega_meas, accel_meas = measurements[name]
                pim = self.imu_pim[name]
                pim.integrateMeasurement(accel_meas, omega_meas, self.dt)

                prev_bias_key = gtsam.symbol(self.imu_bias_prefix[name], self.index - 1)
                curr_bias_key = gtsam.symbol(self.imu_bias_prefix[name], self.index)

                self.graph.add(gtsam.ImuFactor(prev_key, prev_vel_key, curr_key, curr_vel_key, prev_bias_key, pim))

                gyro_walk_std, accel_walk_std = self.imu_bias_walk_std[name]
                bias_noise = gtsam.noiseModel.Diagonal.Sigmas(
                    _bias_noise_sigmas(gyro_walk_std, accel_walk_std, self.dt)
                )
                self.graph.add(gtsam.BetweenFactorConstantBias(
                    prev_bias_key, curr_bias_key, gtsam.imuBias.ConstantBias(), bias_noise
                ))

                prev_bias_est = est.atConstantBias(prev_bias_key)
                self.initial.insert(curr_bias_key, prev_bias_est)
                self.timestamps.insert((curr_bias_key, t))
                pim.resetIntegrationAndSetBias(prev_bias_est)

        # Add NHC factor
        if self.enable_NHC:
            self.graph.add(make_nhc_factor(curr_key, curr_vel_key, self.nhc_noise))

        # Add angular rate factors
        if self.enable_rate:
            r_prev_key = gtsam.symbol('r', self.index - 1)
            r_curr_key = gtsam.symbol('r', self.index)

            self.graph.add(make_rate_tie_factor(prev_key, curr_key, r_prev_key, self.dt, self.rate_tie_noise))
            self.graph.add(make_rate_cv_factor(r_prev_key, r_curr_key, self.rate_cv_noise))

            r_prev_est = est.atVector(r_prev_key)
            self.initial.insert(r_curr_key, r_prev_est)
            self.timestamps.insert((r_curr_key, t))

        # Add gps factor (position only)
        if self.enable_gps:
            gps_pos = measurements["gps"]
            if gps_pos is not None:
                self.graph.add(gtsam.GPSFactor(curr_key, gtsam.Point3(*gps_pos), self.gps_noise))

        
        # Add initial pose guess
        prev_pose = est.atPose3(prev_key)

        curr_pose_guess = prev_pose.compose(dyn_pose)
        self.initial.insert(curr_key, curr_pose_guess)
        self.timestamps.insert((curr_key, t))

        # Velocity initial guess (assume constant in body frame)
        if self.uses_velocity:
            prev_vel = est.atVector(prev_vel_key)
            R_prev = prev_pose.rotation().matrix()
            R_curr = curr_pose_guess.rotation().matrix()
            v_body = R_prev.T @ prev_vel
            vel_guess = R_curr @ v_body
            self.initial.insert(curr_vel_key, vel_guess)
            self.timestamps.insert((curr_vel_key, t))

        # Push into smoother
        self.smoother.update(self.graph, self.initial, self.timestamps)
        self.graph = gtsam.NonlinearFactorGraph()
        self.initial = gtsam.Values()
        self.timestamps = gtsam_unstable.FixedLagSmootherKeyTimestampMap()

        # Solve
        est = self.smoother.calculateEstimate()
        self.state = _pose3_to_state(est.atPose3(curr_key))

        if self.enable_lidar and self.index % 20 == 0:
            drift_pose = est.atPose3(drift_curr_key)
            drift_pos = np.linalg.norm(drift_pose.translation())
            # Logmap gives the rotation vector (axis * angle) -- its norm is
            # the true single rotation angle, unlike e.g. norm(rpy) which
            # isn't rotation-invariant for a combined multi-axis rotation.
            drift_rot_deg = np.degrees(np.linalg.norm(gtsam.Rot3.Logmap(drift_pose.rotation())))
            print(f"[step {self.index}] lidar drift: pos={drift_pos:.4f} m, rot={drift_rot_deg:.4f} deg")

        if self.uses_velocity:
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

















