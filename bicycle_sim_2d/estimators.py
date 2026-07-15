"""Defines estimators that turn sensor measurements into a state estimate."""

from abc import ABC, abstractmethod

import numpy as np
import gtsam



class Estimator(ABC):
    """Base class for anything that maintains a running state estimate."""

    def __init__(self, init_state=(0.0, 0.0, 0.0)):
        self.state = np.array(init_state, dtype=float)

    @abstractmethod
    def update(self, *args, **kwargs):
        """Consume a new measurement and return the updated state estimate."""
        raise NotImplementedError


class DeadReckoner(Estimator):
    """Open-loop strapdown integration of gyro + accelerometer — drifts unbounded, quadratically in position, by design."""

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
    




class GraphReckonerLM(Estimator):
    """Factor-graph estimator, batch-optimized from scratch every keyframe
    for transparency
    """

    def __init__(self, wheelbase, dt, init_state=(0.0, 0.0, 0.0),
                prior_noise_std=0.001, 
                dyn_noise_std=(0.005, 0.005, 0.001), lidar_noise_std=(0.05, 0.05, 0.02),
                enable_lidar = True, 
                #enable_IMU_1 = True, enable_IMU_2 = True
                ):
        super().__init__(init_state)

        # World state
        self.wheelbase = wheelbase
        self.dt = dt  
        self.enable_lidar = enable_lidar
        # self.enable_IMU_1 = enable_IMU_1    
        # self.enable_IMU_2 = enable_IMU_2

        # noise models
        prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([prior_noise_std] * 3))
        self.dyn_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array(dyn_noise_std))
        # self.odom_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array(lidar_noise_std))
        self.lidar_noise_std = lidar_noise_std

        # GTSAM Set up
        self.graph = gtsam.NonlinearFactorGraph()
        self.initial = gtsam.Values()
        self.index = 0

        # GTSAM initialization at origin
        key0 = gtsam.symbol('x', 0)
        self.graph.add(gtsam.PriorFactorPose2(key0, gtsam.Pose2(*init_state), prior_noise))
        self.initial.insert(key0, gtsam.Pose2(*init_state))

        # Initial solve
        self.result = gtsam.LevenbergMarquardtOptimizer(self.graph, self.initial).optimize()
        self.state = self._pose_to_state(self.result.atPose2(key0))


    def update(self, measurements, v, delta):
        # Indexing
        prev_key = gtsam.symbol('x', self.index)
        self.index += 1
        curr_key = gtsam.symbol('x', self.index)

        # Dynamics factor as between factor
        dx, dy, dtheta = self._predict_dynamics(v, delta)
        dyn_pose = gtsam.Pose2(dx, dy, dtheta)
        self.graph.add(gtsam.BetweenFactorPose2(prev_key, curr_key, dyn_pose, self.dyn_noise))

        # Lidar odom as unary factor
        if self.enable_lidar:
            scaled_std = np.array(self.lidar_noise_std) * np.sqrt(self.index)
            odom_noise = gtsam.noiseModel.Diagonal.Sigmas(scaled_std)
            lidar_pose = gtsam.Pose2(*measurements["lidar_odom"])
            self.graph.add(gtsam.PriorFactorPose2(curr_key, lidar_pose, odom_noise))

        # Make new guess based on dynamics
        prev_pose = self.result.atPose2(prev_key)
        self.initial.insert(curr_key, prev_pose.compose(dyn_pose))

        # Solve
        self.result = gtsam.LevenbergMarquardtOptimizer(self.graph, self.initial).optimize()
        self.state = self._pose_to_state(self.result.atPose2(curr_key))

        return self.state.copy()
    

    # Helpers
    def _predict_dynamics(self, v, delta):
        """Bicycle-model relative motion (dx, dy, dtheta) in the local frame."""
        dx = v * self.dt
        dy = 0.0
        dtheta = (v / self.wheelbase) * np.tan(delta) * self.dt
        return dx, dy, dtheta

    @staticmethod
    def _pose_to_state(pose):
        return np.array([pose.x(), pose.y(), pose.theta()])