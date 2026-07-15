import numpy as np
import gtsam
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class BicycleVehicle:
    """Ground truth vehicle motion."""

    def __init__(self, wheelbase, dt, state=(0.0, 0.0, 0.0)):
        self.wheelbase = wheelbase
        self.dt = dt
        self.state = np.array(state, dtype=float)

    def step(self, v, delta):
        prev_state = self.state.copy()
        x, y, theta = self.state
        x += v * np.cos(theta) * self.dt
        y += v * np.sin(theta) * self.dt
        theta += (v / self.wheelbase) * np.tan(delta) * self.dt
        self.state = np.array([x, y, theta])
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


class OdometrySensor:
    """Simulates KISS-ICP style relative pose measurements (noisy scan match)."""

    def __init__(self, noise_std=(0.02, 0.02, 0.01)):
        self.noise_std = noise_std

    def measure(self, prev_state, curr_state):
        dx, dy, dtheta = relative_pose_local_frame(prev_state, curr_state)
        dx += np.random.normal(0, self.noise_std[0])
        dy += np.random.normal(0, self.noise_std[1])
        dtheta += np.random.normal(0, self.noise_std[2])
        return dx, dy, dtheta


class DynamicsModel:
    """Predicts relative motion from (possibly noisy) control input, independent of odometry."""

    def __init__(self, wheelbase, dt, v_std=0.05, delta_std=0.02):
        self.wheelbase = wheelbase
        self.dt = dt
        self.v_std = v_std
        self.delta_std = delta_std

    def predict(self, prev_state, v, delta):
        v_noisy = v + np.random.normal(0, self.v_std)
        delta_noisy = delta + np.random.normal(0, self.delta_std)

        x, y, theta = prev_state
        x += v_noisy * np.cos(theta) * self.dt
        y += v_noisy * np.sin(theta) * self.dt
        theta += (v_noisy / self.wheelbase) * np.tan(delta_noisy) * self.dt
        predicted_state = np.array([x, y, theta])

        return relative_pose_local_frame(prev_state, predicted_state)


class DeadReckoner:
    """Integrates a stream of relative pose measurements into a global trajectory estimate."""

    def __init__(self, state=(0.0, 0.0, 0.0)):
        self.state = np.array(state, dtype=float)

    def integrate(self, dx_local, dy_local, dtheta):
        c, s = np.cos(self.state[2]), np.sin(self.state[2])
        self.state[0] += c * dx_local - s * dy_local
        self.state[1] += s * dx_local + c * dy_local
        self.state[2] += dtheta
        return self.state.copy()


class SimVisualizer:
    def __init__(self, xlim=(-5, 5), ylim=(-1, 9)):
        self.fig, self.ax = plt.subplots()
        self.line_true, = self.ax.plot([], [], 'b-', label='Ground truth')
        self.line_odom, = self.ax.plot([], [], 'r--', label='Odometry only (drifting)')
        self.line_dyn, = self.ax.plot([], [], 'g--', label='Dynamics only (drifting)')
        self.line_graph, = self.ax.plot([], [], 'm-', label='GTSAM fused estimate')
        self.point_true, = self.ax.plot([], [], 'bo')

        self.ax.set_xlim(*xlim)
        self.ax.set_ylim(*ylim)
        self.ax.set_aspect('equal')
        self.ax.set_title("Ground Truth vs. Odometry vs. Dynamics Prediction")
        self.ax.legend(loc='upper right')

        self.xs_true, self.ys_true = [], []
        self.xs_odom, self.ys_odom = [], []
        self.xs_dyn, self.ys_dyn = [], []
        self.xs_graph, self.ys_graph = [], []

    def update(self, true_state, odom_state, dyn_state, graph_state):
        self.xs_true.append(true_state[0]); self.ys_true.append(true_state[1])
        self.xs_odom.append(odom_state[0]); self.ys_odom.append(odom_state[1])
        self.xs_dyn.append(dyn_state[0]); self.ys_dyn.append(dyn_state[1])
        self.xs_graph.append(graph_state[0]); self.ys_graph.append(graph_state[1])

        self.line_true.set_data(self.xs_true, self.ys_true)
        self.line_odom.set_data(self.xs_odom, self.ys_odom)
        self.line_dyn.set_data(self.xs_dyn, self.ys_dyn)
        self.line_graph.set_data(self.xs_graph, self.ys_graph)
        self.point_true.set_data([true_state[0]], [true_state[1]])

        return self.line_true, self.line_odom, self.line_dyn, self.line_graph, self.point_true


class GraphReckoner:
    """Fuses odometry + dynamics relative-pose measurements via a GTSAM factor graph (ISAM2)."""

    def __init__(self):
        self.prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-6, 1e-6, 1e-6]))
        # NOTE: dyn_noise here is a rough (x, y, theta) approximation, not a
        # rigorous propagation of (v_std, delta_std) through the bicycle model.
        self.odom_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.02, 0.02, 0.01]))
        self.dyn_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.005, 0.005, 0.02]))

        self.graph = gtsam.NonlinearFactorGraph()
        self.initial = gtsam.Values()
        self.isam = gtsam.ISAM2()

        self.index = 0
        self.current_estimate = np.array([0.0, 0.0, 0.0])

        # keyframe 0: anchor the graph at the origin
        key0 = gtsam.symbol('x', 0)
        self.graph.add(gtsam.PriorFactorPose2(key0, gtsam.Pose2(0.0, 0.0, 0.0), self.prior_noise))
        self.initial.insert(key0, gtsam.Pose2(0.0, 0.0, 0.0))
        self.isam.update(self.graph, self.initial)
        self.graph = gtsam.NonlinearFactorGraph()
        self.initial = gtsam.Values()

    def add_keyframe(self, odom_meas, dyn_meas):
        prev_key = gtsam.symbol('x', self.index)
        self.index += 1
        curr_key = gtsam.symbol('x', self.index)

        odom_pose = gtsam.Pose2(*odom_meas)
        dyn_pose = gtsam.Pose2(*dyn_meas)

        self.graph.add(gtsam.BetweenFactorPose2(prev_key, curr_key, odom_pose, self.odom_noise))
        self.graph.add(gtsam.BetweenFactorPose2(prev_key, curr_key, dyn_pose, self.dyn_noise))

        # initial guess: propagate last solved pose forward using the odometry measurement
        prev_result = self.isam.calculateEstimate().atPose2(prev_key)
        guess = prev_result.compose(odom_pose)
        self.initial.insert(curr_key, guess)

        self.isam.update(self.graph, self.initial)
        self.graph = gtsam.NonlinearFactorGraph()
        self.initial = gtsam.Values()

        result = self.isam.calculateEstimate().atPose2(curr_key)
        self.current_estimate = np.array([result.x(), result.y(), result.theta()])
        return self.current_estimate.copy()


class Simulation:
    def __init__(self, wheelbase=1.0, dt=0.1, v=1.0, delta=0.3):
        self.v = v
        self.delta = delta

        self.vehicle = BicycleVehicle(wheelbase, dt)
        self.odometry = OdometrySensor()
        self.dynamics = DynamicsModel(wheelbase, dt)

        self.odom_reckoner = DeadReckoner()
        self.dyn_reckoner = DeadReckoner()
        self.graph_reckoner = GraphReckoner()

        self.viz = SimVisualizer()

    def step(self, frame):
        prev_state, curr_state = self.vehicle.step(self.v, self.delta)

        odom_meas = self.odometry.measure(prev_state, curr_state)
        odom_state = self.odom_reckoner.integrate(*odom_meas)

        dyn_meas = self.dynamics.predict(prev_state, self.v, self.delta)
        dyn_state = self.dyn_reckoner.integrate(*dyn_meas)

        graph_state = self.graph_reckoner.add_keyframe(odom_meas, dyn_meas)

        return self.viz.update(curr_state, odom_state, dyn_state, graph_state)

    def run(self):
        self.ani = animation.FuncAnimation(
            self.viz.fig, self.step, interval=50, blit=True, cache_frame_data=False
        )
        plt.show()


if __name__ == "__main__":
    sim = Simulation()
    sim.run()