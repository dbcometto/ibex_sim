"""Entry point: ground truth vs. noisy bicycle vehicle, visualized live."""
import numpy as np

from vehicle import ImperfectBicycleVehicle
from sensors import LidarOdometrySensor, ImuSensor
from estimators import DeadReckoner, GraphReckonerLM
from visualization import SimVisualizer


WHEELBASE = 1.0
DT = 0.1
V = 1.0
DELTA = 0.3

np.random.seed(42)

class Simulation:
    def __init__(self):
        # Vehicle
        self.true_vehicle = ImperfectBicycleVehicle(WHEELBASE, DT, noise_std=(0.005, 0.005, 0.001))

        # Sensors
        self.lidar_odom = LidarOdometrySensor(noise_std=(0.009, 0.009, 0.003))
        self.imu_ouster = ImuSensor(gyro_noise_std=0.001, accel_noise_std=0.01, gyro_bias_walk_std=0.00005, accel_bias_walk_std=0.0001,)
        self.imu_insta = ImuSensor(gyro_noise_std=0.02, accel_noise_std=0.2, gyro_bias_walk_std=0.002, accel_bias_walk_std=0.005,)

        # Estimators
        self.dead_reckoner = DeadReckoner(dt=DT)
        self.graph_reckoner = GraphReckonerLM(wheelbase=WHEELBASE, dt=DT)

        # Visualizer
        self.viz = SimVisualizer(series=["truth", "lidar_odom", "os_dead_reckoner", "GTSAM"], marker_series="truth")

    def step(self, frame):
        _, truth_state = self.true_vehicle.step(V, DELTA)

        # Update sensors
        lidar_odom_meas = self.lidar_odom.measure(self.true_vehicle, V, DELTA)
        imu_ouster_meas = self.imu_ouster.measure(self.true_vehicle, V, DELTA)
        imu_insta_meas = self.imu_insta.measure(self.true_vehicle, V, DELTA)

        # Run estimators
        lidar_odom_state = lidar_odom_meas

        dr_dict = {"imu": imu_ouster_meas}
        dr_state = self.dead_reckoner.update(dr_dict, V, DELTA)

        gr_dict = {"lidar_odom": lidar_odom_meas}
        gr_state = self.graph_reckoner.update(gr_dict, V, DELTA)

        # imu_bad is instantiated but has no consumer yet
        

        return self.viz.update({
            "truth": truth_state,
            "lidar_odom": lidar_odom_state,
            "os_dead_reckoner": dr_state,
            "GTSAM": gr_state,
        })

    def run(self):
        self.viz.animate(self.step)


if __name__ == "__main__":
    Simulation().run()