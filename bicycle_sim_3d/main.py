"""Entry point: 3D bicycle vehicle over terrain, visualized top-down."""
import numpy as np

from vehicle import BicycleVehicle3D, ImperfectBicycleVehicle3D
from world import HeightmapTerrain, FlatTerrain
from sensors import LidarOdometrySensor
from sensors import ImuSensor
from estimators import GraphReckonerLM, GraphReckonerISAM, GraphReckonerPIM
from visualization import SimVisualizer

from enum import IntEnum


WHEELBASE = 1.0
DT = 0.1
V = 1.0
DELTA = 0.3
terrain_path = "/home/dbcometto/workspace/ibex_sim/bicycle_sim_3d/sample_terrain.png"

np.random.seed(42)

class TERRAIN_TYPES(IntEnum):
    FLAT = 1
    IMAGE = 2

terrain = TERRAIN_TYPES.IMAGE



class Simulation:
    def __init__(self):
        # Terrain
        # (resolution=1.0, so extent == pixel shape == (0, 256, 0, 256))
        if terrain == TERRAIN_TYPES.FLAT:   
            starting_state = (10.0, 10.0, 0.0, 0.0, 0.0, 0.0)
            self.terrain = FlatTerrain(elevation=0.0, resolution=0.1, shape=(256,256))

        elif terrain == TERRAIN_TYPES.IMAGE:
            starting_state = (10.0, 10.0, 2.14, 0.0, -0.45, 0.0)
            self.terrain = HeightmapTerrain.from_image(terrain_path, resolution=0.1, z_scale=5.0)

        # Vehicle
        self.true_vehicle = ImperfectBicycleVehicle3D(WHEELBASE, DT, self.terrain, state=starting_state,
                                                      noise_std=(0.003, 0.003, 0.001))

        # Sensors
        self.lidar_odom = LidarOdometrySensor(noise_std=(0.009, 0.009, 0.009, 0.003, 0.003, 0.003), init_state=starting_state)
        self.imu_ouster = ImuSensor(gyro_noise_std=0.001, accel_noise_std=0.01, gyro_bias_walk_std=0.00005, accel_bias_walk_std=0.0001)
        self.imu_insta = ImuSensor(gyro_noise_std=0.02, accel_noise_std=0.2, gyro_bias_walk_std=0.002, accel_bias_walk_std=0.005)

        # Estimators
        # self.graph_reckoner = GraphReckonerLM(wheelbase=WHEELBASE, dt=DT, init_state=starting_state)
        self.graph_reckoner = GraphReckonerISAM(wheelbase=WHEELBASE, dt=DT, init_state=starting_state,
                                                dyn_noise_std=(0.003, 0.003, 0.02, 0.02, 0.02, 0.001),
                                                lidar_noise_std=(0.009, 0.009, 0.009, 0.003, 0.003, 0.003),)
        
        V_x0 = V * np.cos(starting_state[5])  # starting_state[5] = yaw
        V_y0 = V * np.sin(starting_state[5])

        self.graph_reckonerIMU = GraphReckonerPIM(
            wheelbase=WHEELBASE, dt=DT, init_state=starting_state,
            init_velocity=(V_x0, V_y0, 0.0),
            dyn_noise_std=(0.003, 0.003, 0.02, 0.02, 0.02, 0.001),
            lidar_noise_std=(0.009, 0.009, 0.009, 0.003, 0.003, 0.003),
            imu_configs={
                "imu_ouster": {
                    "gyro_noise_std": 0.001,
                    "accel_noise_std": 0.01,
                    "bias_walk_std": 0.0001,
                },
            },
        )

        # Visualizer
        self.viz = SimVisualizer(
            series=["truth", "lidar_odom", "GR_noIMU", "GR_IMU"],
            terrain=self.terrain,
            xlim=(0, 25.6), ylim=(0, 25.6),
            marker_series="truth",
            elevation_range=(0.0, 5.0),
        )

    def step(self, frame):
        # Vehicle
        _, truth_state = self.true_vehicle.step(V, DELTA)

        # Sensors
        lidar_odom_meas = self.lidar_odom.measure(self.true_vehicle, V, DELTA)
        imu_ouster_meas = self.imu_ouster.measure(self.true_vehicle, V, DELTA)
        # imu_insta measured but not yet consumed by any estimator
        # self.imu_insta.measure(self.true_vehicle, V, DELTA)

        # Estimators
        lidar_odom_state = lidar_odom_meas
        graph_reckoner_state = self.graph_reckoner.update(
            {"lidar_odom": lidar_odom_meas}, V, DELTA
        )
        graph_reckoner_IMU_state = self.graph_reckonerIMU.update(
            {"lidar_odom": lidar_odom_meas, "imu_ouster": imu_ouster_meas}, V, DELTA
        )

        return self.viz.update({
            "truth": truth_state,
            "lidar_odom": lidar_odom_state,
            "GR_noIMU": graph_reckoner_state,
            "GR_IMU": graph_reckoner_IMU_state,
        })

    def run(self):
        self.viz.animate(self.step)


if __name__ == "__main__":
    Simulation().run()