"""Entry point: 3D bicycle vehicle over terrain, visualized top-down."""
import numpy as np

from vehicle import BicycleVehicle3D, ImperfectBicycleVehicle3D, ImperfectSuspensionBicycleVehicle3D
from world import HeightmapTerrain, FlatTerrain
from sensors import LidarOdometrySensor
from sensors import ImuSensor, GpsSensor
from estimators import GraphReckonerLM, GraphReckonerISAM, GraphReckonerPIM, GraphReckoner
from visualization import SimVisualizer

from enum import IntEnum


#==========# Setup #==========#
terrain_path = "/home/dbcometto/workspace/ibex_sim/bicycle_sim_3d/sample_terrain.png"

np.random.seed(42)

class TERRAIN_TYPES(IntEnum):
    FLAT = 0
    IMAGE = 1

class TEST_MODES(IntEnum):
    FULL = 0
    MINIMAL = 1

class SENSOR_MODES(IntEnum):
    POOR = 0
    NORMAL = 1

class CONTROL_MODES(IntEnum):
    CONSTANT = 0
    LEASH = 1






#==========# CONFIG #==========#

terrain = TERRAIN_TYPES.IMAGE
test_mode = TEST_MODES.MINIMAL
sensor_mode = SENSOR_MODES.NORMAL
control_mode = CONTROL_MODES.LEASH

WHEELBASE = 1.0
MAX_ELEVATION = 5.0
DT = 0.1


# Constant Mode
DELTA = 0.3
V = 1.0

# Leash Mode
V_LEASH = 1.0
DELTA_AMPLITUDE = 0.3
DELTA_PERIOD_STEPS = 400
LEASH_CENTER = (12.8, 12.8)   # middle of your xlim/ylim area
LEASH_RADIUS = 8.0









#==========# Code #==========#

class Simulation:
    def __init__(self):
        # Terrain
        if terrain == TERRAIN_TYPES.FLAT:   
            starting_state = (10.0, 10.0, 0.0, 0.0, 0.0, 0.0)
            self.terrain = FlatTerrain(elevation=0.0, resolution=0.1, shape=(256,256))

        elif terrain == TERRAIN_TYPES.IMAGE:
            starting_state = (10.0, 10.0, 2.14, 0.0, -0.45, 0.0)
            self.terrain = HeightmapTerrain.from_image(terrain_path, resolution=0.1, z_scale=MAX_ELEVATION)

        # Vehicle (noise tuned by Claude)
        self.true_vehicle = ImperfectSuspensionBicycleVehicle3D(WHEELBASE, DT, self.terrain, state=starting_state, noise_std=(0.003, 0.003, 0.001))
        V_x0 = V * np.cos(starting_state[5])  # starting_state[5] = yaw
        V_y0 = V * np.sin(starting_state[5])

        # Sensors (per Claude based on KISS-ICP benchmark with KITTI and specs for Ouster and Insta360)
        if sensor_mode == SENSOR_MODES.NORMAL:
            self.lidar_odom = LidarOdometrySensor(noise_std=(0.0112, 0.0112, 0.0112, 0.000094, 0.000094, 0.000094), init_state=starting_state) # Per Claude Calculations based on paper
            self.imu_ouster = ImuSensor(gyro_noise_std=0.001, accel_noise_std=0.01, gyro_bias_walk_std=0.00005, accel_bias_walk_std=0.0001)
            self.imu_insta = ImuSensor(gyro_noise_std=0.02, accel_noise_std=0.2, gyro_bias_walk_std=0.002, accel_bias_walk_std=0.005)
            self.gps = GpsSensor(noise_std=(1.5, 1.5, 3.0), dt=DT, freq=1.0)

        # Original (poor) sensors
        elif sensor_mode == SENSOR_MODES.POOR:
            self.lidar_odom = LidarOdometrySensor(noise_std=(0.009, 0.009, 0.009, 0.003, 0.003, 0.003), init_state=starting_state)
            self.imu_ouster = ImuSensor(gyro_noise_std=0.001, accel_noise_std=0.01, gyro_bias_walk_std=0.00005, accel_bias_walk_std=0.0001)
            self.imu_insta = ImuSensor(gyro_noise_std=0.02, accel_noise_std=0.2, gyro_bias_walk_std=0.002, accel_bias_walk_std=0.005)
            self.gps = GpsSensor(noise_std=(1.5, 1.5, 3.0), dt=DT, freq=1.0)

        # Estimators
        if test_mode == TEST_MODES.FULL:
            self.graph_reckoner = GraphReckonerPIM(wheelbase=WHEELBASE, dt=DT, init_state=starting_state,
                enable_IMUs=False, enable_NHC=False, enable_rate=False, enable_gps=False,
                dyn_noise_std=(0.01, 0.01, 1.0, 1.0, 1.0, 0.005),
                lidar_noise_std=(0.1, 0.1, 0.1, 0.005, 0.005, 0.005),
            )
            self.graph_reckonerIMU = GraphReckonerPIM(
                wheelbase=WHEELBASE, dt=DT, init_state=starting_state,
                init_velocity=(V_x0, V_y0, 0.0),
                enable_gps=False,
                dyn_noise_std=(0.01, 0.01, 1.0, 1.0, 1.0, 0.005),
                lidar_noise_std=(0.009, 0.009, 0.009, 0.003, 0.003, 0.003),
                imu_configs={
                    "imu_ouster": {
                        "gyro_noise_std": 0.001,
                        "accel_noise_std": 0.01,
                        "gyro_bias_walk_std": 0.00005, 
                        "accel_bias_walk_std": 0.0001, 
                    },
                },
                nhc_noise_std=(0.001, 0.001),
                rate_prior_std=(1.0, 1.0),
                rate_process_noise_std=(0.5, 0.5),
                rate_tie_noise_std=(0.01, 0.01),
                lag_seconds=5.0,
            )
            self.graph_reckonerIMU2 = GraphReckonerPIM(
                wheelbase=WHEELBASE, dt=DT, init_state=starting_state,
                init_velocity=(V_x0, V_y0, 0.0),
                enable_gps=False,
                dyn_noise_std=(0.01, 0.01, 1.0, 1.0, 1.0, 0.005),
                lidar_noise_std=(0.1, 0.1, 0.1, 0.005, 0.005, 0.005),
                imu_configs={
                    "imu_ouster": {
                        "gyro_noise_std": 0.01,
                        "accel_noise_std": 0.05,
                        "gyro_bias_walk_std": 0.00025, 
                        "accel_bias_walk_std": 0.001, 
                        "integration_noise_std": 0.01,
                    },
                },
                nhc_noise_std=(0.001, 0.001),
                rate_prior_std=(1.0, 1.0),
                rate_process_noise_std=(0.5, 0.5),
                rate_tie_noise_std=(0.01, 0.01),
                lag_seconds=5.0,
            )
            self.graph_reckonerGPS = GraphReckonerPIM(
                wheelbase=WHEELBASE, dt=DT, init_state=starting_state,
                init_velocity=(V_x0, V_y0, 0.0),
                enable_lidar=False,
                dyn_noise_std=(0.01, 0.01, 1.0, 1.0, 1.0, 0.005),
                lidar_noise_std=(0.1, 0.1, 0.1, 0.005, 0.005, 0.005),
                imu_configs={
                    "imu_ouster": {
                        "gyro_noise_std": 0.01,
                        "accel_noise_std": 0.05,
                        "gyro_bias_walk_std": 0.00025, 
                        "accel_bias_walk_std": 0.001, 
                        "integration_noise_std": 0.01,
                    },
                },
                nhc_noise_std=(0.001, 0.001),
                rate_prior_std=(1.0, 1.0),
                rate_process_noise_std=(0.5, 0.5),
                rate_tie_noise_std=(0.01, 0.01),
                lag_seconds=5.0,
            )

            self.graph_reckonerFull = GraphReckonerPIM(
                wheelbase=WHEELBASE, dt=DT, init_state=starting_state,
                init_velocity=(V_x0, V_y0, 0.0),
                dyn_noise_std=(0.01, 0.01, 1.0, 1.0, 1.0, 0.005),
                lidar_noise_std=(0.1, 0.1, 0.1, 0.005, 0.005, 0.005),
                imu_configs={
                    "imu_ouster": {
                        "gyro_noise_std": 0.01,
                        "accel_noise_std": 0.05,
                        "gyro_bias_walk_std": 0.00025, 
                        "accel_bias_walk_std": 0.001, 
                        "integration_noise_std": 0.01,
                    },
                },
                nhc_noise_std=(0.001, 0.001),
                rate_prior_std=(1.0, 1.0),
                rate_process_noise_std=(0.5, 0.5),
                rate_tie_noise_std=(0.01, 0.01),
                lag_seconds=5.0,
            )

        if sensor_mode == SENSOR_MODES.NORMAL:
            self.graph_reckonerBest = GraphReckoner(
                wheelbase=WHEELBASE, dt=DT, init_state=starting_state,
                init_velocity=(V_x0, V_y0, 0.0),
                dyn_noise_std=(0.003, 0.003, 1.0, 1.0, 1.0, 0.001),
                lidar_noise_std=(0.0112, 0.0112, 0.0112, 0.000094, 0.000094, 0.000094),
                lidar_drift_prior_std=(0.01, 0.01, 0.01, 0.005, 0.005, 0.005),
                lidar_drift_process_noise_std=(0.01, 0.01, 0.01, 0.005, 0.005, 0.005),
                imu_configs={
                    # "imu_ouster": {
                    #     "gyro_noise_std": 0.01,
                    #     "accel_noise_std": 0.05,
                    #     "gyro_bias_walk_std": 0.00025, 
                    #     "accel_bias_walk_std": 0.001, 
                    #     "integration_noise_std": 0.01,
                    # },
                    "imu_ouster": {
                        "gyro_noise_std": 0.001,
                        "accel_noise_std": 0.01,
                        "gyro_bias_walk_std": 0.00005, 
                        "accel_bias_walk_std": 0.0001, 
                        "integration_noise_std": 0.001,
                    },
                },
                nhc_noise_std=(0.1, 0.1),
                rate_prior_std=(1.0, 1.0),
                rate_process_noise_std=(0.5, 0.5),
                rate_tie_noise_std=(0.01, 0.01),
                gps_noise_std=(1.5, 1.5, 3.0),
                lag_seconds=10.0,
            )

        elif sensor_mode == SENSOR_MODES.POOR:
            self.graph_reckonerBest = GraphReckoner(
                wheelbase=WHEELBASE, dt=DT, init_state=starting_state,
                init_velocity=(V_x0, V_y0, 0.0),
                dyn_noise_std=(0.01, 0.01, 1.0, 1.0, 1.0, 0.005),
                lidar_noise_std=(0.1, 0.1, 0.1, 0.005, 0.005, 0.005),
                lidar_drift_prior_std=(0.01, 0.01, 0.01, 0.005, 0.005, 0.005),
                lidar_drift_process_noise_std=(0.01, 0.01, 0.01, 0.005, 0.005, 0.005),
                imu_configs={
                    "imu_ouster": {
                        "gyro_noise_std": 0.01,
                        "accel_noise_std": 0.05,
                        "gyro_bias_walk_std": 0.00025, 
                        "accel_bias_walk_std": 0.001, 
                        "integration_noise_std": 0.01,
                    },
                },
                nhc_noise_std=(0.1, 0.1),
                rate_prior_std=(1.0, 1.0),
                rate_process_noise_std=(0.5, 0.5),
                rate_tie_noise_std=(0.01, 0.01),
                gps_noise_std=(1.5, 1.5, 3.0),
                lag_seconds=60.0,
            )

        


        # Visualizer
        series = ["truth", "lidar_odom", "GR_noIMU", "GR_IMU", "GR_IMU2", "GR_GPS", "GR_Full", "GR_Best"] if test_mode == TEST_MODES.FULL else ["truth", "GR_Best"]
        self.viz = SimVisualizer(
            series=series,
            terrain=self.terrain,
            xlim=(0, 25.6), ylim=(0, 25.6),
            marker_series="truth",
            elevation_range=(0.0, MAX_ELEVATION),
        )


    @staticmethod
    def _wrap_to_pi(angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def _compute_leash_delta(self, frame):
        x, y, yaw = self.true_vehicle.state[0], self.true_vehicle.state[1], self.true_vehicle.state[5]
        cx, cy = LEASH_CENTER
        dist_from_center = np.hypot(x - cx, y - cy)

        if dist_from_center > LEASH_RADIUS:
            # outside the leash -- steer back toward center instead of
            # continuing to wander, overriding the varied-curvature pattern
            angle_to_center = np.arctan2(cy - y, cx - x)
            heading_error = self._wrap_to_pi(angle_to_center - yaw)
            return np.clip(heading_error, -DELTA_AMPLITUDE, DELTA_AMPLITUDE)

        return DELTA_AMPLITUDE * np.sin(2 * np.pi * frame / DELTA_PERIOD_STEPS)

    def step(self, frame):
        # Control
        if control_mode == CONTROL_MODES.CONSTANT:
            delta = DELTA
            v = V

        elif control_mode == CONTROL_MODES.LEASH:
            delta = self._compute_leash_delta(frame)
            v = V_LEASH

        # Vehicle
        _, truth_state = self.true_vehicle.step(v, delta)

        # Sensors
        lidar_odom_meas = self.lidar_odom.measure(self.true_vehicle, v, delta)
        imu_ouster_meas = self.imu_ouster.measure(self.true_vehicle, v, delta)
        imu_insta_meas = self.imu_insta.measure(self.true_vehicle, v, delta)
        gps_meas = self.gps.measure(self.true_vehicle, v, delta)

        # Estimators
        lidar_odom_state = lidar_odom_meas
        if test_mode == TEST_MODES.FULL:
            graph_reckoner_state = self.graph_reckoner.update({"lidar_odom": lidar_odom_meas}, v, delta)
            graph_reckoner_IMU_state = self.graph_reckonerIMU.update({"lidar_odom": lidar_odom_meas, "imu_ouster": imu_ouster_meas}, v, delta)
            graph_reckoner_IMU2_state = self.graph_reckonerIMU2.update({"lidar_odom": lidar_odom_meas, "imu_ouster": imu_ouster_meas}, v, delta)
            graph_reckoner_GPS_state = self.graph_reckonerGPS.update({"gps": gps_meas, "imu_ouster": imu_ouster_meas,}, v, delta)
            graph_reckoner_Full_state = self.graph_reckonerFull.update({"lidar_odom": lidar_odom_meas, "gps": gps_meas, "imu_ouster": imu_ouster_meas,}, v, delta)

        graph_reckoner_Best_state = self.graph_reckonerBest.update({"lidar_odom": lidar_odom_meas, "gps": gps_meas, "imu_ouster": imu_ouster_meas,}, v, delta)

        if test_mode == TEST_MODES.FULL:
            viz_update = self.viz.update({
                "truth": truth_state,
                "lidar_odom": lidar_odom_state,
                "GR_noIMU": graph_reckoner_state,
                "GR_IMU": graph_reckoner_IMU_state,
                "GR_IMU2": graph_reckoner_IMU2_state,
                "GR_GPS": graph_reckoner_GPS_state,
                "GR_Full": graph_reckoner_Full_state,
                "GR_Best": graph_reckoner_Best_state,
            })
        elif test_mode == TEST_MODES.MINIMAL:
            viz_update = self.viz.update({
                "truth": truth_state,
                "GR_Best": graph_reckoner_Best_state,
            })

        return viz_update

    def run(self):
        self.viz.animate(self.step)


if __name__ == "__main__":
    Simulation().run()