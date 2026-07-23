"""Sim v3: 3D vehicle with async, jittered, occasionally-lossy sensor
timing, driving GraphReckoner through the same callback shape a ROS2
node would use (timer -> add_primary, subscriptions -> add_lidar /
add_gps / add_imu).

All tuning lives in config/ -- estimator_params.yaml is shared with the
future ROS node, sim_params.yaml is sim-only. See those files' header
comments for the split rationale.

CAVEAT: written but NOT RUN -- no gtsam in this sandbox (see
estimators.py's module docstring). Treat as a first draft to debug on
your machine, same as the rest of this module.
"""
import numpy as np

from vehicle import ImperfectSuspensionBicycleVehicle3D
from world import HeightmapTerrain, FlatTerrain
from visualization import SimVisualizer

from scheduler import EventScheduler, SensorStream
from vehicle_runner import VehicleRunner, leash_control_fn
from sensors import LidarOdometrySensorAsync, ImuSensorAsync, GpsSensorAsync
from estimators import GraphReckoner
from config_loader import load_config


def wrap_to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def main():
    est_cfg = load_config('config/estimator_params.yaml')
    sim_cfg = load_config('config/sim_params.yaml')

    np.random.seed(sim_cfg['run']['random_seed'])

    starting_state = tuple(sim_cfg['vehicle']['starting_state'])
    terrain_cfg = sim_cfg['terrain']
    if terrain_cfg['mode'] == 'flat':
        terrain = FlatTerrain(elevation=terrain_cfg['flat_elevation'], resolution=terrain_cfg['resolution'])
    elif terrain_cfg['mode'] == 'image':
        terrain = HeightmapTerrain.from_image(
            terrain_cfg['path'], resolution=terrain_cfg['resolution'], z_scale=terrain_cfg['max_elevation'],
        )
    else:
        raise ValueError(f"terrain.mode must be 'flat' or 'image', got {terrain_cfg['mode']!r}")
    vehicle = ImperfectSuspensionBicycleVehicle3D(
        est_cfg['graph_reckoner']['wheelbase'], sim_cfg['vehicle']['fine_dt'], terrain,
        state=starting_state,
        noise_std=tuple(sim_cfg['vehicle']['actuation_noise_std']),
    )

    ctrl = sim_cfg['control']
    control_fn = leash_control_fn(
        tuple(ctrl['leash_center']), ctrl['leash_radius'], ctrl['delta_amplitude'],
        ctrl['delta_period_s'], ctrl['v_leash'],
        get_state=lambda: vehicle.state, wrap_to_pi=wrap_to_pi,
    )
    runner = VehicleRunner(vehicle, sim_cfg['vehicle']['fine_dt'], control_fn)

    lidar_cfg = sim_cfg['sensors']['lidar']
    gps_cfg = sim_cfg['sensors']['gps']
    imu_cfg = sim_cfg['sensors']['imu_ouster']

    lidar = LidarOdometrySensorAsync(noise_std=tuple(lidar_cfg['noise_std']), init_state=starting_state)
    imu_ouster = ImuSensorAsync(
        gyro_noise_std=imu_cfg['gyro_noise_std'], accel_noise_std=imu_cfg['accel_noise_std'],
        gyro_bias_walk_std=imu_cfg['gyro_bias_walk_std'], accel_bias_walk_std=imu_cfg['accel_bias_walk_std'],
    )
    gps = GpsSensorAsync(noise_std=tuple(gps_cfg['noise_std']))

    v_x0 = ctrl['v_leash'] * np.cos(starting_state[5])
    v_y0 = ctrl['v_leash'] * np.sin(starting_state[5])

    gr_cfg = est_cfg['graph_reckoner']
    estimator = GraphReckoner(
        wheelbase=gr_cfg['wheelbase'],
        init_state=starting_state,
        init_velocity=(v_x0, v_y0, 0.0),
        prior_noise_std=gr_cfg['prior_noise_std'],
        init_velocity_noise_std=gr_cfg['init_velocity_noise_std'],
        dyn_noise_std=tuple(gr_cfg['dyn_noise_std']),
        lidar_noise_std=tuple(gr_cfg['lidar_noise_std']),
        lidar_drift_prior_std=tuple(gr_cfg['lidar_drift_prior_std']),
        lidar_drift_process_noise_std=tuple(gr_cfg['lidar_drift_process_noise_std']),
        residual_prop_noise_std=tuple(gr_cfg['residual_prop_noise_std']),
        gps_prop_noise_std=tuple(gr_cfg['gps_prop_noise_std']),
        imu_configs=gr_cfg['imu_configs'],
        nhc_noise_std=tuple(gr_cfg['nhc_noise_std']),
        rate_prior_std=tuple(gr_cfg['rate_prior_std']),
        rate_process_noise_std=tuple(gr_cfg['rate_process_noise_std']),
        rate_tie_noise_std=tuple(gr_cfg['rate_tie_noise_std']),
        gps_noise_std=tuple(gr_cfg['gps_noise_std']),
        lag_seconds=gr_cfg['lag_seconds'],
        enable_lidar=gr_cfg['enable_lidar'],
        enable_IMUs=gr_cfg['enable_IMUs'],
        enable_NHC=gr_cfg['enable_NHC'],
        enable_rate=gr_cfg['enable_rate'],
        enable_gps=gr_cfg['enable_gps'],
    )

    rng = np.random.default_rng(sim_cfg['run']['random_seed'])
    scheduler = EventScheduler([
        SensorStream("primary", est_cfg['primary_timer']['period_s'],
                     jitter_std=0.0, dropout_prob=0.0, rng=rng),
        SensorStream("lidar", lidar_cfg['period_s'],
                     jitter_std=lidar_cfg['jitter_std_s'], dropout_prob=lidar_cfg['dropout_prob'], rng=rng),
        SensorStream("gps", gps_cfg['period_s'],
                     jitter_std=gps_cfg['jitter_std_s'], dropout_prob=gps_cfg['dropout_prob'], rng=rng),
        SensorStream("imu_ouster", imu_cfg['period_s'],
                     jitter_std=imu_cfg['jitter_std_s'], dropout_prob=imu_cfg['dropout_prob'], rng=rng),
    ])

    viz_cfg = sim_cfg['visualization']
    viz = SimVisualizer(
        series=["truth", "GR_Best"], terrain=terrain,
        xlim=tuple(viz_cfg['xlim']), ylim=tuple(viz_cfg['ylim']), marker_series="truth",
        elevation_range=(0.0, sim_cfg['terrain']['max_elevation']),
        error_x_label="time (s)",
    )

    horizon_s = sim_cfg['run']['horizon_s']
    state = {"finished": False, "last_artists": []}

    def step(frame):
        # Once the horizon is reached, stop draining/updating entirely --
        # returning the same cached artists instead of re-appending the
        # frozen state every remaining animation frame (which is what
        # produced the flat tail on the error plots before this fix).
        if state["finished"]:
            return state["last_artists"]

        # Drain events until (and including) the next "primary" event --
        # this naturally sweeps up every lidar/gps/imu event that fell
        # inside this primary window first. One animation frame now
        # corresponds to one primary period (~1/6s of sim time) instead
        # of an arbitrary fixed event count, which is what let a fast
        # sensor (200Hz IMU) blow up the apparent "step" count far
        # beyond actual elapsed sim time.
        t_last = None
        while True:
            result = scheduler.pop_next(horizon=horizon_s)
            if result is None:
                state["finished"] = True
                break
            t, name = result
            runner.advance_to(t)
            v, delta = control_fn(t)

            if name == "primary":
                estimator.add_primary(t, v, delta)
                t_last = t
                break  # one primary window handled -- yield back to the animation frame

            elif name == "lidar":
                lidar_pose = lidar.measure(runner)
                estimator.add_lidar(t, lidar_pose)

            elif name == "gps":
                gps_pos = gps.measure(runner)
                estimator.add_gps(t, gps_pos)

            elif name == "imu_ouster":
                result_imu = imu_ouster.measure(runner, v)
                if result_imu is not None:
                    omega_meas, accel_meas, dt = result_imu
                    estimator.add_imu("imu_ouster", omega_meas, accel_meas, dt)

        if t_last is None:
            # Horizon reached before another primary fired -- freeze here.
            return state["last_artists"]

        artists = viz.update({
            "truth": vehicle.state,
            "GR_Best": estimator.get_estimate(),
        }, t=t_last)
        state["last_artists"] = artists
        return artists

    viz.animate(step)


if __name__ == "__main__":
    main()