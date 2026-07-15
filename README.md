# Simulator for IBEX testing



## Usage

Run `main.py` in either simulator, currently set up to work with Ben's WSL system.

## Current Status

Currently the graph estimators work decently well.  The IMU system works on a flat plain because inertial assumptions are not violated by the way I implemented a 3D terrain-snapping bicycle model.  A first oder averaging lag on the roll/pitch/yaw as if there was suspension does not fix the IMU on the heightmap, it might be a limitation of the sensor...

Claude:
```
Built a 3D kinematic bicycle vehicle sim over a heightmap terrain (vehicle.py, world.py), with lidar odometry and dual IMU sensors (sensors.py), plus three GTSAM-based factor-graph estimators (estimators.py): a batch (GraphReckonerLM) and incremental (GraphReckonerISAM) version fusing dynamics + lidar, and a newer GraphReckonerPIM that adds real IMU factors via gtsam's preintegration. The non-IMU estimator tracks truth well; the IMU-fused version tracks reasonably but still drifts, most visibly in elevation, likely due to a units mismatch in the IMU bias noise model and an overly-tight dynamics factor fighting the IMU/lidar on z. Also added a first-order lag to the vehicle's roll/pitch (replacing instant terrain-snapping) since that mismatch was actively corrupting the IMU factor's rigid-body assumptions — confirmed via a flat-vs-heightmap terrain test. Next session's natural starting points: split the bias noise into separate gyro/accel scales, loosen GraphReckonerPIM's z/roll/pitch dynamics noise, and re-test.
```