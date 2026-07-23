import gtsam
import gtsam_unstable
import numpy as np
from vehicle import rotation_body_to_world
from factors import make_nhc_factor, make_rate_tie_factor, make_rate_cv_factor, make_lidar_drift_factor

roll, pitch, yaw = np.radians(20), np.radians(15), np.radians(30)

# Check 1: does gtsam.Rot3.Ypr match our own rotation_body_to_world?
print("Check 1:")
R_ours = rotation_body_to_world(roll, pitch, yaw)
R_gtsam = gtsam.Rot3.Ypr(yaw, pitch, roll).matrix()
print("Rot3.Ypr matches ours:", np.allclose(R_ours, R_gtsam))

# Check 2: round-trip through Pose3 and back
print("\n\nCheck 2:")
pose = gtsam.Pose3(gtsam.Rot3.Ypr(yaw, pitch, roll), gtsam.Point3(1, 2, 3))
r = pose.rotation()
print("roll/pitch/yaw round-trip:", r.roll(), r.pitch(), r.yaw(), "vs", roll, pitch, yaw)

# Check 3: confirm Pose3 noise sigma order is (roll,pitch,yaw,x,y,z), not (x,y,z,roll,pitch,yaw)
# print("Check 3:")
# — hardest to check directly; if in doubt, constrain one axis very tightly
# (e.g. sigma=1e-9 for "x") in each ordering and see which one actually
# pins x during optimization.

# Check 4:
print("\n\nCheck 4:")
bias = gtsam.imuBias.ConstantBias(np.array([1, 2, 3]), np.array([4, 5, 6]))  # (accel, gyro)
print(bias.vector())  # if this prints [1,2,3,4,5,6], accel-first is confirmed;
                       # if [4,5,6,1,2,3], the helper's order needs flipping

# Check 5:
print("\n\nCheck 5:")
m = gtsam_unstable.FixedLagSmootherKeyTimestampMap()
print([x for x in dir(m) if not x.startswith('_')])


# Check 6:
print("\n\nCheck 6:")
# velocity purely forward in body frame -> error should be ~0
pose = gtsam.Pose3(gtsam.Rot3.Ypr(0.3, -0.2, 0.1), gtsam.Point3(0, 0, 0))
v_forward = pose.rotation().matrix() @ np.array([2.0, 0, 0])
values = gtsam.Values()
values.insert(1, pose)
values.insert(2, v_forward)
factor = make_nhc_factor(1, 2, gtsam.noiseModel.Unit.Create(2))
print(factor.error(values))  # expect ~0

# velocity with real sideways component -> error should be nonzero
v_sideways = pose.rotation().matrix() @ np.array([2.0, 1.0, 0])
values2 = gtsam.Values()
values2.insert(1, pose)
values2.insert(2, v_sideways)
print(factor.error(values2))  # expect noticeably nonzero



# Check 7:
print("\n\nCheck 7:")
pose_i = gtsam.Pose3(gtsam.Rot3.Ypr(0.0, 0.1, 0.05), gtsam.Point3(0, 0, 0))
dt = 0.1
r_true = np.array([0.02, -0.03])  # (roll_dot, pitch_dot)
roll_j = 0.05 + r_true[0] * dt
pitch_j = 0.1 + r_true[1] * dt
pose_j = gtsam.Pose3(gtsam.Rot3.Ypr(0.0, pitch_j, roll_j), gtsam.Point3(0, 0, 0))

values = gtsam.Values()
values.insert(1, pose_i); values.insert(2, pose_j); values.insert(3, r_true)
factor = make_rate_tie_factor(1, 2, 3, dt, gtsam.noiseModel.Unit.Create(2))
print(factor.error(values))  # expect ~0

r_wrong = np.array([0.0, 0.0])
values2 = gtsam.Values()
values2.insert(1, pose_i); values2.insert(2, pose_j); values2.insert(3, r_wrong)
print(factor.error(values2))  # expect clearly nonzero



# Check 8
print("\n\nCheck 8:")
true_pose = gtsam.Pose3(gtsam.Rot3.Ypr(0.1, 0.0, 0.0), gtsam.Point3(5, 0, 0))
true_drift = gtsam.Pose3(gtsam.Rot3.Ypr(0.02, 0.0, 0.0), gtsam.Point3(0.3, 0.1, 0.0))
lidar_reading = true_pose.compose(true_drift)  # what a drifting sensor would report

values = gtsam.Values()
values.insert(1, true_pose); values.insert(2, true_drift)
factor = make_lidar_drift_factor(1, 2, lidar_reading, gtsam.noiseModel.Unit.Create(6))
print(factor.error(values))  # expect ~0

values2 = gtsam.Values()
values2.insert(1, true_pose); values2.insert(2, gtsam.Pose3())  # wrong: assume zero drift
factor2 = make_lidar_drift_factor(1, 2, lidar_reading, gtsam.noiseModel.Unit.Create(6))
print(factor2.error(values2))  # expect clearly nonzero