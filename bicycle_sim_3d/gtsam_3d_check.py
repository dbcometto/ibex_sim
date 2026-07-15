import gtsam
import numpy as np
from vehicle import rotation_body_to_world

roll, pitch, yaw = np.radians(20), np.radians(15), np.radians(30)

# Check 1: does gtsam.Rot3.Ypr match our own rotation_body_to_world?
R_ours = rotation_body_to_world(roll, pitch, yaw)
R_gtsam = gtsam.Rot3.Ypr(yaw, pitch, roll).matrix()
print("Rot3.Ypr matches ours:", np.allclose(R_ours, R_gtsam))

# Check 2: round-trip through Pose3 and back
pose = gtsam.Pose3(gtsam.Rot3.Ypr(yaw, pitch, roll), gtsam.Point3(1, 2, 3))
r = pose.rotation()
print("roll/pitch/yaw round-trip:", r.roll(), r.pitch(), r.yaw(), "vs", roll, pitch, yaw)

# Check 3: confirm Pose3 noise sigma order is (roll,pitch,yaw,x,y,z), not (x,y,z,roll,pitch,yaw)
# — hardest to check directly; if in doubt, constrain one axis very tightly
# (e.g. sigma=1e-9 for "x") in each ordering and see which one actually
# pins x during optimization.