"""Fixed gtsam single-character symbol prefixes.

Each prefix + integer index packs into one gtsam Key (see
gtsam.symbol()). The integer always means "which timestep / which
instance in this chain" -- it is NOT a second identifier, so two
IMUs cannot share a prefix and be told apart by index alone. If you
add a new sensor/chain type, add a new prefix here rather than
picking one ad hoc at the call site.
"""

POSE = 'x'              # primary pose nodes (timer-driven backbone)
VELOCITY = 'v'          # velocity nodes, one per primary pose
RATE = 'r'              # angular rate (roll_dot, pitch_dot) chain

LIDAR_DRIFT = 'd'       # latent lidar-odometry drift chain
LIDAR_RESIDUAL = 'l'    # transient residual pose nodes, one per lidar reading

GPS_RESIDUAL = 'g'      # transient residual pose nodes, one per gps reading

IMU_BIAS_OUSTER = 'b'   # Ouster IMU bias chain
IMU_BIAS_INSTA = 'c'    # Insta360 IMU bias chain

# Sanity check: fail loudly at import time if these ever collide,
# rather than silently corrupting a chain the way the old auto-assigned
# ('b', 'c', 'd', ...) scheme could.
_ALL = [POSE, VELOCITY, RATE, LIDAR_DRIFT, LIDAR_RESIDUAL, GPS_RESIDUAL,
        IMU_BIAS_OUSTER, IMU_BIAS_INSTA]
assert len(_ALL) == len(set(_ALL)), "Duplicate gtsam symbol prefix in symbols.py"