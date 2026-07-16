"""Defines the vehicle simulator (3D)."""

from abc import ABC, abstractmethod

import numpy as np


class Vehicle(ABC):
    """Base class for ground-truth vehicle motion models in 3D.

    State convention: (x, y, z, roll, pitch, yaw), all in the world
    frame. Subclasses own this state and advance it by one dt given
    some control input, whose meaning is left to the subclass.
    """

    def __init__(self, dt, state):
        self.dt = dt
        self.state = np.array(state, dtype=float)
        self.prev_state = self.state.copy()

    @abstractmethod
    def step(self, *controls):
        """Advance state by one dt. Returns (prev_state, curr_state), copies."""
        raise NotImplementedError


class BicycleVehicle3D(Vehicle):
    """Kinematic bicycle model in the ground plane (x, y, yaw), with
    roll/pitch first-order-lagging toward whatever Terrain implies at
    the current position: z comes directly from terrain.elevation(x, y)
    (still instantaneous -- only attitude is lagged), and target
    roll/pitch come from the local terrain slope, decomposed into the
    vehicle's forward/lateral directions.

    The lag (attitude_tau) is a cheap improvement over snapping
    instantly, not real suspension physics: there's still no actual
    angular momentum, no mass/spring/damper relationship, and the gyro
    rate implied by this lag isn't tied to any real torque -- it just
    smooths out the least physically defensible part of the old
    behavior (instant teleporting to a new attitude). A real
    spring-damper model would be the next step up if this isn't enough.

    State: (x, y, z, roll, pitch, yaw). Control: (v, delta).
    """

    def __init__(self, wheelbase, dt, terrain, state=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                 attitude_tau=0.3):
        """attitude_tau: time constant (seconds) for roll/pitch to settle
        toward the terrain-implied target. Picked as a placeholder --
        not derived from any real vehicle's suspension response -- so
        treat it as a tuning knob, not a physical parameter. Smaller
        values approach the old instant-snap behavior; tau << dt
        effectively recovers it (the blend factor is clipped at 1.0).
        """
        super().__init__(dt, state)
        self.wheelbase = wheelbase
        self.terrain = terrain
        self.attitude_tau = attitude_tau
        self.state = self._conform_to_terrain(self.state)
        self.prev_state = self.state.copy()

    def step(self, v, delta):
        prev_state = self.state.copy()
        x, y, z, roll, pitch, yaw = self.state

        x += v * np.cos(yaw) * self.dt
        y += v * np.sin(yaw) * self.dt
        yaw += (v / self.wheelbase) * np.tan(delta) * self.dt

        new_state = self._conform_to_terrain(np.array([x, y, z, roll, pitch, yaw]))

        self.state = new_state
        self.prev_state = prev_state
        return prev_state, self.state.copy()

    def _conform_to_terrain(self, state):
        """Body frame: x=forward, y=left, z=up (FLU), with standard
        right-hand-rule Euler angles about each axis. This gives:
        yaw left = positive, roll right (right-side-down) = positive,
        pitch down (nose-down) = positive.

        roll/pitch here are the PREVIOUS values (carried through from
        the state passed in), blended toward the newly-computed target
        by a factor of dt/attitude_tau -- a first-order lag, not an
        instant snap.
        """
        x, y, _, prev_roll, prev_pitch, yaw = state
        z = self.terrain.elevation(x, y)
        dzdx, dzdy = self.terrain.gradient(x, y)

        # slope decomposed into vehicle-forward / vehicle-lateral directions
        slope_fwd = dzdx * np.cos(yaw) + dzdy * np.sin(yaw)
        slope_lat = -dzdx * np.sin(yaw) + dzdy * np.cos(yaw)

        target_pitch = -np.arctan(slope_fwd)  # downhill ahead -> nose down -> positive
        target_roll = np.arctan(slope_lat)    # terrain rising to the left -> right-side-down -> positive

        alpha = min(self.dt / self.attitude_tau, 1.0)
        roll = prev_roll + (target_roll - prev_roll) * alpha
        pitch = prev_pitch + (target_pitch - prev_pitch) * alpha

        return np.array([x, y, z, roll, pitch, yaw])


class ImperfectBicycleVehicle3D(BicycleVehicle3D):
    """Kinematic bicycle model with noise added to the planar (x, y, yaw)
    update, representing actuator/plant imperfection (wheel slip,
    imprecise steering) -- same intent as the 2D ImperfectBicycleVehicle.

    Noise is NOT applied to z/roll/pitch directly: those are always
    recomputed from (x, y, yaw) and terrain in _conform_to_terrain(), so
    perturbing them here would just get silently overwritten one line
    later. Terrain-driven attitude only becomes noisy indirectly, via
    the noisy (x, y) landing on a slightly different patch of terrain
    than the noise-free model would have.
    """

    def __init__(self, wheelbase, dt, terrain, state=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                 noise_std=(0.005, 0.005, 0.001)):
        """noise_std: (x, y, yaw) standard deviations only -- see class docstring."""
        super().__init__(wheelbase, dt, terrain, state)
        self.noise_std = noise_std

    def step(self, v, delta):
        prev_state, curr_state = super().step(v, delta)
        x, y, z, roll, pitch, yaw = curr_state

        x += np.random.normal(0, self.noise_std[0])
        y += np.random.normal(0, self.noise_std[1])
        yaw += np.random.normal(0, self.noise_std[2])

        self.state = self._conform_to_terrain(np.array([x, y, z, roll, pitch, yaw]))
        self.prev_state = prev_state
        return prev_state, self.state.copy()
    


class SuspensionBicycleVehicle3D(BicycleVehicle3D):
    """Kinematic bicycle model with a real second-order mass-spring-
    damper response for roll/pitch, replacing the first-order lag.
    Roll and pitch are now true dynamic states with their own rates
    (self.roll_rate, self.pitch_rate), driven toward the
    terrain-implied target angle by a damped-spring restoring torque.
 
    This is the version whose causal direction actually matches what
    ImuFactor assumes: real rotational dynamics producing the gyro
    reading, rather than an algebraic snap (BicycleVehicle3D) or a
    one-sided lag (the earlier fix) toward a target.
 
    Uses omega_n (natural frequency, rad/s) and zeta (damping ratio) --
    standard 2nd-order system parameters -- instead of raw spring/
    damper constants, since they're more intuitive to tune:
    zeta=1.0 is critically damped (fastest settle, no overshoot);
    zeta<1 oscillates before settling; zeta>1 settles slower without
    oscillating. Neither is derived from any real vehicle's actual
    suspension -- pick values that look/feel reasonable and tune from
    there.
 
    STABILITY CAVEAT: uses semi-implicit (symplectic) Euler,
    sub-stepped n_substeps times per outer dt -- not a single Euler
    step at the full dt. This reduces instability risk but does not
    eliminate it: push omega_n high enough relative to
    dt/n_substeps and explicit integration can still oscillate or blow
    up numerically, independent of the physical zeta you asked for. If
    that happens, increase n_substeps first; the fully robust fix is an
    exact analytic (matrix-exponential) discretization of the linear
    2nd-order system, which is NOT implemented here -- this is a first
    real pass at proper suspension dynamics, not a finished one.
    """
 
    def __init__(self, wheelbase, dt, terrain, state=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                 omega_n=2 * np.pi * 1.5, zeta=0.9, n_substeps=10):
        """
        omega_n: natural frequency in rad/s. 2*pi*1.5 (~1.5 Hz) here as
            a placeholder in the range real vehicle suspensions often
            sit in -- not measured from anything.
        zeta: damping ratio, see class docstring.
        n_substeps: sub-steps per outer dt for the semi-implicit Euler
            integration -- see STABILITY CAVEAT above.
        """
        self.omega_n = omega_n
        self.zeta = zeta
        self.n_substeps = n_substeps
        self.roll_rate = 0.0
        self.pitch_rate = 0.0
        super().__init__(wheelbase, dt, terrain, state)
 
    def _conform_to_terrain(self, state):
        """z still comes directly from terrain (no lag/dynamics there,
        same as the base class) -- only roll/pitch get real 2nd-order
        dynamics. roll/pitch coming in are the PREVIOUS values (this
        method's calling convention, inherited from BicycleVehicle3D,
        is that state carries the prior attitude forward).
        """
        x, y, _, roll, pitch, yaw = state
        z = self.terrain.elevation(x, y)
        dzdx, dzdy = self.terrain.gradient(x, y)
 
        slope_fwd = dzdx * np.cos(yaw) + dzdy * np.sin(yaw)
        slope_lat = -dzdx * np.sin(yaw) + dzdy * np.cos(yaw)
        target_pitch = -np.arctan(slope_fwd)
        target_roll = np.arctan(slope_lat)
 
        sub_dt = self.dt / self.n_substeps
        for _ in range(self.n_substeps):
            roll_accel = (-self.omega_n ** 2 * (roll - target_roll)
                          - 2 * self.zeta * self.omega_n * self.roll_rate)
            pitch_accel = (-self.omega_n ** 2 * (pitch - target_pitch)
                           - 2 * self.zeta * self.omega_n * self.pitch_rate)
 
            # semi-implicit Euler: update rate first, then use the
            # NEW rate to update position -- more stable than plain
            # (explicit) Euler for oscillatory systems, though not
            # unconditionally stable -- see STABILITY CAVEAT.
            self.roll_rate += roll_accel * sub_dt
            self.pitch_rate += pitch_accel * sub_dt
            roll += self.roll_rate * sub_dt
            pitch += self.pitch_rate * sub_dt
 
        return np.array([x, y, z, roll, pitch, yaw])
    


class ImperfectSuspensionBicycleVehicle3D(SuspensionBicycleVehicle3D):
    """Combines noisy planar actuation (see ImperfectBicycleVehicle3D)
    with real 2nd-order roll/pitch suspension dynamics (see
    SuspensionBicycleVehicle3D).
 
    Deliberately does NOT follow ImperfectBicycleVehicle3D's pattern of
    calling super().step() then re-conforming a second time -- that
    would integrate the suspension ODE twice per outer dt (2*n_substeps
    worth of dynamics instead of n_substeps), silently changing the
    suspension's effective response speed. Instead this reimplements
    the planar kinematics directly, adds noise, and calls
    _conform_to_terrain exactly once.
    """
 
    def __init__(self, wheelbase, dt, terrain, state=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                 omega_n=2 * np.pi * 1.5, zeta=0.9, n_substeps=10,
                 noise_std=(0.005, 0.005, 0.001)):
        """noise_std: (x, y, yaw) standard deviations -- same convention
        as ImperfectBicycleVehicle3D. omega_n/zeta/n_substeps: see
        SuspensionBicycleVehicle3D.
        """
        super().__init__(wheelbase, dt, terrain, state, omega_n, zeta, n_substeps)
        self.noise_std = noise_std
 
    def step(self, v, delta):
        prev_state = self.state.copy()
        x, y, z, roll, pitch, yaw = self.state
 
        x += v * np.cos(yaw) * self.dt
        y += v * np.sin(yaw) * self.dt
        yaw += (v / self.wheelbase) * np.tan(delta) * self.dt
 
        x += np.random.normal(0, self.noise_std[0])
        y += np.random.normal(0, self.noise_std[1])
        yaw += np.random.normal(0, self.noise_std[2])
 
        new_state = self._conform_to_terrain(np.array([x, y, z, roll, pitch, yaw]))
 
        self.state = new_state
        self.prev_state = prev_state
        return prev_state, self.state.copy()
    

    


def rotation_body_to_world(roll, pitch, yaw):
    """Body-to-world rotation matrix, FLU frame, standard aerospace ZYX
    composition (R = Rz(yaw) @ Ry(pitch) @ Rx(roll)), consistent with the
    yaw-left/roll-right/pitch-down sign conventions used throughout.
    """
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def relative_pose_local_frame(prev_state, curr_state):
    """Relative motion (dx, dy, dz, droll, dpitch, dyaw) between two 6D
    states. Translation (dx, dy) is rotated into prev_state's local
    frame using yaw only -- roll/pitch are treated as small,
    instantaneous terrain-following angles, not full attitude coupling
    into the frame rotation. This is an approximation, not a rigorous
    SE(3) transform: it's the direct 3D extension of the original 2D
    version, not a new derivation.
    """
    dx_global = curr_state[0] - prev_state[0]
    dy_global = curr_state[1] - prev_state[1]
    dz = curr_state[2] - prev_state[2]
    droll = curr_state[3] - prev_state[3]
    dpitch = curr_state[4] - prev_state[4]
    dyaw = curr_state[5] - prev_state[5]

    yaw = prev_state[5]
    c, s = np.cos(yaw), np.sin(yaw)
    dx_local = c * dx_global + s * dy_global
    dy_local = -s * dx_global + c * dy_global
    return dx_local, dy_local, dz, droll, dpitch, dyaw