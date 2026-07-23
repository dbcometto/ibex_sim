"""Advances ground-truth vehicle state to arbitrary simulation times,
decoupled from any sensor's own sampling rate.

CAVEAT: vehicle.step() only exists at a fixed sub-step dt (needed for
the suspension model's own integration). advance_to(t) rounds UP to the
nearest completed tick >= t_target-ish by stepping fine_dt at a time --
it is not true sub-tick interpolation. Keep fine_dt small relative to
the fastest sensor period (e.g. fine_dt <= 1/50 of the fastest sensor
period) or this rounding error stops being negligible.
"""


class VehicleRunner:
    def __init__(self, vehicle, fine_dt, control_fn):
        """
        vehicle: a Vehicle instance (e.g. ImperfectSuspensionBicycleVehicle3D).
        fine_dt: integration sub-step, should be << fastest sensor period.
        control_fn: callable(t) -> (v, delta). Control as a function of
            elapsed sim time -- replaces the old frame-index-based leash
            logic so control is defined at any t, not just discrete frames.
        """
        self.vehicle = vehicle
        self.fine_dt = fine_dt
        self.control_fn = control_fn
        self.t = 0.0

    def advance_to(self, t_target):
        """Step forward until self.t >= t_target (within one fine_dt).
        No-op if already there. Read self.vehicle.state afterward.
        """
        while self.t < t_target - 1e-12:
            v, delta = self.control_fn(self.t)
            self.vehicle.step(v, delta)
            self.t += self.fine_dt


def leash_control_fn(leash_center, leash_radius, delta_amplitude, delta_period_s,
                      v_leash, get_state, wrap_to_pi):
    """Continuous-time version of Simulation._compute_leash_delta from
    main.py -- driven by elapsed time instead of an integer frame count,
    since events no longer land on fixed frame boundaries.

    get_state: callable() -> current (x, y, ..., yaw) vehicle state,
        called lazily at evaluation time (not captured once), so this
        reflects the true state at whatever t control_fn is asked about.
    """
    import numpy as np

    def control_fn(t):
        state = get_state()
        x, y, yaw = state[0], state[1], state[5]
        cx, cy = leash_center
        dist_from_center = np.hypot(x - cx, y - cy)

        if dist_from_center > leash_radius:
            angle_to_center = np.arctan2(cy - y, cx - x)
            heading_error = wrap_to_pi(angle_to_center - yaw)
            delta = np.clip(heading_error, -delta_amplitude, delta_amplitude)
        else:
            delta = delta_amplitude * np.sin(2 * np.pi * t / delta_period_s)

        return v_leash, delta

    return control_fn