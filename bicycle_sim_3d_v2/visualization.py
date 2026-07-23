"""Defines the simulation visualizer (3D vehicle, top-down view)."""

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


class SimVisualizer:
    """Top-down (x, y) view of named trajectories, with an optional
    terrain image drawn underneath, graphical gauges for yaw/pitch/
    roll/elevation, and a row of error-over-time plots (position
    distance + each angle) measured against a reference series.

    Gauges/error-lines are pre-created for every name in `series`, each
    colored to match that series' trajectory line. Any series whose
    update() state has 6 elements (x, y, z, roll, pitch, yaw) gets its
    gauges/error moved; 2-element (x, y)-only states are skipped for
    both. This means overlaying an estimator's attitude later is just a
    matter of including it in `series` and passing a 6-element state
    for it -- no visualizer changes needed.

    NOTE: the main plot only shows x, y -- it's a top-down position
    view, not a true 3D renderer.
    """

    _STYLES = ['b-', 'r--', 'g--', 'm:', 'c-.', 'y--', 'k-']

    def __init__(self, series, terrain=None,
                 xlim=(-5, 5), ylim=(-1, 9), marker_series=None,
                 pitch_roll_range_deg=45.0, elevation_range=(0.0, 20.0),
                 error_reference_series=None, error_x_label="step"):
        """
        terrain: a HeightmapTerrain instance (or anything with
            .heightmap, .resolution, .origin) to draw as a background.
            The actual elevation array is plotted -- not a re-loaded
            image file -- so matplotlib's built-in hover readout shows
            real elevation in meters, and the extent is derived from the
            terrain object itself instead of passed in by hand (so it
            can't drift out of sync with the data).
        pitch_roll_range_deg: the +/- range the pitch/roll bar gauges
            span before clipping. 45 degrees is a placeholder, not
            derived from your actual terrain's slope distribution --
            tune it if markers are pinning to the ends.
        elevation_range: (min, max) the elevation bar gauge spans before
            clipping. Also a placeholder -- set it to your terrain's
            actual elevation range.
        error_reference_series: name of the series treated as ground
            truth for the error plots -- every other 6-element series
            gets its position distance + |yaw|/|pitch|/|roll| error
            (wrapped to (-180,180] before taking the absolute value)
            plotted against it each step. Defaults to marker_series if
            not given. None disables the error plots entirely. The
            reference series itself is never plotted against itself
            (always-zero line, not useful).
        error_x_label: x-axis label for the four error subplots. Pass
            "time (s)" if you'll be calling update(..., t=<seconds>);
            leave as "step" if relying on the built-in auto-incrementing
            counter (t=None).
        """
        self.pitch_roll_range_deg = pitch_roll_range_deg
        self.elevation_range = elevation_range
        self.error_reference = (
            error_reference_series if error_reference_series is not None else marker_series
        )
        self.error_x_label = error_x_label

        self.fig = plt.figure(figsize=(11, 7.5))
        outer_gs = self.fig.add_gridspec(2, 2, height_ratios=[2.2, 1], width_ratios=[3, 1],
                                          hspace=0.4, wspace=0.3)
        self.ax = self.fig.add_subplot(outer_gs[0, 0])
        inner_gs = outer_gs[0, 1].subgridspec(4, 1, hspace=0.7)
        self.yaw_ax = self.fig.add_subplot(inner_gs[0, 0])
        self.pitch_ax = self.fig.add_subplot(inner_gs[1, 0])
        self.roll_ax = self.fig.add_subplot(inner_gs[2, 0])
        self.elev_ax = self.fig.add_subplot(inner_gs[3, 0])

        if terrain is not None:
            rows, cols = terrain.heightmap.shape
            ox, oy = terrain.origin
            extent = (ox, ox + cols * terrain.resolution,
                      oy, oy + rows * terrain.resolution)
            im = self.ax.imshow(terrain.heightmap, cmap='YlGn_r', origin='lower',
                                 extent=extent, zorder=0)
            self.fig.colorbar(im, ax=self.ax, label='elevation (m)', shrink=0.7)

        self.ax.set_xlim(*xlim)
        self.ax.set_ylim(*ylim)
        self.ax.set_aspect('equal')
        self.ax.set_title("Vehicle Trajectory (top-down)")

        self.marker_series = marker_series
        self.lines = {}
        self.xs = {}
        self.ys = {}
        self.points = {}
        for i, name in enumerate(series):
            style = self._STYLES[i % len(self._STYLES)]
            line, = self.ax.plot([], [], style, label=name, zorder=2)
            self.lines[name] = line
            self.xs[name] = []
            self.ys[name] = []

            is_emphasized = (name == marker_series)
            point, = self.ax.plot(
                [], [], 'o', color=line.get_color(),
                markersize=8 if is_emphasized else 5,
                zorder=4 if is_emphasized else 3,
            )
            self.points[name] = point

        if marker_series is not None and marker_series not in self.lines:
            raise ValueError(f"marker_series '{marker_series}' not in series")
        if self.error_reference is not None and self.error_reference not in self.lines:
            raise ValueError(f"error_reference_series '{self.error_reference}' not in series")

        # Legend moved off the trajectory plot -- lives above the whole
        # figure instead of self.ax.legend(), so it never overlaps data.
        self.fig.legend(
            handles=list(self.lines.values()), labels=list(self.lines.keys()),
            loc='upper center', bbox_to_anchor=(0.5, 1.0),
            ncol=min(len(series), 6), fontsize=9, frameon=True,
        )
        self.fig.subplots_adjust(top=0.90)

        self._setup_yaw_gauge()
        self._setup_bar_gauge(self.pitch_ax, "UP", "DOWN", "Pitch")
        self._setup_bar_gauge(self.roll_ax, "L", "R", "Roll")
        self._setup_elevation_gauge()

        self.yaw_needles = {}
        self.pitch_markers = {}
        self.roll_markers = {}
        self.elev_markers = {}
        for name in series:
            color = self.lines[name].get_color()
            needle, = self.yaw_ax.plot([0, 0], [0, 1], '-', color=color, linewidth=2, zorder=2)
            self.yaw_needles[name] = needle
            pm, = self.pitch_ax.plot([0], [0], marker='^', color=color, markersize=9, zorder=2)
            self.pitch_markers[name] = pm
            rm, = self.roll_ax.plot([0], [0], marker='^', color=color, markersize=9, zorder=2)
            self.roll_markers[name] = rm
            em, = self.elev_ax.plot([0], [elevation_range[0]], marker='<', color=color, markersize=9, zorder=2)
            self.elev_markers[name] = em

        # Error-over-time row
        self.step_count = 0
        self.time_steps = []
        self.pos_error = {name: [] for name in series}
        self.yaw_error = {name: [] for name in series}
        self.pitch_error = {name: [] for name in series}
        self.roll_error = {name: [] for name in series}
        self.pos_error_lines = {}
        self.yaw_error_lines = {}
        self.pitch_error_lines = {}
        self.roll_error_lines = {}

        if self.error_reference is not None:
            error_gs = outer_gs[1, :].subgridspec(1, 4, wspace=0.35)
            self.pos_error_ax = self.fig.add_subplot(error_gs[0, 0])
            self.yaw_error_ax = self.fig.add_subplot(error_gs[0, 1])
            self.pitch_error_ax = self.fig.add_subplot(error_gs[0, 2])
            self.roll_error_ax = self.fig.add_subplot(error_gs[0, 3])

            self._setup_error_axis(self.pos_error_ax, "Position error", "m")
            self._setup_error_axis(self.yaw_error_ax, "Yaw error", "deg")
            self._setup_error_axis(self.pitch_error_ax, "Pitch error", "deg")
            self._setup_error_axis(self.roll_error_ax, "Roll error", "deg")

            for name in series:
                if name == self.error_reference:
                    continue
                color = self.lines[name].get_color()
                l, = self.pos_error_ax.plot([], [], '-', color=color, linewidth=1.3)
                self.pos_error_lines[name] = l
                l, = self.yaw_error_ax.plot([], [], '-', color=color, linewidth=1.3)
                self.yaw_error_lines[name] = l
                l, = self.pitch_error_ax.plot([], [], '-', color=color, linewidth=1.3)
                self.pitch_error_lines[name] = l
                l, = self.roll_error_ax.plot([], [], '-', color=color, linewidth=1.3)
                self.roll_error_lines[name] = l

    def _setup_yaw_gauge(self):
        ax = self.yaw_ax
        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-1.3, 1.3)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.add_patch(plt.Circle((0, 0), 1.0, fill=False, edgecolor='black', linewidth=1))
        ax.text(1.15, 0, "FWD", ha='left', va='center', fontsize=8)
        ax.set_title("Yaw", fontsize=9)

    def _setup_bar_gauge(self, ax, left_label, right_label, title):
        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-0.3, 0.3)
        ax.axis('off')
        ax.plot([-1, 1], [0, 0], '-', color='black', linewidth=1, zorder=1)
        ax.plot([0, 0], [-0.08, 0.08], '-', color='black', linewidth=1, zorder=1)  # center tick
        ax.text(-1.15, 0, left_label, ha='right', va='center', fontsize=8)
        ax.text(1.15, 0, right_label, ha='left', va='center', fontsize=8)
        ax.set_title(title, fontsize=9)

    def _setup_elevation_gauge(self):
        ax = self.elev_ax
        lo, hi = self.elevation_range
        ax.set_xlim(-0.3, 0.3)
        ax.set_ylim(lo, hi)
        ax.axis('off')
        ax.plot([0, 0], [lo, hi], '-', color='black', linewidth=1, zorder=1)
        ax.text(0, lo, f"{lo:g}m", ha='center', va='top', fontsize=8)
        ax.text(0, hi, f"{hi:g}m", ha='center', va='bottom', fontsize=8)
        ax.set_title("Elevation", fontsize=9)

    def _setup_error_axis(self, ax, title, unit):
        ax.set_title(title, fontsize=9)
        ax.set_xlabel(self.error_x_label, fontsize=8)
        ax.set_ylabel(unit, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)

    @staticmethod
    def _wrap_deg(angle_rad):
        """Radians -> degrees, wrapped to (-180, 180]. Needed here (unlike
        the gauges, which are periodic via sin/cos and never needed
        this) because a raw angle difference near +/-180 would otherwise
        show a fake huge error instead of a small wraparound one.
        """
        deg = np.degrees(angle_rad)
        return (deg + 180) % 360 - 180

    def update(self, states, t=None):
        """states: dict mapping series name -> state. state[0:2] is
        (x, y); if len(state) == 6 it's (x, y, z, roll, pitch, yaw) and
        the gauges/error for that series are updated too.

        t: x-axis value for this update's error-plot point. If None,
            falls back to an auto-incrementing integer counter (one per
            update() call) -- the original behavior. Pass real elapsed
            seconds here if you want the error plots' x-axis to mean
            "time" rather than "how many times update() happened to be
            called," which is NOT the same thing once update() calls
            aren't evenly spaced in real time (e.g. one call per
            scheduler event rather than one per unit of sim time).
        """
        artists = []
        for name, state in states.items():
            self.xs[name].append(state[0])
            self.ys[name].append(state[1])
            self.lines[name].set_data(self.xs[name], self.ys[name])
            artists.append(self.lines[name])

            self.points[name].set_data([state[0]], [state[1]])
            artists.append(self.points[name])

            if len(state) == 6:
                _, _, z, roll, pitch, yaw = state

                nx, ny = np.cos(yaw), np.sin(yaw)
                self.yaw_needles[name].set_data([0, nx], [0, ny])
                artists.append(self.yaw_needles[name])

                pitch_frac = np.clip(np.degrees(pitch) / self.pitch_roll_range_deg, -1, 1)
                self.pitch_markers[name].set_data([pitch_frac], [0])
                artists.append(self.pitch_markers[name])

                roll_frac = np.clip(np.degrees(roll) / self.pitch_roll_range_deg, -1, 1)
                self.roll_markers[name].set_data([roll_frac], [0])
                artists.append(self.roll_markers[name])

                z_clipped = np.clip(z, *self.elevation_range)
                self.elev_markers[name].set_data([0], [z_clipped])
                artists.append(self.elev_markers[name])

        if self.error_reference is not None and self.error_reference in states:
            ref_state = states[self.error_reference]
            if len(ref_state) == 6:
                x_val = t if t is not None else self.step_count
                self.time_steps.append(x_val)
                self.step_count += 1

                for name, state in states.items():
                    if name == self.error_reference or len(state) != 6:
                        continue

                    pos_err = np.linalg.norm(np.array(state[0:3]) - np.array(ref_state[0:3]))
                    roll_err = abs(self._wrap_deg(state[3] - ref_state[3]))
                    pitch_err = abs(self._wrap_deg(state[4] - ref_state[4]))
                    yaw_err = abs(self._wrap_deg(state[5] - ref_state[5]))

                    self.pos_error[name].append(pos_err)
                    self.roll_error[name].append(roll_err)
                    self.pitch_error[name].append(pitch_err)
                    self.yaw_error[name].append(yaw_err)

                    self.pos_error_lines[name].set_data(self.time_steps, self.pos_error[name])
                    artists.append(self.pos_error_lines[name])
                    self.yaw_error_lines[name].set_data(self.time_steps, self.yaw_error[name])
                    artists.append(self.yaw_error_lines[name])
                    self.pitch_error_lines[name].set_data(self.time_steps, self.pitch_error[name])
                    artists.append(self.pitch_error_lines[name])
                    self.roll_error_lines[name].set_data(self.time_steps, self.roll_error[name])
                    artists.append(self.roll_error_lines[name])

                # Growing time series need periodic rescaling -- axis
                # limits changing under blit=True is a known-imperfect
                # combination (matplotlib forces a fuller redraw of that
                # axes when limits change), not pure blit performance.
                # Untested on the real TkAgg backend (no display in my
                # sandbox); watch for flicker/slowdown as the run grows.
                for ax in (self.pos_error_ax, self.yaw_error_ax, self.pitch_error_ax, self.roll_error_ax):
                    ax.relim()
                    ax.autoscale_view()

        return artists

    def animate(self, step_fn, interval=50):
        """step_fn(frame) -> iterable of updated artists. Blocks until window closed."""
        self.ani = animation.FuncAnimation(
            self.fig, step_fn, interval=interval, blit=False, cache_frame_data=False
        )
        plt.show()