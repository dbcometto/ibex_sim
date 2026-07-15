"""Defines the simulation visualizer (3D vehicle, top-down view)."""

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


class SimVisualizer:
    """Top-down (x, y) view of named trajectories, with an optional
    terrain image drawn underneath, plus graphical gauges for yaw,
    pitch, roll, and elevation.

    Gauges are pre-created for every name in `series`, each colored to
    match that series' trajectory line. Any series whose update() state
    has 6 elements (x, y, z, roll, pitch, yaw) gets its gauges moved;
    2-element (x, y)-only states are simply skipped for the gauges. This
    means overlaying an estimator's attitude later is just a matter of
    including it in `series` and passing a 6-element state for it --
    no visualizer changes needed.

    NOTE: the main plot only shows x, y -- it's a top-down position
    view, not a true 3D renderer.
    """

    _STYLES = ['b-', 'r--', 'g--', 'm:', 'c-.', 'y:', 'k-']

    def __init__(self, series, terrain=None,
                 xlim=(-5, 5), ylim=(-1, 9), marker_series=None,
                 pitch_roll_range_deg=45.0, elevation_range=(0.0, 20.0)):
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
        """
        self.pitch_roll_range_deg = pitch_roll_range_deg
        self.elevation_range = elevation_range

        self.fig = plt.figure(figsize=(10, 6))
        outer_gs = self.fig.add_gridspec(1, 2, width_ratios=[3, 1])
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
            im = self.ax.imshow(terrain.heightmap, cmap='terrain', origin='lower',
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

        self.ax.legend(loc='upper right')

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

    def update(self, states):
        """states: dict mapping series name -> state. state[0:2] is
        (x, y); if len(state) == 6 it's (x, y, z, roll, pitch, yaw) and
        the gauges for that series are updated too.
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

        return artists

    def animate(self, step_fn, interval=50):
        """step_fn(frame) -> iterable of updated artists. Blocks until window closed."""
        self.ani = animation.FuncAnimation(
            self.fig, step_fn, interval=interval, blit=True, cache_frame_data=False
        )
        plt.show()