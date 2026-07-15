"""Defines the simulation visualizer."""

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class SimVisualizer:
    """Plots an arbitrary number of named (x, y) trajectories live.

    Usage:
        viz = SimVisualizer(series=["truth", "odom", "graph"])
        viz.update({"truth": state1, "odom": state2, "graph": state3})
    """

    _STYLES = ['b-', 'r--', 'g--', 'm-', 'c-.', 'y:', 'k-']

    def __init__(self, series, xlim=(-5, 5), ylim=(-1, 9), marker_series=None):
        """marker_series: name of the series to also draw as a leading point
        marker (e.g. the ground-truth trajectory). None disables it.
        """
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(*xlim)
        self.ax.set_ylim(*ylim)
        self.ax.set_aspect('equal')
        self.ax.set_title("Vehicle Trajectory Estimates")

        self.marker_series = marker_series
        self.lines = {}
        self.xs = {}
        self.ys = {}
        for i, name in enumerate(series):
            style = self._STYLES[i % len(self._STYLES)]
            line, = self.ax.plot([], [], style, label=name)
            self.lines[name] = line
            self.xs[name] = []
            self.ys[name] = []

        if self.marker_series is not None:
            if self.marker_series not in self.lines:
                raise ValueError(f"marker_series '{marker_series}' not in series")
            marker_color = self.lines[self.marker_series].get_color()
            self.point, = self.ax.plot([], [], 'o', color=marker_color)

        self.ax.legend(loc='upper right')

    def update(self, states):
        """states: dict mapping series name -> state (x, y, ...)."""
        artists = []
        for name, state in states.items():
            self.xs[name].append(state[0])
            self.ys[name].append(state[1])
            self.lines[name].set_data(self.xs[name], self.ys[name])
            artists.append(self.lines[name])

        if self.marker_series is not None and self.marker_series in states:
            x, y = states[self.marker_series][0], states[self.marker_series][1]
            self.point.set_data([x], [y])
            artists.append(self.point)

        return artists

    def animate(self, step_fn, interval=50):
        """step_fn(frame) -> iterable of updated artists. Blocks until window closed."""
        self.ani = animation.FuncAnimation(
            self.fig, step_fn, interval=interval, blit=True, cache_frame_data=False
        )
        plt.show()