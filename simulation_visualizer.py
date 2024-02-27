import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


class SimulationVisualizer():
    """This class visualizes bot simulations within a maze,
    constructing the environment, maze, and displaying
    the bot and its sensor rays.
    It also enables plotting sensor values over time."""

    def __init__(self, n_sensors):
        self.fig = plt.figure(figsize=(10, 5), frameon=False)
        self.G = GridSpec(8, 2, width_ratios=(1, 2))
        self.ax =plt.subplot(self.G[:, 0], aspect=1, frameon=False)
        self.P = np.zeros((500, 2))
        self.plots = []
        self.n_sensors = n_sensors
        # Sensor plots
        self.trace, = self.ax.plot([], [], color="0.5", zorder=10, linewidth=1, linestyle=":")
        for i in range(n_sensors):
            ax = plt.subplot(self.G[i, 1])
            ax.set_xticks([]), ax.set_yticks([])
            ax.set_ylabel("Sensor %d" % (i + 1), fontsize="x-small")
            plot, = ax.plot([], [], linewidth=0.75)
            ax.set_xlim(0, 500)
            ax.set_ylim(0, 1.1)
            self.plots.append(plot)
        self.X = np.arange(500)
        self.Y = np.zeros((n_sensors, 500))

    def update_plot(self, frame, bot_position, bot_sensors_val):
        if frame < 500:
            self.P[frame] = bot_position
            self.trace.set_xdata(self.P[:frame, 0])
            self.trace.set_ydata(self.P[:frame, 1])
            for i in range(self.n_sensors):
                self.Y[i, frame] = bot_sensors_val[i]
                self.plots[i].set_ydata(self.Y[i, :frame])
                self.plots[i].set_xdata(self.X[:frame])
        else:
            self.P[:-1] = self.P[1:]
            self.P[-1] = bot_position
            self.trace.set_xdata(self.P[:, 0])
            self.trace.set_ydata(self.P[:, 1])
            self.Y[:, 0:-1] =self.Y[:, 1:]
            for i in range(self.n_sensors):
                self.Y[i, -1] = bot_sensors_val[i]
                self.plots[i].set_ydata(self.Y[i])
                self.plots[i].set_xdata(self.X)

