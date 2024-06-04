import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle
from maze import line_intersect



class Bot:

    def __init__(self, save_bot_sates, sensor_size):
        self.size = 10
        self.position = 150, 250
        self.orientation = 0
        self.n_sensors = 8

        # Direction flag
        self.left_cue = True  # start by going left
        self.right_cue = False
        self.left_cue_prev = True
        self.right_cue_prev = False

        A = np.linspace(-np.pi / 2, +np.pi / 2, self.n_sensors + 2, endpoint=True)[1:-1]
        self.sensors = {
            "angle": A,
            "range": sensor_size*np.ones((self.n_sensors, 1)), # initial sensor size: 60
            "value": np.ones((self.n_sensors, 1))}
        self.sensors["range"][3:5] *= 1.25

        self.save_bot_sates = save_bot_sates
        self.all_orientations = []
        self.all_positions = []
        self.all_sensors_vals = []
        self.all_cues = []

        # For RR-LL
        self.enter_corridor = False
        self.iter_right = 2
        self.iter_left = 0

    def draw(self, ax):
        """Render the bot in the maze."""
        # Sensors
        # Two points per segment
        n = 2 * len(self.sensors["angle"])
        sensors = LineCollection(np.zeros((n, 2, 2)),
                                 colors=["0.75", "0.00"] * n,
                                 linewidths=[0.75, 1.00] * n,
                                 linestyles=["--", "-"] * n)
        # Body
        body = Circle(self.position, self.size, zorder=20, edgecolor="black", facecolor=(1, 1, 1, .75))
        # Head
        P = np.zeros((1, 2, 2))
        P[0, 0] = self.position
        P[0, 1] = P[0, 1] + self.size * np.array([np.cos(self.orientation),
                                                  np.sin(self.orientation)])
        head = LineCollection(P, colors="black", zorder=30)

        # List of artists to be rendered (sensors, body, head)

        # self.artists = [sensors, body, head]
        self.artists = [sensors, body, head]#, direction_KNN]

        ax.add_collection(sensors)
        ax.add_artist(body)
        ax.add_artist(head)

    def set_wall_constraints(self):
        """Imposes restrictions to confine the bot within the walls."""
        x, y = self.position
        size = self.size
        max_x = 300
        min_x = 0
        max_y = 500
        min_y = 0

        # Constrain horizontal movement
        if x + size > max_x:
            x = max_x - size
        elif x - size < min_x:
            x = min_x + size

        # Constrain vertical movement
        if y + size > max_y:
            y = max_y - size
        elif y - size < min_y:
            y = min_y + size

        # Constrain movement around specific walls
        if 100 - size < x < 200 + size:
            if 100 - size < y < 200 + size:  # Bottom walls
                y = max(100 + size, min(y, 200 - size))
                x = max(100 + size, min(x, 200 - size))
            elif 300 - size < y < 400 + size:  # Top walls
                y = max(300 + size, min(y, 400 - size))
                x = max(100 + size, min(x, 200 - size))

        self.position = (x, y)

    def compute_orientation(self):
        """ Calculates the orientation of the bot accoridng to the sensor values."""
        dv = (self.sensors["value"].ravel() * [-4, -3, -2, -1, 1, 2, 3, 4]).sum()
        if abs(dv) > 0.01:  # if 75 sensor size
            self.orientation += 0.015 * dv

    def update_position(self):
        """Updates the position of the bot according to the calculated orientation."""
        self.position += 2 * np.array([np.cos(self.orientation), np.sin(self.orientation)])
        #self.position += np.random.normal(0, 1) * 0.7
        self.set_wall_constraints()

    def update(self, maze, cues):
        """ Update the bot's position and orientation in the maze """
        sensors, body, head = self.artists

        # Sensors
        verts = sensors.get_segments()
        linewidths = sensors.get_linewidth()

        # all angles of the sensors
        A = self.sensors["angle"] + self.orientation

        # cos and sin of the sensors
        T = np.stack([np.cos(A), np.sin(A)], axis=1)

        P1 = self.position + self.size * T
        P2 = P1 + self.sensors["range"] * T
        P3, P4 = maze.walls[:, 0], maze.walls[:, 1]

        for i, (p1, p2) in enumerate(zip(P1, P2)):
            verts[2 * i] = verts[2 * i + 1] = p1, p2
            linewidths[2 * i + 1] = 1
            C = line_intersect(p1, p2, P3, P4)
            index = np.argmin(np.sum((C - p1) ** 2, axis=1))
            p = C[index]
            if p[0] < np.inf:
                verts[2 * i + 1] = p1, p
                self.sensors["value"][i] = np.sqrt(np.sum((p1 - p) ** 2))
                self.sensors["value"][i] /= self.sensors["range"][i]
            else:
                self.sensors["value"][i] = 1

        sensors.set_verts(verts)
        sensors.set_linewidths(linewidths)

        # Update body
        body.set_center(self.position)

        # Update head
        head_verts = np.array([self.position, self.position + self.size * np.array([np.cos(self.orientation),
                                                                                    np.sin(self.orientation)]).reshape(2,)])
        head.set_verts(np.expand_dims(head_verts, axis=0))


        if self.save_bot_sates:
            self.all_orientations.append(self.orientation)
            self.all_sensors_vals.append(self.sensors['value'].ravel())
            self.all_positions.append(self.position)
            if cues:
                self.all_cues.append([int(self.right_cue), int(self.left_cue)])

    def update_cues(self, task):
        if task == 'R-L':
            if (0 <= self.position[0] <= 300) and (200 <= self.position[1] <= 300):
                if self.left_cue_prev:
                    self.right_cue = True
                    self.left_cue = False
                elif self.right_cue_prev:
                    self.left_cue = True
                    self.right_cue = False
                cues = [self.right_cue, self.left_cue]
            elif (200 < self.position[0] <= 300) and (self.position[1] < 200 or self.position[1] > 300):
                if self.left_cue:
                    self.left_cue_prev = True
                    self.right_cue_prev = False
                    self.left_cue = False
                elif self.right_cue:
                    self.right_cue_prev = True
                    self.left_cue_prev = False
                    self.right_cue = False
                cues = [self.right_cue, self.left_cue]
            else:
                cues = [0, 0]
        elif task == 'RR-LL':
            if 200 < self.position[1] < 300:
                if not self.enter_corridor:
                    if self.iter_right == 1 or self.iter_left == 2:
                        self.iter_right += 1
                        self.iter_left = 0
                        self.right_cue = True
                        self.left_cue = False
                    elif self.iter_left == 1 or self.iter_right == 2:
                        self.iter_left += 1
                        self.iter_right = 0
                        self.right_cue = False
                        self.left_cue = True
                self.enter_corridor = True
                if 0 < self.iter_right <= 2 and self.iter_left == 0:
                    self.right_cue = True
                    self.left_cue = False
                elif 0 < self.iter_left <= 2 and self.iter_right == 0:
                    self.right_cue = False
                    self.left_cue = True
                cues = [self.right_cue, self.left_cue]
            else:
                cues = [0, 0]
                self.enter_corridor = False
        return cues





