import numpy as np
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.ticker import MultipleLocator


def line_intersect(p1, p2, P3, P4):
    """ Calculates intersection point of two segments. segment 1: [p1,p2], segment 2: [P3,P4].
    Inputs:
    p1, p2: arrays containing the coordinates of the points of first segments.
    P3, P4: arrays containing the coordinates of the points of second segments
    Outputs: list of X,Y coordinates of the intersection points. Set to np.inf if no intersection.
    """
    p1 = np.atleast_2d(p1)
    p2 = np.atleast_2d(p2)
    P3 = np.atleast_2d(P3)
    P4 = np.atleast_2d(P4)

    x1, y1 = p1[:, 0], p1[:, 1]
    x2, y2 = p2[:, 0], p2[:, 1]
    X3, Y3 = P3[:, 0], P3[:, 1]
    X4, Y4 = P4[:, 0], P4[:, 1]

    D = (Y4 - Y3) * (x2 - x1) - (X4 - X3) * (y2 - y1)

    # Colinearity test
    C = (D != 0)

    # Calculate the distance to the intersection point
    UA = ((X4 - X3) * (y1 - Y3) - (Y4 - Y3) * (x1 - X3))
    UA = np.divide(UA, D, where=C)
    UB = ((x2 - x1) * (y1 - Y3) - (y2 - y1) * (x1 - X3))
    UB = np.divide(UB, D, where=C)

    # Test if intersections are inside each segment
    C = C * (UA > 0) * (UA < 1) * (UB > 0) * (UB < 1)

    # intersection of the point of the two lines
    X = np.where(C, x1 + UA * (x2 - x1), np.inf)
    Y = np.where(C, y1 + UA * (y2 - y1), np.inf)
    return np.stack([X, Y], axis=1)


class Maze:
    """
    A simple 8-maze made of straight walls (line segments)
    """

    def __init__(self, simulation_mode = "esn"):
        self.walls = np.array([
            # Surrounding walls
            [(0, 0), (0, 500)],
            [(0, 500), (300, 500)],
            [(300, 500), (300, 0)],
            [(300, 0), (0, 0)],
            # Bottom hole
            [(100, 100), (200, 100)],
            [(200, 100), (200, 200)],
            [(200, 200), (100, 200)],
            [(100, 200), (100, 100)],
            # Top hole
            [(100, 300), (200, 300)],
            [(200, 300), (200, 400)],
            [(200, 400), (100, 400)],
            [(100, 400), (100, 300)],
            # Moving walls (invisibles) to constraining bot path
            [(0, 250), (100, 200)],
            [(200, 300), (300, 250)]
        ])

        if simulation_mode == "walls":
            self.invisible_walls = True
        else:
            self.invisible_walls = False
            self.walls[12:] = [[(0, 0), (0, 0)],
                               [(0, 0), (0, 0)]]

        self.alternate = None
        self.iter = 0
        self.in_corridor = False

    def draw(self, ax, grid=True, margin=5):
        """
        Render the maze
        """

        # Buidling a filled patch from walls
        V, C, S = [], [], self.walls
        V.extend(S[0 + i, 0] for i in [0, 1, 2, 3, 0])
        V.extend(S[4 + i, 0] for i in [0, 1, 2, 3, 0])
        V.extend(S[8 + i, 0] for i in [0, 1, 2, 3, 0])
        C = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY] * 3
        path = Path(V, C)
        patch = PathPatch(path, clip_on=False, linewidth=1.5,
                          edgecolor="black", facecolor="white")

        # Set figure limits, grid and ticks
        ax.set_axisbelow(True)
        ax.add_artist(patch)
        ax.set_xlim(0 - margin, 300 + margin)
        ax.set_ylim(0 - margin, 500 + margin)
        if grid:
            ax.xaxis.set_major_locator(MultipleLocator(100))
            ax.xaxis.set_minor_locator(MultipleLocator(10))
            ax.yaxis.set_major_locator(MultipleLocator(100))
            ax.yaxis.set_minor_locator(MultipleLocator(10))
            ax.grid(True, "major", color="0.75", linewidth=1.00, clip_on=False)
            ax.grid(True, "minor", color="0.75", linewidth=0.50, clip_on=False)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(axis='both', which='major', size=0)
        ax.tick_params(axis='both', which='minor', size=0)

    def update_walls(self, bot_position):
        """ Add the invisible walls to force the bot alternating right and left direction."""
        if bot_position[1] < 100:
            self.walls[12:] = [[(0, 250), (100, 300)],
                               [(200, 200), (300, 250)]]
        elif bot_position[1] > 400:
            self.walls[12:] = [[(0, 250), (100, 200)],
                               [(200, 300), (300, 250)]]
        else:
            pass

    def update_walls_RR_LL(self, bot_position):
        """ Add the invisible walls to force the bot alternating right and left direction overy other time."""
        if 200 < bot_position[1] < 300:
            if not self.in_corridor:
                if self.iter == 1:
                    self.iter = 0
                else:
                    self.iter += 1
            self.in_corridor = True
        else:
            self.in_corridor = False

        if bot_position[1] < 100 and self.iter < 1:
            self.walls[12:] = [[(0, 250), (100, 300)],
                               [(200, 300), (300, 250)]]

        elif bot_position[1] < 100 and self.iter == 1:
                self.walls[12:] = [[(0, 250), (100, 300)],
                                   [(200, 100), (300, 250)]]

        elif bot_position[1] > 400 and self.iter < 1:
            self.walls[12:] = [[(0, 250), (100, 200)],
                               [(200, 200), (300, 250)]]

        elif bot_position[1] > 400 and self.iter == 1:
            self.walls[12:] = [[(0, 250), (100, 200)],
                               [(200, 300), (300, 250)]]
        else:
            pass
