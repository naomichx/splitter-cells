"""
Reservoir state analysis is conducted at the single-cell level and aims at understanding
the neural dynamics during the bot's navigation. In this script, the reservoir states
beforehand recorded during the bot's navigation are loaded to process to the population analysis.

 The script allows several analytical processes:

- SI Index Computation: Calculates the Selectivity Index (SI) of neurons corresponding to different bot trajectories.

- Place Cells, Head-Direction Cells, and Splitter Cells Activity Plotting: Visualizes the activity patterns of place cells, head-direction cells, and splitter cells, providing insights into spatial and directional encoding in the reservoir.

- Mean Firing Rate Plotting: Plots the mean firing rate of individual neurons during both correct and error trials,
 highlighting different neural activity dynamics in different behavioral contexts.

"""
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import seaborn as sns
plt.rc('font', size=12)


def load_positions(path):
    """Load position data from a specified path.

        Args:
        path (str): The path to the directory containing the position data file.

        Returns:
        numpy.ndarray: An array containing position data.
        """
    return np.load(path + 'positions.npy')


def load_reservoir_states(path):
    """Load reservoir state data from a specified path.

        Args:
        path (str): The path to the directory containing the reservoir state data file.

        Returns:
        numpy.ndarray: An array containing reservoir state data.
        """
    return np.load(path + 'reservoir_states.npy')


def load_orientations(path):
    """Load orientation data from a specified path.

        Args:
        path (str): The path to the directory containing the orientation data file.

        Returns:
        numpy.ndarray: An array containing orientation data.
        """
    return np.load(path + 'output.npy')


def find_location_indexes(y_positions):
    """
        This function identifies location indexes ('m' for middle, 'r' for right, 'l' for left)
        based on the provided y positions. It iterates through the positions and determines
        the location based on specific thresholds.

        Args:
        y_positions (numpy.ndarray): An array containing y positions.

        Returns:
        tuple: A tuple containing two lists:
            - locations: A list of location identifiers ('m', 'r', 'l').
            - locations_indexes: A list of corresponding indexes for each identified location.
        """
    flag_middle = False
    flag_left = False
    flag_right = False
    locations = []
    locations_indexes = []
    for i, pos in enumerate(y_positions):
        if 200 < pos < 300 and not flag_middle:
            locations.append('m')
            locations_indexes.append(i)
            flag_middle = True
            flag_left = False
            flag_right = False
        elif pos < 200 and not flag_right:
            locations.append('r')
            locations_indexes.append(i)
            flag_middle = False
            flag_right = True
            flag_left = False
        elif pos > 300 and not flag_left:
            locations.append('l')
            locations_indexes.append(i)
            flag_middle = False
            flag_right = False
            flag_left = True
    return locations, locations_indexes


def find_activity_ranges(locations, locations_indexes):
    """
    This function determines the ranges of activity indexes for different pathways in the central corridor,
    including:
    - Right to left (R-L)
    - Left to right (L-R)
    - Left to left (L-L)
    - Right to right (R-R)

    Args:
    locations (list): A list of location identifiers ('m', 'r', 'l').
    locations_indexes (list): A list of corresponding indexes for each identified location.

    Returns:
    dict: A dictionary containing activity ranges for each pathway:
        - 'RL': Activity ranges for right to left pathway.
        - 'LR': Activity ranges for left to right pathway.
        - 'RR': Activity ranges for right to right pathway.
        - 'LL': Activity ranges for left to left pathway.
        - 'r_loop': Activity ranges for the right loop.
        - 'l_loop': Activity ranges for the left loop.
    """

    activity_ranges = {'RL': [], 'LR': [], 'RR': [], 'LL': [], 'r_loop':[], 'l_loop':[]}
    for i in range(1, len(locations) - 1):
        if locations[i] == 'm':
            if locations[i-1] == 'r':
                activity_ranges['r_loop'].append([locations_indexes[i-1], locations_indexes[i]])
                if locations[i+1] == 'r':
                    activity_ranges['RR'].append([locations_indexes[i], locations_indexes[i]+100])
                elif locations[i+1] == 'l':
                    activity_ranges['RL'].append([locations_indexes[i], locations_indexes[i]+100])
            elif locations[i-1] == 'l':
                activity_ranges['l_loop'].append([locations_indexes[i - 1], locations_indexes[i]])
                if locations[i+1] == 'r':
                    activity_ranges['LR'].append([locations_indexes[i], locations_indexes[i]+100])
                elif locations[i+1] == 'l':
                    activity_ranges['LL'].append([locations_indexes[i], locations_indexes[i]+100])
    return activity_ranges


def get_activity_ranges(path):
    """ This function retrieves activity ranges for different pathways in the central corridor
        based on the position data obtained from the specified path.

        Args:
        path (str): The path to the directory containing the position data file.

        Returns:
        dict: A dictionary containing activity ranges for each pathway:
            - 'RL': Activity ranges for right to left pathway.
            - 'LR': Activity ranges for left to right pathway.
            - 'RR': Activity ranges for right to right pathway.
            - 'LL': Activity ranges for left to left pathway.
            - 'r_loop': Activity ranges for the right loop.
            - 'l_loop': Activity ranges for the left loop.
        """
    positions = load_positions(path)
    y_positions = positions[:, 1]
    locations, locations_indexes = find_location_indexes(y_positions)
    activity_ranges = find_activity_ranges(locations, locations_indexes)
    return activity_ranges


def get_average_activity(range, reservoir_states):
    """This function calculates the average activity of reservoir states within the specified range.

        Args:
        range: A array containing the start and end indices of the range.
        reservoir_states (numpy.ndarray): An array containing reservoir state data.

        Returns:
        float: The average activity of reservoir states within the specified range.
        """
    selected_states = reservoir_states[range[0]:range[1]]
    return np.mean(selected_states, axis=0)


def compute_SI(vector1, vector2):
    """Compute the Selectivity Index (SI) of two vectors.

    The Selectivity Index (SI) measures the selectivity of two vectors.
    It is calculated as the absolute difference divided by the sum of the vectors.

    Args:
    vector1 (numpy.ndarray): First vector.
    vector2 (numpy.ndarray): Second vector.

    Returns:
    numpy.ndarray: The computed Selectivity Index (SI) for each element.
    """
    diff = vector1 - vector2
    sum_ = vector1 + vector2
    SI = np.abs(diff / sum_)
    return SI


def find_splitter_cells(vector1, vector2, th):
    """Find splitter cells based on the Selectivity Index (SI) of two vectors.

    This function computes the Selectivity Index (SI) of two vectors and returns the indices
    of neurons that have an SI greater than or equal to the specified threshold (th),
    indicating splitter cells.

    Args:
    vector1 (numpy.ndarray): First vector.
    vector2 (numpy.ndarray): Second vector.
    th (float): Threshold value for the Selectivity Index.

    Returns:
    numpy.ndarray: Indices of splitter cells based on the Selectivity Index (SI).
    """
    SI = compute_SI(vector1, vector2)
    sorted_neurons = np.argsort(-SI)
    splitter_cells = sorted_neurons[SI[sorted_neurons] >= th]
    return splitter_cells


def plot_activity(ax, to_plot, act_1, act_2, act_3, label_1, label_2, label_3, title, colors):
    """This function plots the mean activity of neurons for three different trajectories,
        with each condition represented by a different color.

        Args:
        ax (matplotlib.axes.Axes): The axes object to draw the plot on.
        to_plot (numpy.ndarray): Array containing indices of neurons to plot.
        act_1 (numpy.ndarray): Array containing mean activity values for trajectory 1.
        act_2 (numpy.ndarray): Array containing mean activity values for trajectory 2.
        act_3 (numpy.ndarray): Array containing mean activity values for trajectory 3.
        label_1 (str): Label for trajectory 1.
        label_2 (str): Label for trajectory 2.
        label_3 (str): Label for trajectory 3.
        title (str): Title of the plot.
        colors (dict): Dictionary containing colors for different trajectories.

        Returns:
        None
        """
    width = 0.3
    x = np.arange(len(to_plot))
    ax.bar(x, act_1[to_plot], width, color=colors[label_1], label=label_1)
    ax.bar(x - 0.3, act_2[to_plot], width, color=colors[label_2], label=label_2)
    ax.bar(x + 0.3, act_3[to_plot], width, color=colors[label_3], label=label_3)#, alpha=0.5)

    ax.set_xticks(x, to_plot)
    ax.set_title(title)
    ax.set_xlabel('Neuron index')
    ax.set_ylabel('Mean activity')

    plt.tight_layout()
    return


def homogenous_poisson(rate, tmax, bin_size):
    """This function generates spike trains from a homogenous Poisson process
       with a constant firing rate.

       Args:
       rate (float): Firing rate of the Poisson process (spikes per second).
       tmax (float): Maximum time duration of the spike train (in seconds).
       bin_size (float): Size of the time bins for discretization (in seconds).

       Returns:
       numpy.ndarray: Spike train generated from the Poisson process.
       """
    nbins = np.floor(tmax/bin_size).astype(int)
    prob_of_spike = np.repeat(rate * bin_size, 10)
    spikes = np.random.rand(nbins) < prob_of_spike
    return spikes * 1


## Final function
def plot_splitter_cells_count():
    """
    This function computes and visualizes the proportion of prospective and retrospective cells
    in models with and without cues, based on the mean activities of the reservoir in the central corridor.

    Args:
        mean_activities_nc (dict): Mean activities of the reservoir in the central corridor
                                    for the model without cues.
        mean_activities_c (dict): Mean activities of the reservoir in the central corridor
                                   for the model with cues.
    """

    path_nc = "../data/RR-LL/no_cues/reservoir_states/"
    path_c = "../data/RR-LL/cues/reservoir_states/"

    res_activity_nc, res_activity_c = load_reservoir_states(path_nc), load_reservoir_states(path_c)
    activity_ranges_nc, activity_ranges_c = get_activity_ranges(path_nc), get_activity_ranges(path_c)
    mean_activities_nc, mean_activities_c = {}, {}

    for trajectory in ('RL', 'LR', 'RR', 'LL'):
        mean_activities_nc[trajectory] = get_average_activity(activity_ranges_nc[trajectory][1], res_activity_nc)

    for trajectory in ('RL', 'LR', 'RR', 'LL'):
        mean_activities_c[trajectory] = get_average_activity(activity_ranges_c[trajectory][1], res_activity_c)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 5))
    fig.suptitle('Proportion of prospective and retrospective cells')

    # Compute the prospective cells and splitter cells
    pro_nc = len(find_splitter_cells(mean_activities_nc['RL'], mean_activities_nc['RR'], 0.1))
    retro_nc = len(find_splitter_cells(mean_activities_nc['RL'], mean_activities_nc['LL'], 0.1))

    pro_c = len(find_splitter_cells(mean_activities_c['RL'], mean_activities_c['RR'], 0.1))
    retro_c = len(find_splitter_cells(mean_activities_c['RL'], mean_activities_c['LL'], 0.1))

    splitter_nc = [pro_nc * 100 / (pro_nc + retro_nc), retro_nc * 100 / (pro_nc + retro_nc)]
    splitter_c = [pro_c * 100 / (pro_c + retro_c), retro_c * 100 / (pro_c + retro_c)]

    ax1.pie(splitter_nc,  colors=['cornflowerblue', 'tomato'], autopct='%1.1f%%', startangle=90)
    ax2.pie(splitter_c,  colors=['cornflowerblue', 'tomato'], autopct='%1.1f%%', startangle=90)
    ax2.set_title('Model with cues')
    ax1.set_title('Model without cues')

    custom_lines = [Line2D([0], [0], color='cornflowerblue', lw=4),
                    Line2D([0], [0], color='tomato', lw=4)]
    plt.legend(custom_lines, ['Prospective', 'Retrospective'], bbox_to_anchor=(1, 1.01))
    fig.tight_layout()
    plt.show()


def plot_splitter_cells_activity():
    """
    This function visualizes the activity of splitter, retrospective, and prospective cells
        based on their mean activities in different trajectories within the central corridor.

        Returns:
        None
        """
    path = "../data/RR-LL/no_cues/reservoir_states/"
    res_activity = load_reservoir_states(path)
    activity_ranges = get_activity_ranges(path)
    index_range = 2

    mean_activities = {}
    for trajectory in ('RL', 'LR', 'RR', 'LL'):
        mean_activities[trajectory] = get_average_activity(activity_ranges[trajectory][index_range], res_activity)

    mean_activities['l_loop'] = get_average_activity(activity_ranges['l_loop'][index_range], res_activity)
    mean_activities['r_loop'] = get_average_activity(activity_ranges['r_loop'][index_range], res_activity)
    mean_activities['outside_corridor'] = (mean_activities['l_loop'] + mean_activities['r_loop']) / 2

    # Take the 6th neurons related to the biggest SI
    splitter_retro = find_splitter_cells(mean_activities['RL'], mean_activities['LL'], 0.1)[:6]
    splitter_pro = find_splitter_cells(mean_activities['LR'], mean_activities['LL'], 0.1)[:6]

    colors = {}
    colors['R-L'] = 'C0'
    colors['L-L'] = 'C7'
    colors['L-R'] = 'C3'
    colors['R-R'] = 'C7'
    colors['outside corridor'] = 'darkgreen'

    custom_lines = [Line2D([0], [0], color=colors['R-L'], lw=4),
                    Line2D([0], [0], color=colors['L-R'], lw=4),
                    Line2D([0], [0], color=colors['L-L'], lw=4),
                    Line2D([0], [0], color='darkgreen', lw=4), ]

    # Create 2x2 sub plots
    fig = plt.figure(figsize=(14, 9))
    gs = gridspec.GridSpec(2, 2)
    ax = fig.add_subplot(gs[0, :])  # row 0, col 0

    plot_activity(ax, np.concatenate((splitter_retro, splitter_pro)), mean_activities['LR'], mean_activities['RL'],
                  mean_activities['outside_corridor'], label_1='L-R', label_2='R-L', label_3='outside corridor',
                  title='Splitter cells', colors=colors)
    plt.legend(custom_lines, ['R-L', 'L-R', 'L-L', 'Outside corridor'], bbox_to_anchor=(1, 1.01))

    ax = fig.add_subplot(gs[1, 0])  # row 0, col 1
    plot_activity(ax, splitter_retro, mean_activities['LL'], mean_activities['RL'],
                  mean_activities['outside_corridor'], label_1='L-L', label_2='R-L', label_3='outside corridor',
                  title='Retrospective cells', colors=colors)

    ax = fig.add_subplot(gs[1, 1])  # row 1, span all columns
    plot_activity(ax, splitter_pro, mean_activities['LL'], mean_activities['LR'],
                  mean_activities['outside_corridor'], label_1='L-L', label_2='L-R', label_3='outside corridor',
                  title='Prospective cells', colors=colors)

    plt.tight_layout()
    plt.show()


def splitter_cells_activity_compare_level_of_training():
    """
    This function visualizes the activity of splitter, retrospective, and prospective cells
        based on their mean activities in different trajectories within the central corridor.

        Returns:
        None
        """
    path = "../data/R-L_60/cues/reservoir_states/"
    res_activity = load_reservoir_states(path)
    activity_ranges = get_activity_ranges(path)
    index_range = 0

    mean_activities_1 = {}
    for trajectory in ('RL', 'LR'):
        mean_activities_1[trajectory] = get_average_activity(activity_ranges[trajectory][index_range], res_activity)

    mean_activities_1['l_loop'] = get_average_activity(activity_ranges['l_loop'][index_range], res_activity)
    mean_activities_1['r_loop'] = get_average_activity(activity_ranges['r_loop'][index_range], res_activity)
    mean_activities_1['outside_corridor'] = (mean_activities_1['l_loop'] + mean_activities_1['r_loop']) / 2

    # Take the 6th neurons related to the biggest SI
    splitter_1 = find_splitter_cells(mean_activities_1['RL'], mean_activities_1['LR'], 0.1)
    print(len(splitter_1))

    path = "../data/R-L_60/cues/reservoir_states_under_trained/"
    res_activity = load_reservoir_states(path)
    activity_ranges = get_activity_ranges(path)
    mean_activities_2 = {}
    for trajectory in ('RL', 'LR'):
        mean_activities_2[trajectory] = get_average_activity(activity_ranges[trajectory][index_range], res_activity)

    mean_activities_2['l_loop'] = get_average_activity(activity_ranges['l_loop'][index_range], res_activity)
    mean_activities_2['r_loop'] = get_average_activity(activity_ranges['r_loop'][index_range], res_activity)
    mean_activities_2['outside_corridor'] = (mean_activities_2['l_loop'] + mean_activities_2['r_loop']) / 2

    splitter_2 = find_splitter_cells(mean_activities_2['RL'], mean_activities_2['LR'], 0.1)

    colors = {}
    colors['R-L'] = 'C0'
    colors['L-L'] = 'C7'
    colors['L-R'] = 'C3'
    colors['R-R'] = 'C7'
    colors['outside corridor'] = 'darkgreen'

    custom_lines = [Line2D([0], [0], color=colors['R-L'], lw=4),
                    Line2D([0], [0], color=colors['L-R'], lw=4),
                    Line2D([0], [0], color=colors['L-L'], lw=4),
                    Line2D([0], [0], color='darkgreen', lw=4), ]

    # Create 2x2 sub plots
    fig, ax = plt.subplots(2, 1, figsize=(10, 6))

    difference_1 = [abs(lr - rl) for lr, rl in zip(mean_activities_1['LR'], mean_activities_1['RL'])]
    difference_2 = [abs(lr - rl) for lr, rl in zip(mean_activities_2['LR'], mean_activities_2['RL'])]

    all = []
    for split in splitter_1:
        #print((difference_1[split]-difference_2[split])*100/difference_1[split], '%')
        all.append((difference_1[split]-difference_2[split])*100/difference_1[split])

    plot_activity(ax[0], splitter_1[:10], mean_activities_1['LR'], mean_activities_1['RL'],
                  mean_activities_1['outside_corridor'], label_1='L-R', label_2='R-L', label_3='outside corridor',
                  title='Splitter cells over trained', colors=colors)

    plot_activity(ax[1], splitter_1[:10], mean_activities_2['LR'], mean_activities_2['RL'],
                  mean_activities_2['outside_corridor'], label_1='L-R', label_2='R-L', label_3='outside corridor',
                  title='Splitter cells under-trained', colors=colors)
    plt.legend(custom_lines, ['R-L', 'L-R', 'L-L', 'Outside corridor'], bbox_to_anchor=(1, 1.01))

    plt.tight_layout()
    plt.show()

def plot_hippocampal_cells_9():
    """This function visualizes the activity of hippocampal cells in different regions
        including loop cells, corner cells, and place cells.

        Returns:
        None
        """

    def plot_place_cells(axes, place_cells, line, res_activity, positions, start_path, end_path):
        """This function plots the activity of place cells during the 8-trajectory, where each subplot
            represents the activity of a specific place cell.

            Args:
            axes (list): List of matplotlib axes objects to draw the subplots on.
            place_cells (list): List of indices corresponding to place cells.
            line (int): Determines the position of the X-axis label.
                        - 1: X-axis label is shown only on the last subplot.
                        - 2: X-axis label is shown on all subplots.
            res_activity (numpy.ndarray): Array containing reservoir activity data.
            positions (numpy.ndarray): Array containing position data.

            Returns:
            matplotlib.collections.PathCollection: Scatter plot of place cell activity.
            """
        assert len(axes) == len(place_cells)

        x = positions[start_path:end_path:, 0]
        y = positions[start_path:end_path:, 1]
        cmap = plt.get_cmap('coolwarm')
        for i in range(len(place_cells)):
            z = res_activity[start_path:end_path, place_cells[i]]
            scat = axes[i].scatter(x, y, c=z, s=200, cmap=cmap, linewidth=0.1, alpha=0.5)
            axes[i].set_title('Neuron {}'.format(place_cells[i]), fontsize=14)
            if line == 2:
                axes[i].set_xlabel('X', fontsize=14)
            axes[i].margins(0.05)
            axes[i].set_yticks([])
            axes[i].set_xticks([])
            axes[i].set_facecolor('#f0f0f0')
            axes[i].tick_params(axis='both', which='major', labelsize=12)
        axes[0].set_ylabel('Y', fontsize=14)
        return scat


    def create_subtitle(fig, grid, title):
        "Sign sets of subplots with title"
        row = fig.add_subplot(grid)
        # the '\n' is important
        row.set_title(f'{title}\n', fontweight='semibold')
        # hide subplot
        row.set_frame_on(False)
        row.axis('off')

    fig = plt.figure(layout='constrained', figsize=(9,9))
    grid = fig.add_gridspec(3, 3, wspace=0, hspace=0)

    path = "../data/RR-LL/no_cues/reservoir_states/"
    positions = load_positions(path)
    res_activity = load_reservoir_states(path)

    loop_cells = [33, 383, 549]
    corner_cells = [550, 305, 257]
    place_cells = [70, 186, 196]
    all_cells = [loop_cells, corner_cells, place_cells]
    start_path = 364
    end_path = 1089


    for i in range(3):
        axes = []
        for j in range(3):
            axes.append(fig.add_subplot(grid[i, j]))
        scat = plot_place_cells(axes=axes, place_cells=all_cells[i],line=i,
                                res_activity=res_activity, positions=positions,
                                start_path=start_path, end_path=end_path)

    create_subtitle(fig, grid[0, ::], 'Loop cells')
    create_subtitle(fig, grid[1, ::], 'Corner cells')
    create_subtitle(fig, grid[2, ::], 'Place cells')

    cb = plt.colorbar(scat, ax=axes[:], orientation="horizontal")
    # Define custom ticks and labels
    ticks = [np.min(scat.get_array()), (np.max(scat.get_array())+np.min(scat.get_array()))/2, np.max(scat.get_array())]
    tick_labels = ['min', 'medium', 'max']
    cb.set_ticks(ticks)
    cb.set_ticklabels(tick_labels)

    cb.set_label('Activity Level', fontsize=14)
    cb.solids.set_alpha(1)
    plt.show()




def plot_hippocampal_cells_3():
    """This function visualizes the activity of hippocampal cells for neurons 196, 257, and 383.

    Returns:
    None
    """

    def plot_place_cells(ax, place_cells, line, res_activity, positions, start_path, end_path):
        x = positions[start_path:end_path, 0]
        y = positions[start_path:end_path, 1]
        cmap = plt.get_cmap('coolwarm')

        z = res_activity[start_path:end_path, place_cells]
        scat = ax.scatter(x, y, c=z, s=200, cmap=cmap, linewidth=0.1, alpha=0.5)

        ax.set_title('Neuron {}'.format(place_cells), fontsize=14)

        if line == 2:
            ax.set_xlabel('X', fontsize=14)

        ax.set_ylabel('Y', fontsize=14)
        ax.margins(0.05)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_facecolor('#f0f0f0')
        ax.tick_params(axis='both', which='major', labelsize=12)

        return scat
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))  # Increased figure width
    path = "../data/RR-LL/no_cues/reservoir_states/"
    positions = load_positions(path)
    res_activity = load_reservoir_states(path)

    neurons = [196, 257, 383]
    start_path = 364
    end_path = 1089

    for ax, neuron in zip(axes, neurons):
        plot_place_cells(ax=ax, place_cells=neuron, line=2,
                         res_activity=res_activity, positions=positions,
                         start_path=start_path, end_path=end_path)

    axes[0].set_title('Place cells')
    axes[1].set_title('Corner cells')
    axes[2].set_title('Loop cells')

    cbar_ax = fig.add_axes([0.87, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    cb = plt.colorbar(ax.collections[0], cax=cbar_ax)
    cb.set_label('Activity level', fontsize=12, rotation=90,
                 labelpad=5)  # Set label at the top with horizontal orientation
    cb.ax.tick_params(labelsize=10)

    # Define custom ticks and labels
    ticks = [np.min(ax.collections[0].get_array()),
             (np.max(ax.collections[0].get_array()) + np.min(ax.collections[0].get_array())) / 2,
             np.max(ax.collections[0].get_array())]
    tick_labels = ['min', 'medium', 'max']
    cb.set_ticks(ticks)
    cb.set_ticklabels(tick_labels)

    cb.solids.set_alpha(1)

    plt.tight_layout()
    plt.subplots_adjust(right=0.85)
    plt.show()


def plot_head_direction_cells():
    """This function visualizes the activity of head direction cells by correlating
        the reservoir states with the orientation data.

        Returns:
        None
        """
    path = "../data/RR-LL/no_cues/reservoir_states/"
    res_activity = load_reservoir_states(path)
    orientations = load_orientations(path)

    res_activity = np.transpose(res_activity)
    corr_array = []
    for i in range(len(res_activity)):
        res = res_activity[i]
        corr_array.append(np.corrcoef(res, np.squeeze(orientations))[0][1])
    indexes = np.argsort(corr_array)
    most_correlated = indexes[:5]

    inverted_res_activity = np.mean(res_activity) - res_activity

    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.set_xlabel('time (s)')
    ax1.set_ylabel('Orientation', color='orange')
    ax1.plot(orientations, color='orange', markersize=20, linewidth=7, alpha=0.5)
    ax1.tick_params(axis='y', labelcolor='orange')

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'black'

    # Move the ax2.set_ylabel outside of the loop to set it only once
    ax2.set_ylabel('Most correlated cells', color=color)

    for neuron in most_correlated:
        ax2.plot(np.mean(res_activity[neuron]) - res_activity[neuron], color=color, linewidth=0.5, alpha=1)

    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Head direction cells')
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()


def raster_plot():
    """Generate raster plots for retrospective and prospective cells.

    This function generates raster plots for retrospective and prospective cells,
    showing the spike timings of selected neurons for each trajectory.

    Returns:
    None"""
    path = "../data/RR-LL/no_cues/reservoir_states/"
    res_activity = load_reservoir_states(path)
    activity_ranges = get_activity_ranges(path)
    max_firing_rate = 200 / 1000  # Frequency
    bin_size = 1  # time unit: ms
    tmax = 1000  # time_bin= 100 time steps. 100 time steps = 100000 ms.

    colors = {}
    colors['RL'] = 'C0'
    colors['LL'] = 'C7'
    colors['LR'] = 'C3'
    colors['RR'] = 'C8'

    mean_activities = {}
    for trajectory in ('RL', 'LR', 'RR', 'LL'):
        mean_activities[trajectory] = get_average_activity(activity_ranges[trajectory][2], res_activity)

    mean_activities['l_loop'] = get_average_activity(activity_ranges['l_loop'][2], res_activity)
    mean_activities['r_loop'] = get_average_activity(activity_ranges['r_loop'][2], res_activity)
    mean_activities['outside_corridor'] = (mean_activities['l_loop'] + mean_activities['r_loop']) / 2

    splitter_retro = find_splitter_cells(mean_activities['RL'], mean_activities['LL'], 0.1)[0:5]
    splitter_pro = find_splitter_cells(mean_activities['RL'], mean_activities['RR'], 0.1)[0:5]

    # selection of splitter cells by hand
    splitter_pro = [850, 920, 738, 350, 742]
    splitter_retro = [148, 891, 752, 254, 723]

    fig, ax = plt.subplots(2, 2, figsize=(8, len(splitter_retro)), sharex=True)

    plt.suptitle('Retrospective cells                                             Prospective cells')

    for i, cell in enumerate(splitter_retro):
        act = res_activity[:, cell]
        activity_retro = {}
        activity_norm = {}
        firing_rates = {}
        spikes = {}
        spike_times = {}
        for j, trajectory in enumerate(('RL', 'LL')):
            activity_retro[trajectory] = act[activity_ranges[trajectory][1][0]: activity_ranges[trajectory][1][1]]
            scaler = MinMaxScaler()
            activity_norm[trajectory] = scaler.fit_transform(activity_retro[trajectory].reshape(-1, 1))
            firing_rates[trajectory] = max_firing_rate * (activity_norm[trajectory][:100] -
                                                          np.median(activity_norm[trajectory]))
            spikes[trajectory] = homogenous_poisson(firing_rates[trajectory], tmax, bin_size)
            spike_times[trajectory] = np.argwhere(spikes[trajectory] == 1)
            ax[j, 0].vlines(spike_times[trajectory], i - 0.3, i + 0.3, color=colors[trajectory])
            ax[1, 0].set_xlabel('Time')
            ax[j, 0].set_ylabel('Neuron')
            ax[j, 0].set_yticks(np.arange(len(splitter_retro)))
            ax[j, 0].set_yticklabels(splitter_retro)
            ax[j, 0].set_title('{}'.format(trajectory) + ' trajectory')

    for i, cell in enumerate(splitter_pro):
        act = res_activity[:, cell]
        activity_pro = {}
        activity_norm = {}
        firing_rates = {}
        spikes = {}
        spike_times = {}
        for j, trajectory in enumerate(('LR', 'LL')):
            activity_pro[trajectory] = act[activity_ranges[trajectory][1][0]: activity_ranges[trajectory][1][1]]
            scaler = MinMaxScaler()
            activity_norm[trajectory] = scaler.fit_transform(activity_pro[trajectory].reshape(-1, 1))
            firing_rates[trajectory] = max_firing_rate * (activity_norm[trajectory][:100] -
                                                          np.median(activity_norm[trajectory]))
            spikes[trajectory] = homogenous_poisson(firing_rates[trajectory], tmax, bin_size)
            spike_times[trajectory] = np.argwhere(spikes[trajectory] == 1)
            ax[j, 1].vlines(spike_times[trajectory], i - 0.3, i + 0.3, color=colors[trajectory])
            ax[1, 1].set_xlabel('Time')
            ax[j, 1].set_ylabel('Neuron')
            ax[j, 1].set_yticks(np.arange(len(splitter_pro)))
            ax[j, 1].set_yticklabels(splitter_pro)
            ax[j, 1].set_title('{}'.format(trajectory) + ' trajectory')

    plt.tight_layout()
    plt.show()


def plot_splitter_cells_during_error_trial():
    """Plot the activity of splitter cells during an error trial.

        This function loads reservoir states data from the error case directory,
        computes the mean activities for RL, RR, and error trajectories, identifies
        the top splitter cells based on the difference in mean activities between RL and RR,
        and plots the activity of these cells along with error activity.

        Returns:
        None
        """
    # Define the path to the error case data directory
    path = '/Users/nchaix/Documents/PhD/code/splitter_cells/data/RR-LL/no_cues/error_case/'

    # Load reservoir states data
    reservoir_states = np.load(path + 'reservoir_states.npy')

    # RL, RR, error
    activity_ranges = {}
    activity_ranges['RL'] = [1408, 1508]
    activity_ranges['RR'] = [1056, 1156]
    activity_ranges['error'] = [2804, 2904]


    mean_activities = {}
    for trajectory in ('RL', 'RR', 'error'):
        mean_activities[trajectory] = get_average_activity(activity_ranges[trajectory], reservoir_states)


    # Take the 6th neurons related to the biggest SI
    splitter_cells = find_splitter_cells(mean_activities['RL'], mean_activities['RR'], 0.1)[:6]

    # Create 2x2 sub plots
    fig, ax = plt.subplots(figsize=(8, 5))

    colors = {}
    colors['R-L'] = 'b'
    colors['error: RR instead of LL'] = 'yellow'
    colors['R-R'] = 'green'

    plot_activity(ax, splitter_cells, mean_activities['RL'], mean_activities['RR'],
                  mean_activities['error'], label_1='R-L', label_2='R-R', label_3='error: RR instead of LL',
                  title='Splitter cells', colors=colors)

    plt.legend()

    plt.tight_layout()
    plt.show()



def plot_RSA_matrix(cues=False):
    """
    This function allows to compute representational similarity analysis (RSA)
    of the internal activity of the reservoir for different trajectories. The metrics is
    the correlation between internal states.
    """

    if cues:
        path = "../data/RR-LL/cues/reservoir_states/"
    else:
        path = "../data/RR-LL/no_cues/reservoir_states/"

    # Load reservoir states
    reservoir_states = np.load(path + 'reservoir_states_corridor.npy', allow_pickle=True).item()

    mean_activities = {}

    for trajectory in ('RL', 'LR', 'RR', 'LL'):
        mean_activities[trajectory] = []
        for i in range(10):
            mean_activities[trajectory].append(np.mean(reservoir_states[trajectory][i], axis=0))
        mean_activities[trajectory] = np.array(mean_activities[trajectory])

    splitter_cells = find_splitter_cells(mean_activities['LR'][0], mean_activities['RL'][0], 0.1)

    splitter_cells = [38, 312, 498]

    n_neurons = 3

    fig, ax = plt.subplots(1, n_neurons, figsize=(3*n_neurons, 5))

    for i in range(n_neurons):
        neuron = splitter_cells[i]
        my_dict = {}
        my_dict["RL->L"] = mean_activities['LL'][:, neuron]
        my_dict["LL->R"] = mean_activities['LR'][:, neuron]
        my_dict["RR->L"] = mean_activities['RL'][:, neuron]
        my_dict["LR->R"] = mean_activities['RR'][:, neuron]

        df = pd.DataFrame(my_dict)
        corr_matrix = abs(df.corr())
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, annot=True, vmax=1, vmin=-0., center=0,  mask=mask, ax=ax[i],cbar=i==2)

        ax[i].set_title('Neuron {}'.format(neuron))
    fig.suptitle('Ensemble representational similarity')
    plt.tight_layout()
    plt.show()



if __name__ == '__main__':
    #raster_plot()
    #plot_head_direction_cells()
    plot_hippocampal_cells_3()
    #plot_splitter_cells_count()
    #plot_splitter_cells_during_error_trial()
    #plot_RSA_matrix(cues=False)
    #test()
    #plot_splitter_cells_activity()
















