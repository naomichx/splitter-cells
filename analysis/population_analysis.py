"""
Reservoir state analysis is conducted at the population level. In this script, the reservoir states
beforehand recorded during the bot's navigation are loaded to process to the population analysis.


 The script allows several analytical processes:

- 3D PCA Analysis: principal component analysis (PCA) on the reservoir states.
                   Additional information, such as the Euclidean distance between
                  points is incorporated into the analysis.

- SVM Classification:  support vector machine (SVM) classification to categorize
                       the reservoir states based on which direction the bot will take at the next decision point

- UMAP Analysis - Central Corridor: Applies Uniform Manifold Approximation and Projection (UMAP)
                                    on the reservoir states specifically when the bot enters the central corridor.
                                    This analysis helps in distinguishing various trajectories
                                     in a higher-dimensional space.

- UMAP Analysis - Error Case: Implements UMAP on the reservoir states during error cases,
                              providing insights into the internal dynamics of the reservoir when errors occur.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import accuracy_score
from scipy.spatial import distance
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D
import umap
import umap.plot
plt.rc('font', size=12)


# Useful functions
def split_train_test(input, output, nb_train, max_len_test=100000):
    """ Splits the input and output lists in two lists: one for the training phase, one for the testing phase.
    Inputs:
    - input: input list
    - output: output list
    - nb_train:number of element in the training lists. nb_test = len(input)-nb_train
    - max_len_test : max length of the testing list (so it is not too big).
    Outputs:
    - X_train, X_test: two lists from input that were split at the nb_train index. len(X_test) <= max_len_test
    - Y_train, Y_test: two lists from output that were split at the nb_train index. len(Y_test) <= max_len_test"""

    X_train, Y_train, X_test, Y_test = input[:nb_train], output[:nb_train], input[nb_train:], output[nb_train:]
    if len(X_test) > max_len_test:
        X_test, Y_test = X_test[:max_len_test], Y_test[:max_len_test]

    return X_train, Y_train, X_test, Y_test


def generate_legend(y_positions):
    """
    Generate legend based on y_positions.
    'm': middle corridor
    'r': right loop
    'l': left loop
    """
    legend = []
    for pos in y_positions:
        if 200 < pos < 300:
            legend.append('m')  # 'm' for 'Middle loop'
        elif pos < 200:
            legend.append('r')  # 'r' for 'Right loop'
        else:
            legend.append('l')  # 'l' for 'Left loop'
    return np.array(legend)


# For PCA analysis
def plot_PCA_3D(cues=False):
    """
    This function loads reservoir states data and positions, performs PCA to reduce dimensionality to 3D,
    and visualizes the data in a 3D scatter plot.
    Parameters:
    - cues (bool): Whether to include cues in the data path. Defaults to False.

    Returns:
    None
    """
    if cues:
        path = "../data/R-L_60/cues/reservoir_states/"
    else:
        path = "../data/R-L_60/no_cues/reservoir_states/"

    # Load reservoir states and positions
    res_activity = np.load(path + 'reservoir_states.npy')
    positions = np.load(path + 'positions.npy')
    y_positions = positions[:, 1]
    legend = generate_legend(y_positions)
    pca = PCA(n_components=3)
    x = StandardScaler().fit_transform(res_activity)
    principalComponents = pca.fit_transform(x)

    Xax = principalComponents[:, 0]
    Yax = principalComponents[:, 1]
    Zax = principalComponents[:, 2]

    cdict = {"r": 'C3', "m": 'C2', "l": 'C0'}
    labl = {"r": 'Right loop', "m": 'Middle loop', 'l': 'Left loop'}

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d')

    for l in np.unique(legend):
        ix = np.where(legend == l)
        ax.scatter(Xax[ix], Yax[ix], Zax[ix], c=cdict[l], s=10,
                   label=labl[l], marker='o')

    # for loop ends
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.set_zlabel("PCA 3")
    ax.set_title('PCA analysis of the reservoir states ')

    ax.grid(False)
    ax.legend()
    plt.show()


def plot_PCA_3D_with_distance(cues=False):
    """
    This function loads reservoir states data and positions, performs PCA to reduce dimensionality to 3D,
    and visualizes the data in a 3D scatter plot. It calculates the Euclidean distance between specific points
    (C1,C2,C3) in the central stem of the maze.

    Parameters:
    - cues (bool): Whether to include cues in the data path. Defaults to False.

    Returns:
    None
    """
    # Define data path based on cues parameter
    if cues:
        path = "../data/R-L_60/cues/reservoir_states/"
    else:
        path = "../data/R-L_60/no_cues/reservoir_states/"

    # Load reservoir states and positions
    res_activity = np.load(path + 'reservoir_states.npy')
    positions = np.load(path + 'positions.npy')

    # Separate reservoir states and positions for left and right trajectories
    if cues:
        res_left = res_activity[460:700]
        pos_left = positions[460:700]
        res_right = res_activity[80:300]
        pos_right = positions[80:300]
        markers_r = {"C1": 6, "C2": 42, "C3": 77}
        markers_l = {"C1": 238, "C2": 274, "C3": 309}
    else:
        res_right = res_activity[350:600]
        res_left = res_activity[750:1000]
        pos_right = positions[350:600]
        pos_left = positions[750:1000]
        markers_r = {"C1": 0, "C2": 47, "C3": 111}
        markers_l = {"C1": 250, "C2": 300, "C3": 361}

    # Concatenate reservoir states and positions
    res_activity = np.concatenate((res_right, res_left))
    positions = np.concatenate((pos_right, pos_left))

    y_positions = positions[:, 1]
    legend = generate_legend(y_positions)
    x_positions = positions[:, 0]

    # Find C1, C2 and C3 location
    for i, pos in enumerate(y_positions):
        if 220 <= pos <= 275:
            if 70 <= x_positions[i] <= 80:
                # if 110 <= x_positions[i] <= 120:
                print("C1", i)
            elif 140 < x_positions[i] < 150:
                print("C2", i)
            elif 210 < x_positions[i] < 220:
                print("C3", i)

    # Proceed to PCA
    pca = PCA(n_components=3)
    x = StandardScaler().fit_transform(res_activity)
    principalComponents = pca.fit_transform(x)

    Xax = principalComponents[:, 0]
    Yax = principalComponents[:, 1]
    Zax = principalComponents[:, 2]

    cdict = {"r": 'C3', "m": 'C2', "l": 'C0'}
    labl = {"r": 'Right loop', "m": 'Middle loop', 'l': 'Left loop'}

    # Plot PCA
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d')


    for l in np.unique(legend):
        ix = np.where(legend == l)
        ax.scatter(Xax[ix], Yax[ix], Zax[ix], c=cdict[l], s=10,
                   label=labl[l], marker='o')

    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.set_zlabel("PCA 3")
    ax.set_title('PCA Analysis of the Reservoir States')

    colors = {"C1": 'orange', "C2": 'magenta', "C3": 'purple'}

    ax.grid(False)
    ax.legend()

    for key in markers_l:
        ax.scatter3D(Xax[markers_l[key]], Yax[markers_l[key]], Zax[markers_l[key]], s=90, marker='x',
                     color=colors[key])
    for key in markers_r:
        ax.scatter3D(Xax[markers_r[key]], Yax[markers_r[key]], Zax[markers_r[key]], s=90, marker='x',
                     color=colors[key])

    distances = {}
    for key in markers_l:
        distances[key] = distance.euclidean((Xax[markers_r[key]], Yax[markers_r[key]], Zax[markers_r[key]]),
                                            (Xax[markers_l[key]], Yax[markers_l[key]], Zax[markers_l[key]]))

    for key in markers_l:
        ax.plot3D([Xax[markers_r[key]], Xax[markers_l[key]]], [Yax[markers_r[key]], Yax[markers_l[key]]],
                  [Zax[markers_r[key]], Zax[markers_l[key]]], color=colors[key], linestyle='dotted', alpha=0.7,
                  label=f"{key}: {round(distances[key], 2)}")

    plt.legend(prop={'size': 10}, loc=2)
    plt.show()


# For SVM analysis
def SVM_classifier():
    """
    Build and train SVM classifier.
    """
    path = "/Users/nchaix/Documents/PhD/code/splitter_cells/data/SVM_data/"
    input = np.load(path + 'input.npy')
    output = np.load(path + 'output.npy')

    nb_train = 300

    print('Initialisation of SVM...')
    X_train, Y_train, X_test, Y_test = split_train_test(input, output, nb_train=nb_train)
    print('Train data shape classifier:', np.shape(X_train))
    print('Test data shape classifier:', np.shape(X_test))
    Y_train = np.array(Y_train).reshape(nb_train, )
    Y_test = np.array(Y_test).reshape(len(Y_test), )
    SVM = svm.SVC()
    SVM.fit(X_train, Y_train)
    Y_pred_svm = SVM.predict(X_test)
    print('Score SVM:', accuracy_score(Y_pred_svm, Y_test))
    return SVM


def plot_SVM_predictions():
    """Plot SVM predictions on the reservoir states about
    the directions to take at the next decision point.
    Returns:
    None
    """

    # Define the path to the SVM data directory
    path = "/Users/nchaix/Documents/PhD/code/splitter_cells/data/SVM_data/"

    # Load SVM predictions and positions
    SVM_pred = np.squeeze(np.load(path + 'SVM_predictions.npy')) # 0: going right, 1: going left
    positions = np.load(path + 'positions.npy')
    x_positions = positions[:, 0]
    y_positions = positions[:, 1]

    # Create a new figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = ListedColormap(['C0', 'C3'])
    norm = BoundaryNorm([0, 1, 2], cmap.N)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    points = np.array([y_positions, x_positions]).T.reshape(-1, 1, 2)
    ax.set_aspect(1)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(SVM_pred)
    lc.set_linewidth(0.5)
    # Add the LineCollection to the current axes
    plt.gca().add_collection(lc)
    ax.set_xlabel("Y (d.U.)", fontsize=18)
    ax.set_ylabel("x (d.U)", fontsize=18)
    plt.ylim(x_positions.min() - 10, x_positions.max() + 10)
    plt.xlim(y_positions.min() - 10, y_positions.max() + 10)
    plt.show()


# For UMAP analysis
def gather_reservoir_states_central_corridor(cues=False):
    """
        This function collects reservoir states data as the bot navigates into the central corridor.
        It separates the states based on the observed trajectories (RL, LR, LL, RR) and stores them accordingly.

        Parameters:
        - cues (bool): Whether using data including the model with cues. Defaults to False.

        Returns:
        reservoir_states_corridor (dict): A dictionary containing reservoir states for each trajectory.
                                 Keys are trajectory labels (RL, LR, LL, RR), and values are arrays
                                 of corresponding reservoir states.
        """
    if cues:
        path = "../data/RR-LL/cues/reservoir_states/"
    else:
        path = "../data/RR-LL/no_cues/reservoir_states/"

    reservoir_states = np.load(path + 'reservoir_states.npy')
    positions = np.load(path + 'positions.npy')
    trajectories = ('RR', 'LL', 'RL', 'LR')
    list_indexes = {}
    reservoir_states_corridor = {}
    for trajectory in trajectories:
        list_indexes[trajectory] = []
        reservoir_states_corridor[trajectory] = []

    # first time it goes in the central stem
    list_indexes['LR'].append([0, 60])

    enter_corridor = False
    from_R = False
    from_L = False
    to_R = False
    to_L = False

    for i in range(60, len(positions) - 60):
        if 200 < positions[i][1] < 300:
            if not enter_corridor:
                beg = i
                enter_corridor = True
                if positions[i - 60][1] < 200:
                    from_R = True
                elif positions[i - 60][1] > 300:
                    from_L = True
        else:
            if enter_corridor:
                end = i
                enter_corridor = False
                if positions[i + 60][1] < 200:
                    to_R = True
                elif positions[i + 60][1] > 300:
                    to_L = True
                sequence = [beg, end]
                if from_R:
                    if to_L:
                        list_indexes['RL'].append(sequence)
                        from_L = True
                        from_R = False
                    elif to_R:
                        list_indexes['RR'].append(sequence)
                        from_R = True
                        from_L = True
                elif from_L:
                    if to_L:
                        list_indexes['LL'].append(sequence)
                        from_L = True
                        from_R = False
                    elif to_R:
                        list_indexes['LR'].append(sequence)
                        from_L = True
                        from_R = False
                to_R, to_L = False, False

    for trajectory in trajectories:
        for index in list_indexes[trajectory]:
            reservoir_states_corridor[trajectory].append(np.array(reservoir_states[index[0]: index[0] + 100][:]))
    #np.save(arr=reservoir_states_corridor,
    #        file= path + 'reservoir_states_corridor.npy',
    #        allow_pickle=True)
    return reservoir_states_corridor


def plot_UMAP(cues=False, n_neighbors=5):
    """
        Plot UMAP (Uniform Manifold Approximation and Projection) visualization of reservoir states.

        Parameters:
        - cues (bool): Whether using data including the model with cues. Defaults to False.

        This function loads reservoir states data when the bot is going through the central
         corridor of the maze, performs UMAP embedding, and visualizes the embedded data points
         with different trajectories colored differently.

        Returns:
        None
        """
    if cues:
        path = "../data/RR-LL/cues/reservoir_states/"
    else:
        path = "../data/RR-LL/no_cues/reservoir_states/"

    # Load reservoir states
    reservoir_states = np.load(path + 'reservoir_states_corridor.npy', allow_pickle=True).item()

    # Count the number of states for each trajectory
    n_LL = len(reservoir_states['LL'])
    n_LR = len(reservoir_states['LR'])
    n_RL = len(reservoir_states['RL'])
    n_RR = len(reservoir_states['RR'])

    # Concatenate all reservoir states
    all_states = np.concatenate((reservoir_states['RR'], reservoir_states['LR'],
                                 reservoir_states['LL'], reservoir_states['RL'])).reshape((-1, 1000))

    # UMAP embedding
    reducer = umap.UMAP(n_neighbors=n_neighbors)
    embedding = reducer.fit_transform(all_states)

    # Create labels and values for coloring
    labels = ['RR ({})'.format(n_RR), 'LR ({})'.format(n_LR),
              'LL ({})'.format(n_LL), 'RL ({})'.format(n_RL)]
    values = np.repeat([0, 1, 2, 3], [n_RR * 100, n_LR * 100, n_LL * 100, n_RL * 100])

    # Define colormap
    col_traj = {'LL': 'grey', 'RL': 'blue', 'RR': 'green', 'LR': 'red'}
    colours = ListedColormap([col_traj['RR'], col_traj['LR'], col_traj['LL'], col_traj['RL']])

    # Plot
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=values, cmap=colours, alpha=0.8, s=2)
    plt.legend(handles=scatter.legend_elements()[0], labels=labels)
    plt.title("UMAP of the bot's internal state in the central corridor")
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.show()


def plot_UMAP_error_case():
    """Plot UMAP analysis of reservoir states when the bot is entering the central corridor,
    during error cases and correct cases.

    Returns:
    None
    """

    # Define the path to the error case data directory
    path = '/Users/nchaix/Documents/PhD/code/splitter_cells/data/RR-LL/no_cues/error_case/'

    # Load reservoir states data
    reservoir_states = np.load(path + 'reservoir_states.npy')

    # Segment reservoir states into different trajectory groups
    RL = [reservoir_states[0:100], reservoir_states[1408:1508]]
    LR = [reservoir_states[708:808], reservoir_states[2112:2212]]
    RR = [reservoir_states[1056:1156], reservoir_states[2458:2558]]
    LL = [reservoir_states[358:458], reservoir_states[1767:1867]]
    error = [reservoir_states[2804:2904]]

    # Define colors for different trajectories and error cases
    col_traj = {'LL': 'orange', 'RL': 'blue', 'RR': 'green', 'LR': 'green', 'error': 'orange'}

    # Concatenate all trajectory segments and create corresponding labels
    all_data = np.concatenate((RL, RR, error)).reshape((-1, 1000))
    labels = ['RL', 'RR', 'error: RR instead of RL']

    # Assign numeric values to trajectory groups for coloring
    values = np.concatenate([[0] * len(RL) * 100, [1] * len(RR) * 100, [2] * len(error) * 100])

    # Set the number of neighbors for UMAP
    n_neighbors = 2

    # Define colormap based on trajectory colors
    colours = ListedColormap([col_traj['RL'], col_traj['RR'], col_traj['error']])

    # Apply UMAP for dimensionality reduction
    reducer = umap.UMAP(n_neighbors=n_neighbors)
    embedding = reducer.fit_transform(all_data)

    # Plot the UMAP embedding
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=values, cmap=colours, alpha=0.8, s=2, norm=None)

    # Add legend
    plt.legend(handles=scatter.legend_elements()[0], labels=labels)

    # Display the plot
    plt.show()











if __name__ == '__main__':
    #plot_PCA_3D()
    plot_PCA_3D_with_distance()
    #plot_SVM_predictions()
    #plot_UMAP(cues=False, n_neighbors=5)
    #plot_UMAP_error_case()
















