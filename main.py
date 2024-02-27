""" Main script to run.
Before running the script, certain configurations are required:

- task :
        1) 'R-L' (alternation task)
        2) 'RR-LL' (half-alternation task)

- simulation_mode:
                1) 'walls': the bot navigates and takes direction automatically using Braitenberg algorithms.
                            Walls are added to guide the bot in the right direction.
                   Some walls are added so as to force the bot taking the right direction
                2) 'data': the bot is data-driven and navigates based on the provided position file.
                3) 'esn': the bot moves based on ESN predictions, trained using supervised learning.

- save_reservoir_states: set to True if the reservoir states and the bot's positions and orientation need to be recorded
- save_bot_states: set to True if the bot's positions and orientation need to be recorded
- path_to_save: folder to save

"""



import matplotlib.animation as animation
from experiment import Experiment
import matplotlib.pyplot as plt
import numpy as np


task = 'RR-LL' #'RR-LL', 'R-L'
simulation_mode = "walls"  # data, walls, esn
cues = False
save_reservoir_states = False
save_bot_states = False
path_to_save = '/Users/nchaix/Documents/PhD/code/splitter_cells/data/R-L_60/no_cues/reservoir_states/'


if __name__ == '__main__':
    if task == 'R-L':
        print('Run the alternation task (R-L) ...')
        if cues:
            model_file = "model_settings/model_RL_cues.json"
            data_folder = "data/R-L_60/cues/"
        else:
            model_file = "model_settings/model_RL_no_cues.json"
            data_folder = "data/R-L_60/no_cues/"
    elif task == 'RR-LL':
        print('Run the half-alternation task (RR-LL) ...')
        if cues:
            model_file = "model_settings/model_RR-LL_cues.json"
            data_folder = "data/RR-LL/cues/"
        else:
            model_file = "model_settings/model_RR-LL_no_cues.json"
            data_folder = "data/RR-LL/no_cues/"
    else:
        raise Exception("Task name {}".format(task) + " is not recognized.")

    # Set up the experiment
    exp = Experiment(model_file, data_folder, simulation_mode, task, cues,
                     save_reservoir_states=save_reservoir_states,
                     save_bot_states=save_bot_states)

    # Set up the animation
    anim = animation.FuncAnimation(exp.simulation_visualizer.fig, exp.run,
                                   frames=10000, interval=1, repeat=False)
    plt.tight_layout()
    plt.show()

    # Save data
    if save_bot_states:
        np.save(path_to_save + 'positions.npy', exp.bot.all_positions)
        np.save(path_to_save + 'output.npy', exp.bot.all_orientations)

        if cues:
            # concatenate sensors with cues
            input = np.concatenate((exp.bot.all_sensors_vals,  exp.bot.all_cues), axis=1)
            np.save(path_to_save + 'input.npy', input)
        else:
            np.save(path_to_save + 'input.npy', exp.bot.all_sensors_vals)

    if save_reservoir_states:
        np.save(path_to_save + 'reservoir_states_2.npy', exp.model.reservoir_states)


