import numpy as np
import optuna
import mlflow
from typing import Dict
import os
from reservoirpy.nodes import Reservoir, Ridge, Input
from reservoirpy.observables import nrmse, rsquare
import json
from esn_model import split_train_test
SEED = 1



def set_seed(seed):
    """To ensure reproducible runs we fix the seed for different libraries"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    # Deterministic operations for CuDNN, it may impact performances
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_to_disk(path, agent_id, hparams, nrmse):
    try:
        # Create target Directory
        os.mkdir(path + '/'+str(agent_id) + '/')
        print("Directory ", path + '/' + str(agent_id) + '/', " Created ")
    except FileExistsError:
        print("Directory ", path + '/' + str(agent_id) + '/', " already exists")
    with open(path + '/' + str(agent_id) + '/' + 'hparams.json', 'w') as f:
        json.dump(hparams, f)
    np.save(path + '/' + str(agent_id) + '/' + 'nrmse.npy', nrmse)


def get_agent_id(agent_dir) -> str:
    try:
        # Create target Directory
        os.mkdir(agent_dir)
        print("Directory ", agent_dir, " Created ")
    except FileExistsError:
        print("Directory ", agent_dir, " already exists")
    ids = []
    for id in os.listdir(agent_dir):
        try:
            ids.append(int(id))
        except:
            pass
    if ids == []:
        agent_id = 1
    else:
        agent_id = max(ids) + 1
    return str(agent_id)

def sample_hyper_parameters(trial: optuna.trial.Trial) -> Dict:
    # Reservoir
    nb_units = 1000
    warmup = 800
    noise_rc = 0.00
    seed = 1234
    sr = trial.suggest_loguniform("spectral_radius", 0.6, 1.8)
    lr = trial.suggest_loguniform("leak_rate", 1e-4, 0.1)
    ridge = trial.suggest_loguniform("ridge", 1e-5, 1.)
    #rc_connectivity = trial.suggest_loguniform("rc_connectivity", 0.01, 1.)
    rc_connectivity = 0.1
    input_connectivity = 0.1
    #input_connectivity = trial.suggest_loguniform("input_connectivity", 0.00001, 0.5)
    return {
        'reservoir_size': nb_units,
        'spectral_radius': sr,
        'noise_rc': noise_rc,
        'leak_rate': lr,
        'seed': seed,
        'warmup': warmup,
        'ridge': ridge,
        'rc_connectivity': rc_connectivity,
        'input_connectivity': input_connectivity,
    }

def objective(trial: optuna.trial.Trial,X_train, Y_train, X_test, Y_test,agent_dir):
    with mlflow.start_run():
        agent_id = get_agent_id(agent_dir)
        mlflow.log_param('agent_id', agent_id)
        # hyper-parameters
        arg = sample_hyper_parameters(trial)
        mlflow.log_params(trial.params)
        set_seed(arg['seed'])

        reservoir = Reservoir(units=arg['reservoir_size'], lr=arg['leak_rate'],
                              sr=arg['spectral_radius'], noise_rc=arg['noise_rc'],
                              input_connectivity=arg['input_connectivity'],
                              #input_scaling=np.array([1] * 8 + [10] * 2),
                              rc_connectivity=arg['rc_connectivity'], seed=SEED)
        ridge = Ridge(ridge=arg['ridge'])
        esn = reservoir >> ridge
        esn = esn.fit(X_train, Y_train, warmup=arg['warmup'])
        Y_pred = esn.run(X_test)
        rmse = nrmse(Y_test, Y_pred)
        save_to_disk(agent_dir, agent_id, arg, rmse)
        mlflow.log_metric('nrmse', rmse)
    return rmse

def optuna_optim(input,output, title,n_trials = 500):
    print('Start Optuna optimization ...')
    parent_dir = 'model_optimization/'
    SAVED_AGENTS_DIR = parent_dir + 'mlagent/' + title
    MLFLOW_RUNS_DIR = parent_dir + 'mlflows/' + title
    mlflow.set_tracking_uri(MLFLOW_RUNS_DIR)
    mlflow.set_experiment(title)
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(), study_name=title,
                                direction='minimize',
                                load_if_exists=True, storage=f'sqlite:////Users/nchaix/Documents/PhD/code/'
                                                             f'splitter_cells/model_optimization/optuna_db/'
                                                             + title + '.db')
    X_train, Y_train, X_test, Y_test = split_train_test(input, output, 7000)
    func = lambda trial: objective(trial,  X_train, Y_train, X_test, Y_test, agent_dir=SAVED_AGENTS_DIR)
    study.optimize(func, n_trials=n_trials)
    best_trial = study.best_trial
    hparams = {k: best_trial.params[k] for k in best_trial.params if k != 'seed'}


data_folder = "data/R-L/no_cues/"

input = np.load(data_folder + 'input.npy')
output = np.load(data_folder + 'output.npy')
output = output.reshape(len(output), 1)

title = 'RL_no_cues_2'
optuna_optim(input, output, title, n_trials=700)





