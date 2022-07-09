import sys
import warnings
import os

import matplotlib.cbook
import gym
from motion_imitation.envs.a1_training_env import A1GymEnv

warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
from dotmap import DotMap

import mujoco_py
import torch

from envs import *
from gym.wrappers import Monitor

import hydra
import logging

log = logging.getLogger(__name__)

from plot import plot_cp, plot_loss, setup_plotting

from dynamics_model import DynamicsModel


###########################################
#                Datasets                 #
###########################################

def create_dataset_traj(data, control_params=True, train_target=True, threshold=0.0, delta=False, t_range=0):
    """
    Creates a dataset with entries for PID parameters and number of
    timesteps in the future

    Parameters:
    -----------
    data: An array of dotmaps where each dotmap has info about a trajectory
    threshold: the probability of dropping a given data entry
    """
    data_in, data_out = [], []
    for sequence in data:
        states = sequence.states
        actions = sequence.actions
        if t_range:
            states = states[:t_range]
        n = states.shape[0]
        for i in range(n):  # From one state p
            for j in range(i, n):
                # This creates an entry for a given state concatenated
                # with a number t of time steps as well as the PID parameters

                # The randomely continuing is something I thought of to shrink
                # the datasets while still having a large variety
                if np.random.random() < threshold:
                    continue
                dat = [states[i], j - i]
                dat.extend(actions[i])
                data_in.append(np.hstack(dat))
                # data_in.append(np.hstack((states[i], j-i, target)))
                if delta:
                    data_out.append(states[j] - states[i])
                else:
                    data_out.append(states[j])

    data_in = np.array(data_in)
    data_out = np.array(data_out)

    return data_in, data_out


def create_dataset_step(data, delta=True, t_range=0, is_lstm = False, lstm_batch = 0):
    """
    Creates a dataset for learning how one state progresses to the next

    Parameters:
    -----------
    data: A 2d np array. Each row is a state
    """
    data_in = []
    data_out = []
    for sequence in data:
        states = sequence.states
        if t_range:
            states = states[:t_range]
        for i in range(states.shape[0] - 1):
            if 'actions' in sequence.keys():
                actions = sequence.actions
                if t_range:
                    actions = actions[:t_range]
                data_in.append(np.hstack((states[i], actions[i])))
                if delta:
                    data_out.append(states[i + 1] - states[i])
                else:
                    data_out.append(states[i + 1])
            else:
                data_in.append(np.array(states[i]))
                if delta:
                    data_out.append(states[i + 1] - states[i])
                else:
                    data_out.append(states[i + 1])
        if is_lstm:
            remainder = len(data_out)%lstm_batch
            if remainder:
                data_out = data_out[:len(data_out)-remainder]
                data_in = data_in[:len(data_in)-remainder]
    data_in = np.array(data_in)
    data_out = np.array(data_out)

    return data_in, data_out


def collect_data_a1(cfg, plot=False):  # Creates horizon^2/2 points
    """
    Collect data for environment model
    :param nTrials:
    :param horizon:
    :return: an array of DotMaps, where each DotMap contains info about a trajectory
    """

    env_model = 'unitree-a1-v0'
    env = gym.make(env_model)
    # if (cfg.video):
    # env = Monitor(env, hydra.utils.get_original_cwd() + '/trajectories/reacher/video',
    # video_callable = lambda episode_id: episode_id==1,force=True)
    log.info('Initializing env: %s' % env_model)

    # Logs is an array of dotmaps, each dotmap contains 2d np arrays with data
    # about <horizon> steps with actions, rewards and states
    logs = []

    s = np.random.randint(0, 100)
    for i in range(cfg.num_trials):
        log.info('Trial %d' % i)
        if (cfg.PID_test):
            env.seed(0)
        else:
            env.seed(s + i)
        s0 = env.reset()

        lim = cfg.trial_timesteps
        dotmap = run_controller_random(env, horizon=cfg.trial_timesteps, video=cfg.video)
        while len(dotmap.states) < lim:
            env.seed(s)
            env.reset()
            dotmap = run_controller_random(env, horizon=cfg.trial_timesteps, video=cfg.video)
            print(f"- Repeat simulation")
            s += 1
            # if plot and len(dotmap.states)>0: plot_cp(dotmap.states, dotmap.actions)

        if plot: plot_cp(dotmap.states, dotmap.actions, save=True)

        logs.append(dotmap)
        s += 1

    return logs

def run_controller_random(env, horizon, video=False):
    """
    Runs an gym environment for horizon timesteps, taking a single random action

    :param env: A gym object
    :param horizon: The number of states forward to look
    :param policy: A policy object (see other python file)
    """

    logs = DotMap()
    logs.states = []
    logs.actions = []
    logs.rewards = []
    logs.times = []

    observation = env.reset()
    action = env.action_space.sample() # desired action
    for i in range(horizon):
        if (video):
            env.render()
        state = observation

        # print(action)

        observation, reward, done, info = env.step(action)

        if done:
            return logs

        # Log
        # logs.times.append()
        logs.actions.append(action)
        logs.rewards.append(reward)
        logs.states.append(observation.squeeze())

    # Cluster state
    # print(f"Rollout completed, cumulative reward: {np.sum(logs.rewards)}")
    logs.actions = np.array(logs.actions)
    logs.rewards = np.array(logs.rewards)
    logs.states = np.array(logs.states)
    return logs


###########################################
#           Plotting / Output             #
###########################################


def log_hyperparams(cfg):
    log.info(cfg.model.str + ":")
    log.info("  hid_width: %d" % cfg.model.training.hid_width)
    log.info('  hid_depth: %d' % cfg.model.training.hid_depth)
    log.info('  epochs: %d' % cfg.model.optimizer.epochs)
    log.info('  batch size: %d' % cfg.model.optimizer.batch)
    log.info('  optimizer: %s' % cfg.model.optimizer.name)
    log.info('  learning rate: %f' % cfg.model.optimizer.lr)


###########################################
#             Main Functions              #
###########################################

@hydra.main(config_path='conf/a1.yaml')
def contpred(cfg):
    train = cfg.mode == 'train'
    # Collect data
    if not train:
        log.info(f"Collecting new trials")

        exper_data = collect_data_a1(cfg, plot=cfg.plot)
        test_data = collect_data_a1(cfg, plot=cfg.plot)

        log.info("Saving new default data")
        torch.save((exper_data, test_data),
                   hydra.utils.get_original_cwd() + '/trajectories/a1/' + 'raw' + cfg.data_dir)
        log.info(f"Saved trajectories to {'/trajectories/a1/' + 'raw' + cfg.data_dir}")
    # Load data
    else:
        log.info(f"Loading default data")
        # raise ValueError("Current Saved data old format")
        # Todo re-save data
        (exper_data, test_data) = torch.load(
            hydra.utils.get_original_cwd() + '/trajectories/a1/' + 'raw' + cfg.data_dir)

    if train:
        it = range(cfg.copies) if cfg.copies else [0]
        prob = cfg.model.prob
        traj = cfg.model.traj
        ens = cfg.model.ensemble
        delta = cfg.model.delta
        is_lstm = cfg.model.lstm

        log.info(f"Training model P:{prob}, T:{traj}, E:{ens}")

        log_hyperparams(cfg)

        for i in it:
            print('Training model %d' % i)

            # if cfg.model.training.num_traj:
            #     train_data = exper_data[:cfg.model.training.num_traj]
            # else:
            train_data = exper_data

            if traj:
                dataset = create_dataset_traj(exper_data, control_params=cfg.model.training.control_params,
                                              train_target=cfg.model.training.train_target,
                                              threshold=cfg.model.training.filter_rate,
                                              t_range=cfg.model.training.t_range)
            else:
                dataset = create_dataset_step(train_data, delta=delta, is_lstm = is_lstm, lstm_batch = cfg.model.optimizer.batch)

            model = DynamicsModel(cfg)
            train_logs, test_logs = model.train(dataset, cfg)

            setup_plotting({cfg.model.str: model})
            plot_loss(train_logs, test_logs, cfg, save_loc=cfg.env.name + '-' + cfg.model.str, show=False)

            log.info("Saving new default models")
            f = hydra.utils.get_original_cwd() + '/models/a1/'
            if cfg.exper_dir:
                f = f + cfg.exper_dir + '/'
                if not os.path.exists(f):
                    os.mkdir(f)
            copystr = "_%d" % i if cfg.copies else ""
            f = f + cfg.model.str + copystr + '.dat'
            torch.save(model, f)
        # torch.save(model, "%s_backup.dat" % cfg.model.str) # save backup regardless


if __name__ == '__main__':
    sys.exit(contpred())
