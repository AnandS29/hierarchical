"""
The purpose of this file is to load in different trajectories and compare how model types predict control performance.
There are three options:
- Gausian process mapping from control policy and s0 to reward
- One step model rolling out predicted trajectory from initial state, cumulative reward
- trajectory model predicting trajectory and cumulative reward
"""

import sys

import hydra
import logging
import itertools

import torch
import numpy as np

from plot import *
from evaluate import test_models
import gpytorch

log = logging.getLogger(__name__)

def get_reward_reacher(state, action):
    # Copied from the reacher env, without self.state calls
    vec = state[-3:]
    reward_dist = - np.linalg.norm(vec)
    reward_ctrl = - np.square(action).sum() * 0.01
    reward = reward_dist + reward_ctrl
    return reward

def get_reward_cp(state, action):
    # custom reward for sq error from x=0, theta = 0
    reward = state[0]**2 + state[2]**2
    return -reward


def get_reward(predictions, actions, r_function):
    # takes in the predicted trajectory and returns the reward
    rewards = {}
    num_traj = len(actions)
    for m_label, state_data in predictions.items():
        r = []
        for i in range(num_traj):
            r_sub = 0
            cur_states = state_data[i]
            cur_actions = actions[i]
            for s,a in zip(cur_states, cur_actions):
                # TODO need a specific get reward function for the reacher env
                r_sub += r_function(s,a)
            r.append(r_sub)
        rewards[m_label] = (r, np.mean(r), np.std(r))

    return rewards

def pred_traj(model, control):
    # for a one-step model, predicts a trajectory from initial state all in simulation
    return 0

def train_gp(data):
    class ExactGPModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ConstantMean()
            # self.covar_module = gpytorch.kernels.RBFKernel()
            # self.scaled_mod = gpytorch.kernels.ScaleKernel(self.covar_module)
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            # covar_x = self.scaled_mod(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    # initialize likelihood and model
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    train_x = data[0]
    train_y = data[1]
    model = ExactGPModel(train_x, train_y, likelihood)

    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam([
        {'params': model.parameters()},  # Includes GaussianLikelihood parameters
    ], lr=.1) # was .1

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    training_iter = 50
    for i in range(training_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(train_x)
        # Calc loss and backprop gradients
        loss = -mll(output, train_y)
        loss.backward()
        print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
            i + 1, training_iter, loss.item(),
            model.covar_module.base_kernel.lengthscale.item(),
            model.likelihood.noise.item()
        ))
        optimizer.step()

    return model, likelihood

def predict_gp(test_x, model, likelihood, train_x=None, train_y=None):
    # Get into evaluation (predictive posterior) mode
    model.eval()
    likelihood.eval()

    # Test points are regularly spaced along [0,1]
    # Make predictions by feeding model through likelihood
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = likelihood(model(test_x))
        # observed_pred = model(test_x)

    if False:
        with torch.no_grad():
            # Initialize plot
            f, ax = plt.subplots(1, 1, figsize=(4, 3))

            # Get upper and lower confidence bounds
            lower, upper = observed_pred.confidence_region()
            # Plot training data as black stars
            if train_x and train_y:
                ax.plot(train_x.numpy(), train_y.numpy(), 'k*')
            # Plot predictive means as blue line
            ax.plot(test_x.numpy(), observed_pred.mean.numpy(), 'b')
            # Shade between the lower and upper confidence bounds
            ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
            ax.set_ylim([-3, 3])
            ax.legend(['Observed Data', 'Mean', 'Confidence'])

    return observed_pred


@hydra.main(config_path='conf/rewards.yaml')
def reward_rank(cfg):
    # print("here")
    if cfg.env == 'lorenz':
        raise ValueError("No Reward in Lorenz System")

    label = cfg.env.label
    graph_file = 'Plots'
    os.mkdir(graph_file)


    trajectories = torch.load(hydra.utils.get_original_cwd()+'/trajectories/'+label+'/raw'+cfg.data_dir)

    # TODO Three test cases, different datasets
    # create 3 datasets of 50 trajectories.
    # 1. same initial state, same goal. Vary PID, target parameters <- simplest case
    # 2. same initial state, different goals. Vary PID parameters
    # 3. different intial state, different goal. Hardest to order reward because potential reward is different

    f = hydra.utils.get_original_cwd() + '/models/' +label +'/'
    model_one = torch.load(f+cfg.step_model+'.dat')
    model_traj = torch.load(f+cfg.traj_model+'.dat')

    # get rewards, control policy, etc for each type, and control parameters
    data_train = trajectories[0] #+trajectories[1]
    reward = [t['rewards'] for t in data_train]
    states = [np.float32(t['states']) for t in data_train]
    actions = [np.float32(t['actions']) for t in data_train]


    if label == 'reacher':
        control = [np.concatenate((t['D'],t['P'],t['target'])) for t in data_train]
        r_func = get_reward_reacher
    else:
        for vec_s, vec_a in zip(states, actions):
            vec_s[0, 1] = vec_s[0, 1].item()
            vec_s[0, 3] = vec_s[0, 3].item()
            vec_a[1] = vec_a[1].item()
        control = [t['K'] for t in data_train]
        r_func = get_reward_cp
        reward = [np.sum([r_func(s,a) for s,a in zip(sta,act)]) for sta,act in zip(states,actions)]

    from botorch.models import SingleTaskGP
    from botorch.fit import fit_gpytorch_model
    from botorch.utils import standardize
    from gpytorch.mlls import ExactMarginalLogLikelihood

    # train_X = torch.rand(10, 2)
    # Y = 1 - torch.norm(train_X - 0.5, dim=-1, keepdim=True)
    # Y = Y + 0.1 * torch.randn_like(Y)  # add some noise
    # train_Y = standardize(Y)


    # fit GP model to rewards
    split = int(len(data_train)*cfg.split)
    gp_x =  [torch.Tensor(np.concatenate((s[0], c))) for s,c in zip(states,control)]
    #[torch.Tensor(np.concatenate((np.array([np.asscalar(np.array(i)) for i in s[0]])), c)) for s,c in zip(states,control)]
    if label == 'reacher':
        gp_y = torch.Tensor(np.sum(np.stack(reward),axis=1))
    else:
        gp_y = torch.Tensor(reward)

    gp_x_train = gp_x[:split]
    gp_y_train = gp_y[:split]
    # model, likelihood = train_gp((torch.stack(gp_x_train), gp_y_train))

    gp = SingleTaskGP(torch.stack(gp_x_train), gp_y_train.reshape(-1,1))
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_model(mll)

    # gp_x_test = gp_x[split:]
    # gp_y_test = gp_y[split:] # TRUE REWARD
    # gp_pred = predict_gp(torch.stack(gp_x_test), model, likelihood, train_x=torch.stack(gp_x_train), train_y=np.stack(gp_y_train))

    ## TEST SET WORK
    data_test = trajectories[0] #[::5]
    reward = [t['rewards'] for t in data_test]
    states = [np.float32(t['states']) for t in data_test]
    actions = [np.float32(t['actions']) for t in data_test]

    if label == 'reacher':
        control = [np.concatenate((t['D'], t['P'], t['target'])) for t in data_test]
        r_func = get_reward_reacher
    else:
        for vec_s, vec_a in zip(states, actions):
            vec_s[0, 1] = vec_s[0, 1].item()
            vec_s[0, 3] = vec_s[0, 3].item()
            vec_a[1] = vec_a[1].item()
        control = [t['K'] for t in data_test]
        r_func = get_reward_cp
        reward = [np.sum([r_func(s, a) for s, a in zip(sta, act)]) for sta, act in zip(states, actions)]

    gp_x_test = [torch.Tensor(np.concatenate((s[0], c))) for s,c in zip(states,control)]
    # gp_pred_test = predict_gp(torch.stack(gp_x_test), model, likelihood)
    # gp_pred_test = gp.forward(torch.stack(gp_x_test))
    gp_pred_test = gp.posterior(torch.stack(gp_x_test))
    # predict with one step and traj model
    models = {
        'p': model_one,
        't': model_traj,
    }
    MSEs, predictions = test_models(data_test, models, env = cfg.env.label)

    # get dict of rewards for type of model
    pred_rewards = get_reward(predictions, actions, r_func)

    if label == 'reacher':
        cum_reward = [np.sum(rew) for rew in reward]
    else:
        cum_reward = reward

    gp_pr_test = gp_pred_test.mean.detach().numpy()
    nn_step = pred_rewards['p'][0]
    nn_traj = pred_rewards['t'][0]
    # Load test data
    print(f"Mean GP reward err: {np.mean((gp_pr_test-cum_reward)**2)}")
    print(f"Mean one step reward err: {np.mean((cum_reward-np.array(nn_step))**2)}")
    print(f"Mean traj reward err: {np.mean((cum_reward-np.array(nn_traj))**2)}")
    arr = np.stack(sorted(zip(cum_reward, gp_pr_test, np.array(nn_step), np.array(nn_traj))))
    # arr = np.stack((cum_reward, gp_pr_test, nn_step, nn_traj))
    import matplotlib.pyplot as plt
    plt.plot(arr[:, 0], label='gt')
    plt.plot(arr[:, 1], label='gp')
    plt.plot(arr[:, 2], label='step')
    plt.plot(arr[:, 3], label='traj')
    plt.xlabel('Sorted Trajectory')
    plt.ylabel('Cumulative Episode reward')
    plt.legend()
    plt.savefig("Reward Predictions.png")

    plt.plot(arr[:, 0], label='gt')
    plt.plot(arr[:, 1], label='gp')
    plt.plot(arr[:, 2], label='step')
    plt.plot(arr[:, 3], label='traj')
    plt.xlabel('Sorted Trajectory')
    plt.ylabel('Cumulative Episode reward')
    plt.legend()
    plt.savefig("Reward Predictions Error.png")

    quit()
    log.info(f"Loading default data")

    (train_data, test_data) = torch.load(
        hydra.utils.get_original_cwd() + '/trajectories/reacher/' + 'raw' + cfg.data_dir)

    # Load models
    log.info("Loading models")
    if cfg.plotting.copies:
        model_types = list(itertools.product(cfg.plotting.models, np.arange(cfg.plotting.copies)))
    else:
        model_types = cfg.plotting.models
    models = {}
    f = hydra.utils.get_original_cwd() + '/models/reacher/'
    if cfg.exper_dir:
        f = f + cfg.exper_dir + '/'
    for model_type in model_types:
        model_str = model_type if type(model_type) == str else ('%s_%d' % model_type)
        models[model_type] = torch.load(f + model_str + ".dat")


if __name__ == '__main__':
    sys.exit(reward_rank())
