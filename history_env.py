import gym
import numpy as np

import matplotlib.pyplot as plt

def duplicate_list(l, n):
    ll = np.vstack([l.copy() for i in range(n)])
    return ll.flatten()

class HistoryEnv(gym.Env):
    def __init__(self, obs_space, act_space, t_pred, outputs, dt=0.01):
        super(HistoryEnv, self).__init__()
        obs_low, obs_high = duplicate_list(obs_space.low[outputs], t_pred), duplicate_list(obs_space.high[outputs], t_pred)
        act_low, act_high = duplicate_list(act_space.low, t_pred), duplicate_list(act_space.low, t_pred)
        self.outputs = outputs
        self.dt = dt
        self.t_pred = t_pred

        self.state_space = obs_space
        self.single_action_space = act_space
        self.state = None
        self.reset()
        
        self.action_space = gym.spaces.Box(low=act_low, high=act_high)
        self.observation_space = gym.spaces.Box(low=obs_low, high=obs_high)

    def step(self, action):
        u_size = self.single_action_space.shape[0]
        actions = [action[int(i*u_size):int((i+1)*u_size)] for i in range(int(len(action)/u_size))]
        obss = []
        rewards = []
        for action in actions:
            self.state, reward = self.next_state(self.state, action)
            obss.append(self.state[self.outputs])
            rewards.append(reward)
        observation = []
        for ob in obss:
            observation.extend(ob[self.outputs])
        observation = np.array(observation)
        reward = np.sum(rewards, axis=0)
        done = False
        info = {"rewards": rewards}
        return observation, reward, done, info

    def next_state(self, state, action):
        next_state = state + self.dt*self.f(state, action)
        reward = self.reward_fn(next_state, action) # Next state or current state
        return next_state, reward

    def f(self, x, u):
        raise NotImplementedError

    def reward_fn(self, x, u):
        raise NotImplementedError

    def reset(self):
        obs = [self.state_space.sample() for _ in range(self.t_pred)]
        self.state = obs[-1]
        obss = []
        for ob in obs:
            obss.extend(ob[self.outputs])
        observation = np.array(obss)
        return observation
        
    def render(self):
        raise NotImplementedError

class DubinsEnv(HistoryEnv):
    def __init__(self, t_pred, outputs, dt=0.01):
        obs_space = gym.spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        act_space = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        super(DubinsEnv, self).__init__(obs_space, act_space, t_pred, outputs, dt)

    def f(self, x, u):
        return np.array([np.cos(x[2]), np.sin(x[2]), u[0]])

    def reward_fn(self, x, u):
        return -np.linalg.norm(x[:2])

def register_history_env(name, t_pred, outputs, dt=0.01):
    if name == "Dubins":
        gym.envs.register(
            id='DubinsEnv-v0',
            entry_point='__main__:DubinsEnv',
            max_episode_steps=150,
            kwargs={
                't_pred' : t_pred,
                'outputs' : outputs,
                'dt' : dt
            }
        )
    else:
        raise ValueError("Unknown environment name: {}".format(name))