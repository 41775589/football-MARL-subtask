# Copyright (c) 2021, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

"""
Wrapper for making the gather-trade-build environment an OpenAI compatible environment.
This can then be used with reinforcement learning frameworks such as RLlib.
"""

import os
import pickle
import random
import warnings
import tempfile
import gym

import numpy as np
import gfootball.env as football_env
from gym import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv

_BIG_NUMBER = 1e20


def recursive_list_to_np_array(d):
    if isinstance(d, dict):
        new_d = {}
        for k, v in d.items():
            if isinstance(v, list):
                new_d[k] = np.array(v)
            elif isinstance(v, dict):
                new_d[k] = recursive_list_to_np_array(v)
            elif isinstance(v, (float, int, np.floating, np.integer)):
                new_d[k] = np.array([v])
            elif isinstance(v, np.ndarray):
                new_d[k] = v
            else:
                raise AssertionError
        return new_d
    raise AssertionError


def pretty_print(dictionary):
    for key in dictionary:
        print("{:15s}: {}".format(key, dictionary[key].shape))
    print("\n")


class RllibGFootball(MultiAgentEnv):
    """An example of a wrapper for GFootball to make it compatible with rllib."""
    def __init__(self, env_config):
        self.env_config_dict = env_config["env_config_dict"]
        self.env = football_env.create_environment(
            env_name= self.env_config_dict['env_name'],
            stacked= self.env_config_dict['stacked'],
            # logdir=os.path.join(tempfile.gettempdir(), 'rllib_test'),
            write_goal_dumps= self.env_config_dict['write_goal_dumps'],
            write_full_episode_dumps= self.env_config_dict['write_full_episode_dumps'],
            render= self.env_config_dict['render'],
            dump_frequency= self.env_config_dict['dump_frequency'],
            number_of_left_players_agent_controls=self.env_config_dict['number_of_left_players_agent_controls'],
            channel_dimensions=(self.env_config_dict['channel_dimensions_1'],self.env_config_dict['channel_dimensions_2']))

        # Adding env id in the case of multiple environments
        if hasattr(env_config, "worker_index"):
            self.env_id = (
                                  env_config["num_envs_per_worker"] * (env_config.worker_index - 1)
                          ) + env_config.vector_index
        else:
            self.env_id = None

        self.action_space = gym.spaces.Discrete(self.env.action_space.nvec[1])
        self.observation_space = gym.spaces.Box(
            low=self.env.observation_space.low[0],
            high=self.env.observation_space.high[0],
            dtype=self.env.observation_space.dtype)
        self.num_agents = self.env_config_dict['number_of_left_players_agent_controls']

    def reset(self):
        original_obs = self.env.reset()
        obs = {}
        for x in range(self.num_agents):
            if self.num_agents > 1:
                obs['agent_%d' % x] = original_obs[x]
            else:
                obs['agent_%d' % x] = original_obs
        return obs

    def step(self, action_dict):
        actions = []
        for key, value in sorted(action_dict.items()):
            actions.append(value)
        o, r, d, i = self.env.step(actions)
        rewards = {}
        obs = {}
        infos = {}
        for pos, key in enumerate(sorted(action_dict.keys())):
            infos[key] = i
            if self.num_agents > 1:
                rewards[key] = r[pos]
                obs[key] = o[pos]
            else:
                rewards[key] = r
                obs[key] = o
        dones = {'__all__': d}
        # print("RRRRRRRRRRR",rewards)
        return obs, rewards, dones, infos

    @property
    def pickle_file(self):
        if self.env_id is None:
            return "game_object.pkl"
        return "game_object_{:03d}.pkl".format(self.env_id)





