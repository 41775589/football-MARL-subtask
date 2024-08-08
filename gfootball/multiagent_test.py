import os
import tempfile
import gym
import numpy as np

import env as football_env  # 请确保你有正确的 football_env 模块导入

class SimpleGFootballEnv(gym.Env):
    """A simple wrapper for GFootball to make it compatible with gym."""

    def __init__(self, num_agents):
        super(SimpleGFootballEnv, self).__init__()
        self.env = football_env.create_environment(
            env_name='5_vs_5', stacked=False,
            logdir=os.path.join(tempfile.gettempdir(), 'gym_test'),
            write_goal_dumps=False, write_full_episode_dumps=False, render=False,
            dump_frequency=0,
            number_of_left_players_agent_controls=num_agents,
            channel_dimensions=(42, 42))
        self.action_space = gym.spaces.Discrete(self.env.action_space.nvec[1])
        self.observation_space = gym.spaces.Box(
            low=self.env.observation_space.low[0],
            high=self.env.observation_space.high[0],
            dtype=self.env.observation_space.dtype)
        self.num_agents = num_agents

    def reset(self):
        original_obs = self.env.reset()
        obs = {}
        for x in range(self.num_agents):
            if self.num_agents > 1:
                obs[f'agent_{x}'] = original_obs[x]
            else:
                obs[f'agent_{x}'] = original_obs
        return obs

    def step(self, action_dict):
        actions = [action_dict[f'agent_{i}'] for i in range(self.num_agents)]
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
        return obs, rewards, dones, infos

if __name__ == '__main__':
    num_agents = 5  # 你可以调整代理数量
    env = SimpleGFootballEnv(num_agents)

    obs = env.reset()
    for step in range(1000):  # 假设我们运行 1000 步
        # 为每个代理生成一个随机动作
        action_dict = {f'agent_{i}': env.action_space.sample() for i in range(num_agents)}
        obs, rewards, dones, infos = env.step(action_dict)
        print(f"Step {step} - Observations: {obs}, Rewards: {rewards}, Dones: {dones}, Infos: {infos}")

        if dones['__all__']:
            obs = env.reset()

