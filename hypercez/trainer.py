import torch
from tqdm import tqdm

from hypercez import Hparams
from hypercez.agents.agent_base import Agent, ActType
from hypercez.envs.cl_env import CLEnvLoader
from hypercez.util.datautil import DataCollector


class Trainer:
    def __init__(self, agent: Agent, hparams: Hparams, env_loader: CLEnvLoader, device=torch.device("cpu"), plotter=None):
        self.agent = agent
        self.hparams = hparams
        self.env_loader = env_loader
        self.collector = DataCollector(self.hparams)
        self.device = device
        self.agent.to(self.device)
        self.plotter = plotter

    def train(self, dynamic_iters: int):
        print("Running Training sequence on", self.device)
        self.agent.train(dynamic_iters // self.hparams.dynamics_update_every)
        for task_id in range(self.hparams.num_tasks):
            print("Running task {}".format(task_id))
            # random acting to collect data
            env = self.env_loader.get_env(task_id)
            x_t, _ = env.reset()
            self.agent.reset(x_t)
            print("Doing initial random steps...")
            # for _ in tqdm(range(self.hparams.init_rand_steps)):

            for _ in tqdm(range(1200)):
                _, _, u = self.agent.act_init(x_t, task_id=task_id)
                x_tt, reward, terminated, truncated, info = env.step(u.reshape(env.action_space.shape))
                self.agent.collect(x_t, u, reward, x_tt, task_id, done=terminated or truncated)

                if self.plotter is not None:
                    self.plotter.step(reward, terminated or truncated, task_id)

                x_t = x_tt
                if terminated or truncated:
                    x_t, _ = env.reset()
                    self.agent.reset(x_t)

            # trial and error
            x_t, _ = env.reset()
            self.agent.reset(x_t)
            print("Doing training steps...")
            for it in tqdm(range(dynamic_iters)):
                # update when it's do
                if it % self.hparams.dynamics_update_every == 0:
                    self.agent.learn(task_id)

                # exploration
                _, _, u_t = self.agent.act(x_t, task_id=task_id, act_type=ActType.INITIAL)
                x_tt, reward, terminated, truncated, info = env.step(u_t.reshape(env.action_space.shape))
                self.agent.collect(x_t, u_t, reward, x_tt, task_id, done=terminated or truncated)

                if self.plotter is not None:
                    self.plotter.step(reward, terminated or truncated, task_id)

                x_t = x_tt
                if truncated or terminated:
                    x_t, _ = env.reset()
                    self.agent.reset(x_t)

        if self.plotter is not None:
            self.plotter.plot()
            self.plotter.plot_taskwise()
