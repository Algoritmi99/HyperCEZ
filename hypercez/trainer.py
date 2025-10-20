import time

import torch
from tqdm import tqdm

from hypercez import Hparams
from hypercez.agents.agent_base import Agent, ActType
from hypercez.envs.cl_env import CLEnvLoader
from hypercez.evaluator import Evaluator
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

    def train(self, evaluate=False):
        print("Running Training sequence on", self.device)
        self.agent.train()
        train_cnt = 0
        for task_id in range(self.hparams.num_tasks):
            print("Running task {}".format(task_id))
            time.sleep(1)
            # random acting to collect data
            env = self.env_loader.get_env(task_id)
            x_t, _ = env.reset()
            self.agent.reset(x_t)
            print("Doing initial random steps...")
            time.sleep(1)
            pbar = tqdm(desc="Initializing")
            while not self.agent.is_ready_for_training(task_id=task_id, pbar=pbar):
                _, _, u = self.agent.act_init(x_t, task_id=task_id)
                x_tt, reward, terminated, truncated, info = env.step(u.reshape(env.action_space.shape))
                self.agent.collect(x_t, u, reward, x_tt, task_id, done=terminated or truncated)

                if self.plotter is not None:
                    self.plotter.step(reward, terminated or truncated, task_id, split='train')

                x_t = x_tt
                if terminated or truncated:
                    x_t, _ = env.reset()
                    self.agent.reset(x_t)

            pbar.close()
            # trial and error
            x_t, _ = env.reset()
            self.agent.reset(x_t)
            print("Doing training steps...")
            time.sleep(1)
            it = 0
            pbar = tqdm(desc="Training", position=0)
            while not self.agent.done_training(task_id=task_id, pbar=pbar):
                # update when it's do
                if it % self.hparams.dynamics_update_every == 0:
                    self.agent.learn(task_id)

                # exploration
                _, _, u_t = self.agent.act(x_t, task_id=task_id, act_type=ActType.INITIAL)
                x_tt, reward, terminated, truncated, info = env.step(u_t.reshape(env.action_space.shape))
                self.agent.collect(x_t, u_t, reward, x_tt, task_id, done=terminated or truncated)

                if self.plotter is not None:
                    self.plotter.step(reward, terminated or truncated, task_id, split='train')

                x_t = x_tt
                if truncated or terminated:
                    x_t, _ = env.reset()
                    self.agent.reset(x_t)

                if train_cnt % self.hparams.train["eval_interval"] == 0 and evaluate:
                    print("evaluating...")
                    time.sleep(1)
                    evaluator = Evaluator(self.agent, self.hparams, plotter=self.plotter)
                    evaluator.evaluate(self.hparams.train["eval_n_episode"], pbar=pbar)

                it += 1
                train_cnt += 1
            pbar.close()

        if self.plotter is not None:
            self.plotter.plot()
            self.plotter.plot_taskwise()
            self.plotter.save_raw_data()
