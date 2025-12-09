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

    def train(self, evaluate=False, verbose=False, agent_name=""):
        print("Running Training sequence on", self.device)
        self.agent.train()
        train_cnt = 0
        pbar = tqdm()
        evaluated = False
        for task_id in range(self.hparams.num_tasks):
            pbar.set_postfix(task=task_id)
            # random acting to collect data
            env = self.env_loader.get_env(task_id)
            x_t, _ = env.reset()
            self.agent.reset(x_t)
            pbar.set_description("Collecting data")
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

            # trial and error
            x_t, _ = env.reset()
            self.agent.reset(x_t)
            it = 0
            pbar.set_description("Training")
            while not self.agent.done_training(task_id=task_id, pbar=pbar):
                pbar.set_description("Training")
                pbar.set_postfix(task=task_id)
                # update when it's do
                if it % self.hparams.dynamics_update_every == 0:
                    for _ in range(self.hparams.train_dynamic_iters):
                        loss = self.agent.learn(task_id, verbose=verbose)
                        train_cnt += 1
                        evaluated = False
                        if loss is not None and loss < 3:
                            break

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

                if train_cnt % self.hparams.train["eval_interval"] == 0 and evaluate and not evaluated:
                    evaluator = Evaluator(self.agent, self.hparams, plotter=self.plotter)
                    evaluator.evaluate(self.hparams.train["eval_n_episode"], pbar=pbar)
                    self.agent.save("./agents/" + self.hparams.env + "/agent_" + str(it) + agent_name + ".pth")
                    evaluated = True

                it += 1
            pbar.close()

        if self.plotter is not None:
            self.plotter.plot()
            self.plotter.plot_taskwise()
            self.plotter.save_raw_data()
