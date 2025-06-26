from hypercez import Hparams
from hypercez.agents import RandomAgent
from hypercez.agents.agent_base import Agent
from hypercez.envs.cl_env import CLEnvLoader
from hypercez.util.datautil import DataCollector


class Trainer:
    def __init__(self, agent: Agent, hparams: Hparams, env_loader: CLEnvLoader):
        self.agent = agent
        self.hparams = hparams
        self.env_loader = env_loader
        self.collector = DataCollector(self.hparams)

    def train(self, dynamic_iters: int):
        # random acting to collect data
        rand_pi = RandomAgent(self.hparams)
        for task_id in range(self.hparams.num_tasks):
            env = self.env_loader.get_env(task_id)

            x_t = env.reset()

            for it in range(self.hparams.init_rand_steps):
                u = rand_pi.act(x_t)
                x_tt, reward, terminated, truncated, info = env.step(u.reshape(env.action_space.shape))
                self.collector.add(x_t, u, reward, x_tt, task_id)
                x_t = x_tt
                if terminated or truncated:
                    x_t = env.reset()


