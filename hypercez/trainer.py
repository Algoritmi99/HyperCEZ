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
        for task_id in range(self.hparams.num_tasks):
            # random acting to collect data
            rand_pi = RandomAgent(self.hparams)
            env = self.env_loader.get_env(task_id)
            x_t = env.reset()
            for it in range(self.hparams.init_rand_steps):
                u = rand_pi.act(x_t)
                x_tt, reward, terminated, truncated, info = env.step(u.reshape(env.action_space.shape))
                self.collector.add(x_t, u, reward, x_tt, task_id)
                x_t = x_tt
                if terminated or truncated:
                    x_t = env.reset()

            # trial and error
            x_t = env.reset()
            self.agent.train()
            for it in range(dynamic_iters):
                # update when it's do
                if it % self.hparams.dynamics_update_every == 0 and it > 0:
                    train_set, _ = self.collector.get_dataset(task_id)
                    self.agent.learn(train_set, task_id)

                # exploration
                u_t = self.agent.act(x_t, task_id=task_id).detach().cpu().numpy()
                x_tt, reward, terminated, truncated, info = env.step(u_t.reshape(env.action_space.shape))
                self.collector.add(x_t, u_t, reward, x_tt, task_id)
                x_t = x_tt

                if truncated or terminated:
                    x_t = env.reset()
