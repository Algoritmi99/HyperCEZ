import copy

from hypercez.agents.agent_base import Agent, ActType
from hypercez.envs.cl_env import CLEnvLoader


class Evaluator:
    def __init__(self, agent: Agent, hparams, plotter=None):
        self.agent = copy.deepcopy(agent)
        self.hparams = hparams
        self.plotter = plotter

    def evaluate(self, steps, pbar=None, render=False):
        self.agent.eval()
        if pbar is not None:
            pbar.total = steps
            pbar.n = 0
            pbar.set_description("Evaluating", refresh=False)
            pbar.refresh()
        for _ in range(steps):
            env_loader = CLEnvLoader(self.hparams.env)
            for i in range(self.hparams.num_tasks):
                env_loader.add_task(i, render=render)

            for task_id in range(self.hparams.num_tasks):
                if pbar is not None:
                    pbar.set_postfix(task=task_id, refresh=False)
                    pbar.refresh()
                env = env_loader.get_env(task_id)
                x_t, _ = env.reset()

                self.agent.reset(x_t)
                done = False
                while not done:
                    _, _, u_t = self.agent.act(x_t, task_id=task_id, act_type=ActType.INITIAL)
                    x_tt, reward, terminated, truncated, info = env.step(u_t.reshape(env.action_space.shape))
                    x_t = x_tt

                    if render:
                        env.render()

                    if self.plotter is not None:
                        self.plotter.step(reward, terminated or truncated, task_id, split='eval')

                    done = terminated or truncated
            if pbar is not None:
                pbar.n += 1
                pbar.refresh()
