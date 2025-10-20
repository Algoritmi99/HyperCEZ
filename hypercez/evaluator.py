import copy

from numba.np.linalg import outer_impl
from tqdm import tqdm

from hypercez.agents.agent_base import Agent, ActType
from hypercez.envs.cl_env import CLEnvLoader


class Evaluator:
    def __init__(self, agent: Agent, hparams, plotter=None):
        self.agent = copy.deepcopy(agent)
        self.hparams = hparams
        self.plotter = plotter

    def evaluate(self, steps, render=False):
        outer_pbar = tqdm(range(steps), desc="Evaluating", position=1, leave=False)
        for _ in outer_pbar:
            env_loader = CLEnvLoader(self.hparams.env)
            for i in range(self.hparams.num_tasks):
                env_loader.add_task(i, render=render)

            inner_pbar = tqdm(range(self.hparams.num_tasks), desc="Evaluating tasks", position=2, leave=False)
            for task_id in inner_pbar:
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
            inner_pbar.close()
        outer_pbar.close()


