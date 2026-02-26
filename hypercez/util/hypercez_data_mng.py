import copy

from hypercez.agents import EZAgent
from hypercez.util.datautil import DataCollector


class HyperCEZDataManager:
    def __init__(self, ez_agent: EZAgent):
        self.ez_agent = ez_agent
        self.collector_map = {}
        self.traj_pool_map = {}
        self.replay_buffer_map = {}
        self.batch_storage_map = {}
        self.prev_traj_map = {}
        self.trained_steps_map = {}
        self.beta_schedule_map = {}
        self.hyper_crl_collector = DataCollector(ez_agent.hparams)
        self.total_trained_steps = 0

    def update_mem_maps(self):
        assert hasattr(self.ez_agent, "mem_id")
        self.collector_map[self.ez_agent.mem_id] = (self.ez_agent.collector)
        self.traj_pool_map[self.ez_agent.mem_id] = (self.ez_agent.traj_pool)
        self.replay_buffer_map[self.ez_agent.mem_id] = (self.ez_agent.replay_buffer)
        self.batch_storage_map[self.ez_agent.mem_id] = (self.ez_agent.batch_storage)
        self.prev_traj_map[self.ez_agent.mem_id] = (self.ez_agent.prev_traj)
        self.beta_schedule_map[self.ez_agent.mem_id] = (self.ez_agent.beta_schedule)
        self.trained_steps_map[self.ez_agent.mem_id] = self.ez_agent.trained_steps

    def recover_memory(self, task_id):
        assert task_id in self.collector_map, f"No memory found for task id {task_id}"
        self.ez_agent.collector = (self.collector_map[task_id])
        self.ez_agent.traj_pool = (self.traj_pool_map[task_id])
        self.ez_agent.replay_buffer = (self.replay_buffer_map[task_id])
        self.ez_agent.batch_storage = (self.batch_storage_map[task_id])
        self.ez_agent.prev_traj = (self.prev_traj_map[task_id])
        self.ez_agent.beta_schedule = (self.beta_schedule_map[task_id])
        self.ez_agent.trained_steps = self.trained_steps_map[task_id]
        self.ez_agent.mem_id = task_id

    def collect(self, x_t, u_t, reward, x_tt, task_id, done=False):
        if self.ez_agent.mem_id == task_id:
            # No collection or memory change needed
            self.ez_agent.collect(x_t, u_t, reward, x_tt, task_id, done=done)
        else:
            # Memory must be saved
            self.update_mem_maps()

            if task_id in self.replay_buffer_map:
                # Recover the previous memory of the agent on this task if it exists
                self.recover_memory(task_id)
            else:
                # reset the memory of the agent for a fresh new task otherwise
                self.ez_agent.reset_full_memory(x_t)

            self.ez_agent.collect(x_t, u_t, reward, x_tt, task_id, done=done)

        self.hyper_crl_collector.add(x_t, u_t, reward, x_tt, task_id)
        self.ez_agent.mem_id = task_id
        self.update_mem_maps()

    def make_batch(self, task_id, reanalyze_state):
        assert hasattr(self.ez_agent, "mem_id")
        self.update_mem_maps()
        if self.ez_agent.mem_id != task_id:
            self.recover_memory(task_id)
        self.ez_agent.make_batch(model_state=reanalyze_state)
        return self.ez_agent.batch_storage.pop()

    def get_item(self, item_name, task_id):
        if not hasattr(self.ez_agent, "mem_id"):
            self.ez_agent.mem_id = task_id

        self.update_mem_maps()

        return self.__getattribute__(item_name)[task_id]

    def is_ready_for_training(self, task_id, pbar=None):
        self.update_mem_maps()

        if pbar is not None:
            pbar.total = self.ez_agent.hparams.train['start_transitions']

            if task_id in self.replay_buffer_map:
                pbar.n = self.replay_buffer_map[task_id].get_transition_num()
            else:
                pbar.n = 0

            pbar.refresh()

        return (
            self.replay_buffer_map[task_id].get_transition_num() >= self.ez_agent.hparams.train['start_transitions']
            if task_id in self.replay_buffer_map
            else False
        )

    def done_training(self, task_id=None, pbar=None):
        self.update_mem_maps()
        if task_id not in self.trained_steps_map:
            self.trained_steps_map[task_id] = 0

        self.ez_agent.trained_steps = self.trained_steps_map[task_id]
        return self.ez_agent.done_training(pbar=pbar)

    def increment_trained_steps(self, task_id=None):
        if task_id not in self.trained_steps_map:
            self.trained_steps_map[task_id] = 0

        self.trained_steps_map[task_id] += 1
        self.total_trained_steps += 1

        if self.ez_agent.mem_id == task_id:
            self.ez_agent.trained_steps = self.trained_steps_map[task_id]

    def get_trained_steps(self, task_id=None):
        if task_id is None:
            return self.total_trained_steps
        if task_id not in self.trained_steps_map:
            self.trained_steps_map[task_id] = 0
        return self.trained_steps_map[task_id]
