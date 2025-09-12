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
        self.hyper_crl_collector = DataCollector(ez_agent.hparams)

    def collect(self, x_t, u_t, reward, x_tt, task_id, done=False):
        if (
                (not hasattr(self.ez_agent, "mem_id"))
                or
                (hasattr(self.ez_agent, "mem_id") and self.ez_agent.mem_id == task_id)
        ):
            # No collection or memory change needed
            self.ez_agent.collect(x_t, u_t, reward, x_tt, task_id, done=done)
        else:
            # Memory must be saved
            self.collector_map[self.ez_agent.mem_id] = self.ez_agent.collector
            self.traj_pool_map[self.ez_agent.mem_id] = self.ez_agent.traj_pool
            self.replay_buffer_map[self.ez_agent.mem_id] = self.ez_agent.replay_buffer
            self.batch_storage_map[self.ez_agent.mem_id] = self.ez_agent.batch_storage
            self.prev_traj_map[self.ez_agent.mem_id] = self.ez_agent.prev_traj

            if task_id in self.replay_buffer_map:
                # Recover the previous memory of the agent on this task if it exists
                self.ez_agent.collector = self.collector_map[task_id]
                self.ez_agent.traj_pool = self.traj_pool_map[task_id]
                self.ez_agent.replay_buffer = self.replay_buffer_map[task_id]
                self.ez_agent.batch_storage = self.batch_storage_map[task_id]
                self.ez_agent.prev_traj = self.prev_traj_map[task_id]
            else:
                # reset the memory of the agent for a fresh new task otherwise
                self.ez_agent.reset_full_memory(x_t)

            self.ez_agent.collect(x_t, u_t, reward, x_tt, task_id, done=done)

        self.hyper_crl_collector.add(x_t, u_t, reward, x_tt, task_id)

        self.ez_agent.mem_id = task_id

    def make_batch(self, task_id):
        if not hasattr(self.ez_agent, "mem_id"):
            # Save current memory
            self.ez_agent.mem_id = task_id
            self.collector_map[self.ez_agent.mem_id] = self.ez_agent.collector
            self.traj_pool_map[self.ez_agent.mem_id] = self.ez_agent.traj_pool
            self.replay_buffer_map[self.ez_agent.mem_id] = self.ez_agent.replay_buffer
            self.batch_storage_map[self.ez_agent.mem_id] = self.ez_agent.batch_storage
            self.prev_traj_map[self.ez_agent.mem_id] = self.ez_agent.prev_traj

            self.ez_agent.make_batch()
            return self.ez_agent.batch_storage.pop()
        elif self.ez_agent.mem_id != task_id:
            # Save current memory
            self.collector_map[self.ez_agent.mem_id] = self.ez_agent.collector
            self.traj_pool_map[self.ez_agent.mem_id] = self.ez_agent.traj_pool
            self.replay_buffer_map[self.ez_agent.mem_id] = self.ez_agent.replay_buffer
            self.batch_storage_map[self.ez_agent.mem_id] = self.ez_agent.batch_storage
            self.prev_traj_map[self.ez_agent.mem_id] = self.ez_agent.prev_traj

            if task_id in self.replay_buffer_map:
                # recover the memory of the agent on this task
                self.ez_agent.collector = self.collector_map[task_id]
                self.ez_agent.traj_pool = self.traj_pool_map[task_id]
                self.ez_agent.replay_buffer = self.replay_buffer_map[task_id]
                self.ez_agent.batch_storage = self.batch_storage_map[task_id]
                self.ez_agent.prev_traj = self.prev_traj_map[task_id]
            else:
                raise Exception("No memory found for task id {}".format(task_id))

            self.ez_agent.make_batch()
            batch = self.ez_agent.batch_storage.pop()

            # recover agent memory
            self.ez_agent.collector = self.collector_map[self.ez_agent.mem_id]
            self.ez_agent.traj_pool = self.traj_pool_map[self.ez_agent.mem_id]
            self.ez_agent.replay_buffer = self.replay_buffer_map[self.ez_agent.mem_id]
            self.ez_agent.batch_storage = self.batch_storage_map[self.ez_agent.mem_id]
            self.ez_agent.prev_traj = self.prev_traj_map[self.ez_agent.mem_id]
            return batch
        else:
            self.ez_agent.make_batch()
            return self.ez_agent.batch_storage.pop()

