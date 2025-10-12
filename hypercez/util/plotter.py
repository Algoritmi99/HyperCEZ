import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
import os
import csv

class Plotter:
    def __init__(self, save_dir='plots', moving_avg_window=50):
        self.task_rewards = None
        self.episode_rewards = None
        self.current_reward = None
        self.save_dir = save_dir
        self.moving_avg_window = moving_avg_window
        os.makedirs(save_dir, exist_ok=True)

        # TensorBoard related (lazy)
        self.tb_writer = None
        self.tb_log_dir = None
        self.global_step = 0      # generic step counter if you want per-step logs
        self.episode_idx = 0      # episode index for TB scalars

        self.reset()

    # ---------------------- TensorBoard API ----------------------
    def enable_tensorboard(self, log_dir="runs/exp1", **writer_kwargs):
        """
        Lazily enable TensorBoard logging.
        Example: plotter.enable_tensorboard("runs/my_exp", flush_secs=10)
        """
        # Lazy import to avoid hard dependency when TB isn't used
        try:
            from torch.utils.tensorboard import SummaryWriter
        except Exception as e:
            raise RuntimeError(
                "TensorBoard (torch.utils.tensorboard) not available. "
                "Install with `pip install tensorboard torch`."
            ) from e

        self.tb_log_dir = log_dir
        self.tb_writer = SummaryWriter(log_dir=log_dir, **writer_kwargs)

    def tb_log_scalar(self, tag, value, step=None):
        if self.tb_writer is not None:
            if step is None:
                step = self.global_step
            self.tb_writer.add_scalar(tag, value, step)

    def tb_close(self):
        if self.tb_writer is not None:
            self.tb_writer.flush()
            self.tb_writer.close()
            self.tb_writer = None

    # ---------------------- Core logic ----------------------
    def reset(self):
        # For tracking ongoing episode
        self.current_reward = 0.0
        self.episode_rewards = []
        self.task_rewards = defaultdict(list)

    def step(self, reward, done, task_id=None):
        """
        Call this per environment step.
        If `done` is True, logs the finished episode to memory and TensorBoard.
        """
        self.current_reward += reward
        self.global_step += 1  # useful if you want per-step logging elsewhere

        if done:
            ep_reward = self.current_reward
            self.episode_rewards.append(ep_reward)
            if task_id is not None:
                self.task_rewards[task_id].append(ep_reward)

            # ---- TensorBoard episode-level logging ----
            if self.tb_writer is not None:
                # Total reward this episode
                self.tb_log_scalar("episode/total_reward", ep_reward, step=self.episode_idx)

                # Moving average (if enough episodes)
                if len(self.episode_rewards) >= min(self.moving_avg_window, 2):
                    window = min(self.moving_avg_window, len(self.episode_rewards))
                    mov_avg = np.mean(self.episode_rewards[-window:])
                    self.tb_log_scalar(f"episode/moving_avg_{window}", mov_avg, step=self.episode_idx)

                # Optional: per-task episode reward
                if task_id is not None:
                    self.tb_log_scalar(f"task/{task_id}/total_reward", ep_reward,
                                       step=len(self.task_rewards[task_id]) - 1)

            # reset for next episode
            self.current_reward = 0.0
            self.episode_idx += 1

    def plot(self, title='Episode Rewards'):
        if not self.episode_rewards:
            print("No data to plot.")
            return

        rewards = np.array(self.episode_rewards)
        if len(rewards) >= self.moving_avg_window:
            moving_avg = np.convolve(
                rewards, np.ones(self.moving_avg_window) / self.moving_avg_window, mode='valid'
            )
        else:
            # If not enough points, compute a shorter moving average to avoid empty plot
            win = max(1, min(self.moving_avg_window, len(rewards)))
            moving_avg = np.convolve(rewards, np.ones(win) / win, mode='valid')

        fig = plt.figure(figsize=(12, 6))
        plt.plot(rewards, label='Episode Reward', alpha=0.4)
        plt.plot(np.arange(len(moving_avg)) + (len(rewards) - len(moving_avg)),
                 moving_avg, label=f'Moving Avg ({len(rewards) if len(rewards) < self.moving_avg_window else self.moving_avg_window})')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # Save PNG
        path = os.path.join(self.save_dir, 'episode_rewards.png')
        fig.savefig(path)
        plt.close(fig)
        print(f"Saved reward plot to: {path}")

        # Log the same figure to TensorBoard (optional)
        if self.tb_writer is not None:
            # add_figure can take a Matplotlib figure and closes it if close=True
            self.tb_writer.add_figure('plots/episode_rewards', fig, global_step=self.episode_idx, close=False)

    def plot_taskwise(self):
        if not self.task_rewards:
            print("No task-specific reward data.")
            return

        for task_id, rewards in self.task_rewards.items():
            fig = plt.figure(figsize=(10, 4))
            plt.plot(rewards, label=f'Task {task_id} Rewards', alpha=0.6)
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.title(f'Rewards per Episode - Task {task_id}')
            plt.grid(True)
            plt.tight_layout()
            path = os.path.join(self.save_dir, f'task_{task_id}_rewards.png')
            fig.savefig(path)
            plt.close(fig)
            print(f"Saved task {task_id} reward plot to: {path}")

            if self.tb_writer is not None:
                self.tb_writer.add_figure(f'plots/task_{task_id}_rewards', fig,
                                          global_step=len(rewards), close=False)

    def save_raw_data(self):
        """
        Saves all reward data (episode + task-wise) into CSV files inside save_dir.
        """
        # Save episode rewards
        ep_path = os.path.join(self.save_dir, 'episode_rewards.csv')
        with open(ep_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Episode', 'Total_Reward'])
            for i, r in enumerate(self.episode_rewards):
                writer.writerow([i + 1, r])
        print(f"Saved episode rewards to: {ep_path}")

        # Save task-wise rewards
        task_path = os.path.join(self.save_dir, 'task_rewards.csv')
        with open(task_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Task_ID', 'Episode_Index', 'Reward'])
            for task_id, rewards in self.task_rewards.items():
                for i, r in enumerate(rewards):
                    writer.writerow([task_id, i + 1, r])
        print(f"Saved task-wise rewards to: {task_path}")
