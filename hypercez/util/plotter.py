import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
import os
import csv

class Plotter:
    def __init__(self, save_dir='plots', moving_avg_window=50):
        # Per-split containers: 'train' and 'eval'
        self.save_dir = save_dir
        self.moving_avg_window = moving_avg_window
        os.makedirs(save_dir, exist_ok=True)

        # TensorBoard related (lazy)
        self.tb_writer = None
        self.tb_log_dir = None
        self.global_step = {'train': 0, 'eval': 0}   # generic per-split step counters
        self.episode_idx = {'train': 0, 'eval': 0}   # per-split episode indices

        self.reset()

    # ---------------------- TensorBoard API ----------------------
    def enable_tensorboard(self, log_dir="runs/exp1", **writer_kwargs):
        """
        Lazily enable TensorBoard logging.
        Example: plotter.enable_tensorboard("runs/my_exp", flush_secs=10)
        """
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
            self.tb_writer.add_scalar(tag, value, step)

    def tb_close(self):
        if self.tb_writer is not None:
            self.tb_writer.flush()
            self.tb_writer.close()
            self.tb_writer = None

    # ---------------------- Core logic ----------------------
    def reset(self):
        # Track ongoing episode reward per split
        self.current_reward = {'train': 0.0, 'eval': 0.0}
        # Episode rewards per split
        self.episode_rewards = {
            'train': [],
            'eval':  []
        }
        # Task-wise rewards per split
        # dict[task_id] -> {'train': [..], 'eval': [..]}
        self.task_rewards = defaultdict(lambda: {'train': [], 'eval': []})

    def _validate_split(self, split):
        if split not in ('train', 'eval'):
            raise ValueError(f"split must be 'train' or 'eval', got {split!r}")

    def step(self, reward, done, task_id=None, split='train'):
        """
        Call this per environment step.
        If `done` is True, logs the finished episode to memory/CSV/TensorBoard.

        Args:
            reward (float): step reward
            done (bool): whether episode ended
            task_id (hashable|None): optional task identifier
            split (str): 'train' or 'eval'
        """
        self._validate_split(split)

        self.current_reward[split] += reward
        self.global_step[split] += 1  # useful if you want per-step logging elsewhere

        if done:
            ep_reward = self.current_reward[split]
            self.episode_rewards[split].append(ep_reward)
            if task_id is not None:
                self.task_rewards[task_id][split].append(ep_reward)

            # ---- TensorBoard episode-level logging ----
            if self.tb_writer is not None:
                # Total reward this episode (per split)
                self.tb_log_scalar(f"{split}/episode/total_reward",
                                   ep_reward,
                                   step=self.episode_idx[split])

                # Moving average (per split)
                split_eps = self.episode_rewards[split]
                if len(split_eps) >= min(self.moving_avg_window, 2):
                    window = min(self.moving_avg_window, len(split_eps))
                    mov_avg = float(np.mean(split_eps[-window:]))
                    self.tb_log_scalar(f"{split}/episode/moving_avg_{window}",
                                       mov_avg,
                                       step=self.episode_idx[split])

                # Optional: per-task episode reward (per split)
                if task_id is not None:
                    self.tb_log_scalar(f"{split}/task/{task_id}/total_reward",
                                       ep_reward,
                                       step=len(self.task_rewards[task_id][split]) - 1)

            # reset for next episode for this split only
            self.current_reward[split] = 0.0
            self.episode_idx[split] += 1

    # ---------------------- Plotting ----------------------
    def _compute_moving_avg(self, arr, window):
        if len(arr) == 0:
            return np.array([]), 0
        if len(arr) >= window:
            win = window
        else:
            win = max(1, min(window, len(arr)))
        moving = np.convolve(np.array(arr), np.ones(win) / win, mode='valid')
        return moving, win

    def plot(self, title='Episode Rewards'):
        train = self.episode_rewards['train']
        eval_ = self.episode_rewards['eval']
        if not train and not eval_:
            print("No data to plot.")
            return

        fig = plt.figure(figsize=(12, 6))

        # Train series
        if train:
            moving_train, win_t = self._compute_moving_avg(train, self.moving_avg_window)
            plt.plot(train, label='Train: Episode Reward', alpha=0.4)
            if moving_train.size > 0:
                plt.plot(
                    np.arange(len(moving_train)) + (len(train) - len(moving_train)),
                    moving_train,
                    label=f'Train: Moving Avg ({win_t})'
                )

        # Eval series
        if eval_:
            moving_eval, win_e = self._compute_moving_avg(eval_, self.moving_avg_window)
            plt.plot(eval_, label='Eval: Episode Reward', alpha=0.4)
            if moving_eval.size > 0:
                plt.plot(
                    np.arange(len(moving_eval)) + (len(eval_) - len(moving_eval)),
                    moving_eval,
                    label=f'Eval: Moving Avg ({win_e})'
                )

        plt.xlabel('Episode (per split)')
        plt.ylabel('Total Reward')
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # Save PNG
        path = os.path.join(self.save_dir, 'episode_rewards_train_eval.png')
        fig.savefig(path)
        plt.close(fig)
        print(f"Saved reward plot to: {path}")

        # Log the same figure to TensorBoard (optional)
        if self.tb_writer is not None:
            self.tb_writer.add_figure('plots/episode_rewards_train_eval',
                                      fig=None, global_step=max(self.episode_idx.values()))

    def plot_taskwise(self):
        if not self.task_rewards:
            print("No task-specific reward data.")
            return

        for task_id, splits in self.task_rewards.items():
            fig = plt.figure(figsize=(10, 4))
            # Train
            if splits['train']:
                plt.plot(splits['train'], label=f'Task {task_id} Train', alpha=0.6)
            # Eval
            if splits['eval']:
                plt.plot(splits['eval'], label=f'Task {task_id} Eval', alpha=0.6)

            plt.xlabel('Episode (per split)')
            plt.ylabel('Reward')
            plt.title(f'Rewards per Episode - Task {task_id}')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            path = os.path.join(self.save_dir, f'task_{task_id}_rewards_train_eval.png')
            fig.savefig(path)
            plt.close(fig)
            print(f"Saved task {task_id} reward plot to: {path}")

            if self.tb_writer is not None:
                # Use total episodes across both splits as a crude step
                total_eps = len(splits['train']) + len(splits['eval'])
                self.tb_writer.add_figure(f'plots/task_{task_id}_rewards_train_eval',
                                          fig=None, global_step=total_eps)

    # ---------------------- Persistence ----------------------
    def save_raw_data(self):
        """
        Saves all reward data (episode + task-wise) into CSV files inside save_dir,
        with split separation.
        """
        # Episode rewards per split
        for split in ('train', 'eval'):
            ep_path = os.path.join(self.save_dir, f'episode_rewards_{split}.csv')
            with open(ep_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Episode', 'Total_Reward'])
                for i, r in enumerate(self.episode_rewards[split]):
                    writer.writerow([i + 1, r])
            print(f"Saved {split} episode rewards to: {ep_path}")

        # Task-wise rewards with split
        task_path = os.path.join(self.save_dir, 'task_rewards.csv')
        with open(task_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Task_ID', 'Split', 'Episode_Index', 'Reward'])
            for task_id, splits in self.task_rewards.items():
                for split in ('train', 'eval'):
                    for i, r in enumerate(splits[split]):
                        writer.writerow([task_id, split, i + 1, r])
        print(f"Saved task-wise rewards to: {task_path}")
