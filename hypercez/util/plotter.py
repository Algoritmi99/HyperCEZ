import matplotlib.pyplot as plt
from collections import defaultdict, deque
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

        self.reset()

    def reset(self):
        # For tracking ongoing episode
        self.current_reward = 0.0
        self.episode_rewards = []
        self.task_rewards = defaultdict(list)

    def step(self, reward, done, task_id=None):
        self.current_reward += reward
        if done:
            self.episode_rewards.append(self.current_reward)
            if task_id is not None:
                self.task_rewards[task_id].append(self.current_reward)
            self.current_reward = 0.0

    def plot(self, title='Episode Rewards'):
        if not self.episode_rewards:
            print("No data to plot.")
            return

        rewards = np.array(self.episode_rewards)
        moving_avg = np.convolve(rewards, np.ones(self.moving_avg_window) / self.moving_avg_window, mode='valid')

        plt.figure(figsize=(12, 6))
        plt.plot(rewards, label='Episode Reward', alpha=0.4)
        plt.plot(np.arange(len(moving_avg)) + self.moving_avg_window - 1, moving_avg, label=f'Moving Avg ({self.moving_avg_window})', color='orange')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        path = os.path.join(self.save_dir, 'episode_rewards.png')
        plt.savefig(path)
        plt.close()
        print(f"Saved reward plot to: {path}")

    def plot_taskwise(self):
        if not self.task_rewards:
            print("No task-specific reward data.")
            return

        for task_id, rewards in self.task_rewards.items():
            plt.figure(figsize=(10, 4))
            plt.plot(rewards, label=f'Task {task_id} Rewards', alpha=0.6)
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.title(f'Rewards per Episode - Task {task_id}')
            plt.grid(True)
            plt.tight_layout()
            path = os.path.join(self.save_dir, f'task_{task_id}_rewards.png')
            plt.savefig(path)
            plt.close()
            print(f"Saved task {task_id} reward plot to: {path}")

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
