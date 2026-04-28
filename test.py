import os
from pathlib import Path

import torch

from hypercez import Hparams
from hypercez.agents import EZAgent
from hypercez.agents.ez_agent import AgentType
from hypercez.agents.hypercez_delta_agent import HyperCEZDeltaAgent
from hypercez.envs.cl_env import CLEnvLoader
from hypercez.util import extract_observation


def collect_until_ready(agent, env, task_id=0, max_steps=5000):
    obs = extract_observation(env.reset())
    agent.reset(obs)
    steps = 0
    while not agent.is_ready_for_training(task_id=task_id):
        _, _, action = agent.act_init(obs, task_id=task_id)
        next_obs, reward, terminated, truncated, _ = env.step(action.reshape(env.action_space.shape))
        done = terminated or truncated
        agent.collect(obs, action, reward, next_obs, task_id=task_id, done=done)
        obs = extract_observation(env.reset()) if done else next_obs
        if done:
            agent.reset(obs)
        steps += 1
        if steps >= max_steps:
            raise RuntimeError("Reached max_steps before becoming ready for training.")
    return steps


def collect_one_step(agent, env, task_id=0):
    obs = extract_observation(env.reset())
    agent.reset(obs)
    _, _, action = agent.act(obs, task_id=task_id)
    next_obs, reward, terminated, truncated, _ = env.step(action.reshape(env.action_space.shape))
    agent.collect(obs, action, reward, next_obs, task_id=task_id, done=(terminated or truncated))


def main():
    env_name = "half_cheetah"
    hparams = Hparams(env_name)
    hparams.add_ez_hparams(2)
    hparams.add_hnet_hparams()

    print(hparams.max_iteration)

    # Keep this smoke test short.
    hparams.num_tasks = 1
    hparams.train["start_transitions"] = 32
    hparams.train["batch_size"] = 8
    hparams.train["mini_batch_size"] = 8
    hparams.train["training_steps"] = 2
    hparams.train["offline_training_steps"] = 0
    hparams.train["self_play_update_interval"] = 50
    hparams.train["reanalyze_update_interval"] = 50

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cl_env_loader = CLEnvLoader(hparams.env, seed=42)
    cl_env_loader.add_task(0)
    env = cl_env_loader.get_env(0)

    ez_agent = EZAgent(hparams, AgentType.DMC_STATE)
    ez_agent.init_model()

    agent = HyperCEZDeltaAgent(
        hparams,
        ez_agent,
        "representation_model",
        "dynamics_model",
        "reward_prediction_model",
        "value_policy_model",
    )
    agent.init_model()
    agent.to(device)
    agent.train()

    warmup_steps = collect_until_ready(agent, env, task_id=0)
    print(f"Warmup complete with {warmup_steps} transitions.")

    agent.learn(task_id=0)
    print("First training step complete.")

    save_path = Path("agents/test_hypercez_delta_resume.pth")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    agent.save(save_path)
    print(f"Saved checkpoint to {save_path}.")

    loaded_agent = HyperCEZDeltaAgent.load(save_path, map_location=device)
    loaded_agent.to(device)
    print("Loaded checkpoint.")

    # Continue training from loaded state.
    collect_one_step(loaded_agent, env, task_id=0)
    loaded_agent.learn(task_id=0)
    print("Resumed training step complete after load.")

    # Optional cleanup to keep workspace tidy.
    if os.environ.get("KEEP_TEST_CHECKPOINT", "0") != "1":
        save_path.unlink(missing_ok=True)
        print("Deleted temporary checkpoint file.")


if __name__ == "__main__":
    main()