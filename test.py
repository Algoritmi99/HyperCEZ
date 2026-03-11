import torch

from hypercez import Hparams
from hypercez.agents.hypercez_delta_agent import HyperCEZDeltaAgent
from hypercez.envs.cl_env import CLEnvLoader
from hypercez.trainer import Trainer
from hypercez.util.plotter import Plotter


def main():
    env_name = "half_cheetah"
    # env_name = "pendulum"
    hparams = Hparams(env_name)

    hparams.add_ez_hparams(2)
    hparams.add_hnet_hparams()

    print(hparams.hnet_arch)

    hyper_cez_agent = HyperCEZDeltaAgent.load(
        "./agents/half_cheetah/agent_999990hypercez_all_agent_initSafe.pth",
        map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    hyper_cez_agent.hparams = hparams

    cl_env_loader = CLEnvLoader(hparams.env, seed=42)
    for i in range(hparams.num_tasks):
        cl_env_loader.add_task(i)

    plotter = Plotter()
    plotter.enable_tensorboard(log_dir='runs/hypercez' + "_" + env_name)

    trainer = Trainer(
        hyper_cez_agent,
        hparams,
        cl_env_loader,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        plotter=plotter,
    )
    trainer.train(
        evaluate=True,
        # verbose=True,
        agent_name="hypercez_all_agent_initSafe"
    )
    trainer.agent.save('agents/hypercez1' + "_" + env_name + '.agent')


if __name__ == "__main__":
    main()
