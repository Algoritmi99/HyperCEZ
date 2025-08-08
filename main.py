import torch

from hypercez import Hparams
from hypercez.agents import EZAgent
from hypercez.agents.ez_agent import AgentType
from hypercez.envs.cl_env import CLEnvLoader

from hypercez.trainer import Trainer
from hypercez.util.plotter import Plotter


def main():
    hparams = Hparams("half_cheetah")

    hparams.add_ez_hparams(2)
    hparams.add_hnet_hparams()

    ez_agent = EZAgent(
        hparams,
        AgentType.DMC_STATE
    )
    ez_agent.init_model_dmc_state()

    cl_env_loader = CLEnvLoader(hparams.env)
    for i in range(hparams.num_tasks):
        cl_env_loader.add_task(i)

    plotter = Plotter()

    trainer = Trainer(
        ez_agent,
        hparams,
        cl_env_loader,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        plotter=plotter,
    )
    trainer.train(hparams.train["training_steps"])


if __name__ == "__main__":
    main()

