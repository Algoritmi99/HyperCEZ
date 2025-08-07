import torch

from hypercez import Hparams
from hypercez.agents import EZAgent
from hypercez.agents.ez_agent import AgentType
from hypercez.envs.cl_env import CLEnvLoader

from hypercez.trainer import Trainer


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

    trainer = Trainer(
        ez_agent,
        hparams,
        cl_env_loader,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    trainer.train(5000)


if __name__ == "__main__":
    main()

    # TODO:
    #   1. train file that trains an arbitrary agent and resembles the HyperCRL trainer from outside
    #   and EfficientZeroV2 trainer from inside for the agent.
    #       --> An idea is object oriented training subroutines within the agent class of each agent.
    #   2. a run script that starts the training routine for any given parameters (Agent, Env, etc.)
    #   3. The main file must take parameters and serve as the API for the entire program
