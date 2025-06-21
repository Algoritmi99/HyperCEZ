import torch

from hypercez import Hparams
from hypercez.agents import EZAgent
from hypercez.agents.ez_agent import AgentType
from hypercez.hypernet import build_hnet


def main():
    hparams = Hparams("pendulum")
    hparams.add_ez_hparams(2)
    hparams.add_hnet_hparams()

    ez_agent = EZAgent(
        hparams,
        AgentType.DMC_STATE
    )
    ez_agent.init_model_dmc_state()

    hnet = build_hnet(hparams, ez_agent.model.dynamics_model)
    hnet.add_task(0, 1)

    print([list(i.shape) for i in ez_agent.model.dynamics_model.parameters()])

    print([list(i.shape) for i in hnet(0)])

    new_weights = hnet(0)
    with torch.no_grad():
        for p, w in zip(ez_agent.model.dynamics_model.parameters(), new_weights):
            assert p.shape == w.shape
            p.copy_(w)

    for p, w in zip(ez_agent.model.dynamics_model.parameters(), new_weights):
        assert torch.equal(p, w)


if __name__ == "__main__":
    main()

# TODO:
#   1. train file that trains an arbitrary agent and resembles the HyperCRL trainer from outside
#   and EfficientZeroV2 trainer from inside for the agent.
#       --> An idea is object oriented training subroutines within the agent class of each agent.
#   2. a run script that starts the training routine for any given parameters (Agent, Env, etc.)
#   3. The main file must take parameters and serve as the API for the entire program
