import torch

from hypercez import Hparams
from hypercez.agents import EZAgent
from hypercez.agents.ez_agent import AgentType
from hypercez.agents.hypercez_agent import HyperCEZAgent, AgentCtrlType, HNetType
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
    ez_agent.init_model()

    hyper_cez_agent = HyperCEZAgent(
        hparams,
        ez_agent,
        HNetType.HNET,
        None,
        AgentCtrlType.CONTINUOUS,
        "representation_model",
        "dynamics_model",
        "reward_prediction_model",
        "value_policy_model",
        "projection_model",
        "projection_head_model"
    )

    hyper_cez_agent.init_model()

    cl_env_loader = CLEnvLoader(hparams.env)
    for i in range(hparams.num_tasks):
        cl_env_loader.add_task(i)

    plotter = Plotter()
    plotter.enable_tensorboard(log_dir='runs/ez_agent1')

    trainer = Trainer(
        hyper_cez_agent,
        hparams,
        cl_env_loader,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        plotter=plotter,
    )
    trainer.train(hparams.train["training_steps"])
    trainer.agent.save('agents/hypercez1.agent')


if __name__ == "__main__":
    main()

