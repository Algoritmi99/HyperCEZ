import torch

from hypercez import Hparams
from hypercez.agents import EZAgent
from hypercez.agents.hypercez_agent import HyperCEZAgent, HNetType, AgentCtrlType
from hypercez.envs.cl_env import CLEnvLoader
from hypercez.trainer import Trainer
from hypercez.util.plotter import Plotter


def main():
    # env_name = "half_cheetah"
    print([] == "singular")
    env_name = "pendulum"
    hparams = Hparams(env_name)

    hparams.add_ez_hparams(2)
    hparams.add_hnet_hparams()

    print(hparams.hnet_arch)

    ez_agent = EZAgent.load("agents/pendulum/EZagent_1999600.pth")
    assert isinstance(ez_agent, EZAgent)

    ez_agent.total_train_steps = 1500

    # print(ez_agent.model.state_dict())

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
        # "projection_model",
        # "projection_head_model"
    )

    hyper_cez_agent.init_model()

    cl_env_loader = CLEnvLoader(hparams.env)
    for i in range(hparams.num_tasks):
        cl_env_loader.add_task(i)

    plotter = Plotter()
    plotter.enable_tensorboard(log_dir='runs/hypercez_agent_t1' + "_" + env_name)

    trainer = Trainer(
        hyper_cez_agent,
        hparams,
        cl_env_loader,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        plotter=plotter,
    )
    trainer.train(
        # evaluate=True,
        # verbose=True,
        agent_name="hypercez_agent_initSafe_non_scaled_no_amp_pre"
    )
    trainer.agent.save('agents/ez1' + "_" + env_name + '.agent')


if __name__ == "__main__":
    main()
