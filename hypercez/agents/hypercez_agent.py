from enum import IntEnum

import torch
import torch.nn as nn
from sympy import hermite_poly

from hypercez.agents import EZAgent
from hypercez.agents.agent_base import Agent, ActType
from hypercez.agents.ez_agent import AgentType
from hypercez.hypernet import build_hnet


class AgentCtrlType(IntEnum):
    DISCRETE = 0
    IMG_BASED = 1
    CONTINUOUS = 2


class HyperCEZAgent(Agent):
    def __init__(self,
                 hparams,
                 ez_agent: EZAgent = None,
                 hnet_map: dict[str, nn.Module] = None,
                 envs=None,
                 collector=None,
                 ctrl_type=AgentCtrlType.CONTINUOUS,
                 *args: str,
                 ):
        """
        An EfficientZeroV2 agent that uses hypernetworks for continual learning on each of its components.

        The keys of hnet_map or the strings provided as *args can include one or more of the following:
        [
            "representation_model",
            "dynamics_model",
            "reward_prediction_model",
            "value_policy_model",
            "projection_model",
            "projection_head_model"
        ]
        :param hparams: HParams object with added values
        :param ez_agent: EZAgent object correctly initialized
        :param hnet_map: a map with keys being one of the names mentioned above or "all" and values being nn.Module objects
        :param envs:
        :param collector:
        :param ctrl_type: Specifies the type of controller to be used (img based, discrete or continuous)
        :param args: a listing of names of agent including one or more of the names listed above or "all", to create hnets for if hnet_map is not provided.
        """
        super().__init__(hparams)
        self.ez_agent = ez_agent
        self.hnet_map = hnet_map
        self.envs = envs
        self.collector = collector
        self.control_type = ctrl_type
        self.hnet_component_names = list(args) if hnet_map is None else list(hnet_map.keys())

    def init_model(self):
        if self.ez_agent is None:
            self.ez_agent = EZAgent(
                self.hparams,
                AgentType(self.control_type.value)
            )
        if self.hnet_map is None:
            assert len(self.hnet_component_names) > 0, "either hnet_map or hnet_items should be provided!"
            self.hnet_map = {}
            for i in self.hnet_component_names:
                hnet = build_hnet(
                    hparams=self.hparams,
                    model=self.ez_agent.get_model(i)
                )
                for task in range(self.hparams.num_tasks):
                    hnet.add_task(task, self.hparams.std_normal_temb)
                self.hnet_map[i] = hnet


        assert self.ez_agent is not None and self.hnet_map is not None

    def act(self, obs, task_id=None, act_type: ActType = ActType.INITIAL):
        assert task_id is not None
        for hnet_name in self.hnet_component_names:
            new_weights = self.hnet_map[hnet_name](task_id)
            with torch.no_grad():
                for p, w in zip(self.ez_agent.model.__getattr__(hnet_name).parameters(), new_weights):
                    assert p.shape == w.shape
                    p.copy_(w)

            for p, w in zip(self.ez_agent.model.__getattr__(hnet_name).parameters(), new_weights):
                assert torch.equal(p, w)
        return self.ez_agent.act(obs, task_id, act_type)
