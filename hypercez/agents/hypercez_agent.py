from enum import IntEnum

import torch
import torch.nn as nn

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
                 **kwargs
                 ):
        super().__init__(hparams)
        self.ez_agent = ez_agent
        self.hnet_map = hnet_map
        self.envs = envs
        self.collector = collector
        self.control_type = ctrl_type
        self.hnet_items = list(args) if hnet_map is None else list(hnet_map.keys())

    def init_model(self):
        if self.ez_agent is None:
            self.ez_agent = EZAgent(
                self.hparams,
                AgentType(self.control_type.value)
            )
        if self.hnet_map is None:
            assert len(self.hnet_items) > 0, "either hnet_map or hnet_items should be provided!"
            self.hnet_map = {
                i: build_hnet(
                    hparams=self.hparams,
                    model=self.ez_agent.get_model(i[0])
                ) for i in self.hnet_items}

        assert self.ez_agent is not None and self.hnet_map is not None

    def act(self, obs, task_id=None, act_type: ActType = ActType.INITIAL):
        assert task_id is not None
        for hnet_name in self.hnet_items:
            new_weights = self.hnet_map[hnet_name](task_id)
            with torch.no_grad():
                for p, w in zip(self.ez_agent.model.__getattr__(hnet_name).parameters(), new_weights):
                    assert p.shape == w.shape
                    p.copy_(w)

            for p, w in zip(self.ez_agent.model.__getattr__(hnet_name).parameters(), new_weights):
                assert torch.equal(p, w)

        return self.ez_agent.act(obs, task_id, act_type)