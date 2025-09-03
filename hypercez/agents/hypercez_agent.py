from enum import IntEnum

import torch
import torch.nn as nn

from hypercez.agents import EZAgent
from hypercez.agents.agent_base import Agent, ActType
from hypercez.agents.ez_agent import AgentType
from hypercez.hypernet import build_hnet
from hypercez.util.hypercez_data_mng import HyperCEZDataManager
from hypercez.hypernet.hypercl.utils import hnet_regularizer as hreg
from hypercez.hypernet.hypercl.utils import ewc_regularizer as ewc
from hypercez.hypernet.hypercl.utils import si_regularizer as si
from hypercez.hypernet.hypercl.utils import optim_step as opstep


class AgentCtrlType(IntEnum):
    DISCRETE = 0
    IMG_BASED = 1
    CONTINUOUS = 2

class HNetType(IntEnum):
    HNET = 0
    HNET_SI = 1
    HNET_EWC = 2
    HNET_MT = 3
    HNET_REPLAY = 4

class HyperCEZAgent(Agent):
    def __init__(self,
                 hparams,
                 ez_agent: EZAgent = None,
                 hnet_type: HNetType = HNetType.HNET,
                 hnet_map: dict[str, nn.Module] = None,
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
        :param collector:
        :param ctrl_type: Specifies the type of controller to be used (img based, discrete or continuous)
        :param args: a listing of names of agent including one or more of the names listed above or "all", to create hnets for if hnet_map is not provided.
        """
        super().__init__(hparams)
        self.reg_targets = None
        self.mlls = None
        self.theta_optims = None
        self.emb_optims = None
        self.regularized_params = None
        self.fisher_ests = None
        self.si_omegas = None
        self.ez_agent = ez_agent
        self.hnet_type = hnet_type
        self.hnet_map = hnet_map
        self.collector = collector
        self.control_type = ctrl_type
        self.hnet_component_names = list(args) if hnet_map is None else list(hnet_map.keys())
        self.__memory_manager = HyperCEZDataManager(self.ez_agent)
        self.__training_mode = False

    def init_model(self):
        if self.ez_agent is None:
            self.ez_agent = EZAgent(
                self.hparams,
                AgentType(self.control_type.value)
            )
        if self.hnet_map is None:
            assert len(self.hnet_component_names) > 0, "either hnet_map or hnet_items should be provided!"
            self.hnet_map = {}
            for hnet_comp in self.hnet_component_names:
                hnet = build_hnet(
                    hparams=self.hparams,
                    model=self.ez_agent.get_model(hnet_comp)
                )
                for task in range(self.hparams.num_tasks):
                    hnet.add_task(task, self.hparams.std_normal_temb)
                self.hnet_map[hnet_comp] = hnet


        assert self.ez_agent is not None and self.hnet_map is not None

    def generate_main_weights(self, task_id):
        assert self.hnet_map is not None
        for hnet_name in self.hnet_component_names:
            new_weights = self.hnet_map[hnet_name](task_id)
            with torch.no_grad():
                for p, w in zip(self.ez_agent.model.__getattr__(hnet_name).parameters(), new_weights):
                    assert p.shape == w.shape
                    p.copy_(w)
            for p, w in zip(self.ez_agent.model.__getattr__(hnet_name).parameters(), new_weights):
                assert torch.equal(p, w)
        return True

    def act(self, obs, task_id=None, act_type: ActType = ActType.INITIAL):
        assert self.hnet_map is not None
        for hnet_name in self.hnet_component_names:
            new_weights = self.hnet_map[hnet_name](task_id)
            with torch.no_grad():
                for p, w in zip(self.ez_agent.model.__getattr__(hnet_name).parameters(), new_weights):
                    assert p.shape == w.shape
                    p.copy_(w)
            for p, w in zip(self.ez_agent.model.__getattr__(hnet_name).parameters(), new_weights):
                assert torch.equal(p, w)
        return self.ez_agent.act(obs, task_id, act_type)

    def act_init(self, obs, task_id=None, act_type: ActType = ActType.INITIAL):
        return self.act(obs, task_id, act_type)

    def collect(self, x_t, u_t, reward, x_tt, task_id, done=False):
        self.__memory_manager.collect(x_t, u_t, reward, x_tt, task_id, done=done)

    def train(self, total_train_steps):
        # structure of keys: hnet_name -> task_id -> object
        self.reg_targets = {hnet_name: {} for hnet_name in self.hnet_component_names}
        self.mlls = {hnet_name: {} for hnet_name in self.hnet_component_names}
        self.theta_optims = {hnet_name: {} for hnet_name in self.hnet_component_names}
        self.emb_optims = {hnet_name: {} for hnet_name in self.hnet_component_names}
        self.regularized_params = {hnet_name: {} for hnet_name in self.hnet_component_names}
        self.fisher_ests = {hnet_name: {} for hnet_name in self.hnet_component_names}
        self.si_omegas = {hnet_name: {} for hnet_name in self.hnet_component_names}

        for hnet_name in self.hnet_component_names:
            for task_id in range(self.hparams.num_tasks):
                self.reg_targets[hnet_name][task_id] = hreg.get_current_targets(task_id, self.hnet_map[hnet_name])




        self.__training_mode = True

    def __fwd_pass(self, obs, task_id):
        pass

    def learn(self, task_id):
        pass


