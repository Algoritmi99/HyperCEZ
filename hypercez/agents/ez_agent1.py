import copy
import math

from hypercez.agents.agent_base import Agent
from hypercez.ezv2.models.base_model import *
from hypercez.ezv2.models.ez_model import EfficientZero
from hypercez.util.format import DiscreteSupport


class EZAgent(Agent):
    def __init__(self, hparams):
        super(EZAgent, self).__init__(hparams)
        self.num_blocks = hparams.model["num_blocks"]
        self.num_channels = hparams.model["num_channels"]
        self.reduced_channels = hparams.model["reduced_channels"]
        self.fc_layers = hparams.model["fc_layers"]
        self.down_sample = hparams.model["down_sample"]
        self.state_norm = hparams.model["state_norm"]
        self.value_prefix = hparams.model["value_prefix"]
        self.init_zero = hparams.model["init_zero"]
        self.action_embedding = hparams.model["action_embedding"]
        self.action_embedding_dim = hparams.model["action_embedding_dim"]
        self.value_policy_detach = hparams.model["value_policy_detach"]
        self.input_shape = copy.deepcopy(self.obs_shape)


    def init_model(self):
        if self.down_sample:
            state_shape = (self.num_channels, math.ceil(self.obs_shape[1] / 16), math.ceil(self.obs_shape[2] / 16))
        else:
            state_shape = (self.num_channels, self.obs_shape[1], self.obs_shape[2])

        state_dim = state_shape[0] * state_shape[1] * state_shape[2]
        flatten_size = self.reduced_channels * state_shape[1] * state_shape[2]

        representation_model = RepresentationNetwork(self.input_shape, self.num_blocks, self.num_channels, self.down_sample)

        dynamics_model = DynamicsNetwork(self.num_blocks, self.num_channels, self.control_dim,
                                         action_embedding=self.action_embedding, action_embedding_dim=self.action_embedding_dim)

        valueReward_support = DiscreteSupport(self.hparams)
        value_policy_model = ValuePolicyNetwork(self.num_blocks, self.num_channels, self.reduced_channels, flatten_size,
                                                     self.fc_layers, valueReward_support.size,
                                                     self.control_dim, self.init_zero,
                                                     value_policy_detach=self.value_policy_detach,
                                                     v_num=self.hparams.train["v_num"])

        if self.value_prefix:
            reward_prediction_model = SupportLSTMNetwork(self.num_channels, self.reduced_channels,
                                           flatten_size, self.fc_layers, valueReward_support.size,
                                           self.hparams.model["lstm_hidden_size"], self.init_zero)
        else:
            reward_prediction_model = SupportNetwork(self.num_channels, self.reduced_channels,
                                           flatten_size, self.fc_layers, valueReward_support.size,
                                           self.init_zero)

        projection_layers = self.hparams.model["projection_layers"]
        head_layers = self.hparams.model["projection_head_layers"]
        assert projection_layers[1] == head_layers[1]

        projection_model = ProjectionNetwork(state_dim, projection_layers[0], projection_layers[1])
        projection_head_model = ProjectionHeadNetwork(projection_layers[1], head_layers[0], head_layers[1])

        self.model = EfficientZero(representation_model, dynamics_model, reward_prediction_model, value_policy_model,
                                 projection_model, projection_head_model, self.hparams,
                                 state_norm=self.state_norm, value_prefix=self.value_prefix)

        return self.model


