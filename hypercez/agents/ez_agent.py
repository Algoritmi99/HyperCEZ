import copy
import math
from enum import IntEnum
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import L1Loss

from hypercez.agents.agent_base import Agent, ActType
from hypercez.agents.models.alt_model import DynamicsNetwork as AltDynamicsNetwork
from hypercez.agents.models.alt_model import RepresentationNetwork as AltRepresentationNetwork
from hypercez.agents.models.alt_model import RewardNetwork as AltRewardNetwork
from hypercez.agents.models.alt_model import RewardNetworkLSTM as AltRewardNetworkLSTM
from hypercez.agents.models.alt_model import ValuePolicyNetwork as AltValuePolicyNetwork
from hypercez.agents.models.base_model import DynamicsNetwork as BaseDynamicsNetwork
from hypercez.agents.models.base_model import ProjectionHeadNetwork as BaseProjectionHeadNetwork
from hypercez.agents.models.base_model import ProjectionNetwork as BaseProjectionNetwork
from hypercez.agents.models.base_model import RepresentationNetwork as BaseRepresentationNetwork
from hypercez.agents.models.base_model import SupportLSTMNetwork as BaseSupportLSTMNetwork
from hypercez.agents.models.base_model import SupportNetwork as BaseSupportNetwork
from hypercez.agents.models.base_model import ValuePolicyNetwork as BaseValuePolicyNetwork
from hypercez.ezv2.ez_model import EfficientZero
from hypercez.ezv2.mcts.mcts import MCTS
from hypercez.util.format import DiscreteSupport, formalize_obs_lst, LinearSchedule, prepare_obs_lst, FIFOQueue, symexp
from hypercez.util.replay_buffer import ReplayBuffer
from hypercez.util.trajectory import GameTrajectory
from hypercez.util.augmentation import Transforms
from hypercez.ezv2.loss import kl_loss, cosine_similarity_loss, continuous_loss, symlog_loss, Value_loss


def concat_trajs(items, hparams, agent_type):
    obs_lsts, reward_lsts, policy_lsts, action_lsts, pred_value_lsts, search_value_lsts, \
        bootstrapped_value_lsts = items
    traj_lst = []
    for obs_lst, reward_lst, policy_lst, action_lst, pred_value_lst, search_value_lst, bootstrapped_value_lst in \
            zip(obs_lsts, reward_lsts, policy_lsts, action_lsts, pred_value_lsts, search_value_lsts,
                bootstrapped_value_lsts):
        # traj = GameTrajectory(**self.config.env, **self.config.rl, **self.config.model)
        traj = GameTrajectory(
            n_stack=hparams.model["n_stack"],
            discount=hparams.reward_discount,
            gray_scale=False,
            unroll_steps=hparams.train["unroll_steps"],
            td_steps=hparams.train["td_steps"],
            td_lambda=hparams.train["td_lambda"],
            obs_shape=hparams.state_dim,
            max_size=hparams.data["trajectory_size"],
            image_based=agent_type == AgentType.ATARI or agent_type == AgentType.DMC_IMAGE,
            episodic=False,
            GAE_max_steps=hparams.model["GAE_max_steps"]
        )
        traj.obs_lst = obs_lst
        traj.reward_lst = reward_lst
        traj.policy_lst = policy_lst
        traj.action_lst = action_lst
        traj.pred_value_lst = pred_value_lst
        traj.search_value_lst = search_value_lst
        traj.bootstrapped_value_lst = bootstrapped_value_lst
        traj_lst.append(traj)
    return traj_lst


class AgentType(IntEnum):
    ATARI = 0
    DMC_IMAGE = 1
    DMC_STATE = 2


class DecisionModel(IntEnum):
    SELF_PLAY = 0
    REANALYZE = 1
    LATEST = 2
    CURRENT = 3

class EZAgent(Agent):
    def __init__(self, hparams, agent_type: AgentType, env_action_space=None):
        super(EZAgent, self).__init__(hparams)
        self.trained_steps = None
        self.scaler = None
        self.scheduler = None
        self.max_steps = None
        self.optimizer = None
        self.total_train_steps = None
        self.beta_schedule = None
        self.self_play_model = None
        self.reanalyze_model = None
        self.latest_model = None
        self.pool_size = 1
        self.collector: GameTrajectory | None = None
        self.traj_pool : list | None = None
        self.replay_buffer: ReplayBuffer | None = None
        self.batch_storage: FIFOQueue | None = None
        self.prev_traj = None
        self.transforms = None
        self.agent_type = agent_type
        self.stacked_obs = deque(maxlen=hparams.model["n_stack"])
        hparams.add_ez_hparams(agent_type.value)
        self.num_blocks = hparams.model["num_blocks"]
        self.num_channels = hparams.model[
            "num_channels"] if agent_type == AgentType.DMC_IMAGE or agent_type == AgentType.ATARI else None
        self.reduced_channels = hparams.model[
            "reduced_channels"] if agent_type == AgentType.DMC_IMAGE or agent_type == AgentType.ATARI else None
        self.fc_layers = hparams.model["fc_layers"]
        self.down_sample = hparams.model["down_sample"]
        self.state_norm = hparams.model["state_norm"]
        self.value_prefix = hparams.model["value_prefix"]
        self.init_zero = hparams.model["init_zero"]
        self.value_ensemble = hparams.model["value_ensemble"] if agent_type == AgentType.DMC_STATE else None
        self.action_embedding = hparams.model["action_embedding"]
        self.action_embedding_dim = hparams.model[
            "action_embedding_dim"] if not agent_type == AgentType.DMC_STATE else None
        self.value_policy_detach = hparams.model[
            "value_policy_detach"] if not agent_type == AgentType.DMC_STATE else None
        self.v_num = hparams.train["v_num"]
        self.env_action_space = env_action_space

    def init_model_atari(self):
        if isinstance(self.obs_shape, int):
            self.input_shape = copy.deepcopy(self.obs_shape) * self.hparams.model["n_stack"]
        else:
            self.input_shape = copy.deepcopy(self.obs_shape)
            self.input_shape[0] *= self.hparams.model["n_stack"]
        if self.down_sample:
            state_shape = (self.num_channels, math.ceil(self.obs_shape[1] / 16), math.ceil(self.obs_shape[2] / 16))
        else:
            state_shape = (self.num_channels, self.obs_shape[1], self.obs_shape[2])

        state_dim = state_shape[0] * state_shape[1] * state_shape[2]
        flatten_size = self.reduced_channels * state_shape[1] * state_shape[2]

        representation_model = BaseRepresentationNetwork(
            self.input_shape,
            self.num_blocks,
            self.num_channels,
            self.down_sample
        )

        dynamics_model = BaseDynamicsNetwork(
            self.num_blocks,
            self.num_channels,
            self.control_dim,
            action_embedding=self.action_embedding,
            action_embedding_dim=self.action_embedding_dim
        )

        valueReward_support = DiscreteSupport(self.hparams)
        value_policy_model = BaseValuePolicyNetwork(
            self.num_blocks,
            self.num_channels,
            self.reduced_channels,
            flatten_size,
            self.fc_layers,
            valueReward_support.size,
            self.control_dim,
            self.init_zero,
            value_policy_detach=self.value_policy_detach,
            v_num=self.v_num
        )

        if self.value_prefix:
            reward_prediction_model = BaseSupportLSTMNetwork(
                self.num_channels,
                self.reduced_channels,
                flatten_size,
                self.fc_layers,
                valueReward_support.size,
                self.hparams.model["lstm_hidden_size"],
                self.init_zero
            )
        else:
            reward_prediction_model = BaseSupportNetwork(
                self.num_channels,
                self.reduced_channels,
                flatten_size,
                self.fc_layers,
                valueReward_support.size,
                self.init_zero
            )

        projection_layers = self.hparams.model["projection_layers"]
        head_layers = self.hparams.model["projection_head_layers"]
        assert projection_layers[1] == head_layers[1]

        projection_model = BaseProjectionNetwork(
            state_dim, projection_layers[0], projection_layers[1]
        )
        projection_head_model = BaseProjectionHeadNetwork(
            projection_layers[1], head_layers[0], head_layers[1]
        )

        self.model = EfficientZero(
            representation_model,
            dynamics_model,
            reward_prediction_model,
            value_policy_model,
            projection_model,
            projection_head_model,
            self.hparams,
            state_norm=self.state_norm,
            value_prefix=self.value_prefix
        )

        return self.model

    def init_model_dmc_image(self):
        if isinstance(self.obs_shape, int):
            self.input_shape = copy.deepcopy(self.obs_shape) * self.hparams.model["n_stack"]
        else:
            self.input_shape = copy.deepcopy(self.obs_shape)
            self.input_shape[0] *= self.hparams.model["n_stack"]
        if self.down_sample:
            state_shape = (self.num_channels, math.ceil(self.obs_shape[1] / 16), math.ceil(self.obs_shape[2] / 16))
        else:
            state_shape = (self.num_channels, self.obs_shape[1], self.obs_shape[2])

        state_dim = state_shape[0] * state_shape[1] * state_shape[2]
        flatten_size = self.reduced_channels * state_shape[1] * state_shape[2]

        representation_model = BaseRepresentationNetwork(
            self.input_shape, self.num_blocks, self.num_channels, self.down_sample
        )

        is_continuous = (self.hparams.env_type == "DMC")

        value_output_size = self.hparams.model["value_support"]["size"] if self.hparams.model["value_support"][
                                                                               "type"] != 'symlog' else 1

        dynamics_model = BaseDynamicsNetwork(
            self.num_blocks,
            self.num_channels,
            self.control_dim,
            is_continuous,
            action_embedding=self.action_embedding,
            action_embedding_dim=self.action_embedding_dim
        )

        value_policy_model = BaseValuePolicyNetwork(
            self.num_blocks,
            self.num_channels,
            self.reduced_channels,
            flatten_size,
            self.fc_layers,
            value_output_size,
            self.control_dim * 2,
            self.init_zero,
            is_continuous,
            policy_distribution=self.hparams.model["policy_distribution"],
            v_num=self.v_num
        )

        valueReward_support = DiscreteSupport(self.hparams)
        reward_output_size = valueReward_support.size if self.hparams.model["reward_support"]["type"] != 'symlog' else 1

        if self.value_prefix:
            reward_prediction_model = BaseSupportLSTMNetwork(
                self.num_channels,
                self.reduced_channels,
                flatten_size,
                self.fc_layers,
                reward_output_size,
                self.hparams.model["lstm_hidden_size"],
                self.init_zero
            )
        else:
            reward_prediction_model = BaseSupportNetwork(
                self.num_channels,
                self.reduced_channels,
                flatten_size,
                self.fc_layers,
                reward_output_size,
                self.init_zero
            )

        projection_layers = self.hparams.model["projection_layers"]
        head_layers = self.hparams.model["projection_head_layers"]
        assert projection_layers[1] == head_layers[1]

        projection_model = BaseProjectionNetwork(
            state_dim, projection_layers[0], projection_layers[1]
        )

        projection_head_model = BaseProjectionHeadNetwork(
            projection_layers[1], head_layers[0], head_layers[1]
        )

        self.model = EfficientZero(
            representation_model,
            dynamics_model,
            reward_prediction_model,
            value_policy_model,
            projection_model,
            projection_head_model,
            self.hparams,
            state_norm=self.state_norm,
            value_prefix=self.value_prefix
        )

        return self.model

    def init_model_dmc_state(self):
        representation_model = AltRepresentationNetwork(
            self.obs_shape,
            self.hparams.model["n_stack"],
            self.num_blocks,
            self.hparams.model["rep_net_shape"],
            self.hparams.model["hidden_shape"],
        )

        valueReward_support = DiscreteSupport(self.hparams)
        value_output_size = valueReward_support.size if self.hparams.model["reward_support"]["type"] != 'symlog' else 1
        reward_output_size = value_output_size

        dynamics_model = AltDynamicsNetwork(
            self.hparams.model["hidden_shape"],
            self.control_dim,
            self.num_blocks,
            self.hparams.model["dyn_shape"],
            self.hparams.model["act_embed_shape"],
        )

        value_policy_model = AltValuePolicyNetwork(
            self.hparams.model["hidden_shape"],
            self.hparams.model["val_net_shape"],
            self.hparams.model["pi_net_shape"],
            self.control_dim,
            value_output_size,
            init_zero=self.init_zero,
            use_bn=self.hparams.model["use_bn"],
            p_norm=self.hparams.model["use_p_norm"],
            policy_distr=self.hparams.model["policy_distribution"],
            noisy=self.hparams.model["noisy_net"],
            value_support=valueReward_support,
            v_num=self.v_num
        )

        if self.hparams.model["value_prefix"]:
            reward_prediction_model = AltRewardNetworkLSTM(
                self.hparams.model["hidden_shape"],
                self.hparams.model["rew_net_shape"],
                reward_output_size,
                self.hparams.model["lstm_hidden_size"],
                init_zero=self.init_zero,
                use_bn=self.hparams.model["use_bn"],
            )
        else:
            reward_prediction_model = AltRewardNetwork(
                self.hparams.model["hidden_shape"],
                self.hparams.model["rew_net_shape"],
                reward_output_size,
                init_zero=self.init_zero,
                use_bn=self.hparams.model["use_bn"]
            )

        projection_model = nn.Sequential(
            nn.Linear(self.hparams.model["hidden_shape"], self.hparams.model["proj_hid_shape"]),
            nn.LayerNorm(self.hparams.model["proj_hid_shape"]),
            nn.ReLU(),
            nn.Linear(self.hparams.model["proj_hid_shape"], self.hparams.model["proj_hid_shape"]),
            nn.LayerNorm(self.hparams.model["proj_hid_shape"]),
            nn.ReLU(),
            nn.Linear(self.hparams.model["proj_hid_shape"], self.hparams.model["proj_shape"]),
            nn.LayerNorm(self.hparams.model["proj_shape"])
        )
        projection_head_model = nn.Sequential(
            nn.Linear(self.hparams.model["proj_shape"], self.hparams.model["pred_hid_shape"]),
            nn.LayerNorm(self.hparams.model["pred_hid_shape"]),
            nn.ReLU(),
            nn.Linear(self.hparams.model["pred_hid_shape"], self.hparams.model["pred_shape"]),
        )

        self.model = EfficientZero(
            representation_model,
            dynamics_model,
            reward_prediction_model,
            value_policy_model,
            projection_model,
            projection_head_model,
            self.hparams,
            state_norm=self.state_norm,
            value_prefix=self.value_prefix
        )

        return self.model

    def init_model(self):
        init_map = {
            AgentType.ATARI: self.init_model_atari,
            AgentType.DMC_IMAGE: self.init_model_dmc_image,
            AgentType.DMC_STATE: self.init_model_dmc_state,
        }
        return init_map[self.agent_type]()

    def reset(self, obs):
        self.stacked_obs = deque(
            [obs for _ in range(self.hparams.model["n_stack"])], maxlen=self.hparams.model["n_stack"]
        )
        self.prev_traj = None
        self.collector = GameTrajectory(
            n_stack=1,
            discount=self.hparams.reward_discount,
            obs_to_string=False,
            gray_scale=False,
            unroll_steps=self.hparams.train["unroll_steps"],
            td_steps=self.hparams.train["td_steps"],
            td_lambda=self.hparams.train["td_lambda"],
            obs_shape=self.hparams.state_dim,
            max_size=100,
            image_based=self.agent_type == AgentType.ATARI or self.agent_type == AgentType.DMC_IMAGE,
            episodic=False,
            GAE_max_steps=self.hparams.model["GAE_max_steps"],
            trajectory_size=self.hparams.data["trajectory_size"]
        )

    def reset_full_memory(self, obs):
        self.reset(obs)
        self.traj_pool = []

        self.replay_buffer = ReplayBuffer(
            batch_size=self.hparams.train["batch_size"],
            buffer_size=self.hparams.data["buffer_size"],
            top_transitions=self.hparams.data["top_transitions"],
            use_priority=self.hparams.priority["use_priority"],
            env=self.hparams.env,
            total_transitions=self.hparams.data["total_transitions"]
        )
        self.batch_storage = FIFOQueue()

    def get_model(self, model_name: str = None):
        if model_name is None:
            return self.model
        return self.model.__getattr__(model_name)

    def is_ready_for_training(self, task_id=None, pbar=None):
        if pbar is not None:
            pbar.total = self.hparams.train['start_transitions']
            pbar.n = self.replay_buffer.get_transition_num()
            pbar.refresh()
        if self.replay_buffer is None:
            return False
        return self.replay_buffer.get_transition_num() >= self.hparams.train['start_transitions']

    def done_training(self, task_id=None, pbar=None):
        if pbar is not None:
            pbar.total = self.total_train_steps
            pbar.n = self.trained_steps
            pbar.refresh()
        return self.trained_steps >= self.total_train_steps

    def act(self, obs, task_id=None, act_type: ActType = ActType.TRAIN, decision_model: DecisionModel = DecisionModel.SELF_PLAY):
        if decision_model == DecisionModel.SELF_PLAY:
            model = self.self_play_model
        elif decision_model == DecisionModel.LATEST:
            model = self.latest_model
        elif decision_model == DecisionModel.REANALYZE:
            model = self.reanalyze_model
        else:
            model = self.model

        if act_type == ActType.TRAIN:
            model.train()
        else:
            model.eval()

        self.stacked_obs.append(obs)

        current_stacked_obs = formalize_obs_lst(
            list(self.stacked_obs),
            image_based=self.agent_type == AgentType.DMC_IMAGE or self.agent_type == AgentType.ATARI
        ).to(self.device)

        with torch.no_grad():
            with torch.amp.autocast(self.device.type):
                states, values, policies = model.initial_inference(current_stacked_obs)

        values = values.detach().cpu().numpy().flatten()

        mcts = MCTS(
            num_actions=self.control_dim if self.agent_type == AgentType.ATARI
            else self.hparams.mcts["num_sampled_actions"],
            discount=self.reward_discount,
            env=self.hparams.env,
            **self.hparams.mcts,
            **self.hparams.model
        )

        if self.agent_type == AgentType.ATARI:
            if self.hparams.mcts["use_gumbel"]:
                r_values, r_policies, best_actions, _ = mcts.search(
                    model, states.shape[0], states, values, policies, use_gumble_noise=False, verbose=0
                )
            else:
                r_values, r_policies, best_actions, _ = mcts.search_ori_mcts(
                    model, states.shape[0], states, values, policies, use_noise=False
                )
        else:
            r_values, r_policies, best_actions, _, _, _ = mcts.search_continuous(
                model,
                states.shape[0],
                states,
                values,
                policies,
                use_gumble_noise=False,
                verbose=0,
                add_noise=False
            )

        if self.training_mode:
            self.collector.store_search_results(values, r_values, r_policies)
        return r_values, r_policies, best_actions

    def act_init(self, obs, task_id=None, act_type: ActType = ActType.INITIAL):
        return self.act(obs, task_id, act_type)

    def train(self):
        if self.hparams.optimizer["type"] == 'SGD':
            self.optimizer = optim.SGD(self.model.parameters(),
                                  lr=self.hparams.optimizer["lr"],
                                  weight_decay=self.hparams.optimizer["weight_decay"],
                                  momentum=self.hparams.optimizer["momentum"])
        elif self.hparams.optimizer["type"] == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters(),
                                   lr=self.hparams.optimizer["lr"],
                                   weight_decay=self.hparams.optimizer["weight_decay"])
        elif self.hparams.optimizer["type"] == 'AdamW':
            self.optimizer = optim.AdamW(self.model.parameters(),
                                    lr=self.hparams.optimizer["lr"],
                                    weight_decay=self.hparams.optimizer["weight_decay"])
        else:
            raise NotImplementedError

        if self.hparams.optimizer["lr_decay_type"] == 'cosine':
            self.max_steps = self.hparams.train["training_steps"] - int(
                self.hparams.train["training_steps"] * self.hparams.optimizer["lr_warm_up"]
            )
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.max_steps * 3, eta_min=0
            )
        elif self.hparams.optimizer["lr_decay_type"] == 'full_cosine':
            self.max_steps = self.hparams.train["training_steps"] - int(
                self.hparams.train["training_steps"] * self.hparams.optimizer["lr_warm_up"]
            )
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.max_steps // 2, eta_min=0
            )
        else:
            self.scheduler = None

        self.scaler = torch.amp.GradScaler()
        self.trained_steps = 0

        self.collector = GameTrajectory(
            n_stack=1,
            discount=self.hparams.reward_discount,
            obs_to_string=False,
            gray_scale=False,
            unroll_steps=self.hparams.train["unroll_steps"],
            td_steps=self.hparams.train["td_steps"],
            td_lambda=self.hparams.train["td_lambda"],
            obs_shape=self.hparams.state_dim,
            max_size=100,
            image_based=self.agent_type == AgentType.ATARI or self.agent_type == AgentType.DMC_IMAGE,
            episodic=False,
            GAE_max_steps=self.hparams.model["GAE_max_steps"],
            trajectory_size=self.hparams.data["trajectory_size"]
        )

        self.traj_pool = []

        self.replay_buffer = ReplayBuffer(
            batch_size=self.hparams.train["batch_size"],
            buffer_size=self.hparams.data["buffer_size"],
            top_transitions=self.hparams.data["top_transitions"],
            use_priority=self.hparams.priority["use_priority"],
            env=self.hparams.env,
            total_transitions=self.hparams.data["total_transitions"]
        )

        self.prev_traj = None

        self.total_train_steps = self.hparams.train['training_steps'] + self.hparams.train['offline_training_steps']
        self.beta_schedule = LinearSchedule(
            self.total_train_steps,
            initial_p=self.hparams.priority["priority_prob_beta"],
            final_p=1.0
        )

        self.self_play_model = copy.deepcopy(self.model)
        self.reanalyze_model = copy.deepcopy(self.model)
        self.latest_model = copy.deepcopy(self.model)
        self.batch_storage = FIFOQueue()

        if self.hparams.augmentation and self.agent_type == AgentType.DMC_IMAGE or self.agent_type == AgentType.ATARI:
            self.transforms = Transforms(self.hparams.augmentation, image_shape=(self.obs_shape[1], self.obs_shape[2]))

        self.training_mode = True

    def collect(self, x_t, u_t, reward, x_tt, task_id, done=False):
        assert self.training_mode, "collection only in training mode!"
        self.collector.append(u_t, x_tt, reward)
        self.collector.snapshot_lst.append([])

        if self.collector.is_full():
            if self.prev_traj is not None:
                self.save_previous_trajectory(self.prev_traj, self.collector)

            self.prev_traj = copy.deepcopy(self.collector)
            self.collector = GameTrajectory(
                n_stack=1,
                discount=self.hparams.reward_discount,
                obs_to_string=False,
                gray_scale=False,
                unroll_steps=self.hparams.train["unroll_steps"],
                td_steps=self.hparams.train["td_steps"],
                td_lambda=self.hparams.train["td_lambda"],
                obs_shape=self.hparams.state_dim,
                max_size=100,
                image_based=self.agent_type == AgentType.ATARI or self.agent_type == AgentType.DMC_IMAGE,
                episodic=False,
                GAE_max_steps=self.hparams.model["GAE_max_steps"],
                trajectory_size=self.hparams.data["trajectory_size"]
            )
            self.collector.init(list(self.stacked_obs))

        if done:
            if self.prev_traj is not None:
                self.save_previous_trajectory(self.prev_traj, self.collector)

            if len(self.collector) > 0:
                self.collector.pad_over([], [], [], [], [])
                self.collector.save_to_memory()
                self.put_trajs(self.collector)

    def save_previous_trajectory(self, prev_traj, game_traj, padding=True):
        """
        put the previous game trajectory into the pool if the current trajectory is full
        Parameters
        ----------
        :param prev_traj:
            previous game trajectory
        :param game_traj:
            current game trajectory
        :param padding:
            Add padding
        """
        if padding:
            # pad over last block trajectory
            if self.hparams.model["value_target"] == 'bootstrapped':
                gap_step = self.hparams.model["n_stack"] + self.hparams.train["td_steps"]
            else:
                extra = max(
                    0,
                    min(
                        int(1 / (1 - self.hparams.train["td_lambda"])),
                        self.hparams.model["GAE_max_steps"]
                    ) - self.hparams.train["unroll_steps"] - 1
                )
                gap_step = self.hparams.model["n_stack"] + 1 + extra + 1

            beg_index = self.hparams.model["n_stack"]
            end_index = beg_index + self.hparams.train["unroll_steps"]

            pad_obs_lst = game_traj.obs_lst[beg_index:end_index]

            pad_policy_lst = game_traj.policy_lst[0:self.hparams.train["unroll_steps"]]
            pad_reward_lst = game_traj.reward_lst[0:gap_step - 1]
            pad_pred_values_lst = game_traj.pred_value_lst[0:gap_step]
            pad_search_values_lst = game_traj.search_value_lst[0:gap_step]

            # pad over and save
            prev_traj.pad_over(pad_obs_lst, pad_reward_lst, pad_pred_values_lst, pad_search_values_lst,
                                          pad_policy_lst)
        prev_traj.save_to_memory()
        self.put_trajs(prev_traj)

        # reset last block
        prev_traj = None

    def put_trajs(self, traj):
        if self.hparams.priority["use_priority"]:
            traj_len = len(traj)
            pred_values = torch.from_numpy(np.array(traj.pred_value_lst)).to(self.device).float()
            # search_values = torch.from_numpy(np.array(traj.search_value_lst)).to(self.device).float()
            if self.hparams.model["value_target"] == 'bootstrapped':
                target_values = torch.from_numpy(np.asarray(traj.get_bootstrapped_value())).to(self.device).float()
            elif self.hparams.model["value_target"] == 'GAE':
                target_values = torch.from_numpy(np.asarray(traj.get_gae_value())).to(self.device).float()
            else:
                raise NotImplementedError
            priorities = L1Loss(
                reduction='none'
            )(
                pred_values[:traj_len], target_values[:traj_len]
            ).detach().cpu().numpy() + self.hparams.priority["min_prior"]
        else:
            priorities = None
        self.traj_pool.append(traj)
        # save the game histories and clear the pool
        if len(self.traj_pool) >= self.pool_size:
            self.replay_buffer.save_pools(self.traj_pool, priorities)
            del self.traj_pool[:]

    def get_temperature(self, trained_steps):
        if self.hparams.train["change_temperature"]:
            total_steps = self.hparams.train["training_steps"] + self.hparams.train["offline_training_steps"]
            # if self.config.env.env == 'Atari':
            if trained_steps < 0.5 * total_steps:   # prev 0.5
                return 1.0
            elif trained_steps < 0.75 * total_steps:    # prev 0.75
                return 0.5
            else:
                return 0.25
        else:
            return 1.0

    def make_batch(self):
        # ==============================================================================================================
        # Prepare
        # ==============================================================================================================
        beta = self.beta_schedule.value(self.trained_steps)
        batch_size = self.hparams.train["batch_size"]
        batch_context = self.replay_buffer.prepare_batch_context(
            batch_size=batch_size,
            alpha=self.hparams.priority["priority_prob_alpha"],
            beta=beta,
            rank=0,
            cnt=self.trained_steps
        )
        batch_context, validation_flag = batch_context
        (
            traj_lst,
            transition_pos_lst,
            indices_lst,
            weights_lst,
            make_time_lst,
            transition_num,
            prior_lst
        ) = batch_context

        traj_lst = concat_trajs(traj_lst, self.hparams, self.agent_type)

        reanalyze_batch_size = int(batch_size * self.hparams.train["reanalyze_ratio"])
        assert 0 < reanalyze_batch_size <= batch_size

        # ==============================================================================================================
        # Make Inputs
        # ==============================================================================================================
        collected_transitions = self.replay_buffer.get_transition_num()

        # make observations, actions and masks (if unrolled steps are out of trajectory)
        obs_lst, action_lst, mask_lst = [], [], []
        top_new_masks = []

        # prepare the inputs of a batch
        for i in range(batch_size):
            traj = traj_lst[i]
            state_index = transition_pos_lst[i]
            sample_idx = indices_lst[i]
            top_new_masks.append(int(sample_idx > collected_transitions - self.hparams.train["mixed_value_threshold"]))

            if self.agent_type == AgentType.ATARI:
                _actions = traj.action_lst[state_index:state_index + self.hparams.train["unroll_steps"]].tolist()
                _mask = [1. for _ in range(len(_actions))]
                _mask += [0. for _ in range(self.hparams.train["unroll_steps"] - len(_mask))]
                _actions += [np.random.randint(0, self.env_action_space.n) for _ in range(self.hparams.train["unroll_steps"] - len(_actions))]
            else:
                _actions = traj.action_lst[state_index:state_index + self.hparams.train["unroll_steps"]]
                _actions = np.array([a.squeeze() for a in _actions]) if self.control_dim > 1 else np.array([a[0] for a in _actions])
                _unroll_actions = traj.action_lst[state_index + 1:state_index + 1 + self.hparams.train["unroll_steps"]]
                # _unroll_actions = traj.action_lst[state_index:state_index + self.hparams.train["unroll_steps"]]
                _mask = [1. for _ in range(_unroll_actions.shape[0])]
                _mask += [0. for _ in range(self.hparams.train["unroll_steps"] - len(_mask))]
                _rand_actions = np.zeros((self.hparams.train["unroll_steps"] - _actions.shape[0], self.control_dim))
                _actions = np.concatenate((_actions, _rand_actions), axis=0)

            obs_lst.append(traj.get_index_stacked_obs(state_index, padding=True))
            action_lst.append(_actions)
            mask_lst.append(_mask)

        obs_lst = prepare_obs_lst(obs_lst, self.agent_type == AgentType.ATARI or self.agent_type == AgentType.DMC_IMAGE)
        inputs_batch = [obs_lst, action_lst, mask_lst, indices_lst, weights_lst, make_time_lst, prior_lst]
        inputs_batch = [np.asarray(inputs_batch[i]) for i in range(len(inputs_batch))]

        # ==============================================================================================================
        # make targets
        # ==============================================================================================================
        value_target = self.hparams.train["value_target"]
        value_target_type = self.hparams.model["value_target"]

        if value_target in ['sarsa', 'mixed', 'max']:
            if value_target_type == 'GAE':
                prepare_func = self.prepare_reward_value_gae
            elif value_target_type == 'bootstrapped':
                prepare_func = self.prepare_reward_value
            else:
                raise NotImplementedError
        elif value_target == 'search':
            prepare_func = self.prepare_reward
        else:
            raise NotImplementedError

        batch_value_prefixes, batch_values, td_steps, pre_calc, value_masks = prepare_func(
            traj_lst,
            transition_pos_lst,
            indices_lst,
            collected_transitions,
            self.trained_steps
        )

        (
            batch_policies_re,
            sampled_actions,
            best_actions,
            reanalyzed_values,
            pre_lst,
            policy_masks
        ) = self.prepare_policy_reanalyze(
            self.trained_steps,
            traj_lst[:reanalyze_batch_size],
            transition_pos_lst[:reanalyze_batch_size],
            indices_lst[:reanalyze_batch_size],
            state_lst=pre_calc[0],
            value_lst=pre_calc[1],
            policy_lst=pre_calc[2],
            policy_mask=pre_calc[3]
        )

        batch_policies = batch_policies_re

        if self.agent_type == AgentType.ATARI:
            batch_best_actions = np.asarray(best_actions).reshape(batch_size, self.hparams.train["unroll_steps"] + 1)
            batch_actions = sampled_actions.reshape(
                batch_size, self.hparams.train["unroll_steps"] + 1, -1, self.env_action_space.n
            )
        else:
            batch_best_actions = best_actions.reshape(
                batch_size,
                self.hparams.train["unroll_steps"] + 1,
                self.control_dim
            )
            batch_actions = sampled_actions.reshape(
                batch_size, self.hparams.train["unroll_steps"] + 1, -1, self.control_dim
            )

        targets_batch = [batch_value_prefixes,
                         batch_values,
                         batch_actions,
                         batch_policies,
                         batch_best_actions,
                         top_new_masks,
                         policy_masks,
                         reanalyzed_values]
        targets_batch = [np.asarray(targets_batch[i]) for i in range(len(targets_batch))]

        # ==============================================================================================================
        # push batch into batch queue
        # ==============================================================================================================
        batch = [inputs_batch, targets_batch]
        self.batch_storage.push(batch)
        return batch
        
    def prepare_reward_value_gae(self, traj_lst, transition_pos_lst, indices_lst, collected_transitions, trained_steps):
        # value prefix (or reward), value
        batch_value_prefixes, batch_values = [], []
        extra = max(
            0, 
            min(
                int(1 / (1 - self.hparams.train["td_lambda"])), self.hparams.model["GAE_max_steps"]
            ) - self.hparams.train["unroll_steps"] - 1
        )

        # init
        value_obs_lst, td_steps_lst, value_mask = [], [], []  # mask: 0 -> out of traj
        policy_obs_lst, policy_mask = [], []
        zero_obs = traj_lst[0].get_zero_obs(self.hparams.model["n_stack"], channel_first=False)

        # get obs_{t+k}
        for traj, state_index, idx in zip(traj_lst, transition_pos_lst, indices_lst):
            traj_len = len(traj)
            td_steps = 1

            # prepare the corresponding observations for bootstrapped values o_{t+k}
            traj_obs = traj.get_index_stacked_obs(state_index + td_steps, extra=extra)
            game_obs = traj.get_index_stacked_obs(state_index, extra=extra)
            for current_index in range(state_index, state_index + self.hparams.train["unroll_steps"] + 1 + extra):
                bootstrap_index = current_index + td_steps

                if bootstrap_index <= traj_len:
                    value_mask.append(1)
                    beg_index = bootstrap_index - (state_index + td_steps)
                    end_index = beg_index + self.hparams.model["n_stack"]
                    obs = traj_obs[beg_index:end_index]
                else:
                    value_mask.append(0)
                    obs = np.asarray(zero_obs)

                value_obs_lst.append(obs)
                td_steps_lst.append(td_steps)

                if current_index < traj_len:
                    policy_mask.append(1)
                    beg_index = current_index - state_index
                    end_index = beg_index + self.hparams.model["n_stack"]
                    obs = game_obs[beg_index:end_index]
                else:
                    policy_mask.append(0)
                    obs = np.asarray(zero_obs)
                policy_obs_lst.append(obs)

        # reanalyze the bootstrapped value v_{t+k}
        _, value_lst, _ = self.efficient_inference_reanalyze(value_obs_lst, only_value=True)
        state_lst, ori_cur_value_lst, policy_lst = self.efficient_inference_reanalyze(policy_obs_lst, only_value=False)
        # v_{t+k}
        batch_size = len(value_lst)
        value_lst = value_lst.reshape(-1) * (np.array([self.hparams.reward_discount for _ in range(batch_size)]) ** td_steps_lst)
        value_lst = value_lst * np.array(value_mask)
        # value_lst = np.zeros_like(value_lst)  # for unit test, remove if training
        value_lst = value_lst.tolist()

        cur_value_lst = ori_cur_value_lst.reshape(-1) * np.array(policy_mask)
        # cur_value_lst = np.zeros_like(cur_value_lst)    # for unit test, remove if training
        cur_value_lst = cur_value_lst.tolist()

        state_lst_cut, ori_cur_value_lst_cut, policy_lst_cut, policy_mask_cut = [], [], [], []
        for i in range(len(state_lst)):
            if i % (self.hparams.train["unroll_steps"] + extra + 1) < self.hparams.train["unroll_steps"] + 1:
                state_lst_cut.append(state_lst[i].unsqueeze(0))
                ori_cur_value_lst_cut.append(ori_cur_value_lst[i])
                policy_lst_cut.append(policy_lst[i].unsqueeze(0))
                policy_mask_cut.append(policy_mask[i])
        state_lst_cut = torch.cat(state_lst_cut, dim=0)
        ori_cur_value_lst_cut = np.asarray(ori_cur_value_lst_cut)
        policy_lst_cut = torch.cat(policy_lst_cut, dim=0)

        # v_{t} = r + ... + gamma ^ k * v_{t+k}
        value_index = 0
        td_lambdas = []
        for traj, state_index, idx in zip(traj_lst, transition_pos_lst, indices_lst):
            traj_len = len(traj)
            target_values = []
            target_value_prefixs = []

            delta_lambda = 0.1 * (collected_transitions - idx) / self.hparams.train["auto_td_steps"]
            if self.hparams.model["value_target"] in ['mixed', 'max']:
                delta_lambda = 0.0
            td_lambda = self.hparams.train["td_lambda"] - delta_lambda
            td_lambda = np.clip(td_lambda, 0.65, self.hparams.train["td_lambda"])
            td_lambdas.append(td_lambda)

            delta = np.zeros(self.hparams.train["unroll_steps"] + 1 + extra)
            advantage = np.zeros(self.hparams.train["unroll_steps"] + 1 + extra + 1)
            index = self.hparams.train["unroll_steps"] + extra
            for current_index in reversed(range(state_index, state_index + self.hparams.train["unroll_steps"] + 1 + extra)):
                bootstrap_index = current_index + td_steps_lst[value_index + index]

                for i, reward in enumerate(traj.reward_lst[current_index:bootstrap_index]):
                    value_lst[value_index + index] += reward * self.hparams.reward_discount ** i

                delta[index] = value_lst[value_index + index] - cur_value_lst[value_index + index]
                advantage[index] = delta[index] + self.hparams.reward_discount * td_lambda * advantage[index + 1]
                index -= 1

            target_values_tmp = advantage[
                                :self.hparams.train["unroll_steps"] + 1
                                ] + np.asarray(cur_value_lst)[
                                    value_index:value_index + self.hparams.train["unroll_steps"] + 1
                                ]

            horizon_id = 0
            value_prefix = 0.0
            for i, current_index in enumerate(range(state_index, state_index + self.hparams.train["unroll_steps"] + 1)):
                # reset every lstm_horizon_len
                if horizon_id % self.hparams.model["lstm_horizon_len"] == 0 and self.value_prefix:
                    value_prefix = 0.0
                horizon_id += 1

                if current_index < traj_len:
                    # Since the horizon is small and the discount is close to 1.
                    # Compute the reward sum to approximate the value prefix for simplification
                    if self.value_prefix:
                        value_prefix += traj.reward_lst[current_index]
                    else:
                        value_prefix = traj.reward_lst[current_index]
                    target_value_prefixs.append(value_prefix)
                else:
                    target_value_prefixs.append(value_prefix)

                if current_index <= traj_len:
                    target_values.append(target_values_tmp[i])
                else:
                    target_values.append(0)

            value_index += (self.hparams.train["unroll_steps"] + 1 + extra)
            batch_value_prefixes.append(target_value_prefixs)
            batch_values.append(target_values)

        value_index = 0
        value_masks, policy_masks = [], []
        for i, idx in enumerate(indices_lst):
            value_masks.append(int(idx > collected_transitions - self.hparams.train["mixed_value_threshold"]))
            value_index += (self.hparams.train["unroll_steps"] + 1 + extra)

        value_masks = np.asarray(value_masks)
        return np.asarray(batch_value_prefixes), np.asarray(batch_values), np.asarray(td_steps_lst).flatten(), \
               (state_lst_cut, ori_cur_value_lst_cut, policy_lst_cut, policy_mask_cut), value_masks

    def prepare_reward_value(self, traj_lst, transition_pos_lst, indices_lst, collected_transitions, trained_steps):
        # value prefix (or reward), value
        batch_value_prefixes, batch_values = [], []
        # search_values = []

        # init
        value_obs_lst, td_steps_lst, value_mask = [], [], []    # mask: 0 -> out of traj
        zero_obs = traj_lst[0].get_zero_obs(self.hparams.model["n_stack"], channel_first=False)

        # get obs_{t+k}
        for traj, state_index, idx in zip(traj_lst, transition_pos_lst, indices_lst):
            traj_len = len(traj)

            # off-policy correction: shorter horizon of td steps
            delta_td = (collected_transitions - idx) // self.hparams.train["auto_td_steps"]
            if self.hparams.train["value_target"] in ['mixed', 'max']:
                delta_td = 0
            td_steps = self.hparams.train["td_steps"] - delta_td
            # td_steps = self.td_steps  # for test off-policy issue

            td_steps = min(traj_len - state_index, td_steps)
            td_steps = np.clip(td_steps, 1, self.hparams.train["td_steps"]).astype(np.int32)

            obs_idx = state_index + td_steps

            # prepare the corresponding observations for bootstrapped values o_{t+k}
            traj_obs = traj.get_index_stacked_obs(state_index + td_steps)
            for current_index in range(state_index, state_index + self.hparams.train["unroll_steps"] + 1):
                td_steps = min(traj_len - current_index, td_steps)
                td_steps = max(td_steps, 1)
                bootstrap_index = current_index + td_steps

                if bootstrap_index <= traj_len:
                    value_mask.append(1)
                    beg_index = bootstrap_index - obs_idx
                    end_index = beg_index + self.hparams.model["n_stack"]
                    obs = traj_obs[beg_index:end_index]
                else:
                    value_mask.append(0)
                    obs = zero_obs

                value_obs_lst.append(obs)
                td_steps_lst.append(td_steps)

        # reanalyze the bootstrapped value v_{t+k}
        state_lst, value_lst, policy_lst = self.efficient_inference_reanalyze(value_obs_lst, only_value=True)
        batch_size = len(value_lst)
        value_lst = value_lst.reshape(-1) * (np.array([self.hparams.reward_discount for _ in range(batch_size)]) ** td_steps_lst)
        value_lst = value_lst * np.array(value_mask)
        # value_lst = np.zeros_like(value_lst)    # for unit test, remove if training
        value_lst = value_lst.tolist()

        # v_{t} = r + ... + gamma ^ k * v_{t+k}
        value_index = 0
        top_value_masks = []
        for traj, state_index, idx in zip(traj_lst, transition_pos_lst, indices_lst):
            traj_len = len(traj)
            target_values = []
            target_value_prefixs = []

            horizon_id = 0
            value_prefix = 0.0
            top_value_masks.append(int(idx > collected_transitions - self.hparams.train["mixed_value_threshold"]))
            for current_index in range(state_index, state_index + self.hparams.train["unroll_steps"] + 1):
                bootstrap_index = current_index + td_steps_lst[value_index]

                for i, reward in enumerate(traj.reward_lst[current_index:bootstrap_index]):
                    value_lst[value_index] += reward * self.hparams.reward_discount ** i

                # reset every lstm_horizon_len
                if horizon_id % self.hparams.model["lstm_horizon_len"] == 0 and self.value_prefix:
                    value_prefix = 0.0
                horizon_id += 1

                if current_index < traj_len:
                    # Since the horizon is small and the discount is close to 1.
                    # Compute the reward sum to approximate the value prefix for simplification
                    if self.value_prefix:
                        value_prefix += traj.reward_lst[current_index]
                    else:
                        value_prefix = traj.reward_lst[current_index]
                    target_value_prefixs.append(value_prefix)
                else:
                    target_value_prefixs.append(value_prefix)

                if current_index <= traj_len:
                    target_values.append(value_lst[value_index])
                else:
                    target_values.append(0)
                value_index += 1

            batch_value_prefixes.append(target_value_prefixs)
            batch_values.append(target_values)

        value_masks = np.asarray(top_value_masks)
        return np.asarray(batch_value_prefixes), np.asarray(batch_values), np.asarray(td_steps_lst).flatten(), \
               (None, None, None, None), value_masks

    def prepare_reward(self, traj_lst, transition_pos_lst, indices_lst, collected_transitions, trained_steps):
        # value prefix (or reward), value
        batch_value_prefixes = []

        # v_{t} = r + ... + gamma ^ k * v_{t+k}
        value_index = 0
        top_value_masks = []
        for traj, state_index, idx in zip(traj_lst, transition_pos_lst, indices_lst):
            traj_len = len(traj)
            target_value_prefixs = []

            horizon_id = 0
            value_prefix = 0.0
            top_value_masks.append(int(idx > collected_transitions - self.hparams.train["start_use_mix_training_steps"]))
            for current_index in range(state_index, state_index + self.hparams.train["unroll_steps"] + 1):

                # reset every lstm_horizon_len
                if horizon_id % self.hparams.model["lstm_horizon_len"] == 0 and self.value_prefix:
                    value_prefix = 0.0
                horizon_id += 1

                if current_index < traj_len:
                    # Since the horizon is small and the discount is close to 1.
                    # Compute the reward sum to approximate the value prefix for simplification
                    if self.value_prefix:
                        value_prefix += traj.reward_lst[current_index]
                    else:
                        value_prefix = traj.reward_lst[current_index]
                    target_value_prefixs.append(value_prefix)
                else:
                    target_value_prefixs.append(value_prefix)

                value_index += 1

            batch_value_prefixes.append(target_value_prefixs)

        value_masks = np.asarray(top_value_masks)
        batch_value_prefixes = np.asarray(batch_value_prefixes)
        batch_values = np.zeros_like(batch_value_prefixes)
        td_steps_lst = np.ones_like(batch_value_prefixes)
        return batch_value_prefixes, np.asarray(batch_values), td_steps_lst.flatten(), \
               (None, None, None, None), value_masks

    def efficient_inference_reanalyze(self, obs_lst, only_value=False, value_idx=0):
        batch_size = len(obs_lst)
        obs_lst = np.asarray(obs_lst)
        state_lst, value_lst, policy_lst = [], [], []
        # split a full batch into slices of mini_infer_size
        mini_batch = self.hparams.train["mini_batch_size"]
        slices = np.ceil(batch_size / mini_batch).astype(np.int32)
        with torch.no_grad():
            for i in range(slices):
                beg_index = mini_batch * i
                end_index = mini_batch * (i + 1)
                current_obs = obs_lst[beg_index:end_index]
                current_obs = formalize_obs_lst(
                    current_obs,
                    self.agent_type == AgentType.DMC_IMAGE or self.agent_type == AgentType.ATARI
                ).to(self.device)
                # obtain the statistics at current steps
                with torch.amp.autocast(self.device.type):
                    states, values, policies = self.reanalyze_model.initial_inference(current_obs)

                # process outputs
                values = values.detach().cpu().numpy().flatten()
                # concat
                value_lst.append(values)
                if not only_value:
                    state_lst.append(states)
                    policy_lst.append(policies)

        value_lst = np.concatenate(value_lst)
        if not only_value:
            state_lst = torch.cat(state_lst)
            policy_lst = torch.cat(policy_lst)
        return state_lst, value_lst, policy_lst

    def prepare_policy_reanalyze(self,
                                 trained_steps,
                                 traj_lst,
                                 transition_pos_lst,
                                 indices_lst,
                                 state_lst=None,
                                 value_lst=None,
                                 policy_lst=None,
                                 policy_mask=None):
        # policy
        reanalyzed_values = []
        batch_policies = []

        # init
        if value_lst is None:
            policy_obs_lst, policy_mask = [], []   # mask: 0 -> out of traj
            zero_obs = traj_lst[0].get_zero_obs(self.hparams.model["n_stack"], channel_first=False)

            # get obs_{t} instead of obs_{t+k}
            for traj, state_index in zip(traj_lst, transition_pos_lst):
                traj_len = len(traj)

                game_obs = traj.get_index_stacked_obs(state_index)
                for current_index in range(state_index, state_index + self.hparams.train["unroll_steps"] + 1):

                    if current_index < traj_len:
                        policy_mask.append(1)
                        beg_index = current_index - state_index
                        end_index = beg_index + self.hparams.model["n_stack"]
                        obs = game_obs[beg_index:end_index]
                    else:
                        policy_mask.append(0)
                        obs = np.asarray(zero_obs)
                    policy_obs_lst.append(obs)

            # reanalyze the search policy pi_{t}
            state_lst, value_lst, policy_lst = self.efficient_inference_reanalyze(policy_obs_lst, only_value=False)

        # tree search for policies
        batch_size = len(state_lst)

        # temperature
        temperature = self.get_temperature(trained_steps=trained_steps) #* np.ones((batch_size, 1))

        mcts = MCTS(
            num_actions=self.control_dim if self.agent_type == AgentType.ATARI
            else self.hparams.mcts["num_sampled_actions"],
            discount=self.reward_discount,
            env=self.hparams.env,
            **self.hparams.mcts,
            **self.hparams.model
        )

        if self.agent_type == AgentType.ATARI:
            if self.hparams.mcts["use_gumbel"]:
                r_values, r_policies, best_actions, _ = mcts.search(
                    self.reanalyze_model,
                    batch_size,
                    state_lst,
                    value_lst,
                    policy_lst,
                    use_gumble_noise=True,
                    temperature=temperature
                )
            else:
                r_values, r_policies, best_actions, _ = mcts.search_ori_mcts(
                    self.reanalyze_model,
                    batch_size,
                    state_lst,
                    value_lst,
                    policy_lst,
                    use_noise=True,
                    temperature=temperature,
                    is_reanalyze=True
                )
            sampled_actions = best_actions
            search_best_indexes = best_actions
        else:

            r_values, r_policies, best_actions, sampled_actions, search_best_indexes, _ = mcts.search_continuous(
                self.reanalyze_model,
                batch_size,
                state_lst,
                value_lst,
                policy_lst,
                temperature=temperature
            )

        # concat policy
        policy_index = 0
        policy_masks = []
        mismatch_index = []
        for traj, state_index, ind in zip(traj_lst, transition_pos_lst, indices_lst):
            target_policies = []
            search_values = []
            policy_masks.append([])
            for current_index in range(state_index, state_index + self.hparams.train["unroll_steps"] + 1):
                traj_len = len(traj)

                assert (current_index < traj_len) == (policy_mask[policy_index])
                if policy_mask[policy_index]:
                    target_policies.append(r_policies[policy_index])
                    search_values.append(r_values[policy_index])
                    # mask best-action & pi_prime mismatches
                    if r_policies[policy_index].argmax() != search_best_indexes[policy_index]:
                        policy_mask[policy_index] = 0
                        mismatch_index.append(ind + current_index - state_index)
                else:
                    search_values.append(0.0)
                    if self.agent_type != AgentType.ATARI:
                        target_policies.append([0 for _ in range(sampled_actions.shape[1])])
                    else:
                        target_policies.append([0 for _ in range(self.control_dim)])
                policy_masks[-1].append(policy_mask[policy_index])
                policy_index += 1
            batch_policies.append(target_policies)
            reanalyzed_values.append(search_values)

        policy_masks = np.asarray(policy_masks)
        return (batch_policies,
                sampled_actions,
                best_actions,
                reanalyzed_values,
                (state_lst, value_lst, policy_lst, policy_mask),
                policy_masks)

    def prepare_policy_non_reanalyze(self, traj_lst, transition_pos_lst):
        # policy
        batch_policies = []

        # load searched policy in self-play
        for traj, state_index in zip(traj_lst, transition_pos_lst):
            traj_len = len(traj)
            target_policies = []

            for current_index in range(state_index, state_index + self.hparams.train["unroll_steps"] + 1):
                if current_index < traj_len:
                    target_policies.append(traj.policy_lst[current_index])
                else:
                    target_policies.append([0 for _ in range(self.control_dim)])

            batch_policies.append(target_policies)
        return batch_policies

    def adjust_lr(self, optimizer, step_count, scheduler):
        lr_warm_step = int(self.hparams.train["training_steps"] * self.hparams.optimizer["lr_warm_up"])
        optimize_config = self.hparams.optimizer

        # adjust learning rate, step lr every lr_decay_steps
        if step_count < lr_warm_step:
            lr = optimize_config["lr"] * step_count / lr_warm_step
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        else:
            if self.hparams.optimizer["lr_decay_type"] == 'cosine':
                if scheduler is not None:
                    scheduler.step()
                lr = scheduler.get_last_lr()[0] # return a list
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            else:
                lr = optimize_config["lr"] * optimize_config["lr_decay_rate"] ** (
                            (step_count - lr_warm_step) // optimize_config["lr_decay_steps"])
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

        return lr

    def learn(self, task_id: int, verbose=False):
        assert self.optimizer is not None, "The agent must be in training mode. try train()!"
        self.make_batch()
        batch = self.batch_storage.pop()
        self.adjust_lr(self.optimizer, self.trained_steps, self.scheduler)

        if self.trained_steps % 30 == 0:
            self.latest_model = copy.deepcopy(self.model)

        if self.trained_steps % self.hparams.train["self_play_update_interval"] == 0:
            self.self_play_model = copy.deepcopy(self.model)

        if self.trained_steps % self.hparams.train["reanalyze_update_interval"] == 0:
            self.reanalyze_model = copy.deepcopy(self.model)

        weighted_loss, log_data = self.calc_loss(
            self.model,
            batch,
            self.replay_buffer,
            self.trained_steps,
        )

        scalers = self.update_weights(
            weighted_loss,
            self.model,
            self.optimizer,
            self.scaler,
            target_model=copy.deepcopy(self.reanalyze_model)
        )

        self.scaler = scalers[0]
        self.trained_steps += 1

    def calc_loss(self, model, batch, replay_buffer, step_count, extra_loss=None, extra_loss_weight=None, model_state=None):
        # init
        batch_size = self.hparams.train["batch_size"]
        image_channel = self.obs_shape[0] if (
                self.agent_type == AgentType.DMC_IMAGE or self.agent_type == AgentType.ATARI
        ) else self.obs_shape
        unroll_steps = self.hparams.train["unroll_steps"]
        n_stack = self.hparams.model["n_stack"]
        reward_hidden = self.init_reward_hidden(batch_size)
        loss_data = {}
        other_scalar = {}
        other_distribution = {}
        is_continuous = self.agent_type != AgentType.ATARI

        # obtain the batch data
        inputs_batch, targets_batch = batch
        (obs_batch_ori,
         action_batch,
         mask_batch,
         indices,
         weights_lst,
         make_time,
         prior_lst) = inputs_batch
        (target_value_prefixes,
         target_values,
         target_actions,
         target_policies,
         target_best_actions,
         top_value_masks,
         mismatch_masks,
         search_values) = targets_batch
        target_value_prefixes = target_value_prefixes[:, :unroll_steps]

        if self.agent_type == AgentType.DMC_IMAGE or self.agent_type == AgentType.ATARI:
            obs_batch_raw = torch.from_numpy(obs_batch_ori).to(self.device).float() / 255.
        else:
            obs_batch_raw = torch.from_numpy(obs_batch_ori).to(self.device).float()

        obs_batch = obs_batch_raw[:, 0: n_stack * image_channel]  # obs_batch: current observation
        obs_target_batch = obs_batch_raw[:, image_channel:]  # obs_target_batch: observation of next steps

        # augmentation
        obs_batch = self.transform(obs_batch).to(self.device)
        obs_target_batch = self.transform(obs_target_batch)

        # others to gpu
        if is_continuous:
            action_batch = torch.from_numpy(action_batch).float().to(self.device)
        else:
            action_batch = torch.from_numpy(action_batch).to(self.device).unsqueeze(-1).long()
        mask_batch = torch.from_numpy(mask_batch).to(self.device).float()
        weights = torch.from_numpy(weights_lst).to(self.device).float()

        max_value_target = np.array([target_values, search_values]).max(0)

        target_value_prefixes = torch.from_numpy(target_value_prefixes).to(self.device).float()
        target_values = torch.from_numpy(target_values).to(self.device).float()
        target_actions = torch.from_numpy(target_actions).to(self.device).float()
        target_policies = torch.from_numpy(target_policies).to(self.device).float()
        target_best_actions = torch.from_numpy(target_best_actions).to(self.device).float()
        top_value_masks = torch.from_numpy(top_value_masks).to(self.device).float()
        search_values = torch.from_numpy(search_values).to(self.device).float()
        max_value_target = torch.from_numpy(max_value_target).to(self.device).float()

        # transform value and reward to support
        target_value_prefixes_support = DiscreteSupport.scalar_to_vector(target_value_prefixes,
                                                                         **self.hparams.model["reward_support"])

        with torch.amp.autocast(self.device.type):
            states, values, policies = model.initial_inference(obs_batch, training=True, model_state=model_state)

        if self.hparams.model["value_support"]["type"] == 'symlog':
            scaled_value = symexp(values).min(0)[0]
        else:
            scaled_value = DiscreteSupport.vector_to_scalar(values, **self.hparams.model["value_support"]).min(0)[0]
        if is_continuous:
            scaled_value = scaled_value.clip(0, 1e5)

        # loss of first step
        # multi options (Value Loss)
        if self.hparams.train["value_target"] == 'sarsa':
            this_target_values = target_values
        elif self.hparams.train["value_target"] == 'search':
            this_target_values = search_values
        elif self.hparams.train["value_target"] == 'mixed':
            if step_count < self.hparams.train["start_use_mix_training_steps"]:
                this_target_values = target_values
            else:
                this_target_values = target_values * top_value_masks.unsqueeze(1).repeat(1, unroll_steps + 1) \
                                     + search_values * (1 - top_value_masks).unsqueeze(1).repeat(1, unroll_steps + 1)
        elif self.hparams.train["value_target"] == 'max':
            this_target_values = max_value_target
        else:
            raise NotImplementedError

        # update priority
        fresh_priority = L1Loss(reduction='none')(scaled_value.squeeze(-1),
                                                  this_target_values[:, 0]).detach().cpu().numpy()
        fresh_priority += self.hparams.priority["min_prior"]
        replay_buffer.update_priorities(indices, fresh_priority, make_time)

        value_loss = torch.zeros(batch_size).to(self.device)
        value_loss += Value_loss(values, this_target_values[:, 0], self.hparams)

        if is_continuous:
            policy_loss, entropy_loss = continuous_loss(
                policies, target_actions[:, 0], target_policies[:, 0],
                target_best_actions[:, 0],
                distribution_type=self.hparams.model["policy_distribution"]
            )
            mu = policies[:, :policies.shape[-1] // 2].detach().cpu().numpy().flatten()
            sigma = policies[:, policies.shape[-1] // 2:].detach().cpu().numpy().flatten()
            other_distribution.update({
                'dist/policy_mu': mu,
                'dist/policy_sigma': sigma,
            })

        else:
            policy_loss = kl_loss(policies, target_policies[:, 0])
            entropy_loss = torch.zeros(batch_size).to(self.device)

        value_prefix_loss = torch.zeros(batch_size).to(self.device)
        consistency_loss = torch.zeros(batch_size).to(self.device)
        policy_entropy_loss = torch.zeros(batch_size).to(self.device)
        policy_entropy_loss -= entropy_loss

        # unroll k steps recurrently
        with torch.amp.autocast(self.device.type):
            for step_i in range(unroll_steps):
                mask = mask_batch[:, step_i]
                states, value_prefixes, values, policies, reward_hidden = model.recurrent_inference(states,
                                                                                                    action_batch[:,
                                                                                                    step_i],
                                                                                                    reward_hidden,
                                                                                                    training=True,
                                                                                                    model_state=model_state)

                beg_index = image_channel * step_i
                end_index = image_channel * (step_i + n_stack)

                # consistency loss
                gt_next_states = model.do_representation(obs_target_batch[:, beg_index:end_index], model_state=model_state)

                # projection for consistency
                dynamic_states_proj = model.do_projection(states, with_grad=True, model_state=model_state)
                gt_states_proj = model.do_projection(gt_next_states, with_grad=False, model_state=model_state)
                consistency_loss += cosine_similarity_loss(dynamic_states_proj, gt_states_proj) * mask

                # reward, value, policy loss
                if self.hparams.model["reward_support"]["type"] == 'symlog':
                    value_prefix_loss += symlog_loss(value_prefixes, target_value_prefixes[:, step_i]) * mask
                else:
                    value_prefix_loss += kl_loss(value_prefixes, target_value_prefixes_support[:, step_i]) * mask

                value_loss += Value_loss(values, this_target_values[:, step_i + 1], self.hparams) * mask

                if is_continuous:
                    policy_loss_i, entropy_loss_i = continuous_loss(
                        policies, target_actions[:, step_i + 1], target_policies[:, step_i + 1],
                        target_best_actions[:, step_i + 1],
                        mask=mask,
                        distribution_type=self.hparams.model["policy_distribution"]
                    )
                    policy_loss += policy_loss_i
                    policy_entropy_loss -= entropy_loss_i
                else:
                    policy_loss_i = kl_loss(policies, target_policies[:, step_i + 1]) * mask
                    policy_loss += policy_loss_i

                # set half gradient due to two branches of states
                states.register_hook(lambda grad: grad * 0.5)

                # reset reward hidden
                if self.hparams.model["value_prefix"] and (step_i + 1) % self.hparams.model["lstm_horizon_len"] == 0:
                    reward_hidden = self.init_reward_hidden(batch_size)

        # total loss
        loss = (value_prefix_loss * self.hparams.train["reward_loss_coeff"]
                + value_loss * self.hparams.train["value_loss_coeff"]
                + policy_loss * self.hparams.train["policy_loss_coeff"]
                + consistency_loss * self.hparams.train["consistency_coeff"])

        if is_continuous:
            loss += policy_entropy_loss * self.hparams.train["entropy_coeff"]

        weighted_loss = (
                            (1 - (extra_loss_weight if extra_loss_weight is not None else 0)) * (weights * loss).mean()
                        ) +  (
                (extra_loss_weight if extra_loss_weight is not None else 1) * (extra_loss if extra_loss is not None else 0)
        )

        return weighted_loss, (loss_data, other_scalar, other_distribution)

    def update_weights(self, weighted_loss, model, optimizer, scaler, target_model=None):
        target_model.eval()
        # backward
        unroll_steps = self.hparams.train["unroll_steps"]
        gradient_scale = 1. / unroll_steps
        parameters = model.parameters()
        with torch.amp.autocast(self.device.type):
            weighted_loss.register_hook(lambda grad: grad * gradient_scale)
        optimizer.zero_grad()
        scaler.scale(weighted_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(parameters, self.hparams.train["max_grad_norm"])
        scaler.step(optimizer)
        scaler.update()

        if self.hparams.model["noisy_net"]:
            model.value_policy_model.reset_noise()
            target_model.value_policy_model.reset_noise()

        scalers = [scaler]
        return scalers

    def init_reward_hidden(self, batch_size):
        if self.hparams.model["value_prefix"]:
            reward_hidden = (torch.zeros(1, batch_size, self.hparams.model["lstm_hidden_size"]).to(self.device),
                             torch.zeros(1, batch_size, self.hparams.model["lstm_hidden_size"]).to(self.device))
        else:
            reward_hidden = None
        return reward_hidden

    def transform(self, observation):
        if self.transforms is not None:
            return self.transforms(observation)
        else:
            return observation

    def get_model_list(self):
        return [
            component_name for component_name in dir(self.model) if isinstance(
                getattr(self.model, component_name), nn.Module
            )
        ]

    @classmethod
    def load(cls, path, map_location='cpu'):
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        print([(i, v) for i, v in ckpt.items()])
        out = cls(ckpt["hparams"], ckpt["agent_type"])
        for i, v in ckpt.items():
            out.__setattr__(i, v)
        return out
