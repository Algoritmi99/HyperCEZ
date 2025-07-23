import copy
import math
from enum import IntEnum
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler as GradScaler
from torch.cuda.amp import autocast as autocast
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
from hypercez.util.format import DiscreteSupport, formalize_obs_lst, LinearSchedule, prepare_obs_lst
from hypercez.util.replay_buffer import ReplayBuffer
from hypercez.util.trajectory import GameTrajectory


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

# todo: update code to read n_stack from config everywhere

class EZAgent(Agent):
    def __init__(self, hparams, agent_type: AgentType, env_action_space=None):
        super(EZAgent, self).__init__(hparams)
        self.trained_steps = None
        self.eval_counter = None
        self.prev_eval_counter = None
        self.eval_best_score = None
        self.eval_score = None
        self.transition_num = None
        self.traj_num = None
        self.self_play_return = None
        self.recent_weights = None
        self.step_count = None
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
        self.prev_traj = None
        self.agent_type = agent_type
        self.stacked_obs = deque(maxlen=hparams.model["stacked_obs"])
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
        self.mcts = MCTS(
            num_actions=self.control_dim if self.agent_type == AgentType.ATARI
                else self.hparams.mcts["num_sampled_actions"],
            discount=self.reward_discount,
            env=self.hparams.env,
            **self.hparams.mcts,
            **self.hparams.model
        )
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
            [obs for _ in range(self.hparams.model["n_stacked"])], maxlen=self.hparams.model["n_stacked"]
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

    def get_model(self, model_name: str = None):
        if model_name is None:
            return self.model
        return self.model.__getattr__(model_name)

    def act(self, obs, task_id=None, act_type: ActType = ActType.INITIAL, decision_model: DecisionModel = DecisionModel.SELF_PLAY):
        if decision_model == DecisionModel.SELF_PLAY:
            model = self.self_play_model
        elif decision_model == DecisionModel.LATEST:
            model = self.latest_model
        elif decision_model == DecisionModel.REANALYZE:
            model = self.reanalyze_model
        else:
            model = self.model
        self.stacked_obs.append(obs)
        current_stacked_obs = formalize_obs_lst(
            list(self.stacked_obs),
            image_based=self.agent_type == AgentType.DMC_IMAGE or self.agent_type == AgentType.ATARI
        )

        with torch.no_grad():
            with autocast():
                states, values, policies = model.initial_inference(current_stacked_obs)

        values = values.detach().cpu().numpy().flatten()

        if self.agent_type == AgentType.ATARI:
            if self.hparams.mcts["use_gumbel"]:
                r_values, r_policies, best_actions, _ = self.mcts.search(
                    model, states.shape[0], states, values, policies, use_gumble_noise=False, verbose=0
                )
            else:
                r_values, r_policies, best_actions, _ = self.mcts.search_ori_mcts(
                    model, states.shape[0], states, values, policies, use_noise=False
                )
        else:
            r_values, r_policies, best_actions, _, _, _ = self.mcts.search_continuous(
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

    def train(self, total_train_steps):
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

        self.scaler = GradScaler()
        self.step_count = 0
        self.recent_weights = self.model.get_weights()
        self.self_play_return = 0.
        self.traj_num, self.transition_num = 0, 0
        self.eval_score, self.eval_best_score = 0., 0.
        self.prev_eval_counter = -1
        self.eval_counter = 0
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

        self.total_train_steps = total_train_steps
        self.beta_schedule = LinearSchedule(
            self.total_train_steps,
            initial_p=self.hparams.priority["priority_prob_beta"],
            final_p=1.0
        )

        self.self_play_model = copy.deepcopy(self.model)
        self.reanalyze_model = copy.deepcopy(self.model)
        self.latest_model = copy.deepcopy(self.model)

        self.training_mode = True

    def collect(self, x_t, u_t, reward, x_tt, task_id, done=False):
        assert self.training_mode, "collection only in training mode!"
        self.collector.append(u_t, x_tt, reward)
        self.collector.snapshot_lst.append([])

        if self.collector.is_full():
            if self.prev_traj is not None:
                self.save_previous_trajectory(self.prev_traj, self.collector)

            self.prev_traj = self.collector
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
            pred_values = torch.from_numpy(np.array(traj.pred_value_lst)).cuda().float()
            # search_values = torch.from_numpy(np.array(traj.search_value_lst)).cuda().float()
            if self.hparams.model["value_target"] == 'bootstrapped':
                target_values = torch.from_numpy(np.asarray(traj.get_bootstrapped_value())).cuda().float()
            elif self.hparams.model["value_target"] == 'GAE':
                target_values = torch.from_numpy(np.asarray(traj.get_gae_value())).cuda().float()
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
        assert 0 <= reanalyze_batch_size <= batch_size

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

            if self.agent_type == AgentType.ATARI:
                top_new_masks.append(int(sample_idx > collected_transitions - self.hparams.train["mixed_value_threshold"]))
                _actions = traj.action_lst[state_index:state_index + self.hparams.train["unroll_steps"]].tolist()
                _mask = [1. for _ in range(len(_actions))]
                _mask += [0. for _ in range(self.hparams.train["unroll_steps"] - len(_mask))]
                _actions += [np.random.randint(0, self.env_action_space.n) for _ in range(self.hparams.train["unroll_steps"] - len(_actions))]
            else:
                _actions = traj.action_lst[state_index:state_index + self.hparams.train["unroll_steps"]]
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
                )
                # obtain the statistics at current steps
                with autocast():
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


    def learn(self, task_id: int):
        assert self.optimizer is not None, "The agent must be in training mode. try train()!"
        # todo: update decision models



