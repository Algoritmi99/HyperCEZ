# Copyright (c) EVAR Lab, IIIS, Tsinghua University.
#
# This source code is licensed under the GNU License, Version 3.0
# found in the LICENSE file in the root directory of this source tree.

# Refactored by Arash T. Goodarzi

import numpy as np
import torch
import torch.nn as nn
from torch.func import functional_call

from hypercez.util.format import DiscreteSupport, symexp, normalize_state


class EfficientZero(nn.Module):
    def __init__(self,
                 representation_model,
                 dynamics_model,
                 reward_prediction_model,
                 value_policy_model,
                 projection_model,
                 projection_head_model,
                 hparams,
                 **kwargs,
                 ):
        """The basic models in EfficientZero
        Parameters
        ----------
        representation_model: nn.Module
            represent the observations'
        dynamics_model: nn.Module
            dynamics model predicts the next state given the current state and action
        reward_prediction_model: nn.Module
            predict the reward given the next state (Namely, current state and action)
        value_prediction_model: nn.Module
            predict the value given the state
        policy_prediction_model: nn.Module
            predict the policy given the state
        kwargs: dict
            state_norm: bool.
                use state normalization for encoded state
            value_prefix: bool
                predict value prefix instead of reward
        """
        super().__init__()

        self.representation_model = representation_model
        self.dynamics_model = dynamics_model
        self.reward_prediction_model = reward_prediction_model
        self.value_policy_model = value_policy_model
        self.projection_model = projection_model
        self.projection_head_model = projection_head_model
        self.hparams = hparams
        self.state_norm = kwargs.get('state_norm')
        self.value_prefix = kwargs.get('value_prefix')
        self.v_num = hparams.train["v_num"]

    def do_representation(self, obs, model_state=None):
        if self.training and model_state is not None:
            obs = obs.clone().detach().requires_grad_(True)

        state = self.representation_model(obs) if model_state is None or 'representation_model' not in model_state else functional_call(
            self.representation_model, model_state["representation_model"], args=(obs,), kwargs=None
        )
        if self.state_norm:
            state = normalize_state(state)

        return state

    def do_dynamics(self, state, action, model_state=None):
        if self.training and model_state is not None:
            state = state.clone().detach().requires_grad_(True)
            action = action.clone().detach().requires_grad_(True)

        next_state = self.dynamics_model(state, action) if model_state is None or 'dynamics_model' not in model_state else functional_call(
            self.dynamics_model, model_state["dynamics_model"], args=(state, action), kwargs=None
        )
        if self.state_norm:
            next_state = normalize_state(next_state)

        return next_state

    def do_reward_prediction(self, next_state, reward_hidden=None, model_state=None):
        if self.training and model_state is not None:
            next_state = next_state.clone().detach().requires_grad_(True)

        # use the predicted state (Namely, current state + action) for reward prediction
        if self.value_prefix:
            value_prefix, reward_hidden = self.reward_prediction_model(
                next_state, reward_hidden
            ) if model_state is None or 'reward_prediction_model' not in model_state else functional_call(
                self.reward_prediction_model, model_state["reward_prediction_model"], args=(next_state, reward_hidden), kwargs=None
            )
            return value_prefix, reward_hidden
        else:
            reward = self.reward_prediction_model(next_state) if model_state is None or 'reward_prediction_model' not in model_state else functional_call(
                self.reward_prediction_model, model_state["reward_prediction_model"], args=(next_state,), kwargs=None
            )
            return reward, None

    def do_value_policy_prediction(self, state, model_state=None):
        if self.training and model_state is not None:
            state = state.clone().detach().requires_grad_(True)

        value, policy = self.value_policy_model(state) if model_state is None or 'value_policy_model' not in model_state else functional_call(
            self.value_policy_model, model_state["value_policy_model"], args=(state,), kwargs=None
        )
        return value, policy

    def do_projection(self, state, with_grad=True, model_state=None):
        if self.training and model_state is not None:
            state = state.clone().detach().requires_grad_(True)

        # only the branch of proj + pred can share the gradients
        proj = self.projection_model(state) if model_state is None or 'projection_model' not in model_state else functional_call(
            self.projection_model, model_state['projection_model'], args=(state,), kwargs=None
        )

        # with grad, use proj_head
        if with_grad:
            proj = self.projection_head_model(proj) if model_state is None or 'projection_head_model' not in model_state else functional_call(
                self.projection_head_model, model_state['projection_head_model'], args=(proj,), kwargs=None
            )
            return proj
        else:
            return proj.detach()

    def initial_inference(self, obs, training=False, model_state=None):
        state = self.do_representation(obs, model_state=model_state)
        values, policy = self.do_value_policy_prediction(state, model_state=model_state)

        if training:
            return state, values, policy

        if self.v_num > 2:
            values = values[np.random.choice(self.v_num, 2, replace=False)]
        if self.hparams.model["value_support"]["type"] == 'symlog':
            output_values = symexp(values).min(0)[0]
        else:
            output_values = DiscreteSupport.vector_to_scalar(values, **self.hparams.model["value_support"]).min(0)[0]

        if self.hparams.env_type in ['DMC', 'Gym']:
            output_values = output_values.clip(0, 1e5)

        return state, output_values, policy

    def recurrent_inference(self, state, action, reward_hidden, training=False, model_state=None):
        next_state = self.do_dynamics(state, action, model_state=model_state)
        value_prefix, reward_hidden = self.do_reward_prediction(next_state, reward_hidden, model_state=model_state)
        values, policy = self.do_value_policy_prediction(next_state, model_state=model_state)
        if training:
            return next_state, value_prefix, values, policy, reward_hidden

        if self.v_num > 2:
            values = values[np.random.choice(self.v_num, 2, replace=False)]
        if self.hparams.model["value_support"]["type"] == 'symlog':
            output_values = symexp(values).min(0)[0]
        else:
            output_values = DiscreteSupport.vector_to_scalar(values, **self.hparams.model["value_support"]).min(0)[0]

        if self.hparams.env_type in ['DMC', 'Gym']:
            output_values = output_values.clip(0, 1e5)

        if self.hparams.model["reward_support"]["type"] == 'symlog':
            value_prefix = symexp(value_prefix)
        else:
            value_prefix = DiscreteSupport.vector_to_scalar(value_prefix, **self.hparams.model["reward_support"])
        return next_state, value_prefix, output_values, policy, reward_hidden

    def get_weights(self, part='none'):
        if part == 'reward':
            weights = self.reward_prediction_model.state_dict()
        elif part == 'dynamics':
            weights = self.dynamics_model.state_dict()
        elif part == 'projection':
            weights = self.projection_model.state_dict()
        elif part == 'projection_head':
            weights = self.projection_head_model.state_dict()
        elif part == 'value_policy':
            weights = self.value_policy_model.state_dict()
        elif part == 'representation':
            weights = self.representation_model.state_dict()
        else:
            weights = self.state_dict()

        return {k: v.cpu() for k, v in weights.items()}

    def set_weights(self, weights):
        self.load_state_dict(weights)

    def get_gradients(self):
        grads = []
        for p in self.parameters():
            grad = None if p.grad is None else p.grad.data.cpu().numpy()
            grads.append(grad)
        return grads

    def set_gradients(self, gradients):
        for g, p in zip(gradients, self.parameters()):
            if g is not None:
                p.grad = torch.from_numpy(g)
