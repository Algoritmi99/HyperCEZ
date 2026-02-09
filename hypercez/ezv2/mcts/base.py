# Copyright (c) EVAR Lab, IIIS, Tsinghua University.
#
# This source code is licensed under the GNU License, Version 3.0
# found in the LICENSE file in the root directory of this source tree.

import torch
import math
from hypercez.util.distribution import SquashedNormal
import numpy as np


class MCTS_base:
    def __init__(self, num_actions, **kwargs):
        """
        :param num_actions:
        :param num_top_actions:
        :param kwargs:
        """
        self.num_actions = num_actions

        self.num_simulations = kwargs.get('num_simulations')
        self.num_top_actions = kwargs.get('num_top_actions')
        self.c_visit = kwargs.get('c_visit')
        self.c_scale = kwargs.get('c_scale')
        self.c_base = kwargs.get('c_base')
        self.c_init = kwargs.get('c_init')
        self.dirichlet_alpha = kwargs.get('dirichlet_alpha')
        self.explore_frac = kwargs.get('explore_frac')
        self.discount = kwargs.get('discount')
        self.value_minmax_delta = kwargs.get('value_minmax_delta')
        self.value_support = kwargs.get('value_support')
        self.reward_support = kwargs.get('reward_support')
        self.value_prefix = kwargs.get('value_prefix')
        self.lstm_hidden_size = kwargs.get('lstm_hidden_size')
        self.lstm_horizon_len = kwargs.get('lstm_horizon_len')
        self.mpc_horizon = kwargs.get('mpc_horizon')
        self.env = kwargs.get('env')
        self.vis = kwargs.get('vis')                                    # vis: [log, text, graph]
        self.std_magnification = kwargs.get('std_magnification')
        self.policy_action_num = kwargs.get('policy_action_num')
        self.random_action_num = kwargs.get('random_action_num')
        self.policy_distribution = kwargs.get('policy_distribution')

        self.current_num_top_actions = self.num_top_actions             # /2 every phase
        self.current_phase = 0                                          # current phase index
        self.visit_num_for_next_phase = max(
            np.floor(self.num_simulations / (np.log2(self.num_top_actions) * self.current_num_top_actions)), 1) \
                                        * self.current_num_top_actions   # how many visit counts for next phase
        self.used_visit_num = 0
        self.verbose = 0
        assert self.num_top_actions <= self.num_actions

    def search(self, model, batch_size, root_states, root_values, root_policy_logits, **kwargs):
        raise NotImplementedError()

    def search_continuous(self, model, batch_size, root_states, root_values, root_policy_logits,
                          use_gumble_noise=False, temperature=1.0, verbose=0, add_noise=True,
                          input_noises=None, input_dist=None, input_actions=None, prev_mean=None, **kwargs):
        raise NotImplementedError()

    def search_ori_mcts(self, model, batch_size, root_states, root_values, root_policy_logits,
                        use_noise=True, temperature=1.0, verbose=0, is_reanalyze=False, **kwargs):
        raise NotImplementedError()

    def sample_mpc_actions(self, policy):
        is_continuous = (self.env in ['DMC', 'Gym'])
        if is_continuous:
            action_dim = policy.shape[-1] // 2
            mean = policy[:, :action_dim]
            return mean
        else:
            return policy.argmax(dim=-1).unsqueeze(1)

    def sample_actions(self, policy, add_noise=True, temperature=1.0, input_noises=None, input_dist=None, input_actions=None, sample_nums=None, states=None):
        n_policy = self.policy_action_num
        n_random = self.random_action_num
        if sample_nums:
            n_policy = math.ceil(sample_nums / 2)
            n_random = sample_nums - n_policy
        std_magnification = self.std_magnification
        action_dim = policy.shape[-1] // 2

        if input_dist is not None:
            n_policy //= 2
            n_random //= 2

        Dist = SquashedNormal
        mean, std = policy[:, :action_dim], policy[:, action_dim:]
        distr = Dist(mean, std)
        try:
            sampled_actions = distr.sample(torch.Size([n_policy + n_random]))
        except Exception as e:
            print(n_policy)
            print(n_random)
            print(sample_nums)
            print(policy.shape)
            raise e

        policy_actions = sampled_actions[:n_policy]
        random_actions = sampled_actions[-n_random:]

        random_distr = distr
        if add_noise:
            if input_noises is None:
                random_distr = Dist(mean, std_magnification * std)  # more flatten gaussian policy
                random_actions = random_distr.sample(torch.Size([n_random]))
            else:
                noises = torch.from_numpy(input_noises).float().cuda()
                random_actions += noises

        if input_dist is not None:
            refined_mean, refined_std = input_dist[:, :action_dim], input_dist[:, action_dim:]
            refined_distr = Dist(refined_mean, refined_std)
            refined_actions = refined_distr.sample(torch.Size([n_policy + n_random]))

            refined_policy_actions = refined_actions[:n_policy]
            refined_random_actions = refined_actions[-n_random:]

            if add_noise:
                if input_noises is None:
                    refined_random_distr = Dist(refined_mean, std_magnification * refined_std)
                    refined_random_actions = refined_random_distr.sample(torch.Size([n_random]))
                else:
                    noises = torch.from_numpy(input_noises).float().cuda()
                    refined_random_actions += noises

        all_actions = torch.cat((policy_actions, random_actions), dim=0)
        if input_actions is not None:
            all_actions = torch.from_numpy(input_actions).float().cuda()
        if input_dist is not None:
            all_actions = torch.cat((all_actions, refined_policy_actions, refined_random_actions), dim=0)
        all_actions = all_actions.clip(-0.999, 0.999)

        assert (n_policy + n_random) == sample_nums if sample_nums is not None else self.num_actions
        ratio = n_policy / (sample_nums if sample_nums is not None else self.num_actions)
        probs = distr.log_prob(all_actions) - (ratio * distr.log_prob(all_actions) + (1 - ratio) * random_distr.log_prob(all_actions))
        probs = probs.sum(-1).permute(1, 0)
        all_actions = all_actions.permute(1, 0, 2)
        return all_actions, probs

    def update_statistics(self, **kwargs):
        if kwargs.get('prediction'):
            # prediction for next states, rewards, values, logits
            model = kwargs.get('model')
            states = kwargs.get('states')
            last_actions = kwargs.get('actions')
            reward_hidden = kwargs.get('reward_hidden')
            model_state = kwargs.get('model_state')

            next_value_prefixes = 0
            for _ in range(self.mpc_horizon):
                with torch.no_grad():
                    with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                        states, pred_value_prefixes, next_values, next_logits, reward_hidden = \
                            model.recurrent_inference(states, last_actions, reward_hidden, model_state=model_state)
                # last_actions = self.sample_mpc_actions(next_logits)
                next_value_prefixes += pred_value_prefixes

            # process outputs
            next_value_prefixes = next_value_prefixes.detach().cpu().numpy()
            next_values = next_values.detach().cpu().numpy()

            self.log('simulate action {}, r = {:.3f}, v = {:.3f}, logits = {}'
                     ''.format(last_actions[0].tolist(), next_value_prefixes[0].item(), next_values[0].item(), next_logits[0].tolist()),
                     verbose=3)
            return states, next_value_prefixes, next_values, next_logits, reward_hidden
        else:
            # env simulation for next states
            env = kwargs.get('env')
            current_states = kwargs.get('states')
            last_actions = kwargs.get('actions')
            states = env.step(last_actions)
            raise NotImplementedError()

    def estimate_value(self, **kwargs):
        # prediction for value in planning
        model = kwargs.get('model')
        current_states = kwargs.get('states')
        actions = kwargs.get('actions')
        reward_hidden = kwargs.get('reward_hidden')

        Value = 0
        discount = 1
        for i in range(actions.shape[0]):
            current_states_hidden = None
            with torch.no_grad():
                with torch.amp.autocast():
                    next_states, next_value_prefixes, next_values, next_logits, reward_hidden = model.recurrent_inference(current_states, actions[i], reward_hidden)

            next_value_prefixes = next_value_prefixes.detach()
            next_values = next_values.detach()
            current_states = next_states
            Value += next_value_prefixes * discount
            discount *= self.discount

        Value += discount * next_values

        return Value

    def log(self, string, verbose, iteration_begin=False, iteration_end=False):
        if verbose <= self.verbose:
            if iteration_begin:
                print('>' * 50)
            print(string)
            print('-' * 20)
            if iteration_end:
                print('<' * 50)

    def reset(self):
        self.current_num_top_actions = self.num_top_actions
        self.current_phase = 0
        self.visit_num_for_next_phase = max(
            np.floor(self.num_simulations / (np.log2(self.num_top_actions) * self.current_num_top_actions)), 1) \
                                        * self.current_num_top_actions  # how many visit counts for next phase
        self.used_visit_num = 0
        self.verbose = 0
