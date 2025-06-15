import math
import sys


class PSearchResults:
    def __init__(self, num: int = 0):
        self.num = num
        self.search_paths: list[list['PNode']]= [[] for _ in range(num)]
        (
            self.hidden_state_index_x_lst,
            self.hidden_state_index_y_lst,
            self.last_actions,
            self.search_lens
        ) = (
            [], [], [], []
        )
        self.nodes: list['PNode'] = []

class PNode:
    def __init__(self, prior: float = 0, action_num: int = 0, ptr_node_pool: list['PNode'] = None):
        self.prior = prior
        self.action_num = action_num
        self.ptr_node_pool = ptr_node_pool

        self.visit_count = 0
        self.to_play = 0
        self.hidden_state_index_x = -1
        self.hidden_state_index_y = -1
        self.best_action = -1
        self.is_reset = 0
        self.value_prefix = 0.0
        self.value_sum = 0.0
        self.children_index: list[int] = []

    def expand(self, to_play, hidden_state_index_x, hidden_state_index_y, value_prefix, policy_logits):
        self.to_play = to_play
        self.hidden_state_index_x = hidden_state_index_x
        self.hidden_state_index_y = hidden_state_index_y
        self.value_prefix = value_prefix

        action_num = self.action_num
        policy = [0.0] * action_num
        policy_sum = 0.0
        policy_max = sys.float_info.min

        for a in range(action_num):
            if policy_max < policy_logits[a]:
                policy_max = policy_logits[a]

        for a in range(action_num):
            temp_policy = math.exp(policy_logits[a] - policy_max)
            policy_sum += temp_policy
            policy[a] = temp_policy

        for a in range(action_num):
            prior = policy[a] / policy_sum
            index = len(self.ptr_node_pool)
            self.children_index.append(index)
            self.ptr_node_pool.append(PNode(prior, action_num, self.ptr_node_pool))

    def add_exploration_noise(self, exploration_fraction, noises):
        for a in range(self.action_num):
            noise = noises[a]
            child = self.get_child(a)

            prior = child.prior
            child.prior = prior * (1 - exploration_fraction) + noise * exploration_fraction

    def get_mean_q(self, is_root, parent_q, discount):
        total_unsigned_q = 0.0
        total_visits = 0
        parent_value_prefix = self.value_prefix

        for a in range(self.action_num):
            child = self.get_child(a)
            if child.visit_count > 0:
                true_reward = child.value_prefix - parent_value_prefix
                if self.is_reset == 1:
                    true_reward = child.value_prefix
                qsa = true_reward + discount * child.value()
                total_unsigned_q += qsa
                total_visits += 1

        if is_root and total_visits > 0:
            mean_q = total_unsigned_q / total_visits
        else:
            mean_q = (parent_q + total_unsigned_q) / (total_visits + 1)

        return mean_q

    def print_out(self):
        pass

    def expanded(self):
        return len(self.children_index) > 0

    def value(self):
        return self.value_sum / self.visit_count if self.visit_count != 0 else 0

    def get_trajectory(self):
        traj = []

        node = self
        best_action = node.best_action
        while best_action >= 0:
            traj.append(best_action)
            node = node.get_child(best_action)
            best_action = node.best_action

        return traj

    def get_children_distribution(self):
        distribution = []
        if self.expanded():
            for a in range(self.action_num):
                child = self.get_child(a)
                distribution.append(child.visit_count)
        return distribution

    def get_child(self, action):
        return self.ptr_node_pool[self.children_index[action]]


class PRoots:
    def __init__(self, root_num: int = 0, action_num: int = 0, pool_size: int = 0):
        self.root_num = root_num
        self.action_num = action_num
        self.pool_size = pool_size
        self.node_pools: list[list[PNode]] = [[] for _ in range(self.root_num)]
        self.roots: list[PNode] = [PNode(0, action_num, self.node_pools[i]) for i in range(self.root_num)]

    def prepare(self, root_exploration_fraction, noises, value_prefixes, policies):
        for i in range(self.root_num):
            self.roots[i].expand(0, 0, i, value_prefixes[i], policies[i])
            self.roots[i].add_exploration_noise(root_exploration_fraction, noises[i])
            self.roots[i].visit_count += 1

    def prepare_no_noise(self, value_prefixes, policies):
        for i in range(self.root_num):
            self.roots[i].expand(0, 0, i, value_prefixes[i], policies[i])
            self.roots[i].visit_count += 1

    def clear(self):
        self.node_pools.clear()
        self.roots.clear()

    def get_trajectories(self):
        return [self.roots[i].get_trajectory() for i in range(self.root_num)]

    def get_distributions(self):
        return [self.roots[i].get_children_distribution() for i in range(self.root_num)]

    def get_values(self):
        return [self.roots[i].value() for i in range(self.root_num)]



