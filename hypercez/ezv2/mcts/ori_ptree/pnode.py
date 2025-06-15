import math
import random
import sys
import time
from collections import deque

from hypercez.ezv2.mcts.ori_ptree.pminimax import PMinMaxStats, PMinMaxStatsList


class PSearchResults:
    def __init__(self, num: int = 0):
        self.num = num
        self.search_paths: list[list['PNode']] = [[] for _ in range(num)]
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


def update_tree_q(root: PNode, min_max_stats: PMinMaxStats, discount: float):
    node_stack = deque()
    node_stack.append(root)

    parent_value_prefix = 0.0
    is_reset = 0

    while node_stack:
        node = node_stack.pop()

        if node is not root:
            true_reward = node.value_prefix - (0 if is_reset == 1 else parent_value_prefix)
            qsa = true_reward + discount * node.value()
            min_max_stats.update(qsa)

        for a in range(node.action_num):
            child = node.get_child(a)
            if child.expanded():
                node_stack.append(child)

        parent_value_prefix = node.value_prefix
        is_reset = node.is_reset


def pback_propagate(search_path: list[PNode],
                    min_max_stats: PMinMaxStats,
                    to_play: int,
                    value: float,
                    discount: float
                    ):
    bootstrap_value = value
    path_len = len(search_path)

    for i in range(path_len - 1, -1, -1):
        node = search_path[i]
        node.value_sum += bootstrap_value
        node.visit_count += 1

        parent_value_prefix = 0.0
        is_reset = 0
        if i >= 1:
            parent = search_path[i - 1]
            parent_value_prefix = parent.value_prefix
            is_reset = parent.is_reset
            # Optionally track qsa with min_max_stats
            # qsa = (node.value_prefix - parent_value_prefix) + discount * node.value()
            # min_max_stats.update(qsa)

        true_reward = node.value_prefix - parent_value_prefix
        if is_reset == 1:
            true_reward = node.value_prefix

        bootstrap_value = true_reward + discount * bootstrap_value

    min_max_stats.clear()
    root = search_path[0]
    update_tree_q(root, min_max_stats, discount)


def pbatch_back_propagate(hidden_state_index_x: int,
                          discount: float,
                          value_prefixes: list[float],
                          values: list[float],
                          policies: list[list[float]],
                          min_max_stats_lst: PMinMaxStatsList,
                          results: PSearchResults,
                          is_reset_lst: list[int]
                          ):
    for i in range(results.num):
        results.nodes[i].expand(0, hidden_state_index_x, i, value_prefixes[i], policies[i])
        results.nodes[i].is_reset = is_reset_lst[i]

        pback_propagate(
            search_path=results.search_paths[i],
            min_max_stats=min_max_stats_lst.stats_lst[i],
            to_play=0,
            value=values[i],
            discount=discount
        )


def pselect_child(root: PNode,
                  min_max_stats: PMinMaxStats,
                  pb_c_base: int,
                  pb_c_init: float,
                  discount: float,
                  mean_q: float
                  ):
    max_score = -sys.float_info.max
    epsilon = 1e-6
    max_index_list = []

    for a in range(root.action_num):
        child = root.get_child(a)

        temp_score = pucb_score(
            child,
            min_max_stats,
            mean_q,
            root.is_reset,
            root.visit_count - 1,
            root.value_prefix,
            pb_c_base,
            pb_c_init,
            discount
        )

        if temp_score > max_score:
            max_score = temp_score
            max_index_list = [a]
        elif temp_score >= max_score - epsilon:
            max_index_list.append(a)

    action = 0
    if max_index_list:
        action = random.choice(max_index_list)

    return action


def pucb_score(child,
               min_max_stats,
               parent_mean_q,
               is_reset,
               total_children_visit_counts,
               parent_value_prefix,
               pb_c_base,
               pb_c_init,
               discount
               ):
    pb_c = math.log((total_children_visit_counts + pb_c_base + 1) / pb_c_base) + pb_c_init
    pb_c *= math.sqrt(total_children_visit_counts) / (child.visit_count + 1)

    prior_score = pb_c * child.prior

    if child.visit_count == 0:
        value_score = parent_mean_q
    else:
        true_reward = child.value_prefix - (0 if is_reset == 1 else parent_value_prefix)
        value_score = true_reward + discount * child.value()

    value_score = min_max_stats.normalize(value_score)

    # Clamp the normalized value between [0, 1]
    value_score = max(0.0, min(1.0, value_score))

    ucb_value = prior_score + value_score
    return ucb_value


def pbatch_traverse(
    roots: PRoots,
    pb_c_base: int,
    pb_c_init: float,
    discount: float,
    min_max_stats_lst: PMinMaxStatsList,
    results: PSearchResults
):
    # Set seed based on current microseconds
    t1 = time.time()
    seed = int((t1 - int(t1)) * 1_000_000)
    random.seed(seed)

    last_action = -1
    parent_q = 0.0
    results.search_lens = []

    for i in range(results.num):
        node = roots.roots[i]
        is_root = True
        search_len = 0

        results.search_paths[i].append(node)

        while node.expanded():
            mean_q = node.get_mean_q(is_root, parent_q, discount)
            is_root = False
            parent_q = mean_q

            action = pselect_child(
                node,
                min_max_stats_lst.stats_lst[i],
                pb_c_base,
                pb_c_init,
                discount,
                mean_q
            )
            node.best_action = action

            node = node.get_child(action)
            last_action = action
            results.search_paths[i].append(node)
            search_len += 1

        parent = results.search_paths[i][-2]

        results.hidden_state_index_x_lst.append(parent.hidden_state_index_x)
        results.hidden_state_index_y_lst.append(parent.hidden_state_index_y)
        results.last_actions.append(last_action)
        results.search_lens.append(search_len)
        results.nodes.append(node)
