import math
import random
import sys
from collections import deque
import time

from hypercez.ezv2.mcts.ptree_v2.pminimax import PMinMaxStats, PMinMaxStatsList


class PSearchResults:
    def __init__(self, num = 0):
        self.num = num
        self.search_paths = [
            [] for _ in range(num)
        ] if num > 0 else []
        self.hidden_state_index_x_lst: list[int] = []
        self.hidden_state_index_y_lst: list[int] = []
        self.last_actions: list[int] = []
        self.search_lens: list[int] = []
        self.nodes: list[PNode] = []




class PNode:
    def __init__(self, prior: float = 0, action_num: int = 0, ptr_node_pool: list['PNode'] = None):
        self.prior = prior
        self.action_num = action_num
        self.best_action = -1
        self.is_reset = 0
        self.visit_count = 0
        self.value_sum = 0
        self.to_play = 0
        self.reward_sum = 0.0
        self.ptr_node_pool = ptr_node_pool
        self.hidden_state_index_x = -1
        self.hidden_state_index_y = -1
        self.similarity = 0.0
        self.children_index = []

    def expand(self,
               to_play: int,
               hidden_state_index_x: int,
               hidden_state_index_y: int,
               reward_sum: float,
               policy_logits: list[float]):
        self.to_play = to_play
        self.hidden_state_index_x = hidden_state_index_x
        self.hidden_state_index_y = hidden_state_index_y
        self.reward_sum = reward_sum

        action_num = self.action_num
        policy_sum = 0.0
        policy: list = [0.0] * action_num

        policy_max = float('-inf')
        for a in range(self.action_num):
            policy_max = max(policy_max, policy_logits[a])

        for a in range(self.action_num):
            temp_policy = math.exp(policy_logits[a] - policy_max)
            policy_sum += temp_policy
            policy[a] = temp_policy

        ptr_node_pool = self.ptr_node_pool
        for a in range(self.action_num):
            prior = policy[a] / policy_sum
            index = len(ptr_node_pool)
            self.children_index.append(index)
            ptr_node_pool.append(PNode(prior, action_num, ptr_node_pool))

    def get_child(self, action: int):
        return self.ptr_node_pool[self.children_index[action]]

    def value(self):
        true_value = 0.0
        if self.visit_count == 0:
            print("visit count=0, raise value calculation error.\n")
            return true_value
        else:
            true_value = self.value_sum / self.visit_count
            return true_value

    def add_exploration_noise(self, exploration_fraction: float, noises: list[float]):
        for a in range(self.action_num):
            child = self.get_child(a)
            child.prior = child.prior * (1 - exploration_fraction) + exploration_fraction * noises[a]

    def get_mean_q(self, is_root: int, parent_q: float, discount: float):
        total_unsigned_q = 0.0
        total_visits = 0
        parent_reward_sum = self.reward_sum
        for a in range(self.action_num):
            child = self.get_child(a)
            if child.visit_count > 0:
                true_reward = child.reward_sum - parent_reward_sum
                if self.is_reset == 1:
                    true_reward = child.reward_sum
                qsa = true_reward + discount * child.value()
                total_unsigned_q += qsa
                total_visits += 1

        if is_root and total_visits > 0:
            mean_q = total_unsigned_q / total_visits
        else:
            mean_q = (parent_q + total_unsigned_q) / (total_visits + 1)
        return mean_q

    def print_out(self):
        print("*****")
        print(
            f"visit count: {self.visit_count}"
            f" \t hidden_state_index_x: {self.hidden_state_index_x}"
            f" \t hidden_state_index_y: {self.hidden_state_index_y} "
            f"\t reward: {self.reward_sum} \t prior: {self.prior}"
        )
        print(f"children_index size: {len(self.children_index)} \t pool size: {len(self.ptr_node_pool)}")
        print("*****")

    def expanded(self):
        return len(self.children_index) > 0

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


class PRoots:
    def __init__(self, root_num: int = 0, action_num: int = 0, pool_size: int = 0):
        self.root_num = root_num
        self.action_num = action_num
        self.pool_size = pool_size

        self.node_pools: list[list[PNode]] = []
        self.roots: list[PNode] = []

        for i in range(root_num):
            self.node_pools.append([])
            self.roots.append(PNode(0, action_num, self.node_pools[i]))

    def prepare(self,
                 root_exploration_fraction: float,
                 noises: list[list[float]],
                 reward_sums: list[float],
                 policies: list[list[float]]):
        for i in range(self.root_num):
            self.roots[i].expand(0, 0, i, reward_sums[i], policies[i])
            self.roots[i].add_exploration_noise(root_exploration_fraction, noises[i])

    def prepare_no_noise(self, reward_sums: list[float], policies: list[list[float]]):
        for i in range(self.root_num):
            self.roots[i].expand(0, 0, i, reward_sums[i], policies[i])

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
    node_stack = deque([root])
    parent_reward_sum = 0.0
    is_reset = False
    while node_stack:
        node = node_stack.pop()

        if node != root:
            true_reward = node.reward_sum - parent_reward_sum
            if is_reset:
                true_reward = node.reward_sum
            min_max_stats.update(true_reward + discount * node.value())

        for a in range(node.action_num):
            child = node.get_child(a)
            if child.expanded():
                node_stack.append(child)

        parent_reward_sum = node.reward_sum
        is_reset = node.is_reset


def pback_propagate(search_path: list[PNode], min_max_stats: PMinMaxStats, to_play: int, value: float, discount: float):
    bootstrap_value = value
    path_len = len(search_path)
    for i in range(path_len - 1, -1, -1):
        node = search_path[i]
        node.value_sum += bootstrap_value
        node.visit_count += 1

        parent_reward_sum = 0.0
        is_reset = False
        if i > 1:
            parent = search_path[i - 1]
            parent_reward_sum = parent.reward_sum
            is_reset = parent.is_reset

        true_reward = node.reward_sum - (parent_reward_sum if not is_reset else 0)
        bootstrap_value = true_reward + discount * bootstrap_value

    min_max_stats.clear()
    root = search_path[0]
    update_tree_q(root, min_max_stats, discount)


def pmulti_back_propagate(hidden_state_index_x: int,
                          discount: float,
                          reward_sums: list[float],
                          values: list[float],
                          policies: list[list[float]],
                          min_max_stats_lst: PMinMaxStatsList,
                          results: PSearchResults,
                          is_reset_lst: list[int],
                          similarities: list[float]
                          ):
    for i in range(results.num):
        results.nodes[i].expand(0, hidden_state_index_x, i, reward_sums[i], policies[i])
        results.nodes[i].similarity = similarities[i]
        results.nodes[i].is_reset = is_reset_lst[i]
        pback_propagate(results.search_paths[i], min_max_stats_lst.stats_lst[i], 0, values[i], discount)


def pselect_child(root: PNode,
                  min_max_stats: PMinMaxStats,
                  pb_c_base: int,
                  pb_c_init: float,
                  discount: float,
                  mean_q: float,
                  use_mcgs: int):
    max_score = float('-inf')
    epsilon = 0.000001
    max_index_lst: list[int] = []
    for a in range(root.action_num):
        child = root.get_child(a)
        temp_score = pucb_score(child, min_max_stats, mean_q, root.is_reset, root.visit_count, root.reward_sum, pb_c_base, pb_c_init, discount, use_mcgs)

        if max_score < temp_score:
            max_score = temp_score
            max_index_lst.clear()
            max_index_lst.append(a)
        elif temp_score >= max_score - epsilon:
            max_index_lst.append(a)
            max_index_lst.append(a)

    action = 0
    if len(max_index_lst) > 0:
        rand_index = random.randint(0, len(max_index_lst) - 1)
        action = max_index_lst[rand_index]
    else:
        print("[ERROR] max action list is empty!\n")
    return action


def pucb_score(child: PNode,
               min_max_stats: PMinMaxStats,
               parent_mean_q: float,
               is_reset: int,
               parent_visit_count: int,
               parent_reward_sum: float,
               pb_c_base: int,
               pb_c_init: float,
               discount: float,
               use_mcgs: int):
    pb_c = math.log((parent_visit_count + pb_c_base + 1) / pb_c_base) + pb_c_init
    pb_c *= (math.sqrt(parent_visit_count + 1) / (child.visit_count + 1))

    prior_score = pb_c * child.prior
    similarity_score = (1 - child.similarity) * 3

    if child.visit_count == 0:
        value_score = parent_mean_q
    else:
        true_reward = child.reward_sum - parent_reward_sum
        if is_reset == 1:
            true_reward = child.reward_sum
        value_score = true_reward + discount * child.value()

    value_score = min_max_stats.normalize(value_score)
    value_score = max(0.0, min(1.0, value_score))  # Clamp between 0 and 1

    if use_mcgs:
        ucb_value = prior_score + value_score + similarity_score
    else:
        ucb_value = prior_score + value_score

    if not math.isfinite(ucb_value) or ucb_value < sys.float_info.min or ucb_value > sys.float_info.max:
        print("[ERROR] Value: value -> {:.6f}, min/max Q -> {:.6f}/{:.6f}, visit count -> {}({})".format(
            child.value(), min_max_stats.minimum, min_max_stats.maximum, parent_visit_count, child.visit_count
        ))
        print("(prior, value): {:.6f}({:.6f} * {:.6f}) + {:.6f}({:.6f}, [{:.6f}, {:.6f}]) = {:.6f}".format(
            prior_score, pb_c, child.prior, min_max_stats.normalize(value_score), value_score,
            min_max_stats.minimum, min_max_stats.maximum, prior_score + min_max_stats.normalize(value_score)
        ))

    return ucb_value


def pmulti_traverse(roots: PRoots,
                    pb_c_base: int,
                    pb_c_init: float,
                    discount: float,
                    min_max_stats_lst:
                    PMinMaxStatsList,
                    results: PSearchResults,
                    use_mcgs: int):
    # Set random seed using microseconds
    t1 = time.time()
    random.seed(int((t1 - int(t1)) * 1e6))

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
                mean_q,
                use_mcgs
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
