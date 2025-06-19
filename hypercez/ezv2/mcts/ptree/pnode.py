import math

import numpy as np

from hypercez.ezv2.mcts.ptree.pminimax import PMinMaxStats, PMinMaxStatsList
from hypercez.util.format import softmax


def get_transformed_completed_Qs(node: 'PNode', min_max_stats: PMinMaxStats, final: bool) -> list[float]:
    to_normalize = 2 if final else 1
    completed_Qs = node.get_completed_Q(min_max_stats, to_normalize).tolist()
    max_child_visit_count = max(node.get_children_visits())

    # sigma transformation
    for i in range(len(completed_Qs)):
        completed_Qs[i] = (min_max_stats.visit + max_child_visit_count) * min_max_stats.scale * completed_Qs[i]
    return completed_Qs


def p_batch_sequential_halving(roots: 'PRoots',
                               gumble_noises: list[list[float]],
                               min_max_lst: PMinMaxStatsList,
                               current_phase: int,
                               current_num_top_actions: int):
    best_actions = [-1] * roots.num_roots
    for i in range(roots.num_roots):
        action = sequential_halving(
            roots.roots[i], gumble_noises[i], min_max_lst.stats_lst[i], current_phase, current_num_top_actions
        )
        best_actions[i] = action
    return np.asarray(best_actions)


def sequential_halving(root: 'PNode',
                       gumble_noise: list[float],
                       min_max_stats: PMinMaxStats,
                       current_phase: int,
                       current_num_top_actions: int):
    children_prior = root.get_children_priors()
    children_scores: list[float] = []
    transformed_completed_Qs: list[float] = get_transformed_completed_Qs(root, min_max_stats, False)

    selected_children_idx: list[int] = root.selected_children_idx.copy()
    for action in selected_children_idx:
        children_scores.append(gumble_noise[action] + children_prior[action] + transformed_completed_Qs[action])

    idx = list(range(len(children_scores)))
    idx.sort(key=lambda j: children_scores[j], reverse=True)

    root.selected_children_idx.clear()
    for i in range(current_num_top_actions):
        root.selected_children_idx.append(selected_children_idx[idx[i]])

    best_action = root.selected_children_idx[0]
    return best_action


def select_action(node: 'PNode',
                  min_max_stats:
                  PMinMaxStats,
                  num_simulations: int,
                  simulation_idx: int,
                  gumble_noise: list[float],
                  current_num_top_actions):
    if node.is_root():
        if simulation_idx == 0:
            children_prior = node.get_children_priors()
            children_scores: list[float] = []
            for action in range(node.num_actions):
                children_scores.append(gumble_noise[action] + children_prior[action])
            idx = list(range(len(children_scores)))
            idx.sort(key=lambda i: children_scores[i], reverse=True)

            node.selected_children_idx.clear()
            for action in range(current_num_top_actions):
                node.selected_children_idx.append(idx[action])
        action = node.do_equal_visit(num_simulations)
    else:
        transformed_completed_Qs = get_transformed_completed_Qs(node, min_max_stats, False)
        improved_policy = node.get_improved_policy(transformed_completed_Qs)
        children_visits = node.get_children_visits()
        children_scores: list[float] = [0.0] * node.num_actions
        for a in range(node.num_actions):
            score = improved_policy[a] - children_visits[a] / (1. + float(node.visit_count))
            children_scores[a] = score
        action = np.argmax(children_scores)

    return action


def p_batch_traverse(roots: 'PRoots',
                     min_max_stats_lst: PMinMaxStatsList,
                     results: 'PSearchResults',
                     num_simulations: int,
                     simulation_idx: int,
                     gumble_noise: list[list[float]],
                     current_num_top_actions: int):
    last_action = -1
    results.search_lens = []
    for i in range(results.num):
        node = roots.roots[i]
        search_len = 0
        results.search_paths[i].append(node)

        while node.is_expanded():
            action = select_action(
                node,
                min_max_stats_lst.stats_lst[i],
                num_simulations,
                simulation_idx,
                gumble_noise[i],
                current_num_top_actions
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


def back_propagate(search_path: list['PNode'], min_max_stats: PMinMaxStats, value: float):
    bootstrap_value = value
    path_len = len(search_path)
    for i in range(path_len - 1, -1, -1):
        node = search_path[i]
        node.estimated_value_lst.append(bootstrap_value)
        node.visit_count += 1

        bootstrap_value = node.get_reward() + node.discount * bootstrap_value
        min_max_stats.update(bootstrap_value)

def p_batch_back_propagate(hidden_state_index_x: int,
                           value_prefixs: list[float],
                           values: list[float],
                           policies: list[list[float]],
                           min_max_stats_lst: PMinMaxStatsList,
                           results: 'PSearchResults',
                           to_reset_lst: list[int],
                           leaf_action_num: int):
    for i in range(results.num):
        results.nodes[i].expand(
            hidden_state_index_x, i, value_prefixs[i], policies[i], to_reset_lst[i], leaf_action_num
        )
        back_propagate(results.search_paths[i], min_max_stats_lst.stats_lst[i], values[i])



class PNode:
    def __init__(self, prior: float = None,
                 action: int = None,
                 parent: 'PNode' = None,
                 ptr_node_pool: list = None,
                 discount: float = None,
                 num_actions: int = None):
        self.prior = prior if prior is not None else 0.0
        self.action = action if action is not None else -1
        self.parent = parent
        self.ptr_node_pool = ptr_node_pool
        self.discount = discount if discount is not None else 0.0
        self.num_actions = num_actions if num_actions is not None else -1
        self.depth = 0

        self.best_action = -1
        self.reset_value_prefix = 0
        self.depth = 0 if parent is None else parent.depth + 1
        self.visit_count = 0
        self.hidden_state_index_x = 0
        self.hidden_state_index_y = 0

        self.value_prefix = 0.0
        self.children_idx: list[int] = []
        self.selected_children_idx: list[int] = []
        self.estimated_value_lst: list[float] = []

    def expand(self, hidden_state_index_x: int,
               hidden_state_index_y: int,
               value_prefix: float,
               policy_logits: list[float],
               reset_value_prefix: int,
               leaf_action_num: int):
        self.hidden_state_index_x = hidden_state_index_x
        self.hidden_state_index_y = hidden_state_index_y
        self.value_prefix = value_prefix
        self.reset_value_prefix = reset_value_prefix

        for action in range(len(policy_logits)):
            prior = policy_logits[action]
            index = len(self.ptr_node_pool)
            self.children_idx.append(index)
            self.ptr_node_pool.append(PNode(prior, action, self, self.ptr_node_pool, self.discount, leaf_action_num))

    def get_policy(self):
        logits = self.get_children_priors()
        policy = softmax(logits)
        return policy

    def get_improved_policy(self, transformed_completed_Qs: list[float]):
        logits = np.asarray([0.0] * self.num_actions)
        for action in range(self.num_actions):
            child = self.get_child(action)
            logits[action] = child.prior + transformed_completed_Qs[action]

        policy = softmax(logits)
        return policy

    def get_completed_Q(self, min_max_stats: PMinMaxStats, to_normalize: int):
        completed_Qs = [0.0] * self.num_actions
        v_mix = self.get_v_mix()

        for action in range(self.num_actions):
            child = self.get_child(action)

            if child.is_expanded():
                Q = self.get_qsa(action)
            else:
                Q = v_mix

            if to_normalize == 1:
                completed_Qs[action] = min_max_stats.normalize(Q)
                if completed_Qs[action] < 0.0:
                    completed_Qs[action] = 0.0
                if completed_Qs[action] > 1.0:
                    completed_Qs[action] = 1.0
            else:
                completed_Qs[action] = Q

        if to_normalize == 2:
            print("use final normalize\n")
            v_max = max(completed_Qs)
            v_min = min(completed_Qs)

            for action in range(self.num_actions):
                completed_Qs[action] = (completed_Qs[action] - v_min) / (v_max - v_min)

        return np.asarray(completed_Qs)

    def get_children_priors(self) -> np.ndarray:
        priors = [0.0] * self.num_actions
        for action in range(self.num_actions):
            child = self.get_child(action)
            priors[action] = child.prior
        return np.asarray(priors)

    def get_children_visits(self) -> np.ndarray:
        visits = [0.0] * self.num_actions
        for action in range(self.num_actions):
            child = self.get_child(action)
            visits[action] = child.visit_count
        return np.asarray(visits)

    def get_trajectory(self):
        traj = []
        node = self
        best_action = node.best_action
        while best_action >= 0:
            traj.append(best_action)
            node = node.get_child(best_action)
            best_action = node.best_action
        return np.asarray(traj)

    def get_children_visit_sum(self):
        visit_lst = self.get_children_visits()
        return np.sum(visit_lst)

    def get_v_mix(self) -> float:
        pi_lst = self.get_policy()
        pi_sum = 0.0
        pi_qsa_sum = 0.0

        for action in range(self.num_actions):
            child = self.get_child(action)
            if child.is_expanded():
                pi_sum += pi_lst[action]
                pi_qsa_sum += pi_lst[action] * self.get_qsa(action)

        if pi_sum < 1e-6:
            v_mix = self.get_value()
        else:
            v_mix = (1. / (1. + self.visit_count)) * (self.get_value() + self.visit_count * pi_qsa_sum / pi_sum)

        return v_mix

    def get_reward(self):
        if self.reset_value_prefix:
            return self.value_prefix
        else:
            assert self.parent is not None
            return self.value_prefix - self.parent.value_prefix

    def get_value(self) -> float:
        if self.is_expanded():
            return sum(self.estimated_value_lst) / len(self.estimated_value_lst)
        else:
            return self.parent.get_v_mix()

    def get_qsa(self, action: int) -> float:
        child = self.get_child(action)
        return child.get_reward() + self.discount * child.get_value()

    def get_child(self, action: int) -> 'PNode':
        index = self.children_idx[action]
        return self.ptr_node_pool[index]

    def get_root(self) -> 'PNode':
        node = self
        while not node.is_root():
            node = node.parent
        return node

    def get_expanded_children(self):
        children = []
        for action in range(self.num_actions):
            child = self.get_child(action)
            if child.is_expanded():
                children.append(child)
        return children

    def is_root(self):
        return self.parent is None

    def is_leaf(self):
        return len(self.get_expanded_children()) == 0

    def is_expanded(self) -> bool:
        return len(self.children_idx) > 0

    def do_equal_visit(self, num_simulations: int):
        min_visit_count = num_simulations + 1
        action = -1
        for selected_child_idx in self.selected_children_idx:
            visit_count = self.get_child(selected_child_idx).visit_count
            if visit_count < min_visit_count:
                action = selected_child_idx
                min_visit_count = visit_count
        return action

    def print_tree(self, info: list[str]):
        if not self.is_expanded():
            return
        for i in range(self.depth):
            print(info[i], end='')

        is_leaf = self.is_leaf()
        if is_leaf:
            print("└──", end='')
        else:
            print("├──", end='')
        self.print()

        expanded_children = self.get_expanded_children()
        for child in expanded_children:
            string = "|    " if not is_leaf else "   "
            info.append(string)
            child.print_tree(info)

    def print(self):
        action_info = str(self.action)
        if self.is_root():
            action_info = "["
            for a in self.selected_children_idx:
                action_info += str(a) + ", "
            action_info = action_info[:-2] + "]"
        print("[a=", action_info, "reset=", self.reset_value_prefix, "(n=", self.visit_count,
              "vp=", self.value_prefix, "r=", self.get_reward(), "v=", self.get_value(), ")]")


class PRoots:
    def __init__(self, num_roots: int = None, num_actions: int = None, pool_size: int = None, discount: float = None):
        self.num_roots = 0 if num_roots is None else num_roots
        self.num_actions = 0 if num_actions is None else num_actions
        self.pool_size = 0 if pool_size is None else pool_size
        self.discount = 0.0 if discount is None else discount
        self.node_pools: list[list[PNode]] = []
        self.roots: list[PNode] = []

        if num_roots is not None and num_actions is not None and pool_size is not None and discount is not None:
            for i in range(num_roots):
                self.node_pools.append([])
                self.roots.append(PNode(1, -1, None, self.node_pools[i], discount, num_actions))

    def prepare(self, values: list[float], policies: list[list[float]], leaf_action_num: int):
        for i in range(self.num_roots):
            self.roots[i].expand(0, i, 0, policies[i], 1, leaf_action_num)
            self.roots[i].estimated_value_lst.append(values[i])
            self.roots[i].visit_count = 1

    def clear(self):
        self.node_pools.clear()
        self.roots.clear()

    def get_trajectories(self):
        trajs: list[list[int]] = [self.roots[i].get_trajectory().tolist() for i in range(self.num_roots)]
        return np.asarray(trajs)

    def get_distributions(self):
        distributions: list[list[int]] = []
        for i in range(self.num_roots):
            distributions.append(self.roots[i].get_children_visits().tolist())
        return np.asarray(distributions)

    def get_root_policies(self, min_max_stats_lst: PMinMaxStatsList):
        policies: list[list[float]] = []
        for i in range(self.num_roots):
            transformed_completed_Qs = get_transformed_completed_Qs(self.roots[i], min_max_stats_lst.stats_lst[i],
                                                                    False)
            for j in range(self.roots[i].num_actions):
                cq = transformed_completed_Qs[j]
                if math.isnan(cq):
                    print(
                        f"trans_Q NaN, {i}, {j}, "
                        f"{min_max_stats_lst.stats_lst[i].maximum:.2f}, "
                        f"{min_max_stats_lst.stats_lst[i].minimum:.2f}"
                    )
            improved_policy = self.roots[i].get_improved_policy(transformed_completed_Qs)
            policies.append(improved_policy.tolist())
        return np.asarray(policies)

    def get_best_actions(self):
        best_actions = [-1] * self.num_roots
        for i in range(self.num_roots):
            best_actions[i] = self.roots[i].selected_children_idx[0]
        return np.asarray(best_actions)

    def get_values(self):
        values: list[float] = [self.roots[i].get_value() for i in range(self.num_roots)]
        return np.asarray(values)

    def print_tree(self):
        for root in self.roots:
            root.print_tree([])


class PSearchResults:
    def __init__(self, num: int = None):
        self.num = num if num is not None else 0
        (
            self.hidden_state_index_x_lst, self.hidden_state_index_y_lst,
            self.last_actions, self.search_lens
        ) = (
            [], [], [], []
        )
        self.nodes: list[PNode] = []
        self.search_paths: list[list[PNode]] = []

        if self.num > 0:
            for i in range(num):
                self.search_paths.append([])
