from ez.mcts.ptree.pminimax import PMinMaxStatsList
from ez.mcts.ptree.pnode import PRoots, p_batch_sequential_halving, p_batch_traverse, p_batch_back_propagate, PSearchResults


class MinMaxStatsList:
    def __init__(self, num: int):
        self.pmin_max_stats_lst = PMinMaxStatsList(num)

    def set_static_val(self, value_delta_max: float, p_visit: int, p_scale: float):
        self.pmin_max_stats_lst.set_static_val(value_delta_max, p_visit, p_scale)


class ResultsWrapper:
    def __init__(self, num):
        self.presults = PSearchResults(num)

    def get_search_len(self):
        return self.presults.search_lens


class Roots:
    def __init__(self, num_roots: int, action_num: int, tree_nodes: int, discount:float):
        self.num_roots = num_roots
        self.pool_size = action_num * (tree_nodes + 2)
        self.discount = discount
        self.roots = PRoots(num_roots, action_num, self.pool_size, discount)

    def prepare(self, values: list, policies: list, leaf_action_num: int):
        self.roots.prepare(values, policies, leaf_action_num)

    def get_trajectories(self):
        return self.roots.get_trajectories()

    def get_distributions(self):
        return self.roots.get_distributions()

    def get_root_policies(self, min_max_stats_lst: MinMaxStatsList):
        return self.roots.get_root_policies(min_max_stats_lst.pmin_max_stats_lst)

    def get_best_actions(self):
        return self.roots.get_best_actions()

    def get_values(self):
        return self.roots.get_values()

    def print_tree(self):
        return self.roots.print_tree()

    def clear(self):
        self.roots.clear()

    @property
    def num(self):
        return self.num_roots


def batch_sequential_halving(roots: Roots,
                             gumble_noises: list,
                             min_max_stats_lst: MinMaxStatsList,
                             current_phase: int,
                             current_num_top_actions: int):
    return p_batch_sequential_halving(
        roots.roots, gumble_noises, min_max_stats_lst.pmin_max_stats_lst, current_phase, current_num_top_actions
    )


def batch_traverse(roots: Roots,
                   min_max_stats_lst: MinMaxStatsList,
                   results: ResultsWrapper,
                   num_simulations: int,
                   simulation_idx: int,
                   gumbel_noises: list,
                   current_num_top_actions: int):
    p_batch_traverse(
        roots.roots,
        min_max_stats_lst.pmin_max_stats_lst,
        results.presults,
        num_simulations,
        simulation_idx,
        gumbel_noises,
        current_num_top_actions
    )
    return (
        results.presults.hidden_state_index_x_lst,
        results.presults.hidden_state_index_y_lst,
        results.presults.last_actions
    )


def batch_back_propagate(hidden_state_index_x: int,
                         next_value_prefixes: list,
                         next_values: list,
                         next_logits: list,
                         min_max_stats_lst: MinMaxStatsList,
                         results: ResultsWrapper,
                         is_reset_lst: list,
                         leaf_action_num: int):
    p_batch_back_propagate(
        hidden_state_index_x,
        next_value_prefixes,
        next_values,
        next_logits,
        min_max_stats_lst.pmin_max_stats_lst,
        results.presults,
        is_reset_lst,
        leaf_action_num
    )
