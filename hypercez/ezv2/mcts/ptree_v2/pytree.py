from hypercez.ezv2.mcts.ptree_v2.pminimax import PMinMaxStatsList
from hypercez.ezv2.mcts.ptree_v2.gumble_pnode import PSearchResults, PNode, PRoots, pmulti_back_propagate, \
    pmulti_traverse, pmulti_traverse_return_path


class MinMaxStatsList:
    def __init__(self, num: int):
        self.pmin_max_stats_lst = PMinMaxStatsList(num)

    def set_delta(self, value_delta_max: float):
        self.pmin_max_stats_lst.set_delta(value_delta_max)

    def get_min_max(self):
        return self.pmin_max_stats_lst.get_min_max()


class ResultsWrapper:
    def __init__(self, num: int):
        self.presults = PSearchResults(num)

    def get_search_len(self):
        return self.presults.search_lens


class Node:
    def __init__(self):
        self.pnode: PNode = None

    def expand(self,
               to_play: int,
               hidden_state_index_x: int,
               hidden_state_index_y: int,
               reward_sum: float,
               policy_logits: list,
               simulation_num: int):
        self.pnode.expand(
            to_play, hidden_state_index_x, hidden_state_index_y, reward_sum, policy_logits, simulation_num
        )


class Roots:
    def __init__(self, root_num: int, action_num: int, tree_nodes: int):
        self.root_num = root_num
        self.pool_size = action_num * (tree_nodes + 2)
        self.roots = PRoots(root_num, action_num, tree_nodes)

    def prepare(self, reward_sum_pool: list, policy_logits_pool: list, m: int, simulation_num: int, values: list):
        self.roots.prepare(reward_sum_pool, policy_logits_pool, m, simulation_num, values)

    def get_trajectories(self):
        return self.roots.get_trajectories()

    def get_values(self):
        return self.roots.get_values()

    def get_priors(self):
        return self.roots.get_priors()

    def get_advantages(self, discount: float):
        return self.roots.get_advantages(discount)

    def get_pi_primes(self, min_max_stats_lst: MinMaxStatsList, c_visit: float, c_scale: float, discount: float):
        return self.roots.get_pi_primes(min_max_stats_lst.pmin_max_stats_lst, c_visit, c_scale, discount)

    def get_actions(self,
                    min_max_stats_lst: MinMaxStatsList,
                    c_visit: float,
                    c_scale: float,
                    gumbels: list,
                    discount: float):
        return self.roots.get_actions(min_max_stats_lst.pmin_max_stats_lst, c_visit, c_scale, gumbels, discount)

    def get_child_values(self, discount: float):
        return self.roots.get_child_values(discount)

    def clear(self):
        self.roots.clear()

    @property
    def num(self):
        return self.root_num


def multi_back_propagate(hidden_state_index_x: int,
                         discount: float,
                         reward_sums: list,
                         values: list,
                         policies: list,
                         min_max_stats_lst: MinMaxStatsList,
                         results: ResultsWrapper,
                         is_reset_lst: list,
                         simulation_idx: int,
                         gumbels: list,
                         c_visit: float,
                         c_scale: float,
                         simulation_num: int):
    pmulti_back_propagate(
        hidden_state_index_x,
        discount,
        reward_sums,
        values,
        policies,
        min_max_stats_lst.pmin_max_stats_lst,
        results.presults,
        is_reset_lst,
        simulation_idx,
        gumbels,
        c_visit,
        c_scale,
        simulation_num
    )

def multi_traverse(roots: Roots,
                   c_visit: float,
                   c_scale: float,
                   discount: float,
                   min_max_stats_lst: MinMaxStatsList,
                   results: ResultsWrapper,
                   simulation_idx: int,
                   gumbels: list,
                   use_transformer: int):
    if use_transformer == 0:
        pmulti_traverse(
            roots.roots,
            c_visit,
            c_scale,
            discount,
            min_max_stats_lst.pmin_max_stats_lst,
            results.presults,
            simulation_idx,
            gumbels
        )
        return (
            results.presults.hidden_state_index_x_lst,
            results.presults.hidden_state_index_y_lst,
            results.presults.last_actions,
            None,
            None,
            None
        )

    pmulti_traverse_return_path(
        roots.roots,
        c_visit,
        c_scale,
        discount,
        min_max_stats_lst.pmin_max_stats_lst,
        results.presults,
        simulation_idx,
        gumbels
    )
    return (
        results.presults.hidden_state_index_x_lst,
        results.presults.hidden_state_index_y_lst,
        results.presults.last_actions,
        results.presults.search_path_index_x_lst,
        results.presults.search_path_index_y_lst,
        results.presults.search_path_actions
    )
