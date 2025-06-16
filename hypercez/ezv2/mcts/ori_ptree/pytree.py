from hypercez.ezv2.mcts.ori_ptree.pminimax import PMinMaxStatsList
from hypercez.ezv2.mcts.ori_ptree.pnode import PSearchResults, PRoots, PNode, pbatch_back_propagate, pbatch_traverse


class MinMaxStatsList:
    def __init__(self, num: int):
        self.pmin_max_stats_lst = PMinMaxStatsList(num)

    def set_delta(self, value_delta_max: float):
        self.pmin_max_stats_lst.set_delta(value_delta_max)


class ResultsWrapper:
    def __init__(self, num: int):
        self.presults = PSearchResults(num)

    def get_search_len(self):
        return self.presults.search_lens


class Roots:
    def __init__(self, root_num: int, action_num: int, tree_nodes: int):
        self.root_num = root_num
        self.pool_size = action_num * (tree_nodes + 2)
        self.roots = PRoots(root_num, action_num, self.pool_size)

    def prepare(self,
                root_exploration_fraction: float,
                noises: list,
                value_prefix_pool: list,
                policy_logits_pool: list
                ):
        self.roots.prepare(root_exploration_fraction, noises, value_prefix_pool, policy_logits_pool)

    def prepare_no_noise(self, value_prefix_pool: list, policy_logits_pool: list):
        self.roots.prepare_no_noise(value_prefix_pool, policy_logits_pool)

    def get_trajectories(self):
        return self.roots.get_trajectories()

    def get_distributions(self):
        return self.roots.get_distributions()

    def get_values(self):
        return self.roots.get_values()

    def clear(self):
        self.roots.clear()

    @property
    def num(self):
        return self.root_num


class Node:
    def __init__(self):
        self.pnode: PNode = None

    def expand(self,
               to_play: int,
               hidden_state_index_x: int,
               hidden_state_index_y: int,
               value_prefix: float,
               policy_logits: list):
        self.pnode.expand(to_play, hidden_state_index_x, hidden_state_index_y, value_prefix, policy_logits)


def batch_back_propagate(hidden_state_index_x: int,
                         discount: float,
                         value_prefixs: list,
                         values: list,
                         policies: list,
                         min_max_stats_lst: MinMaxStatsList,
                         results: ResultsWrapper,
                         is_reset_lst: list):
    pbatch_back_propagate(hidden_state_index_x, discount, value_prefixs, values, policies,
                          min_max_stats_lst.pmin_max_stats_lst, results.presults, is_reset_lst)


def batch_traverse(roots: Roots,
                   pb_c_base: int,
                   pb_c_init: float,
                   discount: float,
                   min_max_stats_lst: MinMaxStatsList,
                   results: ResultsWrapper):
    pbatch_traverse(roots.roots, pb_c_base, pb_c_init, discount, min_max_stats_lst.pmin_max_stats_lst, results.presults)

    return (
        results.presults.hidden_state_index_x_lst,
        results.presults.hidden_state_index_y_lst,
        results.presults.last_actions
    )
