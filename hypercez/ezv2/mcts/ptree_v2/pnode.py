import math

from hypercez.ezv2.mcts.ptree_v2.pminimax import PMinMaxStats


class PSearchResults:
    def __init__(self, num = 0):
        self.num = num
        self.search_paths = [
            [] for _ in range(num)
        ] if num > 0 else []


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
    pass





