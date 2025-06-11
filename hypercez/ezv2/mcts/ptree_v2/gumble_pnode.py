import math

from hypercez.ezv2.mcts.ptree.pminimax import PMinMaxStats, PMinMaxStatsList


class PSearchResults:
    def __init__(self, num: int = 0):
        self.num = num
        self.hidden_state_index_x_lst: list[int] = []
        self.hidden_state_index_y_lst: list[int] = []
        self.last_actions: list[int] = []
        self.search_lens: list[int] = []
        self.nodes: list[PNode] = []
        self.search_paths: list[list[PNode]] = [[] for _ in range(num)]
        self.search_path_index_x_lst: list[list[int]] = [[] for _ in range(num)]
        self.search_path_index_y_lst: list[list[int]] = [[] for _ in range(num)]
        self.search_path_actions: list[list[int]] = [[] for _ in range(num)]


class PNode:
    def __init__(self, prior: float = 0, action_num: int = 0, ptr_node_pool: list['PNode'] = None):
        self.simulation_num = None
        self.prior = prior
        self.action_num = action_num
        self.best_action = -1
        self.is_reset = 0
        self.visit_count = 0
        self.value_sum = 0
        self.best_action = -1
        self.to_play = 0
        self.reward_sum = 0.0
        self.ptr_node_pool = ptr_node_pool
        self.hidden_state_index_x = -1
        self.hidden_state_index_y = -1
        self.is_root = 0
        self.value_mix = 0
        self.phase_added_flag = 0
        self.current_phase = 0
        self.phase_num = 0
        self.phase_to_visit_num = 0
        self.m = 0
        self.children_index: list[int] = []
        self.selected_children: list[tuple[int, PNode]] = []
        self.parent: PNode = None

    def expand(self,
               to_play: int,
               hidden_state_index_x: int,
               hidden_state_index_y: int,
               reward_sum: float,
               policy_logits: list[float],
               simulation_num: int):
        self.to_play = to_play
        self.simulation_num = simulation_num
        self.hidden_state_index_x = hidden_state_index_x
        self.hidden_state_index_y = hidden_state_index_y
        self.reward_sum = reward_sum

        action_num = self.action_num

        for a in range(action_num):
            index = len(self.ptr_node_pool)
            self.children_index.append(index)
            child_node = PNode(policy_logits[a], action_num, self.ptr_node_pool)
            child_node.parent = self
            self.ptr_node_pool.append(child_node)

    def print_out(self):
        print("*****")
        print(
            f"visit count: {self.visit_count} \t hidden_state_index_x: {self.hidden_state_index_x} \t hidden_state_index_y: {self.hidden_state_index_y} \t reward: {self.reward_sum:.6f} \t prior: {self.prior:.6f}")
        print(f"children_index size: {len(self.children_index)} \t pool size: {len(self.ptr_node_pool)}")
        print("*****")

    def expanded(self):
        return len(self.children_index) > 0

    def value(self):
        if self.visit_count == 0:
            print(f"{self.parent.value_mix:.6f}")
            return self.parent.value_mix
        true_value = self.value_sum / self.visit_count
        return true_value

    def final_value(self):
        final_value_sum = 0.0
        final_action_num = 0.0
        for selected_child in self.selected_children:
            a = selected_child[0]
            final_value_sum += self.get_qsa(a, 0.997)
            final_action_num += 1
        return final_value_sum / final_action_num

    def get_child(self, action: int):
        return self.ptr_node_pool[self.children_index[action]]

    def get_qsa(self, action: int, discount: float):
        child = self.get_child(action)
        true_reward = child.reward_sum - (self.reward_sum if not child.is_reset else 0)
        return true_reward + discount * child.value()

    def v_mix(self, discount):
        pi_dot_sum = 0.0
        pi_value_sum = 0.0
        pi_probs = []

        # Compute un-normalized softmax over priors
        for a in range(self.action_num):
            child = self.get_child(a)
            prob = math.exp(child.prior)
            pi_probs.append(prob)
            pi_dot_sum += prob

        # Normalize and accumulate for visited children
        pi_visited_sum = 0.0
        for a in range(self.action_num):
            pi_probs[a] /= pi_dot_sum
            child = self.get_child(a)
            if child.expanded():
                pi_visited_sum += pi_probs[a]
                pi_value_sum += pi_probs[a] * self.get_qsa(a, discount)

        # Fallback if no visited children
        if abs(pi_visited_sum) < 1e-4:
            return self.value()

        # Compute and return mixed value
        return (1.0 / (1.0 + self.visit_count)) * (
                self.value() + (self.visit_count / pi_visited_sum) * pi_value_sum
        )

    def completedQ(self, discount):
        completed_q = []
        v_mix = self.v_mix(discount)
        self.value_mix = v_mix

        for a in range(self.action_num):
            child = self.get_child(a)
            completed_q.append(self.get_qsa(a, discount) if child.expanded() else v_mix)

        return completed_q

    def get_trajectory(self):
        traj = []
        node = self
        best_action = node.best_action

        while best_action >= 0:
            traj.append(best_action)
            node = node.get_child(best_action)
            best_action = node.best_action

        return traj


def calc_advantage(node: PNode, discount: float):
    advantage = []
    completed_q = node.completedQ(discount)
    node_value = node.value()
    for a in range(node.action_num):
        advantage.append(completed_q[a] - node_value)  # target_V - this_V
    return advantage

def sigma(value: float, root: PNode, c_visit: float, c_scale: float):
    max_visit = 0
    for a in range(root.action_num):
        child = root.get_child(a)
        if child.visit_count > max_visit:
            max_visit = child.visit_count
    return (c_visit + max_visit) * c_scale * value

def calc_pi_prime_dot(node: PNode, min_max_stats: PMinMaxStats, c_visit: float, c_scale: float, discount: float):
    pi_prime = []
    completed_q = node.completedQ(discount)
    pi_prime_max = -10000.0

    for a in range(node.action_num):
        child = node.get_child(a)
        normalized_value = min_max_stats.normalize(completed_q[a])
        visit_count = max(child.visit_count, 1)

        if normalized_value < 0:
            normalized_value = 0
        if normalized_value > 1:
            normalized_value = 1

        score = child.prior + sigma(normalized_value * math.log(visit_count + 1), node, c_visit, c_scale)
        pi_prime_max = max(pi_prime_max, score)
        pi_prime.append(score)

    pi_value_sum = 0.0
    for a in range(node.action_num):
        pi_prime[a] = math.exp(pi_prime[a] - pi_prime_max)
        pi_value_sum += pi_prime[a]

    for a in range(node.action_num):
        pi_prime[a] /= pi_value_sum

    return pi_prime

def calc_gumbel_score(node: PNode,
                      gumbels: list[float],
                      min_max_stats: PMinMaxStats,
                      c_visit: float,
                      c_scale: float,
                      discount: float):
    gumbel_scores = []
    completedQ = node.completedQ(discount)
    for a, child in node.selected_children:
        normalized_value = min_max_stats.normalize(completedQ[a])
        if normalized_value < 0:
            normalized_value = 0
        if normalized_value > 1:
            normalized_value = 1
        score = gumbels[a] + child.prior + sigma(normalized_value, node, c_visit, c_scale)
        gumbel_scores.append((a, score))
    return gumbel_scores


class PRoots:
    def __init__(self, root_num: int = 0, action_num: int = 0, pool_size: int = 0):
        self.root_num = root_num
        self.action_num = action_num
        self.pool_size = pool_size
        self.node_pools: list[list[PNode]] = [[] for _ in range(self.root_num)]
        self.roots: list[PNode] = [PNode(0, action_num, self.node_pools[i]) for i in range(self.root_num)]

    def prepare(self, reward_sums, policies, m, simulation_num, values):
        for i in range(self.root_num):
            self.roots[i].expand(
                to_play=0,
                hidden_state_index_x=0,
                hidden_state_index_y=i,
                reward_sum=reward_sums[i],
                policy_logits=policies[i],
                simulation_num=simulation_num
            )
            self.roots[i].is_root = 1
            self.roots[i].m = min(m, self.action_num)
            self.roots[i].phase_num = math.ceil(math.log2(self.roots[i].m))
            self.roots[i].simulation_num = simulation_num
            self.roots[i].value_sum += values[i]
            self.roots[i].visit_count += 1

    def clear(self):
        self.node_pools.clear()
        self.roots.clear()

    def get_trajectories(self):
        return [self.roots[i].get_trajectory() for i in range(self.root_num)]

    def get_advantages(self, discount: float):
        return [calc_advantage(self.roots[i], discount) for i in range(self.root_num)]

    def get_pi_primes(self, min_max_stats_lst: PMinMaxStatsList, c_visit: float, c_scale: float, discount: float):
        return [
            calc_pi_prime_dot(
                self.roots[i], min_max_stats_lst.stats_lst[i], c_visit, c_scale, discount
            ) for i in range(self.root_num)
        ]

    def get_priors(self):
        return [
            [self.roots[i].get_child(a) for a in range(self.roots[i].action_num)] for i in range(self.root_num)
        ]

    def get_actions(self, min_max_stats_lst: PMinMaxStatsList,
                    c_visit: float,
                    c_scale: float,
                    gumbels: list[list[float]],
                    discount: float):
        actions = []
        for i in range(self.root_num):
            root = self.roots[i]
            scores = calc_gumbel_score(root, gumbels[i], min_max_stats_lst.stats_lst[i], c_visit, c_scale, discount)

            temp_scores = [score[1] for score in scores]
            action_index = temp_scores.index(max(temp_scores))
            action = root.selected_children[action_index][0]
            actions.append(action)
        return actions





