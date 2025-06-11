class PMinMaxStats:
    def __init__(self):
        self.maximum = float('-inf')
        self.minimum = float('inf')
        self.value_delta_max = 0

    def set_delta(self, value_delta_max):
        self.value_delta_max = value_delta_max

    def update(self, value):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def clear(self):
        self.maximum = float('-inf')
        self.minimum = float('inf')

    def normalize(self, value):
        norm_value = value
        delta = self.maximum - self.minimum
        if delta > 0:
            if delta < self.value_delta_max:
                norm_value = (norm_value - self.minimum) / self. value_delta_max
            else:
                norm_value = (norm_value - self.minimum) / delta

        return norm_value


class PMinMaxStatsList:
    def __init__(self, num: int = 0):
        self.num = num
        self.stats_lst = []
        if num > 0:
            self.stats_lst = [PMinMaxStats() for i in range(num)]

    def set_delta(self, value_delta_max):
        for stat in self.stats_lst:
            stat.set_delta(value_delta_max)

    def get_min_max(self):
        min_maxs = []
        for i in range(self.num):
            min_maxs.append(self.stats_lst[i].minimum)
            min_maxs.append(self.stats_lst[i].maximum)
        return min_maxs
