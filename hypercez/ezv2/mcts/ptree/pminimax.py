class PMinMaxStats:
    def __init__(self):
        self.maximum = float('-inf')
        self.minimum = float('inf')
        self.value_delta_max = 0.
        self.scale = 0.
        self.visit = 0

    def set_static_val(self, value_delta_max: float, visit: int, scale: float):
        self.value_delta_max = value_delta_max
        self.visit = visit
        self.scale = scale

    def update(self, value: float):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def clear(self):
        self.maximum = float('-inf')
        self.minimum = float('inf')

    def normalize(self, value: float) -> float:
        norm_value = value
        delta = self.maximum - self.minimum
        if delta > 0:
            if delta < self.value_delta_max:
                norm_value = (norm_value - self.minimum) / self.value_delta_max
            else:
                norm_value = (norm_value - self.minimum) / delta

        if norm_value > 1:
            norm_value = 1
        elif norm_value < 0:
            norm_value = 0
        return norm_value


class PMinMaxStatsList:
    def __init__(self, num:int=None):
        self.num = num if num is not None else 0
        self.stats_lst: list[PMinMaxStats] = []

        if num > 0:
            for i in range(num):
                self.stats_lst.append(PMinMaxStats())

    def set_static_val(self, value_delta_max: float, visit: int, scale: float):
        for i in range(self.num):
            self.stats_lst[i].set_static_val(value_delta_max, visit, scale)
