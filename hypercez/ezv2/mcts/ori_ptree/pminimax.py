class PMinMaxStats:
    def __init__(self):
        self.maximum: float = float('-inf')
        self.minimum: float = float('inf')
        self.value_delta_max: float = 0.0

    def set_delta(self, value_delta_max: float):
        self.value_delta_max = value_delta_max

    def update(self, value: float):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def clear(self):
        self.maximum = float('-inf')
        self.minimum = float('inf')

    def normalize(self, value: float):
        norm_value = value
        delta = self.maximum - self.minimum
        if delta > 0:
            if delta < self.value_delta_max:
                norm_value = (norm_value - self.minimum) / self.value_delta_max
            else:
                norm_value = (norm_value - self.minimum) / delta
        else:
            norm_value = (norm_value - self.minimum) / delta

        return norm_value


class PMinMaxStatsList:
    pass



