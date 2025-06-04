class MinMaxStats:
    def __init__(self, minmax_delta, min_value_bound=None, max_value_bound=None):
        """
        Minimum and Maximum statistics
        :param minmax_delta: float, for soft update
        :param min_value_bound:
        :param max_value_bound:
        """
        self.maximum = min_value_bound if min_value_bound else -float('inf')
        self.minimum = max_value_bound if max_value_bound else float('inf')
        self.minmax_delta = minmax_delta

    def update(self, value: float):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value: float) -> float:
        if self.maximum > self.minimum:
            if value >= self.maximum:
                value = self.maximum
            elif value <= self.minimum:
                value = self.minimum
            # We normalize only when we have set the maximum and minimum values.
            value = (value - self.minimum) / max(self.maximum - self.minimum, self.minmax_delta)  # [-1, 1] range

        value = max(min(value, 1), 0)
        return value

    def clear(self):
        self.maximum = -float('inf')
        self.minimum = float('inf')