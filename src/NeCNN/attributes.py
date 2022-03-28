from neat.attributes import BaseAttribute
import numpy as np

class KernelAttribute(BaseAttribute):
    """
    Class for numeric attributes,
    such as the response of a node or the weight of a connection.
    """
    _config_items = {"mutate_rate_value": [float, None],
                     "mutate_rate_size": [float, None],
                     "max_value": [float, None],
                     "min_value": [float, None],
                     "min-size" : [float, None],
                     "max-size": [float, None]}

    def get_min_max(self, config):
        min_size = getattr(config, self.min_size_name)
        max_size = getattr(config, self.max_size_name)

        min_value = getattr(config, self.min_value_name)
        max_value = getattr(config, self.max_value_name)

        return min_size, max_size, min_value, max_value

    def init_value(self, config):
        min_size, max_size, min_value, max_value = self.get_min_max(config)

        kernel_size =  np.random.randint(min_size, max_size)

        return np.random.uniform(min_value, max_value, (kernel_size, kernel_size))

    def mutate_value(self, value, config):
        min_size, max_size, min_value, max_value = self.get_min_max(config)
        # mutate_rate is usually no lower than replace_rate, and frequently higher -
        # so put first for efficiency
        min_size, max_size, min_value, max_value = self.get_min_max(config)

        mutate_rate_value = getattr(config, self.mutate_rate_value_name)
        mutate_rate_size = getattr(config, self.mutate_rate_size_name)
        r = np.random.random()
        if r < mutate_rate_size:
            return self.init_value(config)

        r = np.random.random()
        if r < mutate_rate_value:
            i1 = np.random.choice(np.shape(value)[0])
            i2 = np.random.choice(np.shape(value)[0])
            value[i1, i2] = np.random.uniform(min_value, max_value)

        return value

    def validate(self, config):  # pragma: no cover
        pass