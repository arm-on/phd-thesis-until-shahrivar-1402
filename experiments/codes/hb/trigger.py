import numpy as np


class TriggerGenerator:
    """
    generates a random trigger of desired shape
    """

    def __init__(self, shape, max_pixel_val=255):
        self.shape = shape
        self.max_pixel_val = max_pixel_val

    def generate(self, num_triggers):
        triggers = []
        for i in range(num_triggers):
            trigger = np.random.randint(self.max_pixel_val, size=self.shape)
            triggers.append(trigger)
        triggers = np.array(triggers)
        return triggers
