import numpy as np


class RandomBaseline:
    def __init__(self):
        pass

    def select_action(self, num_actions):
        action = np.random.randint(0, num_actions-1)
        return action


if __name__ == '__main__':

    rb = RandomBaseline()

    for i in range(10):
        print(rb.select_action(5))
