import numpy as np

class KBandit:
    """
    A k-armed bandit.
    Args:
        - k (int): the number of actions.
    """
    def __init__(self, k):
        self.k = k
        self.action_values = np.random.normal(size=k)
        self.optimal_action = np.argmax(self.action_values)

    def get_reward(self, a):
        """
        Returns the reward after taking the k-th action.
        Args:
            - a (int): the action to take, in the interval [0, k).
        """
        if a < 0 or a >= len(self.action_values):
            raise ValueError('Invalid action number')

        return np.random.normal(loc=self.action_values[a])
