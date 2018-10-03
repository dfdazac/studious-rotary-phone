import numpy as np

def solve_bandit(bandit, eps=0, steps=1000, ucb=False, c=1):
    """
    Runs a greedy action selection method to get the best
    sequence of actions.
    Args:
        - bandit (KBandit): the k-armed bandit.
        - eps (float): probability of taking a random action.
        - steps (int): number of steps to run the method
        - ucb (boolean): whether or not to use the Upper Confidence Bound
            instead of greedy or epsilon-greedy approaches.
        - c (float): confidence level used when ucb is True.
    Returns:
        - numpy array of size (steps x 2). The first column
        contains the action taken, the second the reward.
    """
    # Initialize estimates and counts
    action_values = np.zeros(bandit.k)
    action_counts = np.zeros(bandit.k)
    results = np.zeros((steps, 2))

    for step in range(steps):
        # Select action according to selected method
        if ucb:
            # Use the Upper Confidence Bound criterion

            # An action not chosen so far is the maximizer by default
            zero_idx = np.where(action_counts == 0)[0]
            if len(zero_idx) > 0:
                action = zero_idx[0]
            else:
                action = np.argmax(action_values + c * np.sqrt(np.log(step + 1)/action_counts))
        else:
            # Use an epsilon-greedy approach
            if np.random.random() < eps:
                action = np.random.randint(bandit.k)
            else:
                action = np.argmax(action_values)

        # Get reward from bandit
        reward = bandit.get_reward(action)
        results[step, :] = [action, reward]

        # Update counts and estimate for the selected action
        action_counts[action] += 1
        action_values[action] += (reward - action_values[action])/action_counts[action]

    return results
