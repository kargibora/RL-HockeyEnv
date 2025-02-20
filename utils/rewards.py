def calculate_dense_rewards(info):
    """
    Compute the dense reward using predefined weighting factors.

    This function calculates a dense reward by combining multiple reward components
    provided in the input dictionary with fixed weighting factors:
      - "winner": Weighted by 10 (primary indicator, e.g., 1 for win, 0 for loss)
      - "reward_touch_puck": Weighted by 1.0 (bonus for touching the puck)
      - "reward_closeness_to_puck": Weighted by 0.05 (reward for being close to the puck)
      - "reward_puck_direction": Weighted by 3.0 (bonus for directing the puck correctly)

    Parameters:
        info (dict): A dictionary containing reward components with the following keys:
            - "winner" (int or float): Indicator of winning.
            - "reward_touch_puck" (float): Reward for touching the puck.
            - "reward_closeness_to_puck" (float): Reward based on proximity to the puck.
            - "reward_puck_direction" (float): Reward based on the puck's direction.

    Returns:
        float: The computed dense reward as a weighted sum of the reward components.

    Example:
        >>> info = {
        ...     "winner": 1,
        ...     "reward_touch_puck": 0.5,
        ...     "reward_closeness_to_puck": 0.2,
        ...     "reward_puck_direction": 0.8
        ... }
        >>> calculate_dense_rewards(info)
        12.91
    """
    return (
        10 * info["winner"] +           # Default weight for 'winner'
        1.0 * info["reward_touch_puck"] +  # Additional weight for 'reward_touch_puck'
        0.05 * info["reward_closeness_to_puck"] +  # Default weight for 'reward_closeness_to_puck'
        3.0 * info["reward_puck_direction"]   # Additional weight for 'reward_puck_direction'
    )


def calculate_dense_rewards_with_weights(info, betas):
    """
    Compute the dense reward using customizable weighting factors.

    This function calculates a dense reward by applying a set of dynamic weights to 
    various reward components. The weights are provided in the 'betas' sequence, where:
      - betas[0]: Weight for the "winner" component.
      - betas[1]: Weight for the "reward_touch_puck" component.
      - betas[2]: Weight for the "reward_closeness_to_puck" component.
      - betas[3]: Weight for the "reward_puck_direction" component.

    Parameters:
        info (dict): A dictionary containing the following reward components:
            - "winner" (int or float): Indicator of winning.
            - "reward_touch_puck" (float): Reward for touching the puck.
            - "reward_closeness_to_puck" (float): Reward based on proximity to the puck.
            - "reward_puck_direction" (float): Reward based on the puck's direction.
        betas (list or tuple): A sequence of four numerical weights corresponding to each
            reward component in the same order as listed above.

    Returns:
        float: The computed dense reward as a weighted sum using the provided weights.

    Example:
        >>> info = {
        ...     "winner": 1,
        ...     "reward_touch_puck": 0.5,
        ...     "reward_closeness_to_puck": 0.2,
        ...     "reward_puck_direction": 0.8
        ... }
        >>> betas = [10, 1.0, 0.05, 3.0]
        >>> calculate_dense_rewards_with_weights(info, betas)
        12.91
    """
    return (
        betas[0] * info["winner"] +           # Weight for 'winner'
        betas[1] * info["reward_touch_puck"] +  # Weight for 'reward_touch_puck'
        betas[2] * info["reward_closeness_to_puck"] +  # Weight for 'reward_closeness_to_puck'
        betas[3] * info["reward_puck_direction"]   # Weight for 'reward_puck_direction'
    )
