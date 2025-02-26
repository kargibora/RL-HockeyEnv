"""
The code is mostly from https://github.com/openai/baselines/tree/master
Although the code is inspired from the above source, it has been modified to suit the needs of the project.
"""
import operator
import numpy as np
import torch
import random

class SegmentTree(object):
    def __init__(self, capacity, operation, neutral_element):
        """Build a Segment Tree data structure.

        https://en.wikipedia.org/wiki/Segment_tree

        Can be used as regular array, but with two
        important differences:

            a) setting item's value is slightly slower.
               It is O(lg capacity) instead of O(1).
            b) user has access to an efficient ( O(log segment size) )
               `reduce` operation which reduces `operation` over
               a contiguous subsequence of items in the array.

        Paramters
        ---------
        capacity: int
            Total size of the array - must be a power of two.
        operation: lambda obj, obj -> obj
            and operation for combining elements (eg. sum, max)
            must form a mathematical group together with the set of
            possible values for array elements (i.e. be associative)
        neutral_element: obj
            neutral element for the operation above. eg. float('-inf')
            for max and 0 for sum.
        """
        assert capacity > 0 and capacity & (capacity - 1) == 0, "capacity must be positive and a power of 2."
        self._capacity = capacity
        self._value = [neutral_element for _ in range(2 * capacity)]
        self._operation = operation

    def _reduce_helper(self, start, end, node, node_start, node_end):
        if start == node_start and end == node_end:
            return self._value[node]
        mid = (node_start + node_end) // 2
        if end <= mid:
            return self._reduce_helper(start, end, 2 * node, node_start, mid)
        else:
            if mid + 1 <= start:
                return self._reduce_helper(start, end, 2 * node + 1, mid + 1, node_end)
            else:
                return self._operation(
                    self._reduce_helper(start, mid, 2 * node, node_start, mid),
                    self._reduce_helper(mid + 1, end, 2 * node + 1, mid + 1, node_end)
                )

    def reduce(self, start=0, end=None):
        """Returns result of applying `self.operation`
        to a contiguous subsequence of the array.

            self.operation(arr[start], operation(arr[start+1], operation(... arr[end])))

        Parameters
        ----------
        start: int
            beginning of the subsequence
        end: int
            end of the subsequences

        Returns
        -------
        reduced: obj
            result of reducing self.operation over the specified range of array elements.
        """
        if end is None:
            end = self._capacity
        if end < 0:
            end += self._capacity
        end -= 1
        return self._reduce_helper(start, end, 1, 0, self._capacity - 1)

    def __setitem__(self, idx, val):
        # index of the leaf
        idx += self._capacity
        self._value[idx] = val
        idx //= 2
        while idx >= 1:
            self._value[idx] = self._operation(
                self._value[2 * idx],
                self._value[2 * idx + 1]
            )
            idx //= 2

    def __getitem__(self, idx):
        assert 0 <= idx < self._capacity
        return self._value[self._capacity + idx]


class SumSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super(SumSegmentTree, self).__init__(
            capacity=capacity,
            operation=operator.add,
            neutral_element=0.0
        )

    def sum(self, start=0, end=None):
        """Returns arr[start] + ... + arr[end]"""
        return super(SumSegmentTree, self).reduce(start, end)

    def find_prefixsum_idx(self, prefixsum):
        """Find the highest index `i` in the array such that
            sum(arr[0] + arr[1] + ... + arr[i - i]) <= prefixsum

        if array values are probabilities, this function
        allows to sample indexes according to the discrete
        probability efficiently.

        Parameters
        ----------
        perfixsum: float
            upperbound on the sum of array prefix

        Returns
        -------
        idx: int
            highest index satisfying the prefixsum constraint
        """
        assert 0 <= prefixsum <= self.sum() + 1e-5
        idx = 1
        while idx < self._capacity:  # while non-leaf
            if self._value[2 * idx] > prefixsum:
                idx = 2 * idx
            else:
                prefixsum -= self._value[2 * idx]
                idx = 2 * idx + 1
        return idx - self._capacity


class MinSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super(MinSegmentTree, self).__init__(
            capacity=capacity,
            operation=min,
            neutral_element=float('inf')
        )

    def min(self, start=0, end=None):
        """Returns min(arr[start], ...,  arr[end])"""

        return super(MinSegmentTree, self).reduce(start, end)
    
class PrioritizedReplayBuffer:
    """
    A replay buffer that prioritizes experiences based on their TD error.
    
    This class uses segment trees (for sum and min operations) to efficiently
    sample experiences with probabilities proportional to their priorities.
    
    Attributes:
        max_size (int): Maximum number of transitions to store.
        state (np.ndarray): Pre-allocated array for states.
        action (np.ndarray): Pre-allocated array for actions.
        next_state (np.ndarray): Pre-allocated array for next states.
        reward (np.ndarray): Pre-allocated array for rewards.
        done (np.ndarray): Pre-allocated array for done flags.
        _alpha (float): Degree of prioritization (0 means uniform sampling, 1 means full prioritization).
        _it_sum (SegmentTree): Sum tree for quickly computing cumulative priorities.
        _it_min (SegmentTree): Min tree for finding the smallest priority.
        _max_priority (float): Highest priority seen so far (used for new samples).
        device (torch.device): Device on which tensors are stored.
    """
    def __init__(self, max_size, state_dim, action_dim, alpha):
        # Ensure a nonnegative prioritization factor.
        assert alpha >= 0, "Alpha must be non-negative."
        self.max_size = max_size
        self.ptr = 0     # Next index for inserting a new experience.
        self.size = 0    # Current number of stored experiences.

        # Pre-allocate memory for experience components.
        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.done = np.zeros((max_size, 1))

        self._alpha = alpha
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Determine capacity for the segment trees (next power of 2 ≥ max_size).
        tree_capacity = 1
        while tree_capacity < max_size:
            tree_capacity *= 2

        # Initialize segment trees for priority sums and minimums.
        self._it_sum = SumSegmentTree(tree_capacity)
        self._it_min = MinSegmentTree(tree_capacity)
        self._max_priority = 1.0

    def scale_alpha(self, step, beta):
        """
        Scale the prioritization factor alpha according to the current step.
        
        Parameters:
            step (int): Current training step.
            beta (float): Final value for the prioritization factor.
        """
        self._alpha = min(1.0, self._alpha + (beta - self._alpha) * step / beta)
        
    def add(self, state, action, next_state, reward, done):
        """
        Insert a new transition into the buffer.
        
        Parameters:
            state: Current state.
            action: Action taken.
            next_state: Next state observed.
            reward: Reward received.
            done: Boolean flag indicating episode termination.
        """
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.done[self.ptr] = done
        
        # Use the maximum observed priority as the initial priority for new samples.
        priority = self._max_priority ** self._alpha
        self._it_sum[self.ptr] = priority
        self._it_min[self.ptr] = priority

        # Move the pointer forward and update current buffer size.
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def _sample_proportional(self, batch_size):
        """
        Sample indices based on the relative priority of each transition.
        
        Parameters:
            batch_size (int): Number of indices to sample.
        
        Returns:
            List of indices selected according to their priority.
        """
        indices = []
        total_priority = self._it_sum.sum(0, self.size - 1)
        segment = total_priority / batch_size

        for i in range(batch_size):
            mass = random.random() * segment + i * segment
            idx = self._it_sum.find_prefixsum_idx(mass)
            indices.append(idx)
        return indices

    def sample(self, batch_size, beta):
        """
        Retrieve a batch of experiences along with importance-sampling weights.
        
        Parameters:
            batch_size (int): Number of experiences to sample.
            beta (float): Degree of bias correction for importance sampling (should be > 0).
        
        Returns:
            A tuple containing:
                - A tuple of tensors: (states, actions, next_states, rewards, dones)
                - Importance-sampling weights (numpy array)
                - Indices of the sampled experiences
        """
        assert beta > 0, "Beta must be positive for importance sampling correction."

        indices = self._sample_proportional(batch_size)
        weights = []
        # Compute the minimum sampling probability for scaling weights.
        p_min = self._it_min.min() / self._it_sum.sum(0, self.size - 1)
        max_weight = (p_min * self.size) ** (-beta)

        for idx in indices:
            p_sample = self._it_sum[idx] / self._it_sum.sum(0, self.size - 1)
            weight = (p_sample * self.size) ** (-beta)
            weights.append(weight / max_weight)

        weights = np.array(weights)
        indices = np.array(indices)

        data = self._fetch_data(indices)
        return data, weights, indices

    def _fetch_data(self, indices):
        """
        Retrieve experiences corresponding to given indices.
        
        Parameters:
            indices (list or np.ndarray): Indices of the experiences to retrieve.
        
        Returns:
            Tuple of tensors: (states, actions, next_states, rewards, dones)
        """
        idx_array = np.array(indices)
        return (
            torch.FloatTensor(self.state[idx_array]).to(self.device),
            torch.FloatTensor(self.action[idx_array]).to(self.device),
            torch.FloatTensor(self.next_state[idx_array]).to(self.device),
            torch.FloatTensor(self.reward[idx_array]).to(self.device),
            torch.FloatTensor(self.done[idx_array]).to(self.device)
        )

    def update_priorities(self, indices, priorities):
        """
        Update the priorities for specific experiences in the buffer.
        
        Parameters:
            indices (list): Indices of the experiences to update.
            priorities (list): New priority values for each corresponding experience.
        """
        assert len(indices) == len(priorities), "Indices and priorities must have equal length."
        for idx, priority in zip(indices, priorities):
            assert priority > 0, "Priority must be positive."
            assert 0 <= idx < self.size, "Index is out of the valid range."
            new_priority = priority ** self._alpha
            self._it_sum[idx] = new_priority
            self._it_min[idx] = new_priority
            self._max_priority = max(self._max_priority, priority)