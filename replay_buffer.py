import random
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        """
        Add a transition to the replay buffer.
        """
        state = np.array(state, copy=False)
        action = np.array(action, copy=False)
        next_state = np.array(next_state, copy=False)
        r = float(reward)
        d = 1.0 if done else 0.0  # store done as float (1 or 0)
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, r, next_state, d))
        else:
            # Overwrite oldest
            self.buffer[self.position] = (state, action, r, next_state, d)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        # Convert to numpy arrays
        states = np.stack(states)
        actions = np.stack(actions)
        rewards = np.array(rewards, dtype=np.float32)
        next_states = np.stack(next_states)
        dones = np.array(dones, dtype=np.float32)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)
