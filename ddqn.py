import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from replay_buffer import ReplayBuffer

class DuelingDQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(DuelingDQN, self).__init__()
        # Fully-connected layers
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # Dueling streams: value and advantage
        self.value_layer = nn.Linear(hidden_dim, 1)
        self.advantage_layer = nn.Linear(hidden_dim, action_dim)
        # Initialize weights (Xavier initialization)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.value_layer.weight)
        nn.init.xavier_uniform_(self.advantage_layer.weight)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        # Compute value and advantage
        value = self.value_layer(x)              # shape [batch, 1]
        advantage = self.advantage_layer(x)      # shape [batch, action_dim]
        # Combine into Q-values: Q(s,a) = V(s) + A(s,a) - mean(A(s,*))
        advantage_mean = advantage.mean(dim=1, keepdim=True)
        q_values = value + advantage - advantage_mean
        return q_values

class DDQNAgent:
    def __init__(self, env, hidden_dim=128, batch_size=64, lr=1e-3, gamma=0.95, target_update_freq=1000, memory_capacity=10000):
        """
        Dueling DDQN Agent that interacts with the environment.
        env: Environment object (provides M, N, F, K, etc.).
        """
        self.env = env
        # Dimensions
        self.state_dim = env.N * env.F * env.K + env.M * env.N
        self.action_dim = 2 * env.M * env.F * env.K + env.M * env.N
        # Dueling Q-Network and Target Network
        self.q_network = DuelingDQN(self.state_dim, self.action_dim, hidden_dim)
        self.target_network = DuelingDQN(self.state_dim, self.action_dim, hidden_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        # Replay memory
        self.memory = ReplayBuffer(memory_capacity)
        # Training parameters
        self.batch_size = batch_size
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.update_counter = 0
        # Exploration parameters
        self.epsilon = 1.0
        self.min_epsilon = 0.1
        self.epsilon_decay = 0.99
        # Device (GPU if available)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network.to(self.device)
        self.target_network.to(self.device)

    def select_action(self, state):
        """
        Select action (X, Y, Z) using epsilon-greedy strategy.
        Returns X, Y, Z arrays and action_mask (flattened binary vector of chosen action).
        """
        # Exploration or exploitation
        if np.random.rand() < self.epsilon:
            # Random action: sample a valid combination of X, Y, Z
            M, N, F, K = self.env.M, self.env.N, self.env.F, self.env.K
            Cc, Cr = self.env.C_cache, self.env.C_rec
            X = np.zeros((M, F, K), dtype=int)
            Y = np.zeros((M, F, K), dtype=int)
            Z = np.zeros((M, N), dtype=int)
            # Random caching strategy for each BS (choose C_cache random (f,k) per BS)
            all_pairs = [(f, k) for f in range(F) for k in range(K)]
            for m in range(M):
                if len(all_pairs) <= Cc:
                    chosen_pairs = all_pairs[:]
                else:
                    chosen_idx = np.random.choice(len(all_pairs), size=Cc, replace=False)
                    chosen_pairs = [all_pairs[i] for i in chosen_idx]
                for (f, k) in chosen_pairs:
                    X[m, f, k] = 1
            # Random recommendation strategy for each BS (choose C_rec random (f,k) per BS)
            for m in range(M):
                if len(all_pairs) <= Cr:
                    chosen_pairs = all_pairs[:]
                else:
                    chosen_idx = np.random.choice(len(all_pairs), size=Cr, replace=False)
                    chosen_pairs = [all_pairs[i] for i in chosen_idx]
                for (f, k) in chosen_pairs:
                    Y[m, f, k] = 1
            # Random user association (each user to a random BS)
            for n in range(N):
                m_choice = np.random.randint(0, M)
                Z[:, n] = 0
                Z[m_choice, n] = 1
            action_mask = np.concatenate([X.flatten(), Y.flatten(), Z.flatten()]).astype(np.float32)
            return X, Y, Z, action_mask
        else:
            # Greedy action from Q-network
            state_t = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(state_t).cpu().numpy().flatten()
            M, N, F, K = self.env.M, self.env.N, self.env.F, self.env.K
            Cc, Cr = self.env.C_cache, self.env.C_rec
            X = np.zeros((M, F, K), dtype=int)
            Y = np.zeros((M, F, K), dtype=int)
            Z = np.zeros((M, N), dtype=int)
            # Greedily select caching (top Cc for each BS)
            offset_cache = 0
            for m in range(M):
                segment = q_values[offset_cache + m*F*K : offset_cache + (m+1)*F*K]
                if Cc >= F*K:
                    top_indices = np.arange(F*K)
                else:
                    top_indices = np.argpartition(segment, -Cc)[-Cc:]
                for idx in top_indices:
                    f = idx // K
                    k = idx % K
                    X[m, f, k] = 1
            # Greedily select recommendations (top Cr for each BS)
            offset_rec = M * F * K
            for m in range(M):
                segment = q_values[offset_rec + m*F*K : offset_rec + (m+1)*F*K]
                if Cr >= F*K:
                    top_indices = np.arange(F*K)
                else:
                    top_indices = np.argpartition(segment, -Cr)[-Cr:]
                for idx in top_indices:
                    f = idx // K
                    k = idx % K
                    Y[m, f, k] = 1
            # Greedily select user associations (best BS for each user)
            offset_assoc = 2 * M * F * K
            for n in range(N):
                segment = q_values[offset_assoc + n*M : offset_assoc + (n+1)*M]
                m_best = int(np.argmax(segment))
                Z[:, n] = 0
                Z[m_best, n] = 1
            action_mask = np.concatenate([X.flatten(), Y.flatten(), Z.flatten()]).astype(np.float32)
            return X, Y, Z, action_mask

    def store_transition(self, state, action_mask, reward, next_state, done):
        """Store a transition in replay memory."""
        self.memory.push(state, action_mask, reward, next_state, done)

    def update(self):
        """Perform one DDQN training step (experience replay and network update)."""
        if len(self.memory) < self.batch_size:
            return  # not enough samples yet
        # Sample mini-batch
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        state_t = torch.from_numpy(states).float().to(self.device)
        next_state_t = torch.from_numpy(next_states).float().to(self.device)
        action_mask_t = torch.from_numpy(actions).float().to(self.device)
        reward_t = torch.from_numpy(rewards).float().to(self.device)
        done_t = torch.from_numpy(dones).float().to(self.device)
        # Q-values for current states (for actions taken)
        q_pred_all = self.q_network(state_t)  # [batch, action_dim]
        q_pred = (q_pred_all * action_mask_t).sum(dim=1)
        # Compute target Q values
        with torch.no_grad():
            q_eval_next = self.q_network(next_state_t).cpu().numpy()
            q_target_next = self.target_network(next_state_t).cpu().numpy()
        q_target_vals = []
        B = states.shape[0]
        M, N, F, K = self.env.M, self.env.N, self.env.F, self.env.K
        for i in range(B):
            # If terminal state, target = reward
            if done_t[i].item() == 1.0:
                target_val = reward_t[i].item()
            else:
                # Greedy action selection on next state (using eval network outputs)
                q_eval = q_eval_next[i]
                q_tar = q_target_next[i]
                # Determine best composite action (X, Y, Z) for next state
                X = np.zeros((M, F, K), dtype=int)
                Y = np.zeros((M, F, K), dtype=int)
                Z = np.zeros((M, N), dtype=int)
                # Caching (top C_cache per BS)
                for m in range(M):
                    seg = q_eval[m*F*K:(m+1)*F*K]
                    if self.env.C_cache >= F*K:
                        top_indices = np.arange(F*K)
                    else:
                        top_indices = np.argpartition(seg, -self.env.C_cache)[-self.env.C_cache:]
                    for idx in top_indices:
                        f = idx // K
                        k = idx % K
                        X[m, f, k] = 1
                # Recommendation (top C_rec per BS)
                offset_rec = M * F * K
                for m in range(M):
                    seg = q_eval[offset_rec + m*F*K : offset_rec + (m+1)*F*K]
                    if self.env.C_rec >= F*K:
                        top_indices = np.arange(F*K)
                    else:
                        top_indices = np.argpartition(seg, -self.env.C_rec)[-self.env.C_rec:]
                    for idx in top_indices:
                        f = idx // K
                        k = idx % K
                        Y[m, f, k] = 1
                # User association (best BS per user)
                offset_assoc = 2 * M * F * K
                for n in range(N):
                    seg = q_eval[offset_assoc + n*M : offset_assoc + (n+1)*M]
                    m_best = int(np.argmax(seg))
                    Z[m_best, n] = 1
                # Compute Q value of this action from target network
                action_mask_next = np.concatenate([X.flatten(), Y.flatten(), Z.flatten()])
                total_q = np.dot(q_tar, action_mask_next)
                target_val = reward_t[i].item() + self.gamma * total_q
            q_target_vals.append(target_val)
        q_target_vals = torch.tensor(q_target_vals).float().to(self.device)
        # Loss = MSE(target Q, predicted Q)
        loss = nn.MSELoss()(q_pred, q_target_vals)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # Update target network periodically
        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
