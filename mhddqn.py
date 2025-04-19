import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from replay_buffer import ReplayBuffer
from ddqn import DDQNAgent  # 继承原DDQNAgent

class MultiHeadDuelingDQN(nn.Module):
    def __init__(self, env, hidden_dim=128):
        super(MultiHeadDuelingDQN, self).__init__()
        # 环境参数
        self.M = env.M
        self.N = env.N
        self.F = env.F
        self.K = env.K
        state_dim = env.N * env.F * env.K + env.M * env.N
        # 输出动作空间维度与原DDQN相同
        # 定义共享前馈网络主干
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # 集中式缓存头（Value和Advantage）
        self.value_cache = nn.Linear(hidden_dim, 1)
        self.adv_cache = nn.Linear(hidden_dim, self.M * self.F * self.K)
        # 每个用户的独立推荐和关联头
        self.value_users = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(self.N)])
        self.adv_assoc_users = nn.ModuleList([nn.Linear(hidden_dim, self.M) for _ in range(self.N)])
        self.adv_rec_users = nn.ModuleList([nn.Linear(hidden_dim, self.M * self.F * self.K) for _ in range(self.N)])
        # 参数初始化（Xavier）
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.value_cache.weight)
        nn.init.xavier_uniform_(self.adv_cache.weight)
        for n in range(self.N):
            nn.init.xavier_uniform_(self.value_users[n].weight)
            nn.init.xavier_uniform_(self.adv_assoc_users[n].weight)
            nn.init.xavier_uniform_(self.adv_rec_users[n].weight)

    def forward(self, x):
        # x: [batch_size, state_dim]
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        batch_size = x.shape[0]
        cache_size = self.M * self.F * self.K
        rec_size = self.M * self.F * self.K
        assoc_size = self.M * self.N
        total_actions = 2 * cache_size + assoc_size
        q_values = torch.zeros((batch_size, total_actions), device=x.device)
        # 1. 缓存部分Q值（集中式）
        value_c = self.value_cache(x)                # [batch, 1]
        adv_c = self.adv_cache(x)                    # [batch, cache_size]
        adv_c_mean = adv_c.mean(dim=1, keepdim=True) # [batch, 1]
        q_cache = value_c + adv_c - adv_c_mean       # Dueling合并: V_c + A_c - mean(A_c)
        q_values[:, 0:cache_size] = q_cache
        # 初始化全局推荐优势累计
        rec_adv_global = torch.zeros((batch_size, rec_size), device=x.device)
        # 2. 针对每个用户的推荐和关联部分Q值
        for n in range(self.N):
            # 每个用户n的Value和Advantage
            val_n = self.value_users[n](x)             # [batch, 1]
            adv_assoc_n = self.adv_assoc_users[n](x)   # [batch, M]
            adv_rec_n = self.adv_rec_users[n](x)       # [batch, M*F*K]
            # 对优势进行均值中心化
            adv_assoc_n_center = adv_assoc_n - adv_assoc_n.mean(dim=1, keepdim=True)  # [batch, M]
            adv_rec_n_center = adv_rec_n - adv_rec_n.mean(dim=1, keepdim=True)        # [batch, M*F*K]
            # 用户n的关联动作Q值: Q_assoc_n = V_n + A_assoc_n_center
            q_assoc_n = val_n + adv_assoc_n_center    # [batch, M], val_n对该用户的动作广播
            # 放入总Q向量对应位置 (关联部分起始索引: 2*cache_size)
            assoc_offset = 2 * cache_size + n * self.M
            q_values[:, assoc_offset: assoc_offset + self.M] = q_assoc_n
            # 累加所有用户的推荐优势
            rec_adv_global += adv_rec_n_center
        # 3. 推荐部分Q值（全局合并）
        # 将所有用户优势求和后再中心化，作为全局推荐动作的优势值
        rec_adv_global_center = rec_adv_global - rec_adv_global.mean(dim=1, keepdim=True)
        q_rec_global = rec_adv_global_center  # 未显式使用额外的Value，对齐优势均值后直接作为Q值
        q_values[:, cache_size: cache_size + rec_size] = q_rec_global
        return q_values

class MultiHeadDDQNAgent(DDQNAgent):
    def __init__(self, env, hidden_dim=128, batch_size=64, lr=1e-3, gamma=0.95, target_update_freq=1000, memory_capacity=10000):
        """
        多头Dueling DDQN智能体，继承DDQNAgent以重用选择和更新逻辑。
        """
        # 不调用super().__init__，以免初始化错误的网络
        self.env = env
        # 状态和动作维度
        self.state_dim = env.N * env.F * env.K + env.M * env.N
        self.action_dim = 2 * env.M * env.F * env.K + env.M * env.N
        # 网络初始化为MultiHeadDuelingDQN
        self.q_network = MultiHeadDuelingDQN(env, hidden_dim)
        self.target_network = MultiHeadDuelingDQN(env, hidden_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        # 经验回放
        self.memory = ReplayBuffer(memory_capacity)
        # 训练参数
        self.batch_size = batch_size
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.update_counter = 0
        # 探索参数
        self.epsilon = 1.0
        self.min_epsilon = 0.1
        self.epsilon_decay = 0.99
        # 设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network.to(self.device)
        self.target_network.to(self.device)
        print(f"MultiHeadDDQNAgent initialized with device: {self.device}")
        if torch.cuda.is_available():
            print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
