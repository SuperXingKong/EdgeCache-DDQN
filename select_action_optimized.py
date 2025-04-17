import torch
import numpy as np

def select_action_optimized(self, state):
    """
    优化版select_action方法，减少CPU-GPU数据传输
    仍然使用epsilon-greedy策略，但保持更多计算在GPU上
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
        # Greedy action from Q-network - stay on GPU as much as possible
        state_t = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_t)  # 保持在GPU上
        
        M, N, F, K = self.env.M, self.env.N, self.env.F, self.env.K
        Cc, Cr = self.env.C_cache, self.env.C_rec
        
        # 创建空的动作矩阵
        X = torch.zeros((M, F, K), device=self.device)
        Y = torch.zeros((M, F, K), device=self.device)
        Z = torch.zeros((M, N), device=self.device)
        
        # 初始化缓存策略 X
        cache_size = M * F * K
        for m in range(M):
            start_idx = m * F * K
            end_idx = (m + 1) * F * K
            segment = q_values[0, start_idx:end_idx]  # 批次大小为1
            
            if Cc >= F * K:
                # 如果可以缓存所有内容，则全选
                X[m, :, :] = 1
            else:
                # 选择前Cc个最高值
                values, indices = torch.topk(segment, k=Cc)
                for idx in indices:
                    f = (idx // K).item()
                    k = (idx % K).item()
                    X[m, f, k] = 1
        
        # 初始化推荐策略 Y
        for m in range(M):
            start_idx = cache_size + m * F * K
            end_idx = cache_size + (m + 1) * F * K
            segment = q_values[0, start_idx:end_idx]  # 批次大小为1
            
            if Cr >= F * K:
                # 如果可以推荐所有内容，则全选
                Y[m, :, :] = 1
            else:
                # 选择前Cr个最高值
                values, indices = torch.topk(segment, k=Cr)
                for idx in indices:
                    f = (idx // K).item()
                    k = (idx % K).item()
                    Y[m, f, k] = 1
        
        # 用户关联策略 Z
        assoc_offset = 2 * M * F * K
        for n in range(N):
            start_idx = assoc_offset + n * M
            end_idx = assoc_offset + (n + 1) * M
            segment = q_values[0, start_idx:end_idx]
            
            # 为每个用户选择最好的基站
            best_bs = torch.argmax(segment).item()
            Z[best_bs, n] = 1
        
        # 将张量转换为numpy数组以与环境交互
        X_np = X.cpu().numpy()
        Y_np = Y.cpu().numpy()
        Z_np = Z.cpu().numpy()
        
        # 创建action_mask
        action_mask = torch.cat([X.flatten(), Y.flatten(), Z.flatten()]).cpu().numpy()
        
        return X_np, Y_np, Z_np, action_mask

# 示例如何替换DDQNAgent中的方法:
# 
# # 将此导入到代码中
# from select_action_optimized import select_action_optimized
# 
# # 然后替换DDQNAgent类中的方法
# DDQNAgent.select_action = select_action_optimized 