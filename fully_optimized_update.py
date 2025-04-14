import torch
import torch.nn as nn
import torch.jit as jit

# 使用JIT编译加速批处理选择topk
@torch.jit.script
def batched_topk_selection(values, k: int):
    """高度优化的批量topk选择，无循环"""
    batch_size, dim = values.shape
    if k >= dim:
        # 如果k>=维度，全部选择
        return torch.ones_like(values)
    
    # 获取每个样本中前k大的值的索引
    _, indices = torch.topk(values, k=k, dim=1)
    
    # 创建一个全零张量，然后使用scatter填充1
    mask = torch.zeros_like(values, dtype=torch.float)
    batch_indices = torch.arange(batch_size, device=values.device).unsqueeze(1).expand(-1, k)
    mask[batch_indices, indices] = 1.0
    
    return mask

# 使用JIT编译加速批处理最大值选择
@torch.jit.script
def batched_max_selection(values):
    """高度优化的批量最大值选择，无循环"""
    batch_size, dim = values.shape
    
    # 获取每个样本中最大值的索引
    _, indices = torch.max(values, dim=1)
    
    # 创建一个全零张量，然后使用scatter填充1
    mask = torch.zeros_like(values, dtype=torch.float)
    mask[torch.arange(batch_size, device=values.device), indices] = 1.0
    
    return mask

# 完全优化的更新方法
def fully_optimized_update(self):
    """完全优化的DDQN更新方法，使用向量化操作和JIT加速"""
    if len(self.memory) < self.batch_size:
        return  # 样本不足
    
    # 提取并转换为张量
    states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
    
    # 单次批量转换，减少调用次数
    tensors = [torch.from_numpy(x).float().to(self.device) for x in [states, actions, rewards, next_states, dones]]
    state_t, action_mask_t, reward_t, next_state_t, done_t = tensors
    
    # 计算当前Q值 - 使用预先计算的掩码直接计算
    q_pred_all = self.q_network(state_t)
    q_pred = torch.sum(q_pred_all * action_mask_t, dim=1)
    
    # 计算目标Q值 - 保持在GPU上
    with torch.no_grad():
        # 获取在线网络和目标网络的下一个Q值
        q_eval_next = self.q_network(next_state_t)
        q_target_next = self.target_network(next_state_t)
        
        # 获取环境参数
        M, N, F, K = self.env.M, self.env.N, self.env.F, self.env.K
        Cc, Cr = self.env.C_cache, self.env.C_rec
        
        # 初始化目标值为奖励
        q_targets = reward_t.clone()
        
        # 处理非终止状态
        non_terminal_mask = ~done_t.bool()  # 简化布尔掩码计算
        non_term_count = non_terminal_mask.sum().item()
        
        if non_term_count > 0:
            # 提取非终止状态的相关张量
            non_term_eval = q_eval_next[non_terminal_mask]
            non_term_target = q_target_next[non_terminal_mask]
            non_term_rewards = reward_t[non_terminal_mask]
            
            # 计算动作空间的各部分大小
            cache_size = M * F * K
            rec_size = M * F * K
            action_dim = cache_size + rec_size + M * N
            
            # 预先分配最佳动作张量，避免连接操作
            best_actions = torch.zeros(non_term_count, action_dim, device=self.device)
            
            # 1. 处理缓存决策 - 全向量化处理
            for m in range(M):
                start_idx = m * F * K
                end_idx = (m + 1) * F * K
                
                # 提取相关段
                cache_seg = non_term_eval[:, start_idx:end_idx]
                
                # 使用JIT编译的函数进行topk选择
                cache_mask = batched_topk_selection(cache_seg, Cc)
                
                # 将结果放入最佳动作张量
                best_actions[:, start_idx:end_idx] = cache_mask
            
            # 2. 处理推荐决策 - 全向量化处理
            for m in range(M):
                start_idx = cache_size + m * F * K
                end_idx = cache_size + (m + 1) * F * K
                orig_idx = start_idx - cache_size
                orig_end = end_idx - cache_size
                
                # 提取相关段
                rec_seg = non_term_eval[:, start_idx:end_idx]
                
                # 使用JIT编译的函数进行topk选择
                rec_mask = batched_topk_selection(rec_seg, Cr)
                
                # 将结果放入最佳动作张量
                best_actions[:, start_idx:end_idx] = rec_mask
            
            # 3. 处理用户关联决策 - 全向量化处理
            assoc_offset = cache_size + rec_size
            for n in range(N):
                start_idx = assoc_offset + n * M
                end_idx = assoc_offset + (n + 1) * M
                
                # 提取相关段
                assoc_seg = non_term_eval[:, start_idx:end_idx]
                
                # 使用JIT编译的函数进行最大值选择
                assoc_mask = batched_max_selection(assoc_seg)
                
                # 将结果放入最佳动作张量
                best_actions[:, start_idx:end_idx] = assoc_mask
            
            # 计算目标Q值 - 从目标网络计算最佳动作的值
            best_action_values = torch.sum(non_term_target * best_actions, dim=1)
            
            # 更新非终止状态的Q目标
            q_targets[non_terminal_mask] = non_term_rewards + self.gamma * best_action_values
    
    # 计算损失并更新网络
    loss = nn.MSELoss()(q_pred, q_targets)
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()
    
    # 定期更新目标网络
    self.update_counter += 1
    if self.update_counter % self.target_update_freq == 0:
        self.target_network.load_state_dict(self.q_network.state_dict())

# 如何使用:
# from fully_optimized_update import fully_optimized_update
# DDQNAgent.update = fully_optimized_update 