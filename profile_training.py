import torch
import numpy as np
import time
import json
import os
import sys
from ddqn import DDQNAgent
from advanced_env import AdvancedEnvironment

def profile_step(env, agent, state, step_count=100):
    """Profile a single step of the training process"""
    timings = {
        'select_action': 0,
        'environment_step': 0,
        'store_transition': 0,
        'agent_update': 0,
        'total': 0
    }
    
    start_total = time.time()
    
    for _ in range(step_count):
        # Profile action selection
        start = time.time()
        X, Y, Z, action_mask = agent.select_action(state)
        timings['select_action'] += time.time() - start
        
        # Profile environment step
        start = time.time()
        next_state, reward, done, info = env.step(X, Y, Z)
        timings['environment_step'] += time.time() - start
        
        # Profile storing transition
        start = time.time()
        agent.store_transition(state, action_mask, reward, next_state, False)
        timings['store_transition'] += time.time() - start
        
        # Profile agent update
        start = time.time()
        agent.update()
        timings['agent_update'] += time.time() - start
        
        state = next_state
    
    timings['total'] = time.time() - start_total
    
    # Calculate average times
    for key in timings:
        timings[key] /= step_count
    
    return timings

def profile_network_operations(agent, batch_size=64, repetitions=100):
    """Profile network forward and backward passes"""
    if len(agent.memory) < batch_size:
        print("Not enough samples in memory for profiling")
        return None
    
    # Sample batch
    states, actions, rewards, next_states, dones = agent.memory.sample(batch_size)
    state_t = torch.from_numpy(states).float().to(agent.device)
    action_mask_t = torch.from_numpy(actions).float().to(agent.device)
    next_state_t = torch.from_numpy(next_states).float().to(agent.device)
    reward_t = torch.from_numpy(rewards).float().to(agent.device)
    done_t = torch.from_numpy(dones).float().to(agent.device)
    
    timings = {}
    
    # Warmup
    for _ in range(10):
        q_pred_all = agent.q_network(state_t)
        q_pred = (q_pred_all * action_mask_t).sum(dim=1)
    
    # Profile only update method
    update_time = 0
    for _ in range(repetitions):
        start = time.time()
        # Re-add these samples to memory to ensure we don't run out
        agent.memory.push(states[0], actions[0], rewards[0], next_states[0], dones[0])
        agent.update()
        update_time += time.time() - start
    
    timings['update_method'] = update_time / repetitions
    
    return timings

def main():
    # Load default configuration
    config = {
        "M": 2,               # 基站数
        "N": 4,               # 用户数
        "F": 6,               # 视频数
        "K": 2,               # 每视频层数
        "C_cache": 4,         # 缓存容量(层数)
        "C_rec": 3,           # 推荐容量
        "batch_size": 64,
        "memory_capacity": 10000,
        "seed": 42,
        "env_params": {
            "D_max": 0.2,
            "phi": 100,
            "B": 10.0, 
            "BH_B": 5.0,
            "P_m": 1.0, 
            "w_BH": 0.5
        }
    }
    
    print("=== DDQN优化性能测试 ===")
    print("设置环境和智能体...")
    
    # Create environment and agent
    env = AdvancedEnvironment(
        M=config["M"], N=config["N"], F=config["F"], K=config["K"], 
        C_cache=config["C_cache"], C_rec=config["C_rec"], 
        D_max=config["env_params"]["D_max"], phi=config["env_params"]["phi"],
        B=config["env_params"]["B"], BH_B=config["env_params"]["BH_B"],
        P_m=config["env_params"]["P_m"], w_BH=config["env_params"]["w_BH"],
        seed=config["seed"]
    )
    
    agent = DDQNAgent(
        env, 
        hidden_dim=128, 
        batch_size=config["batch_size"], 
        lr=1e-3, 
        gamma=0.95,
        target_update_freq=100, 
        memory_capacity=config["memory_capacity"]
    )
    
    # Fill memory with some random transitions
    print("收集初始经验样本...")
    state = env.reset()
    for _ in range(min(1000, config["memory_capacity"])):
        X, Y, Z, action_mask = agent.select_action(state)
        next_state, reward, done, _ = env.step(X, Y, Z)
        agent.store_transition(state, action_mask, reward, next_state, False)
        state = next_state if not done else env.reset()
    
    # Profile training steps
    print("\n进行训练步骤分析...")
    step_timings = profile_step(env, agent, state, step_count=100)
    
    # Profile network operations
    print("进行网络操作分析...")
    network_timings = profile_network_operations(agent, repetitions=200)
    
    # Print results
    print("\n=== 训练步骤分析 ===")
    for key, value in step_timings.items():
        print(f"{key}: {value*1000:.2f} ms ({value/step_timings['total']*100:.1f}%)")
    
    if network_timings:
        print("\n=== 网络操作分析 ===")
        for key, value in network_timings.items():
            print(f"{key}: {value*1000:.2f} ms")
    
    # System information
    print("\n=== 系统信息 ===")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    print("\n模型复杂度:")
    total_params = sum(p.numel() for p in agent.q_network.parameters())
    trainable_params = sum(p.numel() for p in agent.q_network.parameters() if p.requires_grad)
    print(f"参数总量: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    
    # Environment complexity
    state_dim = env.N * env.F * env.K + env.M * env.N
    action_dim = 2 * env.M * env.F * env.K + env.M * env.N
    print(f"\n环境复杂度:")
    print(f"状态维度: {state_dim}")
    print(f"动作维度: {action_dim}")
    
    # Specific benchmarking section for update method
    print("\n=== Update方法性能测试 ===")
    print("进行100次更新操作的基准测试...")
    
    # Warm up
    for _ in range(10):
        agent.update()
    
    # Test update 100 times
    total_time = 0
    for _ in range(100):
        start = time.time()
        agent.update()
        total_time += time.time() - start
    
    avg_time = total_time / 100
    print(f"优化后平均更新时间: {avg_time*1000:.2f} ms")
    
    # Test full episodes
    num_episodes = 5
    steps_per_episode = 50
    print(f"\n进行{num_episodes}个回合的训练速度测试...")
    
    start_time = time.time()
    for _ in range(num_episodes):
        state = env.reset()
        for t in range(steps_per_episode):
            X, Y, Z, action_mask = agent.select_action(state)
            next_state, reward, done, info = env.step(X, Y, Z)
            agent.store_transition(state, action_mask, reward, next_state, False)
            agent.update()
            state = next_state
    
    total_time = time.time() - start_time
    steps = num_episodes * steps_per_episode
    print(f"总训练时间: {total_time:.2f} 秒")
    print(f"每步平均时间: {total_time*1000/steps:.2f} ms")
    print(f"每秒步数: {steps/total_time:.2f}")
        
if __name__ == "__main__":
    main() 