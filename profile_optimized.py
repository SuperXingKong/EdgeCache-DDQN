import torch
import numpy as np
import time
import json
from ddqn import DDQNAgent
from advanced_env import AdvancedEnvironment
# 导入优化后的函数
from fully_optimized_update import fully_optimized_update
from select_action_optimized import select_action_optimized

def apply_optimizations(agent):
    """应用高级优化到agent"""
    print("应用最高级别优化...")
    # 替换方法
    agent.__class__.select_action = select_action_optimized
    agent.__class__.update = fully_optimized_update
    print("优化应用完成!")

def profile_step(env, agent, state, step_count=100):
    """分析训练步骤的时间分布"""
    timings = {
        'select_action': 0,
        'environment_step': 0,
        'store_transition': 0,
        'agent_update': 0,
        'total': 0
    }
    
    start_total = time.time()
    
    for _ in range(step_count):
        # 分析动作选择时间
        start = time.time()
        X, Y, Z, action_mask = agent.select_action(state)
        if agent.device.type == 'cuda':
            torch.cuda.synchronize()
        timings['select_action'] += time.time() - start
        
        # 分析环境步骤时间
        start = time.time()
        next_state, reward, done, info = env.step(X, Y, Z)
        timings['environment_step'] += time.time() - start
        
        # 分析存储经验时间
        start = time.time()
        agent.store_transition(state, action_mask, reward, next_state, False)
        timings['store_transition'] += time.time() - start
        
        # 分析网络更新时间
        start = time.time()
        agent.update()
        if agent.device.type == 'cuda':
            torch.cuda.synchronize()
        timings['agent_update'] += time.time() - start
        
        state = next_state
    
    timings['total'] = time.time() - start_total
    
    # 计算平均时间
    for key in timings:
        timings[key] /= step_count
    
    return timings

def profile_update_method(agent, iterations=200):
    """单独分析update方法的性能"""
    # 确保经验回放缓冲区中有足够样本
    if len(agent.memory) < agent.batch_size:
        print("样本不足，无法测试update方法")
        return None
    
    # 预热
    for _ in range(10):
        agent.update()
    
    if agent.device.type == 'cuda':
        torch.cuda.synchronize()
    
    # 测量update方法执行时间
    start_time = time.time()
    for _ in range(iterations):
        agent.update()
    
    if agent.device.type == 'cuda':
        torch.cuda.synchronize()
    
    total_time = time.time() - start_time
    avg_time = total_time / iterations
    return avg_time * 1000  # 转换为毫秒

def main():
    # 创建默认环境和智能体
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
    
    print("=== 创建环境和智能体 ===")
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
    
    print(f"设备: {agent.device}")
    if agent.device.type == 'cuda':
        print(f"GPU型号: {torch.cuda.get_device_name(0)}")
    
    print("\n=== 填充初始经验回放缓冲区 ===")
    state = env.reset()
    for _ in range(500):  # 确保有足够样本
        X, Y, Z, action_mask = agent.select_action(state)
        next_state, reward, done, _ = env.step(X, Y, Z)
        agent.store_transition(state, action_mask, reward, next_state, False)
        state = next_state if not done else env.reset()
    
    # 测试未优化版本
    print("\n=== 未优化版本性能测试 ===")
    
    # 测量普通训练步骤
    unoptimized_timings = profile_step(env, agent, state, step_count=100)
    
    # 测量update方法
    unoptimized_update_time = profile_update_method(agent)
    
    # 应用优化
    print("\n=== 应用最高级优化 ===")
    apply_optimizations(agent)
    
    # 测试优化版本
    print("\n=== 优化后版本性能测试 ===")
    
    # 测量优化后训练步骤
    optimized_timings = profile_step(env, agent, state, step_count=100)
    
    # 测量优化后update方法
    optimized_update_time = profile_update_method(agent)
    
    # 打印结果比较
    print("\n=== 性能对比 ===")
    
    print("\n1. 训练步骤时间分析 (ms)")
    print(f"{'操作':<20} {'优化前':<15} {'优化后':<15} {'速度提升':<10}")
    print("-" * 60)
    
    for key in unoptimized_timings:
        if key != 'total':
            unopt_ms = unoptimized_timings[key] * 1000
            opt_ms = optimized_timings[key] * 1000
            speedup = unopt_ms / opt_ms if opt_ms > 0 else float('inf')
            print(f"{key:<20} {unopt_ms:.3f} ms {opt_ms:.3f} ms {speedup:.2f}x")
    
    # 总时间比较
    unopt_total = unoptimized_timings['total'] * 1000
    opt_total = optimized_timings['total'] * 1000
    total_speedup = unopt_total / opt_total if opt_total > 0 else float('inf')
    print(f"{'total':<20} {unopt_total:.3f} ms {opt_total:.3f} ms {total_speedup:.2f}x")
    
    # update 时间占比比较
    print("\n2. update 时间占比分析")
    unopt_update_pct = (unoptimized_timings['agent_update'] / unoptimized_timings['total']) * 100
    opt_update_pct = (optimized_timings['agent_update'] / optimized_timings['total']) * 100
    
    print(f"优化前 agent_update: {unoptimized_timings['agent_update']*1000:.3f} ms ({unopt_update_pct:.1f}%)")
    print(f"优化后 agent_update: {optimized_timings['agent_update']*1000:.3f} ms ({opt_update_pct:.1f}%)")
    
    # 独立update方法测试
    print("\n3. update方法性能（单独测试）")
    print(f"优化前 update 耗时: {unoptimized_update_time:.3f} ms")
    print(f"优化后 update 耗时: {optimized_update_time:.3f} ms")
    update_speedup = unoptimized_update_time / optimized_update_time if optimized_update_time > 0 else float('inf')
    print(f"update 方法速度提升: {update_speedup:.2f}x")
    
    # 训练速度估算
    print("\n4. 训练速度估算")
    unopt_steps_per_sec = 1.0 / unoptimized_timings['total']
    opt_steps_per_sec = 1.0 / optimized_timings['total']
    
    print(f"优化前: {unopt_steps_per_sec:.2f} steps/s")
    print(f"优化后: {opt_steps_per_sec:.2f} steps/s")
    print(f"训练速度提升: {opt_steps_per_sec/unopt_steps_per_sec:.2f}x")
    
    # 优化后update占比是否合理
    print("\n=== 优化成果评估 ===")
    if opt_update_pct < 30:
        print("优化非常成功: agent_update已不再是训练瓶颈")
    elif opt_update_pct < 50:
        print("优化成功: agent_update占比已大幅降低")
    else:
        print("优化有效但仍有提升空间: agent_update仍占较大比例")
        
    # 环境步骤占比是否成为新瓶颈
    env_pct = (optimized_timings['environment_step'] / optimized_timings['total']) * 100
    if env_pct > 50:
        print(f"环境模拟现在是主要瓶颈 ({env_pct:.1f}%)")
        print("建议: 考虑优化环境模拟代码或使用向量化环境")

if __name__ == "__main__":
    main() 