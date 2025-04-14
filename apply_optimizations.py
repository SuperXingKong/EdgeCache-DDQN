import torch
import numpy as np
import time
import argparse
from ddqn import DDQNAgent
from advanced_env import AdvancedEnvironment
from select_action_optimized import select_action_optimized

def apply_optimizations():
    """应用优化函数到DDQNAgent类"""
    print("应用优化到DDQNAgent类...")
    # 替换select_action方法
    DDQNAgent.select_action = select_action_optimized
    
    print("优化已应用: select_action方法已被优化版本替换")

def run_training_benchmark(episodes=5, steps_per_episode=50):
    """运行训练基准测试"""
    # 默认配置
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
    
    print("创建环境和智能体...")
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
    
    # 收集初始经验
    print("收集初始经验样本...")
    state = env.reset()
    for _ in range(min(500, config["memory_capacity"])):
        X, Y, Z, action_mask = agent.select_action(state)
        next_state, reward, done, _ = env.step(X, Y, Z)
        agent.store_transition(state, action_mask, reward, next_state, False)
        state = next_state if not done else env.reset()
    
    # 运行训练基准测试
    print(f"\n进行{episodes}个回合的训练速度测试...")
    
    update_times = []
    select_action_times = []
    
    start_time = time.time()
    for ep in range(episodes):
        state = env.reset()
        for t in range(steps_per_episode):
            # 测量select_action时间
            sa_start = time.time()
            X, Y, Z, action_mask = agent.select_action(state)
            select_action_times.append(time.time() - sa_start)
            
            # 执行环境步骤
            next_state, reward, done, info = env.step(X, Y, Z)
            agent.store_transition(state, action_mask, reward, next_state, False)
            
            # 测量update时间
            update_start = time.time()
            agent.update()
            update_times.append(time.time() - update_start)
            
            state = next_state
    
    total_time = time.time() - start_time
    steps = episodes * steps_per_episode
    
    # 打印结果
    print(f"总训练时间: {total_time:.2f} 秒")
    print(f"每步平均时间: {total_time*1000/steps:.2f} ms")
    print(f"每秒步数: {steps/total_time:.2f}")
    
    # 打印详细时间统计
    avg_update = np.mean(update_times) * 1000
    avg_select = np.mean(select_action_times) * 1000
    print(f"\n详细时间统计:")
    print(f"平均update时间: {avg_update:.2f} ms")
    print(f"平均select_action时间: {avg_select:.2f} ms")
    
    # GPU使用情况
    print("\nGPU情况:")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"设备: {torch.cuda.get_device_name(0)}")
        # 显示当前内存使用情况
        print(f"分配的GPU内存: {torch.cuda.memory_allocated(0)/1024**2:.1f} MB")
        print(f"缓存的GPU内存: {torch.cuda.memory_reserved(0)/1024**2:.1f} MB")

def main():
    parser = argparse.ArgumentParser(description="应用优化并运行训练基准测试")
    parser.add_argument("--episodes", type=int, default=5, help="测试回合数")
    parser.add_argument("--steps", type=int, default=50, help="每回合步数")
    args = parser.parse_args()
    
    print("=== DDQN优化应用程序 ===")
    # 应用优化
    apply_optimizations()
    
    # 运行基准测试
    run_training_benchmark(episodes=args.episodes, steps_per_episode=args.steps)
    
if __name__ == "__main__":
    main() 