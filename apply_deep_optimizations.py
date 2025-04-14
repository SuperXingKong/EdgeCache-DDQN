import torch
import numpy as np
import time
import argparse
from ddqn import DDQNAgent
from advanced_env import AdvancedEnvironment
from select_action_optimized import select_action_optimized
from fully_optimized_update import fully_optimized_update, batched_topk_selection, batched_max_selection

def apply_optimizations():
    """应用最高级优化到DDQNAgent类"""
    print("正在将最高级别优化应用到DDQNAgent类...")
    # 替换select_action方法
    DDQNAgent.select_action = select_action_optimized
    # 替换update方法
    DDQNAgent.update = fully_optimized_update
    
    print("优化已应用:")
    print("- select_action已替换为GPU优化版本")
    print("- update已替换为完全向量化+JIT编译版本")

def benchmark_single_update(agent, num_iterations=100):
    """基准测试单个update调用的性能"""
    # 确保内存中有足够样本
    if len(agent.memory) < agent.batch_size:
        state = np.random.rand(agent.state_dim).astype(np.float32)
        action = np.random.rand(agent.action_dim).astype(np.float32)
        reward = np.random.rand(1).astype(np.float32)[0]
        next_state = np.random.rand(agent.state_dim).astype(np.float32)
        done = False
        for _ in range(agent.batch_size * 2):
            agent.memory.push(state, action, reward, next_state, done)
    
    # 预热 - 避免首次运行的计时偏差
    for _ in range(10):
        agent.update()
    
    if agent.device.type == 'cuda':
        torch.cuda.synchronize()
    
    # 性能测试
    start_time = time.time()
    for _ in range(num_iterations):
        agent.update()
    
    if agent.device.type == 'cuda':
        torch.cuda.synchronize()
    
    end_time = time.time()
    avg_time = (end_time - start_time) / num_iterations
    return avg_time * 1000  # 转换为毫秒

def run_training_benchmark(episodes=5, steps_per_episode=50):
    """运行完整训练过程的基准测试"""
    # 默认配置
    config = {
        "M": 2,              # 基站数
        "N": 4,              # 用户数
        "F": 6,              # 视频数
        "K": 2,              # 每视频层数
        "C_cache": 4,        # 缓存容量(层数)
        "C_rec": 3,          # 推荐容量
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
    
    print("\n创建环境和智能体...")
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
    
    # 测量单个update调用性能
    print("\n测试agent.update()性能...")
    update_time_ms = benchmark_single_update(agent, num_iterations=200)
    print(f"单次update平均耗时: {update_time_ms:.3f} ms")
    
    # 收集初始经验
    print("\n收集初始经验样本...")
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
    env_step_times = []
    total_start_time = time.time()
    
    for ep in range(episodes):
        episode_start = time.time()
        state = env.reset()
        episode_reward = 0
        
        for t in range(steps_per_episode):
            # 测量select_action时间
            sa_start = time.time()
            X, Y, Z, action_mask = agent.select_action(state)
            if agent.device.type == 'cuda':
                torch.cuda.synchronize()
            select_action_times.append(time.time() - sa_start)
            
            # 测量环境步骤时间
            env_start = time.time()
            next_state, reward, done, info = env.step(X, Y, Z)
            env_step_times.append(time.time() - env_start)
            
            episode_reward += reward
            agent.store_transition(state, action_mask, reward, next_state, False)
            
            # 测量update时间
            update_start = time.time()
            agent.update()
            if agent.device.type == 'cuda':
                torch.cuda.synchronize()
            update_times.append(time.time() - update_start)
            
            state = next_state
        
        episode_time = time.time() - episode_start
        print(f"回合 {ep+1}/{episodes} 完成: 奖励={episode_reward:.2f}, 耗时={episode_time:.2f}秒")
    
    total_time = time.time() - total_start_time
    steps = episodes * steps_per_episode
    
    # 打印详细结果
    avg_update = np.mean(update_times) * 1000
    avg_select = np.mean(select_action_times) * 1000
    avg_env = np.mean(env_step_times) * 1000
    
    print(f"\n=== 训练性能报告 ===")
    print(f"总训练时间: {total_time:.2f} 秒")
    print(f"每步平均时间: {total_time*1000/steps:.2f} ms")
    print(f"每秒步数: {steps/total_time:.2f}")
    
    print(f"\n详细时间分布:")
    print(f"平均update时间: {avg_update:.3f} ms ({avg_update/(avg_update+avg_select+avg_env)*100:.1f}%)")
    print(f"平均select_action时间: {avg_select:.3f} ms ({avg_select/(avg_update+avg_select+avg_env)*100:.1f}%)")
    print(f"平均environment_step时间: {avg_env:.3f} ms ({avg_env/(avg_update+avg_select+avg_env)*100:.1f}%)")
    
    # GPU使用情况
    print("\nGPU情况:")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"设备: {torch.cuda.get_device_name(0)}")
        print(f"分配的GPU内存: {torch.cuda.memory_allocated(0)/1024**2:.1f} MB")
        print(f"缓存的GPU内存: {torch.cuda.memory_reserved(0)/1024**2:.1f} MB")
        
        # 测试JIT加速的函数
        tensor_test = torch.rand(64, 12, device=agent.device)
        
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(1000):
            _ = batched_topk_selection(tensor_test, 3)
        torch.cuda.synchronize()
        jit_time = (time.time() - start) * 1000 / 1000
        
        print(f"\nJIT编译函数性能:")
        print(f"batched_topk_selection (1000次平均): {jit_time:.4f} ms")

def main():
    parser = argparse.ArgumentParser(description="应用深度优化并测试DDQN性能")
    parser.add_argument("--episodes", type=int, default=5, help="测试回合数")
    parser.add_argument("--steps", type=int, default=50, help="每回合步数")
    args = parser.parse_args()
    
    print("=== DDQN深度优化性能测试 ===")
    print("使用了: ")
    print("1. 全向量化操作替代循环")
    print("2. TorchScript JIT编译加速关键函数")
    print("3. 批处理GPU操作")
    print("4. 最小化CPU-GPU数据传输")
    
    # 应用优化
    apply_optimizations()
    
    # 运行基准测试
    run_training_benchmark(episodes=args.episodes, steps_per_episode=args.steps)
    
if __name__ == "__main__":
    main() 