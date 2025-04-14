import torch
import numpy as np
import argparse
import os
import json
from datetime import datetime
from ddqn import DDQNAgent
from advanced_env import AdvancedEnvironment

def random_caching(env):
    """
    随机生成缓存策略 X: shape (M,F,K)
    每个基站 m 只能缓存 C_cache 个 (f,k)，其余置0
    """
    M, F, K, Cc = env.M, env.F, env.K, env.C_cache
    X = np.zeros((M, F, K), dtype=int)
    all_pairs = [(f, k) for f in range(F) for k in range(K)]
    for m in range(M):
        if len(all_pairs) <= Cc:
            chosen_pairs = all_pairs[:]
        else:
            chosen_idx = np.random.choice(len(all_pairs), size=Cc, replace=False)
            chosen_pairs = [all_pairs[i] for i in chosen_idx]
        for (f, kk) in chosen_pairs:
            X[m, f, kk] = 1
    return X

def random_recommendation(env):
    """
    随机生成推荐策略 Y: shape (M,F,K)
    每个基站 m 只能推荐 C_rec 个 (f,k)，其余置0
    """
    M, F, K, Cr = env.M, env.F, env.K, env.C_rec
    Y = np.zeros((M, F, K), dtype=int)
    all_pairs = [(f, k) for f in range(F) for k in range(K)]
    for m in range(M):
        if len(all_pairs) <= Cr:
            chosen_pairs = all_pairs[:]
        else:
            chosen_idx = np.random.choice(len(all_pairs), size=Cr, replace=False)
            chosen_pairs = [all_pairs[i] for i in chosen_idx]
        for (f, kk) in chosen_pairs:
            Y[m, f, kk] = 1
    return Y

def random_association(env):
    """
    随机生成用户关联策略 Z: shape (M,N)
    每个用户只能关联到一个基站
    """
    M, N = env.M, env.N
    Z = np.zeros((M, N), dtype=int)
    for n in range(N):
        m_choice = np.random.randint(0, M)
        Z[m_choice, n] = 1
    return Z

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.json", help="Path to configuration file")
    parser.add_argument("--mode", type=str, default="full", 
                        choices=["full","random_all","random_ua","random_rec","random_ca"],
                        help="Choose the control mode for partial or full random actions")
    args = parser.parse_args()

    # 读取配置文件
    try:
        with open(args.config, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"配置文件 {args.config} 不存在，将创建默认配置文件")
        config = {
            "M": 2,               # 基站数
            "N": 4,               # 用户数
            "F": 6,               # 视频数
            "K": 2,               # 每视频层数
            "C_cache": 4,         # 缓存容量(层数)
            "C_rec": 3,           # 推荐容量
            "episodes": 3000,     # 训练回合数
            "steps_per_episode": 50,
            "gamma": 0.95,
            "lr": 1e-3,
            "batch_size": 64,
            "memory_capacity": 10000,
            "target_update_freq": 100,
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
        # 保存默认配置文件
        os.makedirs(os.path.dirname(args.config) or '.', exist_ok=True)
        with open(args.config, 'w') as f:
            json.dump(config, f, indent=4)
        print(f"已创建默认配置文件: {args.config}")

    # 获取命令行指定的mode
    mode = args.mode
    print(f"使用命令行指定的mode: {mode}")

    # Configuration
    M = config["M"]           # 基站数
    N = config["N"]           # 用户数
    F = config["F"]           # 视频数
    K = config["K"]           # 每视频层数
    C_cache = config["C_cache"]  # 缓存容量(层数)
    C_rec = config["C_rec"]   # 推荐容量
    episodes = config["episodes"]
    steps_per_episode = config["steps_per_episode"]
    gamma = config["gamma"]
    lr = config["lr"]
    batch_size = config["batch_size"]
    memory_capacity = config["memory_capacity"]
    target_update_freq = config["target_update_freq"]
    seed = config["seed"]

    # 获取环境参数
    env_params = config["env_params"]

    # ==================
    # 创建环境 & DDQN智能体
    # ==================
    env = AdvancedEnvironment(
        M=M, N=N, F=F, K=K, 
        C_cache=C_cache, C_rec=C_rec, 
        D_max=env_params["D_max"], phi=env_params["phi"],
        B=env_params["B"], BH_B=env_params["BH_B"],
        P_m=env_params["P_m"], w_BH=env_params["w_BH"],
        seed=seed
    )
    agent = DDQNAgent(env, hidden_dim=128, batch_size=batch_size, lr=lr, gamma=gamma,
                      target_update_freq=target_update_freq, memory_capacity=memory_capacity)

    # 日志记录
    reward_history = []
    energy_history = []
    D_history = []
    hitrate_history = []
    eff_history = []  # 新增: 能量效率

    print(f"=== Training Mode: {mode} ===")
    print(f"Parameters: M={M}, N={N}, F={F}, K={K}, C_cache={C_cache}, C_rec={C_rec}, lr={lr}")

    # ==================
    # 训练循环
    # ==================
    for ep in range(1, episodes+1):
        state = env.reset()
        ep_reward = ep_energy = ep_D = ep_hitrate = ep_eff = 0.0

        for t in range(1, steps_per_episode+1):
            # 1) 根据 mode 选择动作
            if mode == "full":
                # 全部由DDQN做决策
                X, Y, Z, action_mask = agent.select_action(state)
            else:
                # 先用DDQN生成一份完整动作 (X_ddqn, Y_ddqn, Z_ddqn)
                X_ddqn, Y_ddqn, Z_ddqn, action_mask_ddqn = agent.select_action(state)

                # 部分改为随机策略
                if mode == "random_all":
                    X = random_caching(env)
                    Y = random_recommendation(env)
                    Z = random_association(env)
                elif mode == "random_ua":
                    X = X_ddqn
                    Y = Y_ddqn
                    Z = random_association(env)
                elif mode == "random_rec":
                    X = X_ddqn
                    Y = random_recommendation(env)
                    Z = Z_ddqn
                elif mode == "random_ca":
                    X = random_caching(env)
                    Y = Y_ddqn
                    Z = Z_ddqn

                # 将最终 (X,Y,Z) 拼接成 action_mask 用于存储经验
                mask_len = 2*env.M*env.F*env.K + env.M*env.N
                new_mask = np.concatenate([X.flatten(), Y.flatten(), Z.flatten()]).astype(np.float32)
                # 这里的 new_mask 代替 action_mask => 用于 replay buffer
                action_mask = new_mask

            # 2) 执行动作
            next_state, reward, done, info = env.step(X, Y, Z)
            done_flag = (t == steps_per_episode)  # 此处简单episode长度

            # 3) 存储并更新
            agent.store_transition(state, action_mask, reward, next_state, done_flag)
            agent.update()

            # 记录指标
            ep_reward += reward
            ep_energy += info["E_total"]
            ep_D += info["D"]
            ep_hitrate += info["cache_hit_rate"]
            ep_eff += info.get("energy_efficiency", 0.0)

            state = next_state
            if done_flag:
                break

        # epsilon 衰减
        if agent.epsilon > agent.min_epsilon:
            agent.epsilon *= agent.epsilon_decay
            if agent.epsilon < agent.min_epsilon:
                agent.epsilon = agent.min_epsilon

        # 计算当回合平均值
        avg_reward = ep_reward / steps_per_episode
        avg_energy = ep_energy / steps_per_episode
        avg_D = ep_D / steps_per_episode
        avg_hitrate = ep_hitrate / steps_per_episode
        avg_eff = ep_eff / steps_per_episode

        reward_history.append(avg_reward)
        energy_history.append(avg_energy)
        D_history.append(avg_D)
        hitrate_history.append(avg_hitrate)
        eff_history.append(avg_eff)

        # 日志输出
        if ep % 10 == 0 or ep == 1:
            print(f"Episode {ep}/{episodes}: "
                  f"avg_reward={avg_reward:.3f}, avg_energy={avg_energy:.3f}, "
                  f"avg_D={avg_D:.3f}, hit_rate={avg_hitrate:.3f}, "
                  f"energy_eff={avg_eff:.3f}, epsilon={agent.epsilon:.2f}")

    # ==================
    # 训练完成，创建保存目录并保存结果
    # ==================
    # 创建保存目录
    save_dir = "models"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 生成模型文件名，包含所有关键参数
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_name = os.path.splitext(os.path.basename(args.config))[0]
    model_name = f"ddqn_{config_name}_M{M}_N{N}_F{F}_K{K}_Cc{C_cache}_Cr{C_rec}_lr{lr}_mode{mode}_{timestamp}"
    
    # 保存模型和训练指标
    model_path = os.path.join(save_dir, f"{model_name}.pth")
    metrics_path = os.path.join(save_dir, f"{model_name}_metrics.npz")
    
    torch.save({
        'model_state_dict': agent.q_network.state_dict(),
        'target_state_dict': agent.target_network.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'epsilon': agent.epsilon,
        'config': config,
        'config_file': args.config,
        'mode': mode  # 添加命令行指定的mode
    }, model_path)
    
    np.savez(metrics_path,
             reward=np.array(reward_history, dtype=np.float32),
             energy=np.array(energy_history, dtype=np.float32),
             D=np.array(D_history, dtype=np.float32),
             hit_rate=np.array(hitrate_history, dtype=np.float32),
             energy_eff=np.array(eff_history, dtype=np.float32))
    
    print(f"Training completed under mode='{mode}'.")
    print(f"Model saved to {model_path}")
    print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
