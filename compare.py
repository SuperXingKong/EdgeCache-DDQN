import argparse
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
from datetime import datetime
from advanced_env import AdvancedEnvironment
from ddqn import DDQNAgent
from mhddqn import MultiHeadDDQNAgent
import time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.json", help="配置文件路径")
    parser.add_argument("--runs", type=int, default=5, help="独立重复实验次数")
    parser.add_argument("--save_dir", type=str, default="results", help="保存模型和数据的目录")
    args = parser.parse_args()
    
    # 创建保存目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(args.save_dir, timestamp)
    models_dir = os.path.join(save_dir, "models")
    data_dir = os.path.join(save_dir, "data")
    plots_dir = os.path.join(save_dir, "plots")
    
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    # 读取配置
    try:
        with open(args.config, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"配置文件 {args.config} 不存在，将使用默认参数")
        config = {
            "M": 2, "N": 4, "F": 6, "K": 2,
            "C_cache": 4, "C_rec": 3,
            "episodes": 1000, "steps_per_episode": 50,
            "gamma": 0.95, "lr": 1e-3,
            "batch_size": 64, "memory_capacity": 10000,
            "target_update_freq": 1000, "seed": 123,
            "env_params": {
                "D_max": 0.2, "phi": 100,
                "B": 10.0, "BH_B": 5.0,
                "P_m": 1.0, "w_BH": 0.5
            }
        }
    
    # 保存配置
    with open(os.path.join(save_dir, "config.json"), 'w') as f:
        json.dump(config, f, indent=4)
    
    # 提取主要配置
    M = config["M"]; N = config["N"]; F = config["F"]; K = config["K"]
    C_cache = config["C_cache"]; C_rec = config["C_rec"]
    episodes = config["episodes"]; steps_per_episode = config["steps_per_episode"]
    gamma = config["gamma"]; lr = config["lr"]
    batch_size = config["batch_size"]; memory_capacity = config["memory_capacity"]
    target_update_freq = config["target_update_freq"]
    base_seed = config.get("seed", 0)
    env_params = config.get("env_params", {})
    
    # 生成文件名参数字符串，用于保存模型和指标
    config_str = f"M{M}_N{N}_F{F}_K{K}_Cc{C_cache}_Cr{C_rec}_lr{lr}"
    
    print(f"开始训练: 总共 {args.runs} 次独立实验, 每次 {episodes} 个回合, 每个回合 {steps_per_episode} 步")
    print(f"结果将保存到: {save_dir}")
    
    # 准备存储多个实验的指标
    energy_histories_ddqn = []
    hitrate_histories_ddqn = []
    eff_histories_ddqn = []
    reward_histories_ddqn = []
    D_histories_ddqn = []
    
    energy_histories_mh = []
    hitrate_histories_mh = []
    eff_histories_mh = []
    reward_histories_mh = []
    D_histories_mh = []
    
    start_time = time.time()
    # 重复实验
    for run in range(1, args.runs + 1):
        run_start_time = time.time()
        print(f"\n=== 开始运行 {run}/{args.runs} ===")
        
        run_seed = base_seed + run
        # 分别为DDQN和MH-DDQN创建带相同随机种子的独立环境
        env_ddqn = AdvancedEnvironment(
            M=M, N=N, F=F, K=K, C_cache=C_cache, C_rec=C_rec,
            D_max=env_params.get("D_max", 0.2), phi=env_params.get("phi", 100),
            B=env_params.get("B", 10.0), BH_B=env_params.get("BH_B", 5.0),
            P_m=env_params.get("P_m", 1.0), w_BH=env_params.get("w_BH", 0.5),
            seed=run_seed
        )
        env_mh = AdvancedEnvironment(
            M=M, N=N, F=F, K=K, C_cache=C_cache, C_rec=C_rec,
            D_max=env_params.get("D_max", 0.2), phi=env_params.get("phi", 100),
            B=env_params.get("B", 10.0), BH_B=env_params.get("BH_B", 5.0),
            P_m=env_params.get("P_m", 1.0), w_BH=env_params.get("w_BH", 0.5),
            seed=run_seed
        )
        agent_ddqn = DDQNAgent(env_ddqn, hidden_dim=128, batch_size=batch_size, lr=lr,
                               gamma=gamma, target_update_freq=target_update_freq,
                               memory_capacity=memory_capacity)
        agent_mh = MultiHeadDDQNAgent(env_mh, hidden_dim=128, batch_size=batch_size, lr=lr,
                                      gamma=gamma, target_update_freq=target_update_freq,
                                      memory_capacity=memory_capacity)
        # 存储每个episode的指标
        energy_hist_ddqn = []; hitrate_hist_ddqn = []; eff_hist_ddqn = []; reward_hist_ddqn = []; D_hist_ddqn = []
        energy_hist_mh = []; hitrate_hist_mh = []; eff_hist_mh = []; reward_hist_mh = []; D_hist_mh = []
        
        # 计算输出进度的频率
        progress_interval = 10  # 每10个回合输出一次
        
        for ep in range(1, episodes + 1):
            state_ddqn = env_ddqn.reset()
            state_mh = env_mh.reset()
            ep_energy_ddqn = ep_hitrate_ddqn = ep_eff_ddqn = ep_reward_ddqn = ep_D_ddqn = 0.0
            ep_energy_mh = ep_hitrate_mh = ep_eff_mh = ep_reward_mh = ep_D_mh = 0.0
            for t in range(1, steps_per_episode + 1):
                # 由各智能体选择动作
                X_d, Y_d, Z_d, mask_d = agent_ddqn.select_action(state_ddqn)
                X_m, Y_m, Z_m, mask_m = agent_mh.select_action(state_mh)
                # 与环境交互执行动作
                next_state_ddqn, reward_d, _, info_d = env_ddqn.step(X_d, Y_d, Z_d)
                next_state_mh, reward_m, _, info_m = env_mh.step(X_m, Y_m, Z_m)
                done_flag = (t == steps_per_episode)
                # 存储经验并训练更新
                agent_ddqn.store_transition(state_ddqn, mask_d, reward_d, next_state_ddqn, done_flag)
                agent_mh.store_transition(state_mh, mask_m, reward_m, next_state_mh, done_flag)
                agent_ddqn.update()
                agent_mh.update()
                # 累积当前步的指标
                ep_energy_ddqn += info_d["E_total"]
                ep_hitrate_ddqn += info_d["cache_hit_rate"]
                ep_eff_ddqn += info_d.get("energy_efficiency", 0.0)
                ep_reward_ddqn += reward_d
                ep_D_ddqn += info_d.get("D", 0.0)
                
                ep_energy_mh += info_m["E_total"]
                ep_hitrate_mh += info_m["cache_hit_rate"]
                ep_eff_mh += info_m.get("energy_efficiency", 0.0)
                ep_reward_mh += reward_m
                ep_D_mh += info_m.get("D", 0.0)
                
                state_ddqn = next_state_ddqn
                state_mh = next_state_mh
            # 计算每回合平均指标并记录
            energy_hist_ddqn.append(ep_energy_ddqn / steps_per_episode)
            hitrate_hist_ddqn.append(ep_hitrate_ddqn / steps_per_episode)
            eff_hist_ddqn.append(ep_eff_ddqn / steps_per_episode)
            reward_hist_ddqn.append(ep_reward_ddqn / steps_per_episode)
            D_hist_ddqn.append(ep_D_ddqn / steps_per_episode)
            
            energy_hist_mh.append(ep_energy_mh / steps_per_episode)
            hitrate_hist_mh.append(ep_hitrate_mh / steps_per_episode)
            eff_hist_mh.append(ep_eff_mh / steps_per_episode)
            reward_hist_mh.append(ep_reward_mh / steps_per_episode)
            D_hist_mh.append(ep_D_mh / steps_per_episode)
            
            # epsilon衰减（每回合结束）
            agent_ddqn.epsilon = max(agent_ddqn.min_epsilon, agent_ddqn.epsilon * agent_ddqn.epsilon_decay)
            agent_mh.epsilon = max(agent_mh.min_epsilon, agent_mh.epsilon * agent_mh.epsilon_decay)
            
            # 定期输出进度信息
            if ep % progress_interval == 0 or ep == 1 or ep == episodes:
                progress = ep / episodes * 100
                current_time = time.time()
                elapsed = current_time - run_start_time
                if ep > 1:
                    estimated_total = elapsed * episodes / ep
                    remaining = estimated_total - elapsed
                    print(f"运行 {run}/{args.runs}:  回合 {ep}/{episodes} ({progress:.1f}%)  耗时: {elapsed:.1f}s  剩余: {remaining:.1f}s  命中率: {hitrate_hist_ddqn[-1]:.3f}|{hitrate_hist_mh[-1]:.3f}")
                else:
                    print(f"运行 {run}/{args.runs}:  回合 {ep}/{episodes} ({progress:.1f}%)  耗时: {elapsed:.1f}s")
        
        # 保存训练好的模型
        ddqn_model_path = os.path.join(models_dir, f"ddqn_{config_str}_run{run}.pt")
        mhddqn_model_path = os.path.join(models_dir, f"mhddqn_{config_str}_run{run}.pt")
        
        # 保存DDQN模型
        torch.save({
            'policy_net': agent_ddqn.policy_net.state_dict(),
            'target_net': agent_ddqn.target_net.state_dict(),
            'optimizer': agent_ddqn.optimizer.state_dict(),
            'epsilon': agent_ddqn.epsilon
        }, ddqn_model_path)
        
        # 保存MH-DDQN模型
        torch.save({
            'policy_net': agent_mh.policy_net.state_dict(),
            'target_net': agent_mh.target_net.state_dict(),
            'optimizer': agent_mh.optimizer.state_dict(),
            'epsilon': agent_mh.epsilon
        }, mhddqn_model_path)
        
        # 保存训练数据为npz格式
        ddqn_metrics_path = os.path.join(data_dir, f"ddqn_{config_str}_run{run}_metrics.npz")
        mhddqn_metrics_path = os.path.join(data_dir, f"mhddqn_{config_str}_run{run}_metrics.npz")
        
        # 保存DDQN指标
        np.savez(
            ddqn_metrics_path,
            reward=np.array(reward_hist_ddqn),
            energy=np.array(energy_hist_ddqn),
            hit_rate=np.array(hitrate_hist_ddqn),
            energy_eff=np.array(eff_hist_ddqn),
            D=np.array(D_hist_ddqn),
            config=json.dumps(config)
        )
        
        # 保存MH-DDQN指标
        np.savez(
            mhddqn_metrics_path,
            reward=np.array(reward_hist_mh),
            energy=np.array(energy_hist_mh),
            hit_rate=np.array(hitrate_hist_mh),
            energy_eff=np.array(eff_hist_mh),
            D=np.array(D_hist_mh),
            config=json.dumps(config)
        )
        
        energy_histories_ddqn.append(np.array(energy_hist_ddqn))
        hitrate_histories_ddqn.append(np.array(hitrate_hist_ddqn))
        eff_histories_ddqn.append(np.array(eff_hist_ddqn))
        reward_histories_ddqn.append(np.array(reward_hist_ddqn))
        D_histories_ddqn.append(np.array(D_hist_ddqn))
        
        energy_histories_mh.append(np.array(energy_hist_mh))
        hitrate_histories_mh.append(np.array(hitrate_hist_mh))
        eff_histories_mh.append(np.array(eff_hist_mh))
        reward_histories_mh.append(np.array(reward_hist_mh))
        D_histories_mh.append(np.array(D_hist_mh))
        
        run_time = time.time() - run_start_time
        print(f"运行 {run}/{args.runs} 完成，耗时 {run_time:.1f} 秒")
        print(f"模型已保存到: {models_dir}")
        print(f"训练数据已保存到: {data_dir}")
        
    # 保存所有实验的汇总数据（平均值）
    ddqn_avg_metrics_path = os.path.join(data_dir, f"ddqn_{config_str}_avg_metrics.npz")
    mhddqn_avg_metrics_path = os.path.join(data_dir, f"mhddqn_{config_str}_avg_metrics.npz")
    
    # 计算平均及标准差
    energy_histories_ddqn = np.array(energy_histories_ddqn)
    hitrate_histories_ddqn = np.array(hitrate_histories_ddqn)
    eff_histories_ddqn = np.array(eff_histories_ddqn)
    reward_histories_ddqn = np.array(reward_histories_ddqn)
    D_histories_ddqn = np.array(D_histories_ddqn)
    
    energy_histories_mh = np.array(energy_histories_mh)
    hitrate_histories_mh = np.array(hitrate_histories_mh)
    eff_histories_mh = np.array(eff_histories_mh)
    reward_histories_mh = np.array(reward_histories_mh)
    D_histories_mh = np.array(D_histories_mh)
    
    energy_mean_ddqn = energy_histories_ddqn.mean(axis=0)
    energy_std_ddqn = energy_histories_ddqn.std(axis=0)
    hitrate_mean_ddqn = hitrate_histories_ddqn.mean(axis=0)
    hitrate_std_ddqn = hitrate_histories_ddqn.std(axis=0)
    eff_mean_ddqn = eff_histories_ddqn.mean(axis=0)
    eff_std_ddqn = eff_histories_ddqn.std(axis=0)
    reward_mean_ddqn = reward_histories_ddqn.mean(axis=0)
    reward_std_ddqn = reward_histories_ddqn.std(axis=0)
    D_mean_ddqn = D_histories_ddqn.mean(axis=0)
    D_std_ddqn = D_histories_ddqn.std(axis=0)
    
    energy_mean_mh = energy_histories_mh.mean(axis=0)
    energy_std_mh = energy_histories_mh.std(axis=0)
    hitrate_mean_mh = hitrate_histories_mh.mean(axis=0)
    hitrate_std_mh = hitrate_histories_mh.std(axis=0)
    eff_mean_mh = eff_histories_mh.mean(axis=0)
    eff_std_mh = eff_histories_mh.std(axis=0)
    reward_mean_mh = reward_histories_mh.mean(axis=0)
    reward_std_mh = reward_histories_mh.std(axis=0)
    D_mean_mh = D_histories_mh.mean(axis=0)
    D_std_mh = D_histories_mh.std(axis=0)
    
    # 保存DDQN平均指标
    np.savez(
        ddqn_avg_metrics_path,
        reward=reward_mean_ddqn,
        reward_std=reward_std_ddqn,
        energy=energy_mean_ddqn,
        energy_std=energy_std_ddqn,
        hit_rate=hitrate_mean_ddqn,
        hit_rate_std=hitrate_std_ddqn,
        energy_eff=eff_mean_ddqn,
        energy_eff_std=eff_std_ddqn,
        D=D_mean_ddqn,
        D_std=D_std_ddqn,
        config=json.dumps(config),
        mode="ddqn"
    )
    
    # 保存MH-DDQN平均指标
    np.savez(
        mhddqn_avg_metrics_path,
        reward=reward_mean_mh,
        reward_std=reward_std_mh,
        energy=energy_mean_mh,
        energy_std=energy_std_mh,
        hit_rate=hitrate_mean_mh,
        hit_rate_std=hitrate_std_mh,
        energy_eff=eff_mean_mh,
        energy_eff_std=eff_std_mh,
        D=D_mean_mh,
        D_std=D_std_mh,
        config=json.dumps(config),
        mode="mhddqn"
    )
    
    # 为方便plot_compare.py使用，将模型文件复制到models目录
    import shutil
    models_common_dir = "models"
    os.makedirs(models_common_dir, exist_ok=True)
    
    # 复制平均指标文件到公共models目录
    ddqn_common_metrics_path = os.path.join(models_common_dir, f"ddqn_{config_str}_mode{args.runs}runs_metrics.npz")
    mhddqn_common_metrics_path = os.path.join(models_common_dir, f"mhddqn_{config_str}_mode{args.runs}runs_metrics.npz")
    
    shutil.copy(ddqn_avg_metrics_path, ddqn_common_metrics_path)
    shutil.copy(mhddqn_avg_metrics_path, mhddqn_common_metrics_path)
    
    total_time = time.time() - start_time
    print(f"\n所有实验完成! 总耗时: {total_time:.1f} 秒")
    print("正在生成图表...")
    
    # 绘制指标对比图
    episodes_range = np.arange(1, episodes+1)
    # 图1: 总能耗 vs 回合
    plt.figure()
    plt.plot(episodes_range, energy_mean_ddqn, label="DDQN")
    plt.fill_between(episodes_range, energy_mean_ddqn - energy_std_ddqn, energy_mean_ddqn + energy_std_ddqn, alpha=0.3)
    plt.plot(episodes_range, energy_mean_mh, label="MH-DDQN")
    plt.fill_between(episodes_range, energy_mean_mh - energy_std_mh, energy_mean_mh + energy_std_mh, alpha=0.3)
    plt.xlabel("Episode")
    plt.ylabel("Average E_total per Step")
    plt.title("Energy Consumption vs Episodes")
    plt.legend()
    plt.savefig(os.path.join(plots_dir, "comparison_energy.png"))
    
    # 图2: 缓存命中率 vs 回合
    plt.figure()
    plt.plot(episodes_range, hitrate_mean_ddqn, label="DDQN")
    plt.fill_between(episodes_range, hitrate_mean_ddqn - hitrate_std_ddqn, hitrate_mean_ddqn + hitrate_std_ddqn, alpha=0.3)
    plt.plot(episodes_range, hitrate_mean_mh, label="MH-DDQN")
    plt.fill_between(episodes_range, hitrate_mean_mh - hitrate_std_mh, hitrate_mean_mh + hitrate_std_mh, alpha=0.3)
    plt.xlabel("Episode")
    plt.ylabel("Average Cache Hit Rate")
    plt.title("Cache Hit Rate vs Episodes")
    plt.legend()
    plt.savefig(os.path.join(plots_dir, "comparison_hitrate.png"))
    
    # 图3: 能量效率 vs 回合
    plt.figure()
    plt.plot(episodes_range, eff_mean_ddqn, label="DDQN")
    plt.fill_between(episodes_range, eff_mean_ddqn - eff_std_ddqn, eff_mean_ddqn + eff_std_ddqn, alpha=0.3)
    plt.plot(episodes_range, eff_mean_mh, label="MH-DDQN")
    plt.fill_between(episodes_range, eff_mean_mh - eff_std_mh, eff_mean_mh + eff_std_mh, alpha=0.3)
    plt.xlabel("Episode")
    plt.ylabel("Average Energy Efficiency")
    plt.title("Energy Efficiency vs Episodes")
    plt.legend()
    plt.savefig(os.path.join(plots_dir, "comparison_energy_efficiency.png"))
    
    # 图4: 奖励 vs 回合
    plt.figure()
    plt.plot(episodes_range, reward_mean_ddqn, label="DDQN")
    plt.fill_between(episodes_range, reward_mean_ddqn - reward_std_ddqn, reward_mean_ddqn + reward_std_ddqn, alpha=0.3)
    plt.plot(episodes_range, reward_mean_mh, label="MH-DDQN")
    plt.fill_between(episodes_range, reward_mean_mh - reward_std_mh, reward_mean_mh + reward_std_mh, alpha=0.3)
    plt.xlabel("Episode")
    plt.ylabel("Average Reward per Step")
    plt.title("Reward vs Episodes")
    plt.legend()
    plt.savefig(os.path.join(plots_dir, "comparison_reward.png"))
    
    # 图5: 偏好偏差 vs 回合
    plt.figure()
    plt.plot(episodes_range, D_mean_ddqn, label="DDQN")
    plt.fill_between(episodes_range, D_mean_ddqn - D_std_ddqn, D_mean_ddqn + D_std_ddqn, alpha=0.3)
    plt.plot(episodes_range, D_mean_mh, label="MH-DDQN")
    plt.fill_between(episodes_range, D_mean_mh - D_std_mh, D_mean_mh + D_std_mh, alpha=0.3)
    plt.axhline(y=env_params.get("D_max", 0.2), color='red', linestyle='--', label=f"D_max={env_params.get('D_max', 0.2)}")
    plt.xlabel("Episode")
    plt.ylabel("Average Preference Deviation D")
    plt.title("Preference Deviation vs Episodes")
    plt.legend()
    plt.savefig(os.path.join(plots_dir, "comparison_preference_deviation.png"))
    
    print(f"对比图表已保存到: {plots_dir}")
    print(f"所有结果已保存到: {save_dir}")
    print(f"平均指标文件已复制到公共目录 {models_common_dir}，可直接用于 plot_compare.py")
    
    # 添加模型加载示例代码到README
    with open(os.path.join(save_dir, "README.md"), 'w') as f:
        f.write(f"""# 训练结果 - {timestamp}

## 目录结构
- models/ - 保存的模型权重
- data/ - 训练数据和统计信息
- plots/ - 性能对比图表
- config.json - 训练配置

## 模型加载示例
```python
import torch
from ddqn import DDQNAgent
from mhddqn import MultiHeadDDQNAgent
from advanced_env import AdvancedEnvironment

# 创建环境和智能体
env = AdvancedEnvironment(...)
agent = DDQNAgent(env, ...)

# 加载模型权重
checkpoint = torch.load('models/ddqn_{config_str}_run1.pt')
agent.policy_net.load_state_dict(checkpoint['policy_net'])
agent.target_net.load_state_dict(checkpoint['target_net'])
agent.optimizer.load_state_dict(checkpoint['optimizer'])
agent.epsilon = checkpoint['epsilon']

# 测试模型
state = env.reset()
action = agent.select_action(state, evaluate=True)  # evaluate=True 表示测试模式
```

## 数据分析示例
```python
import numpy as np
import matplotlib.pyplot as plt

# 加载指标数据
data = np.load('data/ddqn_{config_str}_avg_metrics.npz')

# 访问数据
reward = data['reward']
reward_std = data['reward_std']
hit_rate = data['hit_rate']
energy = data['energy']
D = data['D']

# 绘制图表
plt.figure()
plt.plot(reward)
plt.fill_between(np.arange(len(reward)), reward - reward_std, reward + reward_std, alpha=0.3)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.show()
```

## 使用 plot_compare.py 进行不同配置的比较
此次训练的平均指标文件已复制到公共目录 `{models_common_dir}`，可以使用 plot_compare.py 进行比较：

```bash
# 比较不同模式的结果
python plot_compare.py --modes ddqn mhddqn

# 特定配置的比较
python plot_compare.py --M {M} --N {N} --F {F} --K {K}
```
""")

if __name__ == "__main__":
    main()
