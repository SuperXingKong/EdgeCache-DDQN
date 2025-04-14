import torch
import numpy as np
import argparse
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
    parser.add_argument("--mode", type=str, default="full", 
                        choices=["full","random_all","random_ua","random_rec","random_ca"],
                        help="Choose the control mode for partial or full random actions")
    args = parser.parse_args()

    # Configuration
    M = 2        # 基站数
    N = 4        # 用户数
    F = 6        # 视频数
    K = 2        # 每视频层数
    C_cache = 4  # 缓存容量(层数)
    C_rec = 3    # 推荐容量
    episodes = 3000
    steps_per_episode = 50
    gamma = 0.95
    lr = 1e-3
    batch_size = 64
    memory_capacity = 10000
    target_update_freq = 100
    seed = 42

    # ==================
    # 创建环境 & DDQN智能体
    # ==================
    env = AdvancedEnvironment(
        M=M, N=N, F=F, K=K, 
        C_cache=C_cache, C_rec=C_rec, 
        D_max=0.2, phi=100,
        B=10.0, BH_B=5.0,
        P_m=1.0, w_BH=0.5,
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

    print(f"=== Training Mode: {args.mode} ===")

    # ==================
    # 训练循环
    # ==================
    for ep in range(1, episodes+1):
        state = env.reset()
        ep_reward = ep_energy = ep_D = ep_hitrate = ep_eff = 0.0

        for t in range(1, steps_per_episode+1):
            # 1) 根据 mode 选择动作
            if args.mode == "full":
                # 全部由DDQN做决策
                X, Y, Z, action_mask = agent.select_action(state)
            else:
                # 先用DDQN生成一份完整动作 (X_ddqn, Y_ddqn, Z_ddqn)
                X_ddqn, Y_ddqn, Z_ddqn, action_mask_ddqn = agent.select_action(state)

                # 部分改为随机策略
                if args.mode == "random_all":
                    X = random_caching(env)
                    Y = random_recommendation(env)
                    Z = random_association(env)
                elif args.mode == "random_ua":
                    X = X_ddqn
                    Y = Y_ddqn
                    Z = random_association(env)
                elif args.mode == "random_rec":
                    X = X_ddqn
                    Y = random_recommendation(env)
                    Z = Z_ddqn
                elif args.mode == "random_ca":
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
    # 训练完成，保存结果
    # ==================
    save_name = f"ddqn_model_{args.mode}.pth"
    torch.save(agent.q_network.state_dict(), save_name)
    np.savez(f"training_metrics_{args.mode}.npz",
             reward=np.array(reward_history, dtype=np.float32),
             energy=np.array(energy_history, dtype=np.float32),
             D=np.array(D_history, dtype=np.float32),
             hit_rate=np.array(hitrate_history, dtype=np.float32),
             energy_eff=np.array(eff_history, dtype=np.float32))
    print(f"Training completed under mode='{args.mode}'.")
    print(f"Model saved to {save_name} and metrics to training_metrics_{args.mode}.npz")


if __name__ == "__main__":
    main()
