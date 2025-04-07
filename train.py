import torch
import numpy as np
import argparse
from env import Environment
from ddqn import DDQNAgent

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
    M = 2       # number of base stations
    N = 4       # number of users
    F = 6       # number of videos
    K = 2       # layers per video
    C_cache = 4  # cache capacity per BS (number of items)
    C_rec = 3    # recommendation list size per BS
    episodes = 1000
    steps_per_episode = 50
    gamma = 0.95
    lr = 1e-3
    batch_size = 64
    memory_capacity = 10000
    target_update_freq = 100
    seed = 42

    # Initialize environment and agent
    env = Environment(M=M, N=N, F=F, K=K, C_cache=C_cache, C_rec=C_rec, seed=seed)
    agent = DDQNAgent(env, hidden_dim=128, batch_size=batch_size, lr=lr, gamma=gamma,
                      target_update_freq=target_update_freq, memory_capacity=memory_capacity)

    # Logging
    reward_history = []
    energy_history = []
    D_history = []
    hitrate_history = []

    print(f"=== Training Mode: {args.mode} ===")

    # Training loop
    for ep in range(1, episodes+1):
        state = env.reset()
        ep_reward = ep_energy = ep_D = ep_hitrate = 0.0
        for t in range(1, steps_per_episode+1):
            if args.mode == "full":
                # 全部DDQN优化
                X, Y, Z, action_mask = agent.select_action(state)
            else:
                # 部分随机：先让agent选出完整的动作mask
                X_ddqn, Y_ddqn, Z_ddqn, action_mask_ddqn = agent.select_action(state)

                # 然后根据mode覆盖部分随机
                if args.mode == "random_all":
                    X = random_caching(env)
                    Y = random_recommendation(env)
                    Z = random_association(env)
                elif args.mode == "random_ua":
                    # 缓存、推荐 = DDQN，用户关联=随机
                    X = X_ddqn
                    Y = Y_ddqn
                    Z = random_association(env)
                elif args.mode == "random_rec":
                    # 缓存、用户关联=DDQN，推荐=随机
                    X = X_ddqn
                    Y = random_recommendation(env)
                    Z = Z_ddqn
                elif args.mode == "random_ca":
                    # 推荐、用户关联=DDQN，缓存=随机
                    X = random_caching(env)
                    Y = Y_ddqn
                    Z = Z_ddqn
                # 最后合成一个 action_mask 以存储，但不会真正影响环境
                # (因为 env.step() 用的是 X,Y,Z)
                # 我们只需要把真实使用的 X,Y,Z 拼成 flatten 用来做 experience replay
                mask_len = 2*env.M*env.F*env.K + env.M*env.N
                # flatten X, Y, Z
                action_mask_parts = [
                    X.flatten(),
                    Y.flatten(),
                    Z.flatten()
                ]
                action_mask = np.concatenate(action_mask_parts).astype(np.float32)
            
            # Take action in env
            next_state, reward, done, info = env.step(X, Y, Z)
            done_flag = True if t == steps_per_episode else False
            # Store transition and update agent
            agent.store_transition(state, action_mask, reward, next_state, done_flag)
            
            # 注意：无论是否 random，一律用 DQN 来更新（因为要继续学习 Q 函数）
            # 这样只不过是环境实际动作部分随机，不影响 Q 更新的过程。
            # 这也意味着在 random_X 模式下，Q 学到的东西不一定真实→ 只用于对比baseline
            agent.update()

            # Accumulate metrics
            ep_reward += reward
            ep_energy += info["E_total"]
            ep_D += info["D"]
            ep_hitrate += info["cache_hit_rate"]

            state = next_state
            if done_flag:
                break

        # Decay epsilon after each episode
        if agent.epsilon > agent.min_epsilon:
            agent.epsilon *= agent.epsilon_decay
            if agent.epsilon < agent.min_epsilon:
                agent.epsilon = agent.min_epsilon

        # Record average metrics for the episode
        avg_reward = ep_reward / steps_per_episode
        avg_energy = ep_energy / steps_per_episode
        avg_D = ep_D / steps_per_episode
        avg_hitrate = ep_hitrate / steps_per_episode
        reward_history.append(avg_reward)
        energy_history.append(avg_energy)
        D_history.append(avg_D)
        hitrate_history.append(avg_hitrate)

        # Progress printout
        if ep % 10 == 0 or ep == 1:
            print(f"Episode {ep}/{episodes}: avg_reward={avg_reward:.3f}, avg_energy={avg_energy:.3f}, "
                  f"avg_D={avg_D:.3f}, hit_rate={avg_hitrate:.3f}, epsilon={agent.epsilon:.2f}")

    # Save model and metrics (include mode info)
    save_name = f"ddqn_model_{args.mode}.pth"
    torch.save(agent.q_network.state_dict(), save_name)
    np.savez(f"training_metrics_{args.mode}.npz", 
             reward=np.array(reward_history, dtype=np.float32),
             energy=np.array(energy_history, dtype=np.float32),
             D=np.array(D_history, dtype=np.float32),
             hit_rate=np.array(hitrate_history, dtype=np.float32))
    print(f"Training completed under mode='{args.mode}'.")
    print(f"Model saved to {save_name} and metrics to training_metrics_{args.mode}.npz")

if __name__ == "__main__":
    main()
