import torch
import numpy as np
from env import Environment
from ddqn import DDQNAgent

def main():
    # Configuration
    M = 2       # number of base stations
    N = 4       # number of users
    F = 6       # number of videos
    K = 2       # layers per video
    C_cache = 4  # cache capacity per BS (number of items)
    C_rec = 3    # recommendation list size per BS
    episodes = 1000
    steps_per_episode = 50
    gamma = 0.98
    lr = 5e-4
    batch_size = 64
    memory_capacity = 10000
    target_update_freq = 200
    seed = 42

    # Initialize environment and agent
    env = Environment(M=M, N=N, F=F, K=K, C_cache=C_cache, C_rec=C_rec, seed=seed)
    agent = DDQNAgent(env, hidden_dim=256, batch_size=batch_size, lr=lr, gamma=gamma,
                      target_update_freq=target_update_freq, memory_capacity=memory_capacity)

    # Logging
    reward_history = []
    energy_history = []
    D_history = []
    hitrate_history = []

    # Training loop
    for ep in range(1, episodes+1):
        state = env.reset()
        ep_reward = ep_energy = ep_D = ep_hitrate = 0.0
        for t in range(1, steps_per_episode+1):
            # Choose action
            X, Y, Z, action_mask = agent.select_action(state)
            # Take action in env
            next_state, reward, done, info = env.step(X, Y, Z)
            done_flag = True if t == steps_per_episode else False  # episode ends after fixed steps
            # Store transition and update agent
            agent.store_transition(state, action_mask, reward, next_state, done_flag)
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
    # Save model and metrics
    torch.save(agent.q_network.state_dict(), "ddqn_model.pth")
    np.savez("training_metrics.npz", 
             reward=np.array(reward_history, dtype=np.float32),
             energy=np.array(energy_history, dtype=np.float32),
             D=np.array(D_history, dtype=np.float32),
             hit_rate=np.array(hitrate_history, dtype=np.float32))
    print("Training completed. Model saved to ddqn_model.pth and metrics saved to training_metrics.npz")

if __name__ == "__main__":
    main()
