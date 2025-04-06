import torch
import numpy as np
from env import Environment
from ddqn import DDQNAgent

def main():
    # (Use same environment configuration as training)
    M = 2; N = 4; F = 6; K = 2
    C_cache = 4; C_rec = 3
    episodes = 10
    steps_per_episode = 50
    seed = 123  # seed for evaluation environment

    env = Environment(M=M, N=N, F=F, K=K, C_cache=C_cache, C_rec=C_rec, seed=seed)
    agent = DDQNAgent(env)
    # Load trained model weights
    agent.q_network.load_state_dict(torch.load("ddqn_model.pth", map_location=torch.device('cpu')))
    agent.target_network.load_state_dict(agent.q_network.state_dict())
    agent.epsilon = 0.0  # no exploration during evaluation

    total_energy = total_D = total_hitrate = 0.0
    for ep in range(episodes):
        state = env.reset()
        ep_energy = ep_D = ep_hitrate = 0.0
        for t in range(steps_per_episode):
            X, Y, Z, action_mask = agent.select_action(state)
            next_state, reward, done, info = env.step(X, Y, Z)
            ep_energy += info["E_total"]
            ep_D += info["D"]
            ep_hitrate += info["cache_hit_rate"]
            state = next_state
        total_energy += ep_energy / steps_per_episode
        total_D += ep_D / steps_per_episode
        total_hitrate += ep_hitrate / steps_per_episode
    avg_energy = total_energy / episodes
    avg_D = total_D / episodes
    avg_hitrate = total_hitrate / episodes
    print(f"Average Energy Consumption: {avg_energy:.3f}")
    print(f"Average Preference Deviation: {avg_D:.3f}")
    print(f"Average Cache Hit Rate: {avg_hitrate:.3f}")

if __name__ == "__main__":
    main()
