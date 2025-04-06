import numpy as np
import matplotlib.pyplot as plt

def main():
    data = np.load("training_metrics.npz")
    reward_history = data["reward"]
    energy_history = data["energy"]
    D_history = data["D"]
    hitrate_history = data["hit_rate"]
    episodes = np.arange(1, reward_history.shape[0] + 1)
    # Plot average reward
    plt.figure()
    plt.plot(episodes, reward_history, label="Average Reward")
    plt.title("Training Reward per Episode")
    plt.xlabel("Episode"); plt.ylabel("Average Reward")
    plt.grid(True); plt.legend()
    plt.savefig("reward_curve.png")
    # Plot energy consumption
    plt.figure()
    plt.plot(episodes, energy_history, label="Average Energy", color='orange')
    plt.title("Average Energy Consumption per Episode")
    plt.xlabel("Episode"); plt.ylabel("Energy Consumption")
    plt.grid(True); plt.legend()
    plt.savefig("energy_curve.png")
    # Plot preference deviation
    plt.figure()
    plt.plot(episodes, D_history, label="Preference Deviation", color='green')
    plt.axhline(y=0.2, color='red', linestyle='--', label="D_max")
    plt.title("Average Preference Deviation per Episode")
    plt.xlabel("Episode"); plt.ylabel("Preference Deviation D")
    plt.grid(True); plt.legend()
    plt.savefig("preference_deviation_curve.png")
    # Plot cache hit rate
    plt.figure()
    plt.plot(episodes, hitrate_history, label="Cache Hit Rate", color='purple')
    plt.title("Cache Hit Rate per Episode")
    plt.xlabel("Episode"); plt.ylabel("Cache Hit Rate")
    plt.ylim(0, 1); plt.grid(True); plt.legend()
    plt.savefig("cache_hit_rate_curve.png")
    print("Plots saved as reward_curve.png, energy_curve.png, preference_deviation_curve.png, cache_hit_rate_curve.png")

if __name__ == "__main__":
    main()
