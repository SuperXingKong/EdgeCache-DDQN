import numpy as np
import matplotlib.pyplot as plt
import os

def main():
    # 要对比的多种模式
    modes = [
        "full",
        "random_all",
        "random_ca",
        "random_rec",
        "random_ua"
    ]

    # 尝试加载对应的 metrics 文件
    metrics_dict = {}
    for mode in modes:
        filename = f"training_metrics_{mode}.npz"
        if os.path.exists(filename):
            data = np.load(filename)
            reward_history = data["reward"]
            energy_history = data["energy"]
            D_history = data["D"]
            hitrate_history = data["hit_rate"]
            metrics_dict[mode] = {
                "reward": reward_history,
                "energy": energy_history,
                "D": D_history,
                "hitrate": hitrate_history
            }
        else:
            print(f"Warning: {filename} not found. Skipping mode='{mode}'.")

    if not metrics_dict:
        print("No metrics files found. Please run training first or check filenames.")
        return

    # 统一 episodes 数量（假设每种模式都跑了相同 episodes）
    any_mode = next(iter(metrics_dict))  # 获取一个可用的 mode
    max_len = len(metrics_dict[any_mode]["reward"])
    episodes = np.arange(1, max_len + 1)

    # 创建存放图片的文件夹
    output_dir = "comparison_plots"
    os.makedirs(output_dir, exist_ok=True)

    # 1) Plot average reward
    plt.figure(figsize=(8,6))
    for mode, mdata in metrics_dict.items():
        plt.plot(episodes, mdata["reward"], label=f"{mode}")
    plt.title("Training Reward per Episode (Comparison)")
    plt.xlabel("Episode"); plt.ylabel("Average Reward")
    plt.grid(True); plt.legend()
    plt.savefig(os.path.join(output_dir, "reward_compare.png"))
    plt.close()

    # 2) Plot energy consumption
    plt.figure(figsize=(8,6))
    for mode, mdata in metrics_dict.items():
        plt.plot(episodes, mdata["energy"], label=f"{mode}")
    plt.title("Average Energy Consumption per Episode (Comparison)")
    plt.xlabel("Episode"); plt.ylabel("Energy Consumption")
    plt.grid(True); plt.legend()
    plt.savefig(os.path.join(output_dir, "energy_compare.png"))
    plt.close()

    # 3) Plot preference deviation
    plt.figure(figsize=(8,6))
    for mode, mdata in metrics_dict.items():
        plt.plot(episodes, mdata["D"], label=f"{mode}")
    plt.axhline(y=0.2, color='red', linestyle='--', label="D_max=0.2")
    plt.title("Preference Deviation per Episode (Comparison)")
    plt.xlabel("Episode"); plt.ylabel("Preference Deviation D")
    plt.grid(True); plt.legend()
    plt.savefig(os.path.join(output_dir, "preference_deviation_compare.png"))
    plt.close()

    # 4) Plot cache hit rate
    plt.figure(figsize=(8,6))
    for mode, mdata in metrics_dict.items():
        plt.plot(episodes, mdata["hitrate"], label=f"{mode}")
    plt.title("Cache Hit Rate per Episode (Comparison)")
    plt.xlabel("Episode"); plt.ylabel("Cache Hit Rate")
    plt.ylim(0, 1)
    plt.grid(True); plt.legend()
    plt.savefig(os.path.join(output_dir, "cache_hit_rate_compare.png"))
    plt.close()

    print(f"Comparison plots saved into '{output_dir}' folder:")
    print("  - reward_compare.png")
    print("  - energy_compare.png")
    print("  - preference_deviation_compare.png")
    print("  - cache_hit_rate_compare.png")

if __name__ == "__main__":
    main()
