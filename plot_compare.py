import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import re
from glob import glob

def extract_params_from_filename(filename):
    """从文件名中提取参数信息"""
    basename = os.path.basename(filename)
    # 提取各个参数
    m_match = re.search(r'M(\d+)', basename)
    n_match = re.search(r'N(\d+)', basename)
    f_match = re.search(r'F(\d+)', basename)
    k_match = re.search(r'K(\d+)', basename)
    cc_match = re.search(r'Cc(\d+)', basename)
    cr_match = re.search(r'Cr(\d+)', basename)
    lr_match = re.search(r'lr(\d+\.\d+)', basename)
    mode_match = re.search(r'mode([a-zA-Z_]+)', basename)
    
    params = {}
    if m_match: params['M'] = int(m_match.group(1))
    if n_match: params['N'] = int(n_match.group(1))
    if f_match: params['F'] = int(f_match.group(1))
    if k_match: params['K'] = int(k_match.group(1))
    if cc_match: params['C_cache'] = int(cc_match.group(1))
    if cr_match: params['C_rec'] = int(cr_match.group(1))
    if lr_match: params['lr'] = float(lr_match.group(1))
    if mode_match: params['mode'] = mode_match.group(1)
    
    return params

def find_metrics_files_by_mode(models_dir="models", target_modes=None):
    """查找指定模式的指标文件"""
    all_metrics_files = glob(os.path.join(models_dir, "*_metrics.npz"))
    
    if not all_metrics_files:
        print(f"未在 {models_dir} 目录中找到任何指标文件")
        return []
    
    if target_modes is None:
        return all_metrics_files
    
    filtered_files = []
    for file_path in all_metrics_files:
        params = extract_params_from_filename(file_path)
        if 'mode' in params and params['mode'] in target_modes:
            filtered_files.append(file_path)
    
    return filtered_files

def find_metrics_files_by_config(models_dir="models", config_name=None, common_params=None):
    """查找满足特定配置的指标文件"""
    all_metrics_files = glob(os.path.join(models_dir, "*_metrics.npz"))
    
    if not config_name and not common_params:
        return all_metrics_files
    
    filtered_files = []
    for file_path in all_metrics_files:
        if config_name and config_name not in os.path.basename(file_path):
            continue
            
        if common_params:
            params = extract_params_from_filename(file_path)
            match = True
            for key, value in common_params.items():
                if key == 'mode':  # 模式名称需要完全匹配
                    if key not in params or params[key] != value:
                        match = False
                        break
                elif key not in params or params[key] != value:
                    match = False
                    break
            
            if not match:
                continue
        
        filtered_files.append(file_path)
    
    return filtered_files

def main():
    parser = argparse.ArgumentParser(description="比较不同实验配置的训练结果")
    parser.add_argument("--modes", nargs="+", default=None, 
                        help="要比较的模式 (例如: full random_all random_ca)")
    parser.add_argument("--config", type=str, default=None,
                        help="配置文件名称，用于筛选结果")
    parser.add_argument("--models_dir", type=str, default="models",
                        help="模型保存目录")
    parser.add_argument("--output_dir", type=str, default="comparison_plots",
                        help="比较图表保存目录")
    parser.add_argument("--M", type=int, default=None, help="基站数")
    parser.add_argument("--N", type=int, default=None, help="用户数")
    parser.add_argument("--F", type=int, default=None, help="视频数")
    parser.add_argument("--K", type=int, default=None, help="每视频层数")
    parser.add_argument("--C_cache", type=int, default=None, help="缓存容量")
    parser.add_argument("--C_rec", type=int, default=None, help="推荐容量")
    parser.add_argument("--lr", type=float, default=None, help="学习率")
    args = parser.parse_args()
    
    # 收集要筛选的参数
    filter_params = {}
    if args.M is not None: filter_params['M'] = args.M
    if args.N is not None: filter_params['N'] = args.N
    if args.F is not None: filter_params['F'] = args.F
    if args.K is not None: filter_params['K'] = args.K
    if args.C_cache is not None: filter_params['C_cache'] = args.C_cache
    if args.C_rec is not None: filter_params['C_rec'] = args.C_rec
    if args.lr is not None: filter_params['lr'] = args.lr

    # 查找符合条件的指标文件
    if args.modes:
        metrics_files = find_metrics_files_by_mode(args.models_dir, args.modes)
    else:
        metrics_files = find_metrics_files_by_config(args.models_dir, args.config, filter_params)
    
    if not metrics_files:
        print("未找到符合条件的指标文件。请检查筛选条件或确保已运行训练。")
        return
    
    print(f"找到 {len(metrics_files)} 个符合条件的指标文件:")
    for f in metrics_files:
        print(f"  - {os.path.basename(f)}")
    
    # 加载指标文件
    metrics_dict = {}
    for file_path in metrics_files:
        basename = os.path.basename(file_path)
        params = extract_params_from_filename(file_path)
        
        # 使用模式作为键，如果多个文件同模式，则添加区分标识
        mode_key = params.get('mode', 'unknown')
        if 'M' in params and 'N' in params:
            mode_key = f"{mode_key}_M{params['M']}_N{params['N']}"
        
        # 避免键重复
        counter = 1
        original_key = mode_key
        while mode_key in metrics_dict:
            counter += 1
            mode_key = f"{original_key}_{counter}"
        
        # 加载数据
        data = np.load(file_path)
        metrics_dict[mode_key] = {
            "reward": data["reward"],
            "energy": data["energy"],
            "D": data["D"],
            "hitrate": data["hit_rate"],
            "energy_eff": data["energy_eff"] if "energy_eff" in data else data["hit_rate"] / (data["energy"] + 1e-10)
        }
        print(f"已加载 {basename} 的指标数据")

    if not metrics_dict:
        print("没有成功加载任何指标数据。")
        return

    # 统一 episodes 数量（找到最短的序列长度）
    min_len = min(len(mdata["reward"]) for mdata in metrics_dict.values())
    episodes = np.arange(1, min_len + 1)
    
    # 创建存放图片的文件夹
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # 1) Plot average reward
    plt.figure(figsize=(10,6))
    for mode, mdata in metrics_dict.items():
        plt.plot(episodes, mdata["reward"][:min_len], label=f"{mode}")
    plt.title("Training Reward per Episode")
    plt.xlabel("Episode"); plt.ylabel("Average Reward")
    plt.grid(True); plt.legend()
    plt.savefig(os.path.join(output_dir, "reward_compare.png"), dpi=300)
    plt.close()

    # 2) Plot energy consumption
    plt.figure(figsize=(10,6))
    for mode, mdata in metrics_dict.items():
        plt.plot(episodes, mdata["energy"][:min_len], label=f"{mode}")
    plt.title("Average Energy Consumption per Episode")
    plt.xlabel("Episode"); plt.ylabel("Energy Consumption")
    plt.grid(True); plt.legend()
    plt.savefig(os.path.join(output_dir, "energy_compare.png"), dpi=300)
    plt.close()

    # 3) Plot preference deviation
    plt.figure(figsize=(10,6))
    for mode, mdata in metrics_dict.items():
        plt.plot(episodes, mdata["D"][:min_len], label=f"{mode}")
    plt.axhline(y=0.2, color='red', linestyle='--', label="D_max=0.2")
    plt.title("Preference Deviation per Episode")
    plt.xlabel("Episode"); plt.ylabel("Preference Deviation D")
    plt.grid(True); plt.legend()
    plt.savefig(os.path.join(output_dir, "preference_deviation_compare.png"), dpi=300)
    plt.close()

    # 4) Plot cache hit rate
    plt.figure(figsize=(10,6))
    for mode, mdata in metrics_dict.items():
        plt.plot(episodes, mdata["hitrate"][:min_len], label=f"{mode}")
    plt.title("Cache Hit Rate per Episode")
    plt.xlabel("Episode"); plt.ylabel("Cache Hit Rate")
    plt.ylim(0, 1)
    plt.grid(True); plt.legend()
    plt.savefig(os.path.join(output_dir, "cache_hit_rate_compare.png"), dpi=300)
    plt.close()

    # 5) Plot energy efficiency
    plt.figure(figsize=(10,6))
    for mode, mdata in metrics_dict.items():
        plt.plot(episodes, mdata["energy_eff"][:min_len], label=f"{mode}")
    plt.title("Energy Efficiency per Episode")
    plt.xlabel("Episode"); plt.ylabel("Energy Efficiency")
    plt.grid(True); plt.legend()
    plt.savefig(os.path.join(output_dir, "energy_efficiency_compare.png"), dpi=300)
    plt.close()

    print(f"比较图表已保存到 '{output_dir}' 文件夹:")
    print("  - reward_compare.png")
    print("  - energy_compare.png")
    print("  - preference_deviation_compare.png")
    print("  - cache_hit_rate_compare.png")
    print("  - energy_efficiency_compare.png")

if __name__ == "__main__":
    main()
