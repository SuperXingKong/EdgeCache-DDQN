#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
import re
from glob import glob

def extract_mode(filename):
    """从文件名中提取模式参数"""
    basename = os.path.basename(filename)
    if "full" in basename:
        return "full"
    elif "random_all" in basename:
        return "random_all"
    elif "random_ua" in basename:
        return "random_ua"
    elif "random_rec" in basename:
        return "random_rec"
    elif "random_ca" in basename:
        return "random_ca"
    return "unknown"

def main():
    # 查找所有包含能耗数据的指标文件
    models_dir = "models"
    metrics_files = glob(os.path.join(models_dir, "*_metrics.npz"))
    
    if not metrics_files:
        print("未找到包含能耗数据的指标文件")
        return
    
    print(f"找到 {len(metrics_files)} 个指标文件")
    
    # 整理数据: {mode: energy_array}
    data = {}
    mode_display_names = {
        "full": "DDQN-based CA, UA and REC",
        "random_all": "Random CA, UA and REC",
        "random_ua": "DDQN-based CA and REC only",
        "random_rec": "DDQN-based CA and UA only",
        "random_ca": "DDQN-based UA and REC only"
    }
    
    # 尝试从stdout文件中提取能耗数据
    results_dirs = [
        "experiment_results_20250414_154215",
        "experiment_results_20250414_115106"
    ]
    
    stdout_files = []
    for base_dir in results_dirs:
        if os.path.exists(base_dir):
            for root, dirs, files in os.walk(base_dir):
                for file in files:
                    if "_stdout.txt" in file:
                        stdout_files.append(os.path.join(root, file))
    
    print(f"找到 {len(stdout_files)} 个stdout文件进行分析")
    
    # 从stdout文件中提取能耗数据
    for file_path in stdout_files:
        mode = extract_mode(file_path)
        
        if mode == "unknown":
            continue
        
        energy_values = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            for line in lines:
                if "avg_energy=" in line:
                    energy_match = re.search(r'avg_energy=(\d+\.\d+)', line)
                    if energy_match:
                        energy_values.append(float(energy_match.group(1)))
        except Exception as e:
            print(f"读取文件 {file_path} 时出错: {str(e)}")
        
        if energy_values:
            if mode not in data:
                data[mode] = energy_values
            elif len(energy_values) > len(data[mode]):
                # 保留最长的能耗序列
                data[mode] = energy_values
    
    # 如果没有从stdout文件中找到数据，尝试从metrics文件中加载
    if not data:
        for file_path in metrics_files:
            mode = extract_mode(file_path)
            
            if mode == "unknown":
                continue
            
            try:
                metrics = np.load(file_path)
                if "energy" in metrics:
                    energy_values = metrics["energy"]
                    if mode not in data:
                        data[mode] = energy_values
                    elif len(energy_values) > len(data[mode]):
                        # 保留最长的能耗序列
                        data[mode] = energy_values
            except Exception as e:
                print(f"加载文件 {file_path} 时出错: {str(e)}")
    
    if not data:
        print("未能提取有效的能耗数据")
        return
    
    # 绘制收敛曲线
    plt.figure(figsize=(12, 8))
    plt.title("Figure 3: Convergence of the proposed algorithm")
    plt.xlabel("Episode")
    plt.ylabel("System energy consumption")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 定义线条样式和颜色，匹配图像中的样式
    styles = {
        "full": ("red", "-"),
        "random_ca": ("green", "-"),
        "random_ua": ("blue", "-"),
        "random_rec": ("cyan", "-"),
        "random_all": ("magenta", "-")
    }
    
    max_episodes = 0
    for mode, energy_values in data.items():
        if len(energy_values) > max_episodes:
            max_episodes = len(energy_values)
    
    # 绘制每种模式的收敛曲线
    for mode, energy_values in data.items():
        color, style = styles.get(mode, ("black", "-"))
        display_name = mode_display_names.get(mode, mode)
        
        # 确保所有曲线的长度一致，不足的使用NaN填充
        if len(energy_values) < max_episodes:
            energy_values = np.concatenate([energy_values, np.full(max_episodes - len(energy_values), np.nan)])
        
        episodes = np.arange(1, len(energy_values) + 1)
        plt.plot(episodes, energy_values, style, color=color, label=display_name, linewidth=2)
        
        print(f"模式: {mode}, 能耗数据点数量: {len(energy_values)}")
    
    # 设置x轴范围为0-1600
    plt.xlim(0, 1600)
    
    # 设置图例和保存
    plt.legend(loc='best')
    output_dir = "figures"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "fig3_convergence.png"), dpi=300, bbox_inches='tight')
    print(f"图表已保存至 {output_dir}/fig3_convergence.png")
    
    # 显示图表
    plt.show()

if __name__ == "__main__":
    main() 