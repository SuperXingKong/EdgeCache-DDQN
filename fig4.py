#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
import re
from glob import glob

def extract_cache_capacity(filename):
    """从文件名中提取缓存容量参数"""
    basename = os.path.basename(filename)
    cc_match = re.search(r'Cc(\d+)', basename)
    if cc_match:
        return int(cc_match.group(1))
    return None

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
    # 查找所有Figure 4相关的指标文件
    models_dir = "models"
    metrics_files = glob(os.path.join(models_dir, "*Figure_4*_metrics.npz"))
    
    if not metrics_files:
        print("未找到与Figure 4相关的指标文件")
        return
    
    print(f"找到 {len(metrics_files)} 个Figure 4相关的指标文件")
    
    # 组织数据结构: {mode: {cache_size: energy_value}}
    data = {}
    mode_display_names = {
        "full": "DDQN-based CA, UA and REC",
        "random_all": "Random CA, UA and REC",
        "random_ua": "DDQN-based CA and REC only",
        "random_rec": "DDQN-based CA and UA only",
        "random_ca": "DDQN-based UA and REC only"
    }
    
    for file_path in metrics_files:
        cache_capacity = extract_cache_capacity(file_path)
        mode = extract_mode(file_path)
        
        if cache_capacity is None or mode == "unknown":
            continue
        
        # 加载数据
        metrics = np.load(file_path)
        if "energy" in metrics:
            # 计算平均能耗（使用最后100个episode的平均值作为稳定值）
            energy_data = metrics["energy"]
            avg_energy = np.mean(energy_data[-min(100, len(energy_data)):])
            
            if mode not in data:
                data[mode] = {}
            data[mode][cache_capacity] = avg_energy
            print(f"模式: {mode}, 缓存容量: {cache_capacity}, 平均能耗: {avg_energy:.2f}")
    
    if not data:
        print("未能提取有效数据")
        return
    
    # 绘制图表
    plt.figure(figsize=(12, 8))
    plt.title("Figure 4: Total energy consumption under different values of the caching capacity of each BS")
    plt.xlabel("Caching capacity")
    plt.ylabel("System energy consumption")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 定义线条样式和颜色
    styles = {
        "full": ("red", "x-"),
        "random_ca": ("green", "s-"),
        "random_ua": ("cyan", "^-"),
        "random_rec": ("blue", "o-"),
        "random_all": ("magenta", "*-")
    }
    
    # 绘制每种模式的数据
    for mode, capacities in data.items():
        if not capacities:
            continue
            
        # 按缓存容量排序
        sorted_items = sorted(capacities.items())
        x_values = [item[0] for item in sorted_items]
        y_values = [item[1] for item in sorted_items]
        
        color, style = styles.get(mode, ("black", "-"))
        display_name = mode_display_names.get(mode, mode)
        plt.plot(x_values, y_values, style, color=color, label=display_name, linewidth=2)
    
    # 设置图例和保存
    plt.legend(loc='best')
    output_dir = "figures"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "fig4_cache_capacity_comparison.png"), dpi=300, bbox_inches='tight')
    print(f"图表已保存至 {output_dir}/fig4_cache_capacity_comparison.png")
    
    # 显示图表
    plt.show()

if __name__ == "__main__":
    main()