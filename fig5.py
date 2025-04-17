#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
import re
from glob import glob

def extract_user_count(filename):
    """从文件名中提取用户数量参数"""
    basename = os.path.basename(filename)
    user_count_match = re.search(r'User_count_(\d+)', basename)
    if user_count_match:
        return int(user_count_match.group(1))
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

def extract_energy_from_stdout(file_path):
    """从stdout文件中提取最后100个episode的平均能耗值"""
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 收集最后100个episode的能耗数据
    energy_values = []
    for line in lines:
        if "avg_energy=" in line:
            energy_match = re.search(r'avg_energy=(\d+\.\d+)', line)
            if energy_match:
                energy_values.append(float(energy_match.group(1)))
    
    # 取最后100个数据的平均值，如果数据不足100个则取所有可用数据的平均值
    if energy_values:
        energy_values = energy_values[-min(100, len(energy_values)):]
        return sum(energy_values) / len(energy_values)
    return None

def main():
    # 查找所有Figure 5相关的stdout文件
    results_dir = "experiment_results_20250414_154215/fig5_user_count"
    stdout_files = glob(os.path.join(results_dir, "*_stdout.txt"))
    
    if not stdout_files:
        print("未找到与Figure 5相关的stdout文件")
        return
    
    print(f"找到 {len(stdout_files)} 个Figure 5相关的stdout文件")
    
    # 组织数据结构: {mode: {user_count: energy_value}}
    data = {}
    mode_display_names = {
        "full": "DDQN-based CA, UA and REC",
        "random_all": "Random CA, UA and REC",
        "random_ua": "DDQN-based CA and REC only",
        "random_rec": "DDQN-based CA and UA only",
        "random_ca": "DDQN-based UA and REC only"
    }
    
    for file_path in stdout_files:
        user_count = extract_user_count(file_path)
        mode = extract_mode(file_path)
        
        if user_count is None or mode == "unknown":
            continue
        
        # 提取能耗数据
        energy_value = extract_energy_from_stdout(file_path)
        if energy_value is not None:
            if mode not in data:
                data[mode] = {}
            data[mode][user_count] = energy_value
            print(f"模式: {mode}, 用户数: {user_count}, 平均能耗: {energy_value:.2f}")
    
    if not data:
        print("未能提取有效数据")
        return
    
    # 绘制图表
    plt.figure(figsize=(12, 8))
    plt.title("Figure 5: Total energy consumption under different values of the number of users")
    plt.xlabel("The number of users")
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
    
    # 检查是否所有模式都有完整的用户数量数据
    all_user_counts = set()
    for mode, user_counts in data.items():
        all_user_counts.update(user_counts.keys())
    
    print(f"发现的用户数量: {sorted(all_user_counts)}")
    for mode in data:
        missing_counts = all_user_counts - set(data[mode].keys())
        if missing_counts:
            print(f"警告: 模式 {mode} 缺少用户数量数据: {sorted(missing_counts)}")
            
    # 绘制每种模式的数据
    for mode, user_counts in data.items():
        if not user_counts:
            continue
            
        # 按用户数量排序
        sorted_items = sorted(user_counts.items())
        x_values = [item[0] for item in sorted_items]
        y_values = [item[1] for item in sorted_items]
        
        color, style = styles.get(mode, ("black", "-"))
        display_name = mode_display_names.get(mode, mode)
        plt.plot(x_values, y_values, style, color=color, label=display_name, linewidth=2)
    
    # 设置x轴的刻度为整数
    plt.xticks(sorted(list(all_user_counts)))
    
    # 设置图例和保存
    plt.legend(loc='best')
    output_dir = "figures"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "fig5_user_number_comparison.png"), dpi=300, bbox_inches='tight')
    print(f"图表已保存至 {output_dir}/fig5_user_number_comparison.png")
    
    # 显示图表
    plt.show()

if __name__ == "__main__":
    main() 