#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
import re
from glob import glob

def extract_deviation_tolerance(filename):
    """从文件名中提取最大偏差容忍度参数"""
    basename = os.path.basename(filename)
    # 尝试多种模式匹配偏差容忍度值
    patterns = [
        r'Deviation_tolerance_(\d+\.\d+)',
        r'Deviation_tolerance_0?(\d+)',  # 可能的整数格式
        r'D_tolerance_(\d+\.\d+)',
        r'tolerance_(\d+\.\d+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, basename)
        if match:
            return float(match.group(1))
    return None

def extract_mode(filename):
    """从文件名中提取模式参数"""
    basename = os.path.basename(filename)
    if "_full_" in basename:
        return "full"
    elif "_random_all_" in basename:
        return "random_all"
    elif "_random_ua_" in basename:
        return "random_ua"
    elif "_random_rec_" in basename:
        return "random_rec"
    elif "_random_ca_" in basename:
        return "random_ca"
    return "unknown"

def extract_energy_from_stdout(file_path):
    """从stdout文件中提取最后100个episode的平均能耗值"""
    try:
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
        else:
            print(f"警告: 在文件 {file_path} 中找不到能耗数据")
    except Exception as e:
        print(f"读取文件 {file_path} 时出错: {str(e)}")
    return None

def main():
    # 查找所有Figure 9相关的stdout文件
    # 首先检查fig9_deviation_tolerance目录
    possible_dirs = [
        "experiment_results_20250414_154215/fig9_deviation_tolerance",
        "experiment_results_20250414_115106/fig9_deviation_tolerance",
        "experiment_results_20250414_154215/fig_9_deviation_tolerance",
        "experiment_results_20250414_115106/fig_9_deviation_tolerance"
    ]
    
    stdout_files = []
    results_dir = None
    
    for dir_path in possible_dirs:
        if os.path.exists(dir_path):
            results_dir = dir_path
            stdout_files = glob(os.path.join(dir_path, "*_stdout.txt"))
            if stdout_files:
                break
    
    if not stdout_files:
        print("未找到与Figure 9相关的stdout文件，尝试在上级目录中查找")
        # 尝试在实验结果目录中查找包含'Deviation_tolerance'的文件
        for base_dir in ["experiment_results_20250414_154215", "experiment_results_20250414_115106"]:
            if os.path.exists(base_dir):
                for root, dirs, files in os.walk(base_dir):
                    for file in files:
                        if "_stdout.txt" in file and "Deviation_tolerance" in file:
                            stdout_files.append(os.path.join(root, file))
                if stdout_files:
                    results_dir = base_dir
                    break
    
    # 组织数据结构: {mode: {tolerance: energy_value}}
    data = {}
    mode_display_names = {
        "full": "DDQN-based CA, UA and REC",
        "random_all": "Random CA, UA and REC",
        "random_ua": "DDQN-based CA and REC only",
        "random_rec": "DDQN-based CA and UA only",
        "random_ca": "DDQN-based UA and REC only"
    }
    
    # 预期的偏差容忍度值列表，用于图表中的对照
    expected_tolerances = [0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13]
    
    if stdout_files:
        print(f"找到 {len(stdout_files)} 个Figure 9相关的stdout文件")
        
        # 输出所有文件以便调试
        for file_path in stdout_files:
            print(f"文件: {file_path}")
            
        for file_path in stdout_files:
            tolerance = extract_deviation_tolerance(file_path)
            mode = extract_mode(file_path)
            
            print(f"分析文件: {os.path.basename(file_path)}, 偏差容忍度: {tolerance}, 模式: {mode}")
            
            if tolerance is None or mode == "unknown":
                print(f"  无法提取有效的偏差容忍度或模式: {os.path.basename(file_path)}")
                continue
            
            # 提取能耗数据
            energy_value = extract_energy_from_stdout(file_path)
            if energy_value is not None:
                if mode not in data:
                    data[mode] = {}
                data[mode][tolerance] = energy_value
                print(f"  模式: {mode}, 偏差容忍度: {tolerance}, 平均能耗: {energy_value:.2f}")
            else:
                print(f"  未能从文件提取能耗数据: {os.path.basename(file_path)}")
    else:
        print("未找到任何Figure 9相关的stdout文件")
    
    if not data:
        print("未能从实际数据文件中提取能耗数据，程序退出")
        return
    
    # 绘制图表
    plt.figure(figsize=(10, 6))
    plt.title("Figure 9: Total energy consumption under different values of the maximum deviation tolerance")
    plt.xlabel("The maximum deviation tolerance")
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
    
    # 获取所有偏差容忍度值
    all_tolerances = set()
    for mode, tolerances in data.items():
        all_tolerances.update(tolerances.keys())
    
    print(f"发现的偏差容忍度值: {sorted(all_tolerances)}")
    for mode in data:
        missing_tolerances = sorted([tol for tol in expected_tolerances if tol not in data[mode]])
        if missing_tolerances:
            print(f"警告: 模式 {mode} 缺少偏差容忍度数据: {missing_tolerances}")
    
    # 绘制每种模式的数据
    for mode, tolerances in data.items():
        if not tolerances:
            continue
            
        # 按偏差容忍度排序
        sorted_items = sorted(tolerances.items())
        x_values = [item[0] for item in sorted_items]
        y_values = [item[1] for item in sorted_items]
        
        color, style = styles.get(mode, ("black", "-"))
        display_name = mode_display_names.get(mode, mode)
        plt.plot(x_values, y_values, style, color=color, label=display_name, linewidth=2)
    
    # 设置x轴的刻度为预期的偏差容忍度值
    plt.xticks(sorted(list(all_tolerances)))
    
    # 设置图例和保存
    plt.legend(loc='best')
    output_dir = "figures"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "fig9_deviation_tolerance_comparison.png"), dpi=300, bbox_inches='tight')
    print(f"图表已保存至 {output_dir}/fig9_deviation_tolerance_comparison.png")
    
    # 显示图表
    plt.show()

if __name__ == "__main__":
    main() 