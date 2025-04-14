# EdgeCache-DDQN: Edge Caching and Recommendation System with Deep Reinforcement Learning

[English](#english) | [中文](#chinese)

<a name="english"></a>
## Edge Caching and Recommendation System with Deep Reinforcement Learning

This project implements a Deep Reinforcement Learning (DRL) solution for joint edge caching, content recommendation, and user association in wireless networks. It uses a Dueling Double Deep Q-Network (DDQN) to optimize network performance, reduce energy consumption, and improve user experience.

### Features

- Joint optimization of caching, recommendation, and user association policies
- Advanced environment modeling with Finite State Markov Chain (FSMC) for user preferences and wireless channel states
- Support for Scalable Video Coding (SVC) with multi-layer video transmission
- Energy consumption modeling with distinction between local caching and backhaul fetching
- Preference deviation penalty to maintain user satisfaction

### Components

- `ddqn.py`: Implementation of the Dueling DDQN agent
- `env.py`: Basic environment implementation
- `advanced_env.py`: Advanced environment with detailed system modeling
- `replay_buffer.py`: Experience replay buffer for DDQN training
- `train.py`: Main training script with various training modes
- `plot_results.py` and `plot_compare.py`: Utilities for visualizing training results
- `test.py`: Testing script for evaluating trained models

### Training Modes

- `full`: Full DDQN-based decision making for all policies
- `random_all`: Random decisions for all policies
- `random_ua`: Random user association with DDQN-based caching and recommendation
- `random_rec`: Random recommendations with DDQN-based caching and user association
- `random_ca`: Random caching with DDQN-based recommendation and user association

### Usage

Train a model with default configuration:
```bash
python train.py
```

Train with a specific configuration and mode:
```bash
python train.py --config your_config.json --mode full
```

Test a trained model:
```bash
python test.py --model models/your_model.pth
```

Visualize results:
```bash
python plot_results.py --metrics models/your_model_metrics.npz
```

### Requirements

- Python 3.6+
- PyTorch
- NumPy
- Matplotlib (for plotting)

### Configuration

The configuration file (default: `config.json`) allows you to customize parameters such as:
- Network topology (M base stations, N users)
- Content library (F videos, K layers per video)
- Caching and recommendation capacities
- Learning parameters (discount factor, learning rate, etc.)
- Environment parameters (backhaul bandwidth, wireless bandwidth, etc.)

<a name="chinese"></a>
## 边缘缓存与推荐系统的深度强化学习实现

本项目实现了一种基于深度强化学习（DRL）的解决方案，用于无线网络中的联合边缘缓存、内容推荐和用户关联优化。它使用对偶深度Q网络（Dueling DDQN）来优化网络性能，降低能源消耗，并提升用户体验。

### 功能特点

- 缓存策略、推荐策略和用户关联策略的联合优化
- 使用有限状态马尔可夫链（FSMC）建模用户偏好和无线信道状态
- 支持可伸缩视频编码（SVC）的多层视频传输
- 能源消耗建模，区分本地缓存和回程获取的能耗差异
- 偏好偏离度惩罚机制，维持用户满意度

### 组件结构

- `ddqn.py`：对偶深度Q网络（Dueling DDQN）智能体实现
- `env.py`：基础环境实现
- `advanced_env.py`：高级环境实现，包含详细的系统建模
- `replay_buffer.py`：DDQN训练所需的经验回放缓冲区
- `train.py`：主训练脚本，支持多种训练模式
- `plot_results.py` 和 `plot_compare.py`：训练结果可视化工具
- `test.py`：用于评估训练模型的测试脚本

### 训练模式

- `full`：所有策略均由DDQN完全决策
- `random_all`：所有策略均采用随机决策
- `random_ua`：随机用户关联，DDQN决策缓存和推荐
- `random_rec`：随机推荐，DDQN决策缓存和用户关联
- `random_ca`：随机缓存，DDQN决策推荐和用户关联

### 使用方法

使用默认配置训练模型：
```bash
python train.py
```

使用特定配置和模式训练：
```bash
python train.py --config your_config.json --mode full
```

测试训练好的模型：
```bash
python test.py --model models/your_model.pth
```

可视化结果：
```bash
python plot_results.py --metrics models/your_model_metrics.npz
```

### 环境要求

- Python 3.6+
- PyTorch
- NumPy
- Matplotlib（用于绘图）

### 配置说明

配置文件（默认：`config.json`）允许自定义以下参数：
- 网络拓扑（M个基站，N个用户）
- 内容库（F个视频，每个视频K层）
- 缓存和推荐容量
- 学习参数（折扣因子，学习率等）
- 环境参数（回程带宽，无线带宽等）
