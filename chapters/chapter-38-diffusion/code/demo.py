"""
38.8 代码实现详解

本节提供完整的DDPM代码实现，包含：
1. 简化的UNet架构
2. DDPM训练和采样器
3. 完整的训练流程
4. 可视化工具

文件说明：
- ddpm_from_scratch.py: 核心DDPM实现
- train_ddpm.py: 训练脚本
- demo.py: 演示脚本（本文件）
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

# 演示代码可以直接运行
print("扩散模型代码实现")
print("=" * 50)
print("\n核心组件：")
print("1. SimpleUNet - 噪声预测网络")
print("2. DDPM - 训练和采样器")
print("3. DiffusionTrainer - 训练流程封装")
print("\n使用方法：")
print("  python ddpm_from_scratch.py  # 测试核心实现")
print("  python train_ddpm.py         # 完整训练流程")
print("  python train_ddpm.py sample  # 生成样本")
