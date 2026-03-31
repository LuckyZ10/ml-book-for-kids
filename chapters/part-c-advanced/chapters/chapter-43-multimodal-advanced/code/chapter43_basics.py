"""
第43章：基础代码实现
包含核心概念演示和基础算法
"""

import numpy as np
import matplotlib.pyplot as plt


def demo():
    """基础演示"""
    print("=" * 60)
    print("第43章：基础演示")
    print("=" * 60)
    
    # 基础示例
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    
    plt.figure(figsize=(8, 4))
    plt.plot(x, y)
    plt.title("基础示例")
    plt.grid(True)
    plt.savefig('demo.png')
    plt.show()
    
    print("演示完成！")


if __name__ == "__main__":
    demo()
