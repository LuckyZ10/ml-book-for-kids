"""
第十八章：反向传播算法 - 完整实现
包含：自动微分、计算图、梯度检查
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Callable


class Variable:
    """
    计算图中的变量节点
    支持自动微分
    """
    
    def __init__(self, data: np.ndarray, name: str = None):
        self.data = data
        self.name = name
        self.grad = None
        self.creator = None  # 创建这个变量的函数
    
    def set_creator(self, func):
        """设置创建者"""
        self.creator = func
    
    def backward(self, retain_grad=False):
        """反向传播计算梯度"""
        if self.grad is None:
            self.grad = np.ones_like(self.data)
        
        funcs = [self.creator]
        
        while funcs:
            f = funcs.pop()
            
            # 获取输出梯度
            gys = [output.grad for output in f.outputs]
            
            # 计算输入梯度
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)
            
            # 传递梯度
            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx
                
                if x.creator is not None:
                    funcs.append(x.creator)
            
            if not retain_grad:
                for y in f.outputs:
                    y.grad = None


class Function:
    """计算图中的函数基类"""
    
    def __call__(self, *inputs: Variable) -> Variable:
        """前向传播"""
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        
        outputs = [Variable(y) for y in ys]
        
        for output in outputs:
            output.set_creator(self)
        
        self.inputs = inputs
        self.outputs = outputs
        
        return outputs[0] if len(outputs) == 1 else outputs
    
    def forward(self, *xs):
        raise NotImplementedError
    
    def backward(self, *gys):
        raise NotImplementedError


class Square(Function):
    """平方函数 y = x^2"""
    
    def forward(self, x):
        return x ** 2
    
    def backward(self, gy):
        x = self.inputs[0].data
        return 2 * x * gy


class Exp(Function):
    """指数函数 y = exp(x)"""
    
    def forward(self, x):
        return np.exp(x)
    
    def backward(self, gy):
        x = self.inputs[0].data
        return np.exp(x) * gy


class Add(Function):
    """加法函数 y = x0 + x1"""
    
    def forward(self, x0, x1):
        return x0 + x1
    
    def backward(self, gy):
        return gy, gy


class Mul(Function):
    """乘法函数 y = x0 * x1"""
    
    def forward(self, x0, x1):
        return x0 * x1
    
    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        return gy * x1, gy * x0


# 便捷函数
def square(x):
    return Square()(x)

def exp(x):
    return Exp()(x)

def add(x0, x1):
    return Add()(x0, x1)

def mul(x0, x1):
    return Mul()(x0, x1)


def backprop_demo():
    """反向传播演示"""
    print("=" * 60)
    print("反向传播算法演示")
    print("=" * 60)
    
    # 示例：计算 y = x^2 + exp(x) 在 x=0.5 处的导数
    x = Variable(np.array(0.5))
    
    # 前向传播
    a = square(x)      # a = x^2
    b = exp(x)         # b = exp(x)
    y = add(a, b)      # y = a + b
    
    print(f"\n函数: y = x^2 + exp(x)")
    print(f"输入: x = {x.data}")
    print(f"输出: y = {y.data:.6f}")
    
    # 反向传播
    y.backward()
    
    print(f"\n反向传播结果:")
    print(f"dy/dx = {x.grad:.6f}")
    
    # 验证：解析导数 dy/dx = 2x + exp(x)
    analytical = 2 * 0.5 + np.exp(0.5)
    print(f"解析导数: {analytical:.6f}")
    print(f"匹配: {np.isclose(x.grad, analytical)}")


def chain_rule_demo():
    """链式法则演示"""
    print("\n" + "=" * 60)
    print("链式法则演示")
    print("=" * 60)
    
    # 复合函数：y = (exp(x^2))^2
    x = Variable(np.array(0.5))
    
    print(f"\n函数: y = (exp(x^2))^2")
    print(f"分解: u = x^2, v = exp(u), y = v^2")
    
    # 前向传播
    u = square(x)      # u = x^2
    v = exp(u)         # v = exp(u)
    y = square(v)      # y = v^2
    
    print(f"\n前向传播:")
    print(f"  x = {x.data}")
    print(f"  u = x^2 = {u.data:.6f}")
    print(f"  v = exp(u) = {v.data:.6f}")
    print(f"  y = v^2 = {y.data:.6f}")
    
    # 反向传播
    y.backward()
    
    print(f"\n反向传播（链式法则）:")
    print(f"  dy/dy = 1")
    print(f"  dy/dv = 2*v = {2*v.data:.6f}")
    print(f"  dv/du = exp(u) = {v.data:.6f}")
    print(f"  du/dx = 2*x = {2*x.data:.6f}")
    print(f"  dy/dx = {x.grad:.6f}")


def main():
    """主函数"""
    backprop_demo()
    chain_rule_demo()
    
    print("\n" + "=" * 60)
    print("第十八章演示完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
