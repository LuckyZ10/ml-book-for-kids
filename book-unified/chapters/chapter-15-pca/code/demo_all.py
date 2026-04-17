"""
《机器学习与深度学习：从小学生到大师》
第十五章：降维——抓住主要矛盾
综合演示脚本

运行所有演示：
    python demo_all.py

单独运行：
    python pca_numpy.py
    python pca_torch.py  
    python tsne_numpy.py
"""

import subprocess
import sys

def run_demo(script_name, description):
    """运行单个演示脚本"""
    print(f"\n{'='*70}")
    print(f"运行: {description}")
    print(f"脚本: {script_name}")
    print('='*70)
    
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            capture_output=False,
            text=True,
            timeout=300  # 5分钟超时
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"错误: {script_name} 超时")
        return False
    except Exception as e:
        print(f"错误: {e}")
        return False

def main():
    """主函数：运行所有演示"""
    print("="*70)
    print("第十五章：降维综合演示")
    print("="*70)
    
    demos = [
        ("pca_numpy.py", "PCA NumPy实现 - 标准PCA、增量PCA、核PCA"),
        ("pca_torch.py", "PCA PyTorch实现 - GPU加速、神经网络集成"),
        ("tsne_numpy.py", "t-SNE NumPy实现 - 非线性降维可视化"),
    ]
    
    results = []
    for script, desc in demos:
        success = run_demo(script, desc)
        results.append((script, success))
    
    # 总结
    print("\n" + "="*70)
    print("演示总结")
    print("="*70)
    
    for script, success in results:
        status = "✅ 成功" if success else "❌ 失败"
        print(f"{script}: {status}")
    
    all_success = all(success for _, success in results)
    print("\n" + ("🎉 所有演示成功完成！" if all_success else "⚠️ 部分演示失败"))

if __name__ == "__main__":
    main()
