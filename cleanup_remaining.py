#!/usr/bin/env python3
"""
Second pass - clean up remaining loose files in chapters/ directory
"""

import os
import shutil
from pathlib import Path

BASE_DIR = Path("/root/.openclaw/workspace/ml-book-for-kids")
CHAPTERS_DIR = BASE_DIR / "chapters"
DEPRECATED_DIR = BASE_DIR / "deprecated"

def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)
    return path

def get_chapter_number(filename):
    import re
    match = re.search(r'chapter[-_]?0?(\d+)', filename.lower())
    if match:
        try:
            return int(match.group(1))
        except:
            pass
    return None

def cleanup_remaining_files():
    """清理 chapters/ 目录中剩余的非规范文件"""
    
    print("=" * 60)
    print("第二阶段 - 清理剩余零散文件")
    print("=" * 60)
    
    # 章节主题映射（从 BOOK_TOC.md 提取）
    CHAPTER_TOPICS = {
        1: "what-is-learning",
        2: "seeing-data",
        3: "prediction-and-loss",
        4: "gradient-descent",
        5: "python-warmup",
        6: "knn",
        7: "linear-regression",
        8: "logistic-regression",
        9: "decision-tree",
        10: "svm",
        11: "naive-bayes",
        12: "ensemble-learning",
        13: "kmeans-clustering",
        14: "hierarchical-clustering",
        15: "dimensionality-reduction",
        16: "perceptron",
        17: "multilayer-neural-network",
        18: "backpropagation",
        19: "activation-functions",
        20: "optimizers",
        21: "regularization",
        22: "cnn",
        23: "rnn",
        24: "attention-transformer",
        25: "pretraining-finetuning",
        26: "llm-prompting",
        27: "rag",
        28: "multimodal-learning",
        29: "generative-models",
        30: "reinforcement-learning",
        31: "deep-rl-advanced",
        32: "gnn",
        33: "temporal-forecasting",
        34: "nas-basics",
        35: "self-supervised-learning",
        36: "federated-learning",
        37: "nas-advanced",
        38: "diffusion-models",
        39: "3d-vision-nerf",
        40: "rl-frontier",
        41: "xai",
        42: "diffusion-advanced",
        43: "multimodal-frontier",
        44: "ai-agents",
        45: "uncertainty-quantification",
        46: "neuro-symbolic",
        47: "bayesian-optimization",
        48: "deep-ensemble",
        49: "physics-informed-ml",
        50: "probabilistic-graphical-models",
        51: "causal-inference",
        52: "generative-models-advanced",
        53: "gnn-geometric",
        54: "neuro-symbolic-advanced",
        55: "continual-learning",
        56: "nas-hardware-aware",
        57: "meta-learning",
        58: "few-shot-learning",
        59: "mlops",
        60: "complete-project",
    }
    
    moved = 0
    
    for item in list(CHAPTERS_DIR.iterdir()):
        name = item.name
        
        # 跳过已经是规范格式的目录
        if item.is_dir() and name.startswith('chapter-') and len(name.split('-')) >= 3:
            # 检查是否有 code/ 子目录
            code_dir = item / "code"
            if not code_dir.exists():
                ensure_dir(code_dir)
            continue
        
        # 处理文件
        if item.is_file():
            num = get_chapter_number(name)
            if num:
                topic = CHAPTER_TOPICS.get(num, f"topic-{num}")
                target_dir = CHAPTERS_DIR / f"chapter-{num:02d}-{topic}"
                ensure_dir(target_dir)
                code_dir = ensure_dir(target_dir / "code")
                
                if name.endswith('.py'):
                    dst = code_dir / name
                elif name.endswith('.md'):
                    dst = target_dir / "README.md"
                else:
                    dst = DEPRECATED_DIR / name
                
                if not dst.exists():
                    shutil.copy2(item, dst)
                    print(f"[COPY] {name} -> {dst.relative_to(BASE_DIR)}")
                
                # 移动原文件到 deprecated
                deprecated_dst = DEPRECATED_DIR / name
                if not deprecated_dst.exists():
                    shutil.move(str(item), str(deprecated_dst))
                    print(f"[MOVE] {name} -> deprecated/")
                    moved += 1
        
        # 处理不规范的目录
        elif item.is_dir():
            num = get_chapter_number(name)
            if num:
                topic = CHAPTER_TOPICS.get(num, f"topic-{num}")
                target_dir = CHAPTERS_DIR / f"chapter-{num:02d}-{topic}"
                
                # 如果目标目录已存在，合并内容
                if target_dir.exists():
                    ensure_dir(target_dir / "code")
                    
                    # 复制内容
                    for subitem in item.iterdir():
                        if subitem.is_file():
                            if subitem.suffix == '.py':
                                dst = target_dir / "code" / subitem.name
                            elif subitem.name == 'README.md':
                                dst = target_dir / subitem.name
                            elif subitem.suffix == '.md':
                                dst = target_dir / subitem.name
                            else:
                                dst = DEPRECATED_DIR / subitem.name
                            
                            if not dst.exists():
                                shutil.copy2(subitem, dst)
                                print(f"[COPY] {name}/{subitem.name} -> {dst.relative_to(BASE_DIR)}")
                    
                    # 移动旧目录到 deprecated
                    deprecated_dst = DEPRECATED_DIR / name
                    if not deprecated_dst.exists():
                        shutil.move(str(item), str(deprecated_dst))
                        print(f"[MOVE] {name}/ -> deprecated/")
                        moved += 1
                else:
                    # 重命名为规范格式
                    ensure_dir(target_dir)
                    code_dir = ensure_dir(target_dir / "code")
                    
                    for subitem in item.iterdir():
                        if subitem.is_file():
                            if subitem.suffix == '.py':
                                dst = code_dir / subitem.name
                            elif subitem.name == 'README.md':
                                dst = target_dir / subitem.name
                            elif subitem.suffix == '.md':
                                dst = target_dir / subitem.name
                            else:
                                dst = DEPRECATED_DIR / subitem.name
                            
                            if not dst.exists():
                                shutil.copy2(subitem, dst)
                                print(f"[COPY] {name}/{subitem.name} -> {dst.relative_to(BASE_DIR)}")
                    
                    # 移动旧目录到 deprecated
                    deprecated_dst = DEPRECATED_DIR / name
                    if not deprecated_dst.exists():
                        shutil.move(str(item), str(deprecated_dst))
                        print(f"[MOVE] {name}/ -> deprecated/")
                        moved += 1
    
    print(f"\n第二阶段完成，移动了 {moved} 个文件/目录")
    return moved

if __name__ == "__main__":
    cleanup_remaining_files()
