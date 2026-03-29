#!/usr/bin/env python3
"""
ML Book for Kids - Chapter Structure Organizer
整理 ml-book-for-kids 教材的章节结构
"""

import os
import shutil
import json
from pathlib import Path
from collections import defaultdict

# 基础配置
BASE_DIR = Path("/root/.openclaw/workspace/ml-book-for-kids")
CHAPTERS_DIR = BASE_DIR / "chapters"
DEPRECATED_DIR = BASE_DIR / "deprecated"

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

def ensure_dir(path):
    """确保目录存在"""
    path.mkdir(parents=True, exist_ok=True)
    return path

def get_chapter_number(filename):
    """从文件名提取章节号"""
    import re
    # 匹配 chapter-XX, chapter_XX, chapterXX 等格式
    patterns = [
        r'chapter[-_]?0?(\d+)',
        r'chapter([-_]\d+)',
    ]
    for pattern in patterns:
        match = re.search(pattern, filename.lower())
        if match:
            try:
                return int(match.group(1))
            except:
                continue
    return None

def scan_all_files():
    """扫描所有章节相关文件"""
    files_map = defaultdict(list)
    
    # 扫描根目录
    for item in BASE_DIR.iterdir():
        if item.is_file():
            num = get_chapter_number(item.name)
            if num:
                files_map[num].append({
                    'path': item,
                    'type': 'file',
                    'source': 'root'
                })
        elif item.is_dir() and 'chapter' in item.name.lower():
            num = get_chapter_number(item.name)
            if num:
                files_map[num].append({
                    'path': item,
                    'type': 'dir',
                    'source': 'root'
                })
    
    # 扫描 chapters/ 目录
    if CHAPTERS_DIR.exists():
        for item in CHAPTERS_DIR.iterdir():
            num = get_chapter_number(item.name)
            if num:
                files_map[num].append({
                    'path': item,
                    'type': 'dir' if item.is_dir() else 'file',
                    'source': 'chapters'
                })
    
    return files_map

def organize_chapters():
    """主整理函数"""
    print("=" * 60)
    print("ML Book for Kids - 章节结构整理")
    print("=" * 60)
    
    # 1. 扫描所有文件
    print("\n[1/6] 扫描所有章节文件...")
    files_map = scan_all_files()
    print(f"发现 {len(files_map)} 个章节的相关文件")
    
    # 2. 确保 deprecated 目录存在
    ensure_dir(DEPRECATED_DIR)
    
    # 3. 创建新的规范目录结构
    print("\n[2/6] 创建规范目录结构...")
    organized_count = 0
    deprecated_count = 0
    
    for chapter_num in sorted(files_map.keys()):
        topic = CHAPTER_TOPICS.get(chapter_num, f"topic-{chapter_num}")
        target_dir = CHAPTERS_DIR / f"chapter-{chapter_num:02d}-{topic}"
        
        files = files_map[chapter_num]
        print(f"\n  章节 {chapter_num}: 发现 {len(files)} 个文件/目录")
        
        # 确保目标目录存在
        ensure_dir(target_dir)
        code_dir = ensure_dir(target_dir / "code")
        
        # 处理每个文件/目录
        for f in files:
            src = f['path']
            
            # 跳过已经在新位置的文件
            if 'chapter-' in src.name and src.parent == CHAPTERS_DIR:
                if src.is_dir() and src.name.startswith(f"chapter-{chapter_num:02d}"):
                    continue
            
            if f['type'] == 'file':
                # 根据文件类型决定处理方式
                if src.suffix == '.py':
                    # Python 代码移到 code/
                    dst = code_dir / src.name
                    if not dst.exists():
                        shutil.copy2(src, dst)
                        print(f"    [COPY] {src.name} -> code/")
                elif src.suffix == '.md':
                    # Markdown 文件作为主内容
                    if 'README' not in src.name:
                        dst = target_dir / "README.md"
                        if not dst.exists() or src.stat().st_size > dst.stat().st_size:
                            shutil.copy2(src, dst)
                            print(f"    [COPY] {src.name} -> README.md")
                else:
                    # 其他文件移到 deprecated
                    dst = DEPRECATED_DIR / src.name
                    if not dst.exists():
                        shutil.move(str(src), str(dst))
                        print(f"    [MOVE] {src.name} -> deprecated/")
                        deprecated_count += 1
            else:
                # 处理目录
                if src.name.startswith(f"chapter-{chapter_num:02d}") and src.parent == CHAPTERS_DIR:
                    continue  # 已经是规范目录
                
                # 检查目录内容
                if src.is_dir():
                    for subitem in src.iterdir():
                        if subitem.is_file():
                            if subitem.suffix == '.py':
                                dst = code_dir / subitem.name
                                if not dst.exists():
                                    shutil.copy2(subitem, dst)
                                    print(f"    [COPY] {src.name}/{subitem.name} -> code/")
                            elif subitem.suffix == '.md' and 'README' in subitem.name:
                                dst = target_dir / "README.md"
                                if not dst.exists():
                                    shutil.copy2(subitem, dst)
                                    print(f"    [COPY] {src.name}/{subitem.name} -> README.md")
                            elif subitem.suffix == '.md':
                                dst = target_dir / subitem.name
                                if not dst.exists():
                                    shutil.copy2(subitem, dst)
                                    print(f"    [COPY] {src.name}/{subitem.name}")
                    
                    # 移动旧目录到 deprecated
                    deprecated_chapter_dir = DEPRECATED_DIR / src.name
                    if not deprecated_chapter_dir.exists():
                        shutil.move(str(src), str(deprecated_chapter_dir))
                        print(f"    [MOVE] {src.name}/ -> deprecated/")
                        deprecated_count += 1
        
        organized_count += 1
    
    print(f"\n[3/6] 整理了 {organized_count} 个章节")
    print(f"[4/6] 移动了 {deprecated_count} 个废弃文件/目录")
    
    # 4. 清理根目录的零散文件
    print("\n[5/6] 清理根目录零散文件...")
    cleaned = 0
    for item in BASE_DIR.iterdir():
        if item.is_file() and 'chapter' in item.name.lower():
            num = get_chapter_number(item.name)
            if num:
                dst = DEPRECATED_DIR / item.name
                if not dst.exists():
                    shutil.move(str(item), str(dst))
                    cleaned += 1
                    print(f"  [MOVE] {item.name} -> deprecated/")
    
    print(f"清理了 {cleaned} 个根目录文件")
    
    # 5. 生成报告
    print("\n[6/6] 生成整理报告...")
    report = {
        "organized_chapters": organized_count,
        "deprecated_files": deprecated_count + cleaned,
        "chapter_list": sorted(files_map.keys()),
        "new_structure": []
    }
    
    for chapter_num in sorted(files_map.keys()):
        topic = CHAPTER_TOPICS.get(chapter_num, f"topic-{chapter_num}")
        chapter_dir = CHAPTERS_DIR / f"chapter-{chapter_num:02d}-{topic}"
        if chapter_dir.exists():
            contents = list(chapter_dir.iterdir())
            report["new_structure"].append({
                "chapter": chapter_num,
                "name": f"chapter-{chapter_num:02d}-{topic}",
                "contents": [c.name for c in contents]
            })
    
    # 保存报告
    report_file = BASE_DIR / "chapter_organization_report.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"\n报告已保存: {report_file}")
    
    return organized_count, deprecated_count + cleaned

if __name__ == "__main__":
    organized, deprecated = organize_chapters()
    print("\n" + "=" * 60)
    print("整理完成!")
    print(f"- 整理了 {organized} 个章节")
    print(f"- 移动了 {deprecated} 个重复/废弃文件")
    print("=" * 60)
