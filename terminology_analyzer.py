#!/usr/bin/env python3
"""
ML教材术语一致性分析工具
"""

import os
import re
import json
from collections import defaultdict, Counter
from pathlib import Path

# 工作目录
WORK_DIR = "/root/.openclaw/workspace/ml-book-for-kids"

# 核心术语定义（中英文对照）
CORE_TERMS = {
    # 基础术语
    "机器学习": ["机器学习", "machine learning", "ml"],
    "深度学习": ["深度学习", "deep learning"],
    "人工智能": ["人工智能", "artificial intelligence", "ai"],
    
    # 神经网络相关
    "神经网络": ["神经网络", "神经网路", "neural network", "neural networks"],
    "神经元": ["神经元", "neuron"],
    "感知机": ["感知机", "感知器", "perceptron"],
    "多层感知机": ["多层感知机", "多层感知器", "mlp", "multilayer perceptron"],
    
    # 核心算法
    "梯度下降": ["梯度下降", "梯度下將", "gradient descent"],
    "反向传播": ["反向传播", "反向傳播", "backpropagation", "back propagation", "bp"],
    "损失函数": ["损失函数", "loss function", "cost function", "代价函数", "目标函数"],
    "激活函数": ["激活函数", "activation function", "激励函数"],
    "优化器": ["优化器", "optimizer", "optimiser"],
    "学习率": ["学习率", "learning rate"],
    
    # 正则化
    "正则化": ["正则化", "regularization", "regularisation"],
    "过拟合": ["过拟合", "overfitting", "过适配"],
    "欠拟合": ["欠拟合", "underfitting", "欠适配"],
    "dropout": ["dropout", "随机失活"],
    
    # 网络架构
    "卷积神经网络": ["卷积神经网络", "cnn", "convolutional neural network"],
    "循环神经网络": ["循环神经网络", "rnn", "recurrent neural network"],
    "长短期记忆": ["长短期记忆", "lstm", "long short-term memory"],
    "transformer": ["transformer", "变换器", "转换器"],
    "注意力机制": ["注意力机制", "attention", "attention mechanism"],
    "自注意力": ["自注意力", "self-attention"],
    
    # 数据相关
    "嵌入": ["嵌入", "embedding", "词嵌入", "word embedding"],
    "特征": ["特征", "feature"],
    "标签": ["标签", "label"],
    "训练集": ["训练集", "training set"],
    "验证集": ["验证集", "validation set"],
    "测试集": ["测试集", "test set"],
    "批量": ["批量", "batch", "批次"],
    "epoch": ["epoch", "轮次", "轮"],
    "超参数": ["超参数", "hyperparameter"],
    
    # 应用
    "推理": ["推理", "inference"],
    "部署": ["部署", "deployment"],
    "微调": ["微调", "fine-tuning", "finetuning"],
    "预训练": ["预训练", "pretraining", "pre-training"],
    
    # 其他常用
    "卷积": ["卷积", "convolution"],
    "池化": ["池化", "pooling"],
    "全连接": ["全连接", "fully connected", "dense"],
    "softmax": ["softmax"],
    "relu": ["relu"],
    "归一化": ["归一化", "normalization", "batch normalization", "bn", "层归一化"],
}

# 易混淆/常见错误术语
AMBIGUOUS_TERMS = {
    "神经网路": "神经网络",  # 错误写法
    "梯度下將": "梯度下降",  # 错误写法  
    "反向傳播": "反向传播",  # 繁体
    "感知器": "感知机",      # 变体
    "激励函数": "激活函数",  # 变体
    "代价函数": "损失函数",  # 变体
    "目标函数": "损失函数",  # 变体
    "过适配": "过拟合",      # 变体
    "欠适配": "欠拟合",      # 变体
    "随机失活": "dropout",   # 需要统一
    "变换器": "transformer", # 需要统一
    "转换器": "transformer", # 需要统一
    "词嵌入": "embedding",   # 变体
    "批次": "批量",          # 需要统一
    "轮": "epoch",          # 需要统一
    "finetuning": "fine-tuning",
    "pretraining": "pre-training",
    "regularisation": "regularization",
    "optimiser": "optimizer",
}

def find_all_markdown_files(work_dir):
    """查找所有markdown文件"""
    md_files = []
    for root, dirs, files in os.walk(work_dir):
        # 排除一些目录
        dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__', 'node_modules', 'deprecated']]
        for file in files:
            if file.endswith('.md') and ('chapter' in file.lower() or 'chapter' in root.lower()):
                md_files.append(os.path.join(root, file))
    return sorted(set(md_files))

def extract_text_from_markdown(file_path):
    """从markdown文件中提取文本内容"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return content
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return ""

def clean_text(text):
    """清理文本：去除代码块、链接等"""
    # 去除代码块
    text = re.sub(r'```[\s\S]*?```', '', text)
    text = re.sub(r'`[^`]*`', '', text)
    # 去除markdown链接
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    # 去除html标签
    text = re.sub(r'<[^>]+>', '', text)
    # 去除图片
    text = re.sub(r'!\[([^\]]*)\]\([^\)]+\)', '', text)
    return text

def count_term_frequencies(text, terms_dict):
    """统计术语频率"""
    results = defaultdict(lambda: defaultdict(int))
    text_lower = text.lower()
    
    for standard_name, variants in terms_dict.items():
        for variant in variants:
            # 使用单词边界匹配英文，灵活匹配中文
            if re.match(r'^[a-zA-Z]', variant):  # 英文
                pattern = r'\b' + re.escape(variant.lower()) + r'\b'
            else:  # 中文
                pattern = re.escape(variant)
            
            matches = re.findall(pattern, text_lower)
            if matches:
                results[standard_name][variant] += len(matches)
    
    return results

def detect_inconsistencies(file_results):
    """检测术语不一致"""
    inconsistencies = []
    
    for file_path, term_counts in file_results.items():
        for standard_term, variants in term_counts.items():
            # 检查是否有多个变体同时出现
            active_variants = {k: v for k, v in variants.items() if v > 0}
            if len(active_variants) > 1:
                # 检查是否有错误写法
                has_error = any(v in AMBIGUOUS_TERMS for v in active_variants.keys())
                inconsistencies.append({
                    'file': file_path,
                    'standard_term': standard_term,
                    'variants': active_variants,
                    'has_error': has_error
                })
    
    return inconsistencies

def generate_frequency_table(file_results):
    """生成术语频率统计表"""
    total_counts = defaultdict(int)
    term_variants_used = defaultdict(set)
    
    for file_path, term_counts in file_results.items():
        for standard_term, variants in term_counts.items():
            for variant, count in variants.items():
                if count > 0:
                    total_counts[standard_term] += count
                    term_variants_used[standard_term].add(variant)
    
    return total_counts, term_variants_used

def main():
    print("=" * 60)
    print("ML教材术语一致性分析")
    print("=" * 60)
    
    # 1. 查找所有markdown文件
    print("\n[1/5] 正在查找所有章节文件...")
    md_files = find_all_markdown_files(WORK_DIR)
    print(f"找到 {len(md_files)} 个markdown文件")
    
    # 2. 读取并分析每个文件
    print("\n[2/5] 正在分析术语使用情况...")
    file_results = {}
    all_text = ""
    
    for file_path in md_files:
        content = extract_text_from_markdown(file_path)
        if content:
            cleaned = clean_text(content)
            all_text += cleaned + "\n"
            file_results[file_path] = count_term_frequencies(cleaned, CORE_TERMS)
    
    # 3. 生成频率统计
    print("\n[3/5] 生成术语频率统计...")
    total_counts, variants_used = generate_frequency_table(file_results)
    
    # 4. 检测不一致
    print("\n[4/5] 检测术语不一致...")
    inconsistencies = detect_inconsistencies(file_results)
    
    # 5. 生成报告
    print("\n[5/5] 生成报告...")
    
    # 输出结果
    print("\n" + "=" * 60)
    print("术语使用频率统计 (Top 30)")
    print("=" * 60)
    sorted_terms = sorted(total_counts.items(), key=lambda x: x[1], reverse=True)[:30]
    for term, count in sorted_terms:
        variants_str = ", ".join(variants_used[term])
        print(f"{term:20s} | {count:5d} 次 | 变体: {variants_str}")
    
    print("\n" + "=" * 60)
    print("术语不一致问题")
    print("=" * 60)
    if inconsistencies:
        # 按文件分组
        by_file = defaultdict(list)
        for inc in inconsistencies:
            by_file[inc['file']].append(inc)
        
        for file_path, issues in sorted(by_file.items()):
            print(f"\n📄 {file_path}")
            for issue in issues:
                variants_str = ", ".join([f"'{k}'({v}次)" for k, v in issue['variants'].items()])
                error_mark = " ⚠️ 包含错误写法" if issue['has_error'] else ""
                print(f"   • {issue['standard_term']}: {variants_str}{error_mark}")
    else:
        print("未发现术语不一致问题")
    
    # 检查特定错误写法
    print("\n" + "=" * 60)
    print("特定错误写法检查")
    print("=" * 60)
    error_patterns = {
        "神经网路": "神经网络",
        "梯度下將": "梯度下降",
        "反向傳播": "反向传播",
    }
    for error, correct in error_patterns.items():
        count = all_text.lower().count(error.lower())
        if count > 0:
            print(f"❌ 发现错误 '{error}' (应改为 '{correct}') 出现 {count} 次")
    
    # 保存详细报告
    report = {
        "summary": {
            "total_files": len(md_files),
            "total_terms_tracked": len(CORE_TERMS),
            "inconsistencies_found": len(inconsistencies)
        },
        "frequency_table": dict(sorted(total_counts.items(), key=lambda x: x[1], reverse=True)),
        "variants_used": {k: list(v) for k, v in variants_used.items()},
        "inconsistencies": inconsistencies,
        "files_analyzed": md_files
    }
    
    report_path = os.path.join(WORK_DIR, "terminology_analysis_report.json")
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"\n📊 详细报告已保存至: {report_path}")
    
    return report

if __name__ == "__main__":
    report = main()
