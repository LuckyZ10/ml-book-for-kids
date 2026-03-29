#!/usr/bin/env python3
"""
ML教材术语批量替换脚本
"""
import os
import re
import json
from pathlib import Path
from collections import defaultdict

WORK_DIR = "/root/.openclaw/workspace/ml-book-for-kids"

# 替换规则: (模式, 替换为, 描述)
REPLACEMENT_RULES = [
    # 1. 代价函数 -> 损失函数
    (r'(?<![\w])代价函数(?![\w])', '损失函数', '代价函数→损失函数'),
    
    # 2. 目标函数 -> 损失函数 (在ML语境中)
    # 注意：这里需要小心，数学优化语境中的"目标函数"应该保留
    # 我们只在明确ML语境的行中替换
    
    # 3. 批次 -> 批量
    (r'(?<![\w])批次(?![\w])', '批量', '批次→批量'),
    
    # 4. pretraining -> pre-training (各种大小写)
    (r'(?<![\w])Pretraining(?![\w])', 'Pre-training', 'Pretraining→Pre-training'),
    (r'(?<![\w])pretraining(?![\w])', 'pre-training', 'pretraining→pre-training'),
    
    # 5. finetuning -> fine-tuning
    (r'(?<![\w])Finetuning(?![\w])', 'Fine-tuning', 'Finetuning→Fine-tuning'),
    (r'(?<![\w])finetuning(?![\w])', 'fine-tuning', 'finetuning→fine-tuning'),
    
    # 6. 感知器 -> 感知机
    (r'(?<![\w])感知器(?![\w])', '感知机', '感知器→感知机'),
    
    # 7. 词嵌入 -> 嵌入
    (r'(?<![\w])词嵌入(?![\w])', '嵌入', '词嵌入→嵌入'),
]

# 目标函数的特殊处理 - 需要上下文判断
# 在训练、优化、机器学习语境中替换为"损失函数"
# 在数学规划语境中保留
OBJECT_FUNCTION_CONTEXTS = [
    r'训练.*目标函数',
    r'优化.*目标函数',
    r'最小化.*目标函数',
    r'损失.*目标函数',
    r'目标函数.*损失',
    r'目标函数.*优化',
    r'目标函数.*训练',
]

def find_all_md_files(work_dir):
    """查找所有markdown文件"""
    md_files = []
    for root, dirs, files in os.walk(work_dir):
        # 排除一些目录
        dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__', 'node_modules']]
        for file in files:
            if file.endswith('.md'):
                md_files.append(os.path.join(root, file))
    return sorted(md_files)

def replace_in_file(file_path, rules):
    """在文件中执行替换"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        return None, f"Error reading: {e}"
    
    original_content = content
    changes = []
    
    # 执行标准替换
    for pattern, replacement, desc in rules:
        matches = list(re.finditer(pattern, content))
        if matches:
            count = len(matches)
            content = re.sub(pattern, replacement, content)
            changes.append({
                'rule': desc,
                'count': count,
                'examples': [m.group(0) for m in matches[:3]]  # 记录前3个示例
            })
    
    # 特殊处理：目标函数 -> 损失函数（只在ML语境）
    # 简单策略：如果行中包含"训练"、"损失"、"优化"等关键词，则替换
    lines = content.split('\n')
    new_lines = []
    target_function_changes = 0
    
    for line in lines:
        original_line = line
        # 检查是否包含目标函数以及ML相关上下文
        if '目标函数' in line:
            is_ml_context = any(re.search(ctx, line) for ctx in OBJECT_FUNCTION_CONTEXTS)
            if is_ml_context:
                line = re.sub(r'目标函数', '损失函数', line)
                if line != original_line:
                    target_function_changes += 1
        new_lines.append(line)
    
    if target_function_changes > 0:
        content = '\n'.join(new_lines)
        changes.append({
            'rule': '目标函数→损失函数(ML语境)',
            'count': target_function_changes,
            'examples': ['目标函数→损失函数']
        })
    
    # 如果有修改，写回文件
    if content != original_content:
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return changes, None
        except Exception as e:
            return None, f"Error writing: {e}"
    
    return [], None  # 无修改

def main():
    print("=" * 60)
    print("ML教材术语批量替换")
    print("=" * 60)
    
    # 查找所有md文件
    print("\n[1/4] 查找所有Markdown文件...")
    md_files = find_all_md_files(WORK_DIR)
    print(f"找到 {len(md_files)} 个文件")
    
    # 执行替换
    print("\n[2/4] 执行术语替换...")
    results = []
    modified_count = 0
    
    for i, file_path in enumerate(md_files):
        if i % 20 == 0:
            print(f"  处理中... {i+1}/{len(md_files)}")
        
        changes, error = replace_in_file(file_path, REPLACEMENT_RULES)
        
        if error:
            results.append({
                'file': file_path,
                'status': 'error',
                'error': error
            })
        elif changes:
            results.append({
                'file': file_path,
                'status': 'modified',
                'changes': changes
            })
            modified_count += 1
        else:
            results.append({
                'file': file_path,
                'status': 'unchanged'
            })
    
    # 统计
    print("\n[3/4] 生成统计报告...")
    
    rule_stats = defaultdict(int)
    for r in results:
        if r['status'] == 'modified':
            for change in r['changes']:
                rule_stats[change['rule']] += change['count']
    
    # 输出统计
    print("\n" + "=" * 60)
    print("替换统计")
    print("=" * 60)
    print(f"总文件数: {len(md_files)}")
    print(f"修改文件数: {modified_count}")
    print(f"未修改文件数: {len(md_files) - modified_count}")
    print(f"错误文件数: {sum(1 for r in results if r['status'] == 'error')}")
    
    print("\n各规则替换次数:")
    for rule, count in sorted(rule_stats.items(), key=lambda x: x[1], reverse=True):
        print(f"  {rule}: {count} 次")
    
    # 保存日志
    print("\n[4/4] 保存日志...")
    
    log_data = {
        'summary': {
            'total_files': len(md_files),
            'modified_files': modified_count,
            'rule_stats': dict(rule_stats)
        },
        'modified_files': [r for r in results if r['status'] == 'modified'],
        'errors': [r for r in results if r['status'] == 'error']
    }
    
    log_path = os.path.join(WORK_DIR, 'replacement_log.json')
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, ensure_ascii=False, indent=2)
    print(f"详细日志: {log_path}")
    
    # 生成Markdown报告
    report_path = os.path.join(WORK_DIR, 'replacement_summary.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# 术语替换报告\n\n")
        f.write(f"**执行时间**: 2026-03-27\n\n")
        f.write("## 统计摘要\n\n")
        f.write(f"- **总文件数**: {len(md_files)}\n")
        f.write(f"- **修改文件数**: {modified_count}\n")
        f.write(f"- **替换总次数**: {sum(rule_stats.values())}\n\n")
        
        f.write("## 各规则替换次数\n\n")
        f.write("| 规则 | 次数 |\n|------|------|\n")
        for rule, count in sorted(rule_stats.items(), key=lambda x: x[1], reverse=True):
            f.write(f"| {rule} | {count} |\n")
        
        f.write("\n## 修改的文件列表\n\n")
        for r in log_data['modified_files'][:50]:  # 只显示前50个
            file_short = r['file'].replace(WORK_DIR + '/', '')
            f.write(f"### {file_short}\n")
            for change in r['changes']:
                f.write(f"- {change['rule']}: {change['count']} 次\n")
            f.write("\n")
        
        if len(log_data['modified_files']) > 50:
            f.write(f"*... 还有 {len(log_data['modified_files']) - 50} 个文件 ...*\n")
    
    print(f"摘要报告: {report_path}")
    print("\n" + "=" * 60)
    print("替换完成!")
    print("=" * 60)

if __name__ == '__main__':
    main()
