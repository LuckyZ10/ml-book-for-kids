#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ML教材术语批量修正脚本 - 改进版
"""

import os
import re
import json
from pathlib import Path
from typing import List, Dict, Tuple

class TerminologyReplacer:
    """术语替换器"""
    
    # 中文术语替换规则 (直接使用字符串替换)
    CHINESE_RULES = [
        ('代价函数', '损失函数'),
        ('目标函数', '损失函数'),
        ('批次', '批量'),
        ('感知器', '感知机'),
        ('词嵌入', '嵌入'),
    ]
    
    # 英文术语替换规则 (使用正则表达式保留大小写)
    ENGLISH_RULES = [
        # pretraining -> pre-training
        (r'\bpretraining\b', 'pre-training'),
        (r'\bPretraining\b', 'Pre-training'),
        (r'\bPRETRAINING\b', 'PRE-TRAINING'),
        # finetuning -> fine-tuning
        (r'\bfinetuning\b', 'fine-tuning'),
        (r'\bFinetuning\b', 'Fine-tuning'),
        (r'\bFINETUNING\b', 'FINE-TUNING'),
    ]
    
    def __init__(self):
        self.log = []
        self.rule_stats = {}
        self._init_stats()
    
    def _init_stats(self):
        """初始化统计信息"""
        for old, new in self.CHINESE_RULES:
            self.rule_stats[f'{old} → {new}'] = 0
        for pattern, new in self.ENGLISH_RULES:
            old = pattern.replace(r'\b', '').replace('\\b', '')
            self.rule_stats[f'{old} → {new}'] = 0
    
    def replace_chinese(self, content: str) -> Tuple[str, List[Dict]]:
        """替换中文术语，返回修改后的内容和变更记录"""
        changes = []
        modified = content
        
        for old, new in self.CHINESE_RULES:
            count = modified.count(old)
            if count > 0:
                # 找到所有出现的位置
                pos = 0
                while True:
                    idx = modified.find(old, pos)
                    if idx == -1:
                        break
                    # 计算行号
                    line_num = modified[:idx].count('\n') + 1
                    changes.append({
                        'line': line_num,
                        'original': old,
                        'replacement': new,
                        'rule': f'{old} → {new}'
                    })
                    pos = idx + len(old)
                
                # 执行替换
                modified = modified.replace(old, new)
                self.rule_stats[f'{old} → {new}'] += count
        
        return modified, changes
    
    def replace_english(self, content: str) -> Tuple[str, List[Dict]]:
        """替换英文术语，返回修改后的内容和变更记录"""
        changes = []
        modified = content
        
        for pattern, replacement in self.ENGLISH_RULES:
            regex = re.compile(pattern)
            matches = list(regex.finditer(modified))
            
            if matches:
                # 记录变更
                for match in matches:
                    line_num = modified[:match.start()].count('\n') + 1
                    changes.append({
                        'line': line_num,
                        'original': match.group(),
                        'replacement': replacement,
                        'rule': f"{pattern.replace(r'\b', '')} → {replacement}"
                    })
                
                # 执行替换
                modified = regex.sub(replacement, modified)
                self.rule_stats[f"{pattern.replace(r'\b', '')} → {replacement}"] += len(matches)
        
        return modified, changes
    
    def process_file(self, file_path: Path) -> Tuple[str, int, Dict]:
        """处理单个文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            return None, 0, {'error': str(e), 'file': str(file_path)}
        
        # 中文替换
        modified, cn_changes = self.replace_chinese(content)
        
        # 英文替换
        modified, en_changes = self.replace_english(modified)
        
        total_changes = cn_changes + en_changes
        
        return modified, len(total_changes), {
            'file': str(file_path),
            'replacements': len(total_changes),
            'changes': total_changes
        }
    
    def process_directory(self, directory: Path) -> Dict:
        """处理目录中的所有.md文件"""
        results = {
            'files_processed': 0,
            'files_modified': 0,
            'total_replacements': 0,
            'file_logs': [],
            'rule_stats': {}
        }
        
        md_files = sorted(directory.rglob('*.md'))
        
        for file_path in md_files:
            # 跳过我自己的报告文件
            if file_path.name in ['terminology_consistency_report.md', 'replacement_summary.md', 'replacement_log.json']:
                continue
                
            results['files_processed'] += 1
            modified_content, count, file_log = self.process_file(file_path)
            
            if modified_content is None:
                continue
            
            if count > 0:
                results['files_modified'] += 1
                results['total_replacements'] += count
                results['file_logs'].append(file_log)
                
                # 写回文件
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(modified_content)
        
        # 复制最终的规则统计
        results['rule_stats'] = self.rule_stats.copy()
        
        return results


def generate_summary(results: Dict, output_path: Path, base_dir: Path):
    """生成替换统计报告"""
    lines = [
        "# ML教材术语批量修正报告",
        "",
        "## 执行摘要",
        "",
        f"- **处理文件总数**: {results['files_processed']}",
        f"- **修改文件数**: {results['files_modified']}",
        f"- **总替换次数**: {results['total_replacements']}",
        "",
        "## 替换规则统计",
        "",
        "| 规则 | 替换次数 |",
        "|------|----------|",
    ]
    
    # 过滤掉0次的规则并按次数排序
    sorted_rules = sorted(
        [(k, v) for k, v in results['rule_stats'].items() if v > 0],
        key=lambda x: x[1],
        reverse=True
    )
    
    for rule, count in sorted_rules:
        lines.append(f"| {rule} | {count} |")
    
    lines.extend([
        "",
        "## 文件级修改概览",
        "",
        "| 文件 | 替换次数 |",
        "|------|----------|",
    ])
    
    # 按替换次数排序
    sorted_logs = sorted(results['file_logs'], key=lambda x: x['replacements'], reverse=True)
    for log in sorted_logs:
        rel_path = os.path.relpath(log['file'], base_dir)
        lines.append(f"| {rel_path} | {log['replacements']} |")
    
    lines.extend([
        "",
        "## 详细修改记录",
        "",
    ])
    
    for log in sorted_logs[:50]:  # 只显示前50个文件的详细记录
        rel_path = os.path.relpath(log['file'], base_dir)
        lines.append(f"### {rel_path}")
        lines.append("")
        
        # 按规则分组显示
        changes_by_rule = {}
        for change in log['changes']:
            rule = change['rule']
            if rule not in changes_by_rule:
                changes_by_rule[rule] = []
            changes_by_rule[rule].append(change)
        
        for rule, changes in changes_by_rule.items():
            lines.append(f"- **{rule}** ({len(changes)}次)")
            # 显示前5个示例
            for change in changes[:5]:
                lines.append(f"  - 第{change['line']}行: `{change['original']}` → `{change['replacement']}`")
            if len(changes) > 5:
                lines.append(f"  - ... 还有 {len(changes) - 5} 处")
        lines.append("")
    
    if len(sorted_logs) > 50:
        lines.append(f"\n*还有 {len(sorted_logs) - 50} 个文件未显示详细记录*")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def main():
    base_dir = Path('/root/.openclaw/workspace/ml-book-for-kids')
    
    print(f"="*60)
    print(f"ML教材术语批量修正")
    print(f"="*60)
    print(f"工作目录: {base_dir}")
    
    replacer = TerminologyReplacer()
    
    print(f"\n替换规则:")
    print(f"  中文规则:")
    for old, new in replacer.CHINESE_RULES:
        print(f"    - {old} → {new}")
    print(f"  英文规则:")
    for pattern, new in replacer.ENGLISH_RULES:
        print(f"    - {pattern.replace(r'\b', '')} → {new}")
    
    print(f"\n开始处理...")
    results = replacer.process_directory(base_dir)
    
    # 保存详细日志
    log_path = base_dir / 'replacement_log.json'
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n详细日志已保存: {log_path}")
    
    # 生成统计报告
    summary_path = base_dir / 'replacement_summary.md'
    generate_summary(results, summary_path, base_dir)
    print(f"统计报告已保存: {summary_path}")
    
    print(f"\n{'='*60}")
    print(f"处理完成!")
    print(f"  - 处理文件数: {results['files_processed']}")
    print(f"  - 修改文件数: {results['files_modified']}")
    print(f"  - 总替换次数: {results['total_replacements']}")
    print(f"{'='*60}")
    
    # 显示各规则的使用情况
    print(f"\n各规则替换次数:")
    for rule, count in sorted(results['rule_stats'].items(), key=lambda x: x[1], reverse=True):
        if count > 0:
            print(f"  - {rule}: {count}次")


if __name__ == '__main__':
    main()
