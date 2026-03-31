"""
基于强化学习的神经架构搜索(RL-based NAS)
实现NASNet和ENAS的核心思想
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import List, Tuple, Dict
import random

# ==================== 控制器网络 ====================

class NASController(nn.Module):
    """
    NAS控制器：使用LSTM生成架构描述
    每个架构被编码为一系列决策
    """
    
    def __init__(
        self,
        num_blocks=5,
        num_ops=7,
        controller_hid=100,
        controller_temperature=None,
        controller_tanh_constant=2.5,
    ):
        super().__init__()
        
        self.num_blocks = num_blocks
        self.num_ops = num_ops
        self.controller_hid = controller_hid
        self.controller_temperature = controller_temperature
        self.controller_tanh_constant = controller_tanh_constant
        
        # 嵌入维度
        self.num_tokens = num_ops + num_blocks + 2
        
        # LSTM控制器
        self.encoder = nn.Embedding(self.num_tokens, controller_hid)
        self.lstm = nn.LSTMCell(controller_hid, controller_hid)
        
        # 预测头
        self.num_decisions_per_block = 4
        self.decoders = nn.ModuleList([
            nn.Linear(controller_hid, self._get_decoder_size(i))
            for i in range(num_blocks * self.num_decisions_per_block)
        ])
        
        self.reset_parameters()
    
    def _get_decoder_size(self, decision_idx):
        """获取每个决策的输出维度"""
        decision_in_block = decision_idx % self.num_decisions_per_block
        if decision_in_block in [0, 2]:  # 选择前驱节点
            block_id = decision_idx // self.num_decisions_per_block
            return 2 + block_id
        else:  # 选择操作
            return self.num_ops
    
    def reset_parameters(self):
        """初始化参数"""
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.uniform_(m.weight, -0.1, 0.1)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
    
    def forward(self, batch_size=1):
        """生成一个批次架构"""
        inputs = self.encoder(torch.zeros(batch_size, dtype=torch.long))
        hidden = (
            torch.zeros(batch_size, self.controller_hid),
            torch.zeros(batch_size, self.controller_hid)
        )
        
        if next(self.parameters()).is_cuda:
            inputs = inputs.cuda()
            hidden = (hidden[0].cuda(), hidden[1].cuda())
        
        sample_log_probs = []
        sample_entropy = []
        archs = [[] for _ in range(batch_size)]
        
        for block_id in range(self.num_blocks):
            for dec_type in range(self.num_decisions_per_block):
                log_prob, entropy, action, hidden = self._sample_decision(
                    inputs, hidden, block_id, dec_type
                )
                sample_log_probs.append(log_prob)
                sample_entropy.append(entropy)
                
                dec_name = ['pre1', 'op1', 'pre2', 'op2'][dec_type]
                for i, a in enumerate(action):
                    archs[i].append((dec_name, a.item()))
                
                inputs = self.encoder(action)
        
        return sample_log_probs, sample_entropy, archs
    
    def _sample_decision(self, inputs, hidden, block_id, decision_type):
        """采样一个决策"""
        hx, cx = self.lstm(inputs, hidden)
        
        decision_idx = block_id * self.num_decisions_per_block + decision_type
        logits = self.decoders[decision_idx](hx)
        
        if self.controller_temperature is not None:
            logits /= self.controller_temperature
        
        if self.controller_tanh_constant is not None:
            logits = self.controller_tanh_constant * torch.tanh(logits)
        
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        
        action = torch.multinomial(probs, 1).squeeze(-1)
        selected_log_prob = log_probs.gather(1, action.view(-1, 1)).squeeze(-1)
        entropy = -(log_probs * probs).sum(-1)
        
        return selected_log_prob, entropy, action, (hx, cx)


# ==================== 简化的子网络 ====================

class SimpleChildNetwork(nn.Module):
    """简化的子网络，用于演示RL-based NAS"""
    
    def __init__(self, arch, num_classes=10, num_channels=16):
        super().__init__()
        self.arch = arch
        
        layers = []
        in_channels = 3
        
        layers.append(nn.Conv2d(in_channels, num_channels, 3, padding=1))
        layers.append(nn.BatchNorm2d(num_channels))
        layers.append(nn.ReLU())
        
        num_layers = max(2, min(8, len(arch) // 2))
        for i in range(num_layers):
            out_channels = num_channels * (2 ** (i // 3))
            stride = 2 if i % 3 == 2 else 1
            layers.append(nn.Conv2d(
                num_channels if i == 0 else out_channels//2, 
                out_channels, 3, padding=1, stride=stride
            ))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU())
            num_channels = out_channels
        
        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(num_channels, num_classes)
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def build_child_network(arch):
    """构建子网络的工厂函数"""
    return SimpleChildNetwork(arch)


# ==================== REINFORCE训练器 ====================

class REINFORCETrainer:
    """使用REINFORCE算法训练NAS控制器"""
    
    def __init__(
        self,
        controller,
        child_network_builder,
        entropy_weight=0.0001,
        baseline_decay=0.99,
        controller_lr=0.00035,
    ):
        self.controller = controller
        self.child_network_builder = child_network_builder
        self.entropy_weight = entropy_weight
        self.baseline = 0.0
        self.baseline_decay = baseline_decay
        
        self.controller_optimizer = optim.Adam(
            controller.parameters(), lr=controller_lr
        )
    
    def train_step(self, train_loader, val_loader, num_samples=1):
        """执行一步REINFORCE训练"""
        self.controller.train()
        
        sample_log_probs, sample_entropy, archs = self.controller(
            batch_size=num_samples
        )
        
        rewards = []
        for arch in archs:
            reward = self._evaluate_architecture(arch, train_loader, val_loader)
            rewards.append(reward)
        
        rewards = torch.tensor(rewards)
        if next(self.controller.parameters()).is_cuda:
            rewards = rewards.cuda()
        
        self.baseline = self.baseline_decay * self.baseline + \
                       (1 - self.baseline_decay) * rewards.mean().item()
        
        loss = 0
        num_decisions = len(sample_log_probs) // num_samples
        for i in range(num_samples):
            start = i * num_decisions
            end = start + num_decisions
            log_prob_sum = torch.stack(sample_log_probs[start:end]).sum()
            entropy_sum = torch.stack(sample_entropy[start:end]).sum()
            advantage = rewards[i] - self.baseline
            loss = loss - log_prob_sum * advantage - self.entropy_weight * entropy_sum
        
        loss = loss / num_samples
        
        self.controller_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.controller.parameters(), 0.25)
        self.controller_optimizer.step()
        
        return {
            'loss': loss.item(),
            'mean_reward': rewards.mean().item(),
            'baseline': self.baseline,
        }
    
    def _evaluate_architecture(self, arch, train_loader, val_loader, epochs=5):
        """评估一个架构的性能（简化版）"""
        return random.uniform(60, 95)  # 模拟准确率


# ==================== 使用示例 ====================

if __name__ == '__main__':
    print("=" * 60)
    print("RL-based NAS测试")
    print("=" * 60)
    
    controller = NASController(num_blocks=3, num_ops=5)
    print(f"控制器参数量: {sum(p.numel() for p in controller.parameters()):,}")
    
    log_probs, entropy, archs = controller(batch_size=2)
    print(f"\n采样架构数量: {len(archs)}")
    print(f"架构示例: {archs[0][:4]}...")
    
    child_net = build_child_network(archs[0])
    print(f"\n子网络参数量: {sum(p.numel() for p in child_net.parameters()):,}")
    
    x = torch.randn(2, 3, 32, 32)
    y = child_net(x)
    print(f"输入: {x.shape}, 输出: {y.shape}")
