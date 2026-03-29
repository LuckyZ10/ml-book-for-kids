"""
RLHF与DPO实现
第44章：大语言模型对齐与安全
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class RewardModel(nn.Module):
    """
    奖励模型 - 预测人类偏好
    
    架构: 基础语言模型 + 评分头
    """
    def __init__(self, base_model_name=None, hidden_size=768):
        super().__init__()
        # 简化实现，实际应加载预训练模型
        self.embedding = nn.Embedding(50000, hidden_size)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_size, nhead=8, batch_first=True),
            num_layers=6
        )
        self.score_head = nn.Linear(hidden_size, 1)
    
    def forward(self, input_ids, attention_mask=None):
        """
        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)
        
        Returns:
            scores: (batch,) 奖励分数
        """
        x = self.embedding(input_ids)
        
        if attention_mask is not None:
            # 将attention_mask转换为key_padding_mask
            key_mask = ~attention_mask.bool()
            x = self.transformer(x, src_key_padding_mask=key_mask)
        else:
            x = self.transformer(x)
        
        # 取最后一个token的表示
        last_hidden = x[:, -1, :]
        score = self.score_head(last_hidden).squeeze(-1)
        
        return score


class DPOTrainer:
    """
    直接偏好优化(DPO)训练器
    
    核心思想: 直接优化语言模型，无需奖励模型和RL
    """
    def __init__(self, model, ref_model, beta=0.1, lr=1e-5):
        """
        Args:
            model: 要训练的策略模型
            ref_model: 参考模型（SFT后的模型，冻结）
            beta: DPO温度参数
            lr: 学习率
        """
        self.model = model
        self.ref_model = ref_model
        self.beta = beta
        
        # 冻结参考模型
        for param in self.ref_model.parameters():
            param.requires_grad = False
        self.ref_model.eval()
        
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    def compute_log_probs(self, model, input_ids, attention_mask, labels):
        """
        计算序列的平均log概率
        
        Args:
            model: 语言模型
            input_ids: 输入token IDs
            attention_mask: 注意力掩码
            labels: 标签（与input_ids相同，用于计算loss）
        
        Returns:
            log_probs: (batch,) 每个样本的log概率
        """
        # 简化实现，实际应使用模型的forward
        # 这里返回一个占位值
        outputs = model(input_ids, attention_mask=attention_mask)
        
        # 假设outputs.logits存在
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs
        
        # 计算每个token的log概率
        log_probs = F.log_softmax(logits, dim=-1)
        
        # 获取实际token的log概率
        token_log_probs = torch.gather(
            log_probs, 
            dim=-1, 
            index=labels.unsqueeze(-1)
        ).squeeze(-1)
        
        # 只对非padding部分求平均
        mask = (labels != -100).float()
        sequence_log_probs = (token_log_probs * mask).sum(dim=1) / mask.sum(dim=1)
        
        return sequence_log_probs
    
    def compute_loss(self, batch):
        """
        计算DPO损失
        
        batch包含:
        - prompt_input_ids: 提示的token IDs
        - prompt_attention_mask: 提示的attention mask
        - chosen_input_ids: 偏好回答的token IDs
        - chosen_attention_mask: 偏好回答的attention mask
        - rejected_input_ids: 不喜欢回答的token IDs
        - rejected_attention_mask: 不喜欢回答的attention mask
        
        Returns:
            loss: DPO损失
            metrics: 包含chosen_reward, rejected_reward等
        """
        # 获取策略模型的log概率
        chosen_logps = self.compute_log_probs(
            self.model,
            batch['chosen_input_ids'],
            batch['chosen_attention_mask'],
            batch['chosen_labels']
        )
        rejected_logps = self.compute_log_probs(
            self.model,
            batch['rejected_input_ids'],
            batch['rejected_attention_mask'],
            batch['rejected_labels']
        )
        
        # 获取参考模型的log概率（不计算梯度）
        with torch.no_grad():
            ref_chosen_logps = self.compute_log_probs(
                self.ref_model,
                batch['chosen_input_ids'],
                batch['chosen_attention_mask'],
                batch['chosen_labels']
            )
            ref_rejected_logps = self.compute_log_probs(
                self.ref_model,
                batch['rejected_input_ids'],
                batch['rejected_attention_mask'],
                batch['rejected_labels']
            )
        
        # 计算隐式奖励: r(x,y) = beta * log(pi(y|x) / pi_ref(y|x))
        chosen_rewards = self.beta * (chosen_logps - ref_chosen_logps)
        rejected_rewards = self.beta * (rejected_logps - ref_rejected_logps)
        
        # DPO损失: -log(sigmoid(chosen_rewards - rejected_rewards))
        logits = chosen_rewards - rejected_rewards
        loss = -F.logsigmoid(logits).mean()
        
        # 计算指标
        metrics = {
            'loss': loss.item(),
            'chosen_reward': chosen_rewards.mean().item(),
            'rejected_reward': rejected_rewards.mean().item(),
            'reward_margin': (chosen_rewards - rejected_rewards).mean().item(),
            'accuracy': (logits > 0).float().mean().item()
        }
        
        return loss, metrics
    
    def train_step(self, batch):
        """单次训练步骤"""
        self.model.train()
        self.optimizer.zero_grad()
        
        loss, metrics = self.compute_loss(batch)
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        
        self.optimizer.step()
        
        return metrics


class RLHFTrainer:
    """
    RLHF训练器 (使用PPO)
    
    简化版实现，展示核心思想
    """
    def __init__(self, policy_model, ref_model, reward_model, 
                 kl_coef=0.2, lr=1e-5, gamma=1.0, lam=0.95):
        """
        Args:
            policy_model: 策略模型（要训练的）
            ref_model: 参考模型（SFT模型，冻结）
            reward_model: 奖励模型（冻结）
            kl_coef: KL散度惩罚系数
            lr: 学习率
            gamma: 折扣因子
            lam: GAE lambda
        """
        self.policy_model = policy_model
        self.ref_model = ref_model
        self.reward_model = reward_model
        
        self.kl_coef = kl_coef
        self.gamma = gamma
        self.lam = lam
        
        # 冻结参考模型和奖励模型
        for param in self.ref_model.parameters():
            param.requires_grad = False
        for param in self.reward_model.parameters():
            param.requires_grad = False
        self.ref_model.eval()
        self.reward_model.eval()
        
        self.optimizer = torch.optim.AdamW(policy_model.parameters(), lr=lr)
    
    def compute_rewards(self, input_ids, attention_mask, generated_ids):
        """
        计算奖励（奖励模型分数 - KL惩罚）
        """
        # 奖励模型分数
        with torch.no_grad():
            reward = self.reward_model(generated_ids, attention_mask)
        
        # 计算KL散度惩罚
        policy_logits = self.policy_model(generated_ids, attention_mask).logits
        ref_logits = self.ref_model(generated_ids, attention_mask).logits
        
        policy_logprobs = F.log_softmax(policy_logits, dim=-1)
        ref_logprobs = F.log_softmax(ref_logits, dim=-1)
        
        # 近似KL: log(pi) - log(pi_ref)
        kl_div = (policy_logprobs - ref_logprobs).sum(dim=-1).mean()
        
        # 最终奖励
        final_reward = reward - self.kl_coef * kl_div
        
        return final_reward
    
    def ppo_update(self, old_logprobs, returns, advantages, batch):
        """
        PPO更新（简化版）
        """
        # 计算新的log概率
        logits = self.policy_model(
            batch['input_ids'], 
            attention_mask=batch['attention_mask']
        ).logits
        
        new_logprobs = F.log_softmax(logits, dim=-1)
        
        # 策略损失（简化，实际需要更复杂的实现）
        ratio = torch.exp(new_logprobs - old_logprobs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 0.8, 1.2) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # 价值损失（如果有critic）
        value_loss = 0  # 简化
        
        loss = policy_loss + value_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), 1.0)
        self.optimizer.step()
        
        return {'policy_loss': policy_loss.item()}


class PreferenceDataset(Dataset):
    """
    偏好数据集
    
    格式: (prompt, chosen_response, rejected_response)
    """
    def __init__(self, data, tokenizer, max_length=512):
        """
        Args:
            data: 列表，每项是dict包含'prompt', 'chosen', 'rejected'
            tokenizer: 分词器
            max_length: 最大序列长度
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 编码prompt
        prompt_encoding = self.tokenizer(
            item['prompt'],
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        # 编码chosen回答
        chosen_encoding = self.tokenizer(
            item['prompt'] + item['chosen'],
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        # 编码rejected回答
        rejected_encoding = self.tokenizer(
            item['prompt'] + item['rejected'],
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'prompt_input_ids': prompt_encoding['input_ids'].squeeze(0),
            'prompt_attention_mask': prompt_encoding['attention_mask'].squeeze(0),
            'chosen_input_ids': chosen_encoding['input_ids'].squeeze(0),
            'chosen_attention_mask': chosen_encoding['attention_mask'].squeeze(0),
            'chosen_labels': chosen_encoding['input_ids'].squeeze(0),
            'rejected_input_ids': rejected_encoding['input_ids'].squeeze(0),
            'rejected_attention_mask': rejected_encoding['attention_mask'].squeeze(0),
            'rejected_labels': rejected_encoding['input_ids'].squeeze(0),
        }


def constitutional_ai_response(model, tokenizer, prompt, constitution, max_length=256):
    """
    Constitutional AI推理
    
    Args:
        model: 语言模型
        tokenizer: 分词器
        prompt: 用户提示
        constitution: 宪法原则列表
        max_length: 最大生成长度
    
    Returns:
        dict: 包含initial、critique、revised三个阶段的结果
    """
    model.eval()
    
    # 第一阶段：生成初始回答
    inputs = tokenizer(prompt, return_tensors='pt')
    with torch.no_grad():
        initial_ids = model.generate(**inputs, max_length=max_length)
    initial_response = tokenizer.decode(initial_ids[0], skip_special_tokens=True)
    
    # 第二阶段：自我批评
    critique_prompt = f"""Human: {prompt}

Assistant: {initial_response}

Critique: Does this response follow these principles?
{chr(10).join(['- ' + p for p in constitution])}

If not, what is wrong?

Critique:"""
    
    critique_inputs = tokenizer(critique_prompt, return_tensors='pt')
    with torch.no_grad():
        critique_ids = model.generate(**critique_inputs, max_length=max_length + 200)
    critique = tokenizer.decode(critique_ids[0], skip_special_tokens=True)
    
    # 第三阶段：生成改进回答
    revision_prompt = f"""Human: {prompt}

Assistant: {initial_response}

Critique: {critique}

Based on this critique, provide an improved response.

Improved Response:"""
    
    revision_inputs = tokenizer(revision_prompt, return_tensors='pt')
    with torch.no_grad():
        revised_ids = model.generate(**revision_inputs, max_length=max_length + 400)
    revised_response = tokenizer.decode(revised_ids[0], skip_special_tokens=True)
    
    return {
        'initial': initial_response,
        'critique': critique,
        'revised': revised_response
    }


if __name__ == "__main__":
    print("=" * 60)
    print("RLHF与DPO实现")
    print("=" * 60)
    print("\n核心类:")
    print("  1. RewardModel - 奖励模型")
    print("  2. DPOTrainer - DPO训练器")
    print("  3. RLHFTrainer - RLHF训练器(PPO)")
    print("  4. PreferenceDataset - 偏好数据集")
    print("\n工具函数:")
    print("  - constitutional_ai_response - Constitutional AI推理")
    print("\n使用方法:")
    print("  from alignment import DPOTrainer, RewardModel")
    print("  trainer = DPOTrainer(model, ref_model, beta=0.1)")
    print("  metrics = trainer.train_step(batch)")
