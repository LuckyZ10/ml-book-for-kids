"""
第四十七章：测试时计算与推理优化 - 完整代码实现
包含：
1. 测试时计算推理引擎 (Best-of-N、Beam Search、MCTS、迭代修正)
2. 过程奖励模型 (PRM训练与推理)
3. TTT-Linear层 (从零实现TTT序列建模层)
4. 思维链生成器 (CoT提示与自我修正)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional, Callable
import numpy as np
from dataclasses import dataclass
from collections import deque
import heapq
import random


# ==================== 47.6.1 测试时计算推理引擎 ====================

@dataclass
class SearchConfig:
    """搜索配置"""
    num_samples: int = 16          # Best-of-N的N
    beam_width: int = 4            # Beam search的宽度
    max_length: int = 512          # 最大生成长度
    temperature: float = 1.0       # 采样温度
    top_p: float = 0.95            # Nucleus sampling参数
    use_verifier: bool = True      # 是否使用验证器
    num_iterations: int = 3        # 迭代修正次数


class Verifier(nn.Module):
    """
    验证器模型：评估答案或推理步骤的质量
    基于Transformer编码器架构
    """
    def __init__(self, hidden_size: int = 768, num_layers: int = 3, num_heads: int = 12):
        super().__init__()
        self.hidden_size = hidden_size
        
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=num_heads,
                dim_feedforward=hidden_size * 4,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=num_layers
        )
        self.score_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, input_ids: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        计算输入序列的质量分数
        Args:
            input_ids: [batch_size, seq_len, hidden_size] (假设已是embedding)
            attention_mask: [batch_size, seq_len]
        Returns:
            scores: [batch_size, 1]，0-1之间的质量分数
        """
        if attention_mask is None:
            attention_mask = torch.ones(input_ids.shape[:2], dtype=torch.bool, device=input_ids.device)
        
        # 编码
        encoded = self.encoder(input_ids, src_key_padding_mask=~attention_mask)
        
        # 全局平均池化
        mask_expanded = attention_mask.unsqueeze(-1).float()
        pooled = (encoded * mask_expanded).sum(dim=1) / (mask_expanded.sum(dim=1) + 1e-8)
        
        # 计算分数
        score = self.score_head(pooled)
        return score


class ProcessRewardModel(nn.Module):
    """
    过程奖励模型(PRM)：评估每一步推理的质量
    使用双向LSTM捕获步骤间的依赖关系
    """
    def __init__(self, hidden_size: int = 768, vocab_size: int = 50000):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        
        # 双向LSTM编码器
        self.encoder = nn.LSTM(
            hidden_size, hidden_size // 2,
            num_layers=2, bidirectional=True,
            batch_first=True, dropout=0.1
        )
        
        # 步骤评分器
        self.step_scorer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, input_ids: torch.Tensor,
                step_boundaries: List[List[int]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算每步的奖励
        Args:
            input_ids: [batch_size, seq_len]
            step_boundaries: 每一步的结束位置列表
        Returns:
            step_rewards: 每步的奖励分数 [batch_size, max_num_steps]
            total_reward: 总奖励（聚合） [batch_size, 1]
        """
        batch_size = input_ids.shape[0]
        embedded = self.embedding(input_ids)
        encoded, _ = self.encoder(embedded)
        
        # 计算每步的奖励
        batch_step_rewards = []
        max_steps = max(len(b) for b in step_boundaries)
        
        for i in range(batch_size):
            boundaries = step_boundaries[i]
            step_rewards = []
            
            for j, end in enumerate(boundaries):
                start = boundaries[j-1] if j > 0 else 0
                step_repr = encoded[i, start:end, :].mean(dim=0, keepdim=True)  # [1, hidden]
                reward = self.step_scorer(step_repr)  # [1, 1]
                step_rewards.append(reward.squeeze())
            
            # 填充到最大步数
            while len(step_rewards) < max_steps:
                step_rewards.append(torch.tensor(0.0, device=input_ids.device))
            
            batch_step_rewards.append(torch.stack(step_rewards))
        
        step_rewards_tensor = torch.stack(batch_step_rewards)  # [batch, max_steps]
        
        # 聚合（使用最小奖励，因为链的强度取决于最弱环节）
        mask = (step_rewards_tensor > 0).float()
        masked_rewards = step_rewards_tensor * mask + (1 - mask) * 1.0  # 填充位置设为1
        total_reward = masked_rewards.min(dim=1, keepdim=True)[0]
        
        return step_rewards_tensor, total_reward
    
    def get_step_scores(self, reasoning_chain: List[str], tokenizer=None) -> List[float]:
        """
        获取每步推理的分数（用于推理时）
        简化实现，实际使用时需要完整tokenize
        """
        scores = []
        for i, step in enumerate(reasoning_chain):
            # 启发式评分
            score = 0.5 + 0.3 * (1.0 / (1.0 + 0.1 * i))  # 随着步骤推进，基础分略降
            if any(kw in step.lower() for kw in ['正确', '答案', '结论', '因此']):
                score += 0.2
            if len(step) > 20:
                score += 0.1
            scores.append(min(score, 1.0))
        return scores


class MCTSNode:
    """MCTS树节点"""
    
    def __init__(self, state: str, parent=None, prior: float = 1.0):
        self.state = state
        self.parent = parent
        self.children: List['MCTSNode'] = []
        self.visits = 0
        self.value_sum = 0.0
        self.prior = prior
        self.is_terminal = "[END]" in state or "答案" in state
    
    def value(self) -> float:
        """平均价值"""
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits
    
    def ucb_score(self, c: float = 1.414) -> float:
        """UCB1分数"""
        if self.visits == 0:
            return float('inf')
        
        # 利用项
        exploitation = self.value()
        
        # 探索项
        if self.parent:
            exploration = c * self.prior * np.sqrt(np.log(self.parent.visits + 1) / self.visits)
        else:
            exploration = 0
        
        return exploitation + exploration
    
    def is_expanded(self) -> bool:
        """是否已扩展"""
        return len(self.children) > 0


class TestTimeInferenceEngine:
    """
    测试时计算推理引擎
    支持多种测试时计算策略：Best-of-N、Beam Search、MCTS、迭代修正
    """
    
    def __init__(self, 
                 model: Optional[nn.Module] = None,
                 verifier: Optional[Verifier] = None,
                 prm: Optional[ProcessRewardModel] = None,
                 config: Optional[SearchConfig] = None):
        self.model = model
        self.verifier = verifier
        self.prm = prm
        self.config = config or SearchConfig()
    
    def generate(self, prompt: str, strategy: str = "best_of_n") -> str:
        """
        使用指定的测试时计算策略生成答案
        Args:
            prompt: 输入提示
            strategy: 策略名称 ("best_of_n", "beam_search", "mcts", "iterative_refinement")
        Returns:
            生成的答案
        """
        strategies = {
            "best_of_n": self.best_of_n,
            "beam_search": self.beam_search,
            "mcts": self.mcts_search,
            "iterative_refinement": self.iterative_refinement
        }
        
        if strategy not in strategies:
            raise ValueError(f"Unknown strategy: {strategy}. Available: {list(strategies.keys())}")
        
        return strategies[strategy](prompt)
    
    def best_of_n(self, prompt: str) -> str:
        """
        Best-of-N策略：生成N个答案，选择验证器评分最高的
        """
        candidates = []
        scores = []
        
        print(f"[Best-of-N] 生成 {self.config.num_samples} 个候选...")
        
        for i in range(self.config.num_samples):
            # 生成候选答案
            candidate = self._sample_answer(prompt)
            candidates.append(candidate)
            
            # 验证器评分
            if self.verifier is not None and self.config.use_verifier:
                score = self._verify_answer(prompt, candidate)
            else:
                score = self._heuristic_score(candidate)
            
            scores.append(score)
            print(f"  候选 {i+1}: score={score:.4f}, length={len(candidate)}")
        
        # 选择最佳答案
        best_idx = int(np.argmax(scores))
        best_answer = candidates[best_idx]
        
        print(f"[Best-of-N] 选择候选 {best_idx+1} (score={scores[best_idx]:.4f})")
        
        return best_answer
    
    def beam_search(self, prompt: str) -> str:
        """
        Beam Search：在每一步保留top-k候选
        """
        beam_width = self.config.beam_width
        beams: List[Tuple[float, str]] = [(0.0, "")]  # (累积分数, 部分序列)
        
        print(f"[Beam Search] beam_width={beam_width}, max_length={self.config.max_length}")
        
        for step in range(self.config.max_length):
            candidates: List[Tuple[float, str]] = []
            
            for score, seq in beams:
                # 生成下一个token的候选
                next_tokens = self._get_next_tokens(prompt + seq, k=beam_width * 2)
                
                for token, token_prob in next_tokens:
                    new_seq = seq + token
                    new_score = score + np.log(token_prob + 1e-10)
                    
                    # 如果使用PRM，加上步骤奖励
                    if self.prm is not None:
                        step_reward = self._get_step_reward(new_seq)
                        new_score += 0.5 * step_reward
                    
                    candidates.append((new_score, new_seq))
            
            # 保留top-k
            candidates.sort(reverse=True, key=lambda x: x[0])
            beams = candidates[:beam_width]
            
            # 检查是否都完成了
            if all(self._is_complete(seq) for _, seq in beams):
                break
            
            if step % 50 == 0 and step > 0:
                print(f"  Step {step}: best_score={beams[0][0]:.4f}")
        
        # 返回最佳完整序列
        best_seq = max(beams, key=lambda x: x[0])[1]
        print(f"[Beam Search] 完成，最终长度={len(best_seq)}")
        
        return best_seq
    
    def mcts_search(self, prompt: str, num_simulations: int = 100) -> str:
        """
        蒙特卡洛树搜索用于推理
        """
        root = MCTSNode(prompt, parent=None)
        
        print(f"[MCTS] 进行 {num_simulations} 次模拟...")
        
        for i in range(num_simulations):
            # 选择
            node = self._mcts_select(root)
            
            # 扩展
            if not node.is_terminal and not node.is_expanded():
                node = self._mcts_expand(node)
            
            # 模拟
            value = self._mcts_simulate(node)
            
            # 回溯
            self._mcts_backpropagate(node, value)
            
            if (i + 1) % 20 == 0:
                print(f"  模拟 {i+1}/{num_simulations}, root_value={root.value():.4f}")
        
        # 选择访问次数最多的路径
        if root.children:
            best_child = max(root.children, key=lambda c: c.visits)
            result = best_child.state
            print(f"[MCTS] 完成，选择访问次数={best_child.visits}的节点")
        else:
            result = prompt
            print("[MCTS] 警告：没有扩展任何节点")
        
        return result
    
    def iterative_refinement(self, prompt: str) -> str:
        """
        迭代修正：生成初始答案后多次改进
        """
        # 初始生成
        answer = self._sample_answer(prompt)
        print(f"[Iterative Refinement] 初始答案长度={len(answer)}")
        
        for iteration in range(self.config.num_iterations):
            # 自我批评
            critique = self._generate_critique(prompt, answer)
            print(f"  迭代 {iteration+1}: 批评='{critique[:50]}...'")
            
            # 检查是否需要继续
            if any(phrase in critique for phrase in ["无需修改", "完美", "正确"]):
                print(f"  迭代 {iteration+1}: 无需进一步修改")
                break
            
            # 生成改进版
            refinement_prompt = f"""问题：{prompt}
当前答案：{answer}
批评意见：{critique}
请根据批评意见改进答案。"""
            
            improved = self._sample_answer(refinement_prompt)
            
            # 验证改进
            old_score = self._verify_answer(prompt, answer) if self.verifier else 0.5
            new_score = self._verify_answer(prompt, improved) if self.verifier else 0.6
            
            if new_score > old_score:
                print(f"  迭代 {iteration+1}: 分数 {old_score:.4f} → {new_score:.4f}")
                answer = improved
            else:
                print(f"  迭代 {iteration+1}: 分数未提升，保持原答案")
        
        return answer
    
    # ==================== 辅助方法 ====================
    
    def _sample_answer(self, prompt: str) -> str:
        """从模型采样一个答案（占位符实现）"""
        # 实际实现应调用语言模型
        templates = [
            f"根据问题'{prompt[:30]}...'，我的答案是：42",
            f"让我分析一下...{prompt[:20]}的答案是17。",
            f"经过计算，答案是：{random.randint(1, 100)}",
            f"这是一个复杂的问题，需要多步推理...[答案: {random.randint(10, 99)}]"
        ]
        return random.choice(templates)
    
    def _verify_answer(self, prompt: str, answer: str) -> float:
        """使用验证器评分"""
        if self.verifier is None:
            return 0.5 + 0.4 * np.random.random()
        # 实际实现应调用验证器模型
        return 0.5 + 0.4 * np.random.random()
    
    def _heuristic_score(self, answer: str) -> float:
        """启发式评分（无验证器时使用）"""
        score = 0.5
        if len(answer) > 50:
            score += 0.1
        if any(kw in answer for kw in ["答案", "结论", "因此", "所以"]):
            score += 0.2
        if any(char.isdigit() for char in answer):
            score += 0.1
        return min(score, 1.0)
    
    def _get_next_tokens(self, prompt: str, k: int = 5) -> List[Tuple[str, float]]:
        """获取下一个token的top-k候选"""
        # 占位符实现，返回概率递减的token
        return [(f"token_{i}", max(0.5 - i * 0.05, 0.1)) for i in range(k)]
    
    def _is_complete(self, seq: str) -> bool:
        """检查序列是否完整"""
        end_markers = ["[END]", "答案", "结论", "完成"]
        return any(marker in seq for marker in end_markers) or len(seq) > self.config.max_length
    
    def _get_step_reward(self, seq: str) -> float:
        """获取步骤奖励（用于PRM）"""
        if self.prm is None:
            return 0.5 + 0.3 * np.random.random()
        return 0.5 + 0.3 * np.random.random()
    
    def _generate_critique(self, prompt: str, answer: str) -> str:
        """生成批评意见"""
        critiques = [
            "推理过程清晰，但可以更详细地说明中间步骤。",
            "答案正确，但缺少关键假设的说明。",
            "计算过程有误，请重新检查中间步骤。",
            "无需修改，答案正确且完整。",
            "建议增加验证步骤以确保答案的准确性。"
        ]
        return random.choice(critiques)
    
    # ==================== MCTS辅助方法 ====================
    
    def _mcts_select(self, root: MCTSNode) -> MCTSNode:
        """使用UCB1选择节点"""
        node = root
        while node.is_expanded() and not node.is_terminal:
            node = max(node.children, key=lambda c: c.ucb_score())
        return node
    
    def _mcts_expand(self, node: MCTSNode) -> MCTSNode:
        """扩展节点"""
        # 生成几个可能的延续
        continuations = self._get_next_tokens(node.state, k=3)
        for token, prob in continuations:
            child = MCTSNode(node.state + token, parent=node, prior=prob)
            node.children.append(child)
        return node.children[0] if node.children else node
    
    def _mcts_simulate(self, node: MCTSNode) -> float:
        """从节点进行模拟"""
        # 快速rollout到终止
        seq = node.state
        for _ in range(50):  # 最大rollout长度
            if self._is_complete(seq):
                break
            tokens = self._get_next_tokens(seq, k=1)
            seq += tokens[0][0]
        
        # 评估最终结果
        return self._verify_answer("", seq)
    
    def _mcts_backpropagate(self, node: MCTSNode, value: float):
        """回溯更新统计"""
        while node is not None:
            node.visits += 1
            node.value_sum += value
            node = node.parent


# ==================== 47.6.2 过程奖励模型训练 ====================

class PRMTrainer:
    """
    过程奖励模型训练器
    """
    
    def __init__(self, model: ProcessRewardModel, lr: float = 1e-4, device: str = "cpu"):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.BCELoss()
    
    def train_step(self, batch: Dict) -> float:
        """
        训练步骤
        Args:
            batch: {
                'input_ids': [batch_size, seq_len],
                'step_boundaries': List[List[int]],
                'step_labels': [batch_size, num_steps]  # 每步的正确性标签
            }
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(self.device)
        step_boundaries = batch['step_boundaries']
        step_labels = batch['step_labels'].to(self.device)
        
        # 前向传播
        step_rewards, _ = self.model(input_ids, step_boundaries)
        
        # 计算损失（只计算有效步骤）
        mask = (step_labels >= 0).float()
        loss = self.criterion(step_rewards * mask, step_labels * mask)
        
        # 反向传播
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def prepare_step_labels(self, reasoning_chains: List[List[str]], 
                           correct_answers: List[str]) -> List[List[int]]:
        """
        自动标注推理链的每步标签
        使用启发式方法，实际应用可以使用更强的验证器或人工标注
        """
        labels = []
        
        for chain, correct in zip(reasoning_chains, correct_answers):
            chain_labels = []
            for step in chain:
                label = 1 if self._is_valid_step(step, correct) else 0
                chain_labels.append(label)
            labels.append(chain_labels)
        
        return labels
    
    def _is_valid_step(self, step: str, correct_answer: str) -> bool:
        """检查步骤是否有效（启发式）"""
        if len(step) < 5:
            return False
        # 检查是否包含数字或数学表达式
        has_number = any(c.isdigit() for c in step)
        has_logic = any(kw in step for kw in ["因为", "所以", "如果", "则", "="])
        return has_number or has_logic


# ==================== 47.6.3 TTT-Linear层实现 ====================

class TTTLinnerLayer(nn.Module):
    """
    TTT-Linear层：使用线性内部模型的测试时训练层
    通过在每个序列上动态更新内部模型参数实现线性复杂度
    """
    
    def __init__(self, hidden_size: int = 768, mini_batch_size: int = 16):
        super().__init__()
        self.hidden_size = hidden_size
        self.mini_batch_size = mini_batch_size
        
        # 慢权重：学习到的投影矩阵
        self.W_init = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.02)
        self.b_init = nn.Parameter(torch.zeros(hidden_size))
        
        # 额外的MLP层（类似Transformer的FFN）
        self.gate_proj = nn.Linear(hidden_size, hidden_size * 4)
        self.up_proj = nn.Linear(hidden_size, hidden_size * 4)
        self.down_proj = nn.Linear(hidden_size * 4, hidden_size)
        
        # Layer Norm
        self.norm = nn.LayerNorm(hidden_size)
        
    def forward(self, x: torch.Tensor, use_ttt: bool = True) -> torch.Tensor:
        """
        前向传播
        Args:
            x: [batch_size, seq_len, hidden_size]
            use_ttt: 是否使用TTT（测试时训练）
        Returns:
            output: [batch_size, seq_len, hidden_size]
        """
        if use_ttt:
            # 使用TTT
            outputs = []
            for b in range(x.shape[0]):
                output = self._ttt_forward(x[b])  # [seq_len, hidden_size]
                outputs.append(output)
            return torch.stack(outputs, dim=0)
        else:
            # 标准前向（无TTT）
            return self._standard_forward(x)
    
    def _ttt_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        TTT前向传播（单序列）
        Args:
            x: [seq_len, hidden_size]
        """
        seq_len, hidden_size = x.shape
        
        # 初始化内部模型参数（快权重）
        W = self.W_init.clone()  # [hidden_size, hidden_size]
        b = self.b_init.clone()  # [hidden_size]
        
        outputs = []
        
        # 处理输入
        for t in range(seq_len):
            x_t = x[t]  # [hidden_size]
            
            # 使用当前内部模型进行预测
            pred = torch.matmul(W, x_t) + b  # [hidden_size]
            
            # 应用门控MLP
            gated = torch.sigmoid(self.gate_proj(pred)) * self.up_proj(pred)
            output = self.down_proj(gated)
            output = self.norm(output + pred)  # 残差连接
            
            outputs.append(output)
            
            # 更新内部模型（梯度下降）
            if t < seq_len - 1:
                target = x[t + 1]
                
                # 计算梯度
                diff = pred - target  # [hidden_size]
                grad_W = torch.outer(diff, x_t)  # [hidden, hidden]
                grad_b = diff
                
                # 更新参数（学习率作为超参数）
                lr = 0.01
                W = W - lr * grad_W
                b = b - lr * grad_b
        
        return torch.stack(outputs, dim=0)
    
    def _standard_forward(self, x: torch.Tensor) -> torch.Tensor:
        """标准前向传播（无TTT）"""
        # 投影
        pred = torch.matmul(x, self.W_init.t()) + self.b_init
        
        # 门控MLP
        gated = torch.sigmoid(self.gate_proj(pred)) * self.up_proj(pred)
        output = self.down_proj(gated)
        
        return self.norm(output + pred)
    
    def forward_batch_optimized(self, x: torch.Tensor) -> torch.Tensor:
        """
        批量化优化的TTT前向
        使用向量化操作提高效率
        """
        batch_size, seq_len, hidden_size = x.shape
        
        # 批量初始化
        W_batch = self.W_init.unsqueeze(0).expand(batch_size, -1, -1).clone()
        b_batch = self.b_init.unsqueeze(0).expand(batch_size, -1).clone()
        
        outputs = []
        
        for t in range(seq_len):
            x_t = x[:, t, :]  # [batch_size, hidden_size]
            
            # 批量预测
            pred = torch.bmm(W_batch, x_t.unsqueeze(-1)).squeeze(-1) + b_batch
            
            # 门控MLP
            gated = torch.sigmoid(self.gate_proj(pred)) * self.up_proj(pred)
            output = self.down_proj(gated)
            output = self.norm(output + pred)
            
            outputs.append(output)
            
            # 批量更新
            if t < seq_len - 1:
                target = x[:, t + 1, :]
                
                diff = pred - target  # [batch_size, hidden_size]
                grad_W = torch.bmm(diff.unsqueeze(-1), x_t.unsqueeze(1))  # [batch, hidden, hidden]
                grad_b = diff
                
                lr = 0.01
                W_batch = W_batch - lr * grad_W
                b_batch = b_batch - lr * grad_b
        
        return torch.stack(outputs, dim=1)  # [batch, seq, hidden]


class TTTMLPLayer(nn.Module):
    """
    TTT-MLP层：使用MLP作为内部模型的TTT层
    表达能力更强，但计算开销更大
    """
    
    def __init__(self, hidden_size: int = 768, inner_dim: int = 512):
        super().__init__()
        self.hidden_size = hidden_size
        self.inner_dim = inner_dim
        
        # 内部MLP的初始参数（慢权重）
        self.W1_init = nn.Parameter(torch.randn(inner_dim, hidden_size) * 0.02)
        self.b1_init = nn.Parameter(torch.zeros(inner_dim))
        self.W2_init = nn.Parameter(torch.randn(hidden_size, inner_dim) * 0.02)
        self.b2_init = nn.Parameter(torch.zeros(hidden_size))
        
        # 输出变换
        self.output_proj = nn.Linear(hidden_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        batch_size, seq_len, _ = x.shape
        
        outputs = []
        
        for b in range(batch_size):
            # 初始化内部MLP参数（快权重）
            W1 = self.W1_init.clone()
            b1 = self.b1_init.clone()
            W2 = self.W2_init.clone()
            b2 = self.b2_init.clone()
            
            seq_outputs = []
            
            for t in range(seq_len):
                x_t = x[b, t]
                
                # 前向传播通过内部MLP
                h = F.relu(torch.matmul(W1, x_t) + b1)
                pred = torch.matmul(W2, h) + b2
                
                # 输出变换
                output = self.output_proj(pred)
                output = self.norm(output)
                
                seq_outputs.append(output)
                
                # 更新内部参数（多步梯度下降）
                if t < seq_len - 1:
                    target = x[b, t + 1]
                    
                    # 简单的单步梯度下降（实际应用可使用多步）
                    h = F.relu(torch.matmul(W1, x_t) + b1)
                    pred = torch.matmul(W2, h) + b2
                    
                    loss = F.mse_loss(pred, target)
                    
                    # 简化的梯度计算（示意）
                    lr = 0.001
                    with torch.enable_grad():
                        h_temp = F.relu(torch.matmul(W1, x_t) + b1)
                        pred_temp = torch.matmul(W2, h_temp) + b2
                        loss_temp = F.mse_loss(pred_temp, target)
                    
                    # 注意：这里简化了，实际应使用autograd
                    # 或手动计算完整梯度
            
            outputs.append(torch.stack(seq_outputs, dim=0))
        
        return torch.stack(outputs, dim=0)


class TTTSequentialModel(nn.Module):
    """
    使用TTT层的完整序列模型
    """
    
    def __init__(self, vocab_size: int = 50000, 
                 hidden_size: int = 768,
                 num_layers: int = 6,
                 use_ttt: bool = True):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        
        # TTT层或Transformer层
        if use_ttt:
            self.layers = nn.ModuleList([
                TTTLinnerLayer(hidden_size) for _ in range(num_layers)
            ])
        else:
            # 使用标准Transformer层作为对比
            self.layers = nn.ModuleList([
                nn.TransformerEncoderLayer(
                    d_model=hidden_size,
                    nhead=8,
                    dim_feedforward=hidden_size * 4,
                    batch_first=True
                ) for _ in range(num_layers)
            ])
        
        self.output_head = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        x = self.embedding(input_ids)
        
        for layer in self.layers:
            if isinstance(layer, TTTLinnerLayer):
                x = layer(x)
            else:
                x = layer(x)
        
        return self.output_head(x)
    
    def count_parameters(self) -> int:
        """统计参数数量"""
        return sum(p.numel() for p in self.parameters())
    
    def estimate_memory(self, seq_len: int, batch_size: int = 1) -> Dict[str, float]:
        """
        估算内存使用（MB）
        """
        # 参数内存
        param_memory = self.count_parameters() * 4 / (1024 ** 2)  # MB (float32)
        
        # 激活内存
        num_layers = len(self.layers)
        hidden_size = self.embedding.embedding_dim
        activation_memory = batch_size * seq_len * hidden_size * num_layers * 4 / (1024 ** 2)
        
        # TTT特有的内部模型内存
        has_ttt = any(isinstance(l, TTTLinnerLayer) for l in self.layers)
        ttt_memory = batch_size * hidden_size * hidden_size * 4 / (1024 ** 2) if has_ttt else 0
        
        return {
            'param_memory_mb': param_memory,
            'activation_memory_mb': activation_memory,
            'ttt_memory_mb': ttt_memory,
            'total_mb': param_memory + activation_memory + ttt_memory
        }


# ==================== 47.6.4 思维链生成器 ====================

class ChainOfThoughtGenerator:
    """
    思维链生成器：生成带推理过程的答案
    支持多种解码策略：标准CoT、自一致性、验证引导、迭代修正
    """
    
    def __init__(self, model: Optional[nn.Module] = None, tokenizer=None):
        self.model = model
        self.tokenizer = tokenizer
    
    def generate_cot(self, question: str, 
                     max_reasoning_steps: int = 10,
                     temperature: float = 0.7) -> Dict:
        """
        生成带思维链的答案
        Args:
            question: 问题
            max_reasoning_steps: 最大推理步数
            temperature: 采样温度
        Returns:
            {
                'reasoning_chain': List[str],
                'final_answer': str,
                'confidence': float
            }
        """
        reasoning_chain = []
        
        # 构建提示
        prompt = f"问题：{question}\n让我们逐步思考：\n"
        
        for step in range(max_reasoning_steps):
            # 生成下一步推理
            step_context = "\n".join(reasoning_chain)
            step_prompt = f"{prompt}{step_context}\n步骤{step + 1}:"
            
            # 实际实现应调用模型生成
            step_output = self._generate_step(step_prompt, temperature)
            
            # 检查是否应该停止
            if any(marker in step_output for marker in ["答案是", "最终答案", "结论"]):
                reasoning_chain.append(step_output)
                break
            
            reasoning_chain.append(step_output)
            
            # 检查停止条件
            if self._should_stop(reasoning_chain):
                break
        
        # 提取最终答案
        final_answer = self._extract_answer(reasoning_chain[-1]) if reasoning_chain else ""
        confidence = self._calculate_confidence(reasoning_chain)
        
        return {
            'reasoning_chain': reasoning_chain,
            'final_answer': final_answer,
            'confidence': confidence
        }
    
    def generate_self_consistency(self, question: str, 
                                   num_paths: int = 5) -> Dict:
        """
        自一致性解码：生成多条推理路径，选择最一致的答案
        (Self-Consistency Improves Chain of Thought Reasoning in Language Models)
        """
        paths = []
        answers = []
        
        for i in range(num_paths):
            result = self.generate_cot(question, temperature=0.9)
            paths.append(result['reasoning_chain'])
            answers.append(result['final_answer'])
        
        # 选择最常见的答案
        from collections import Counter
        answer_counts = Counter(answers)
        most_common = answer_counts.most_common(1)[0]
        
        best_answer = most_common[0]
        confidence = most_common[1] / num_paths
        
        # 找到对应路径
        best_path_idx = answers.index(best_answer)
        best_path = paths[best_path_idx]
        
        return {
            'final_answer': best_answer,
            'confidence': confidence,
            'reasoning_chain': best_path,
            'all_paths': paths,
            'answer_distribution': dict(answer_counts)
        }
    
    def generate_with_verification(self, question: str,
                                    verifier: Optional[Callable] = None) -> Dict:
        """
        生成带验证的思维链
        每生成一步就验证，如果出错则回溯
        """
        reasoning_chain = []
        max_attempts = 3
        
        prompt = f"问题：{question}\n让我们逐步思考：\n"
        
        while len(reasoning_chain) < 10:  # 最大步数限制
            step_context = "\n".join(reasoning_chain)
            if reasoning_chain:
                step_prompt = f"{prompt}{step_context}\n下一步:"
            else:
                step_prompt = f"{prompt}步骤1:"
            
            # 生成候选步骤
            candidates = []
            for _ in range(max_attempts):
                candidate = self._generate_step(step_prompt, temperature=0.8)
                candidates.append(candidate)
            
            # 验证并选择最佳
            if verifier:
                best_step = max(candidates, 
                               key=lambda c: verifier(question, reasoning_chain, c))
                step_valid = verifier(question, reasoning_chain, best_step) > 0.5
            else:
                # 启发式选择（最长且包含逻辑词）
                best_step = max(candidates, 
                               key=lambda c: len(c) + sum(kw in c for kw in ["因为", "所以"]))
                step_valid = True
            
            if step_valid:
                reasoning_chain.append(best_step)
                
                # 检查是否完成
                if any(marker in best_step for marker in ["答案", "结论"]):
                    break
            else:
                # 回溯：移除上一步
                if reasoning_chain:
                    reasoning_chain.pop()
                    print(f"[验证失败] 回溯一步")
        
        return {
            'reasoning_chain': reasoning_chain,
            'final_answer': self._extract_answer(reasoning_chain[-1]) if reasoning_chain else "",
            'num_steps': len(reasoning_chain)
        }
    
    def iterative_self_refinement(self, question: str,
                                   max_iterations: int = 3) -> Dict:
        """
        迭代自我修正
        基于STaR (Self-Taught Reasoner) 思想
        """
        # 初始生成
        result = self.generate_cot(question)
        answer = result['final_answer']
        reasoning = result['reasoning_chain']
        
        print(f"[初始答案] {answer}")
        
        for iteration in range(max_iterations):
            # 自我批评
            critique = self._self_critique(question, reasoning, answer)
            print(f"[第{iteration+1}轮批评] {critique}")
            
            # 检查是否需要改进
            if any(phrase in critique for phrase in ["无需修改", "正确", "完美"]):
                print("[完成] 无需进一步改进")
                break
            
            # 生成改进版本
            refinement_prompt = f"""问题：{question}

当前推理过程：
{chr(10).join(reasoning)}

批评意见：{critique}

请根据批评意见改进答案，提供修正后的完整推理过程："""
            
            improved = self.generate_cot(refinement_prompt)
            new_answer = improved['final_answer']
            
            print(f"[第{iteration+1}轮改进] {new_answer}")
            
            # 更新
            if new_answer != answer:
                answer = new_answer
                reasoning = improved['reasoning_chain']
            else:
                print("[收敛] 答案未改变")
                break
        
        return {
            'final_answer': answer,
            'reasoning_chain': reasoning,
            'num_iterations': iteration + 1
        }
    
    # ==================== 辅助方法 ====================
    
    def _generate_step(self, prompt: str, temperature: float) -> str:
        """生成单步推理（占位符实现）"""
        steps = [
            "首先，我们需要理解问题的核心要求。",
            "根据已知条件，我们可以建立等式。",
            "通过代入数值，我们得到中间结果。",
            "验证这个结果是否符合所有条件。",
            "因此，最终答案是...",
            "让我们分解这个问题。",
            "关键步骤是识别变量。",
            "通过逻辑推导，我们得出..."
        ]
        return random.choice(steps)
    
    def _should_stop(self, reasoning_chain: List[str]) -> bool:
        """判断是否停止生成"""
        if not reasoning_chain:
            return False
        last = reasoning_chain[-1].lower()
        stop_signals = ['答案', '结论', '综上所述', '因此', '所以', '最终', '结束']
        return any(sig in last for sig in stop_signals)
    
    def _extract_answer(self, final_step: str) -> str:
        """从最后一步提取答案"""
        markers = ['答案是', '答案为', '答案是：', '结论：', '最终答案：']
        for marker in markers:
            if marker in final_step:
                idx = final_step.index(marker) + len(marker)
                return final_step[idx:].strip()
        return final_step
    
    def _calculate_confidence(self, reasoning_chain: List[str]) -> float:
        """计算置信度"""
        if not reasoning_chain:
            return 0.0
        # 启发式：步骤越多、每步越长，置信度越高
        avg_len = sum(len(s) for s in reasoning_chain) / len(reasoning_chain)
        num_steps = len(reasoning_chain)
        return min((avg_len / 100 + num_steps / 10) / 2, 1.0)
    
    def _self_critique(self, question: str, reasoning: List[str], answer: str) -> str:
        """生成自我批评"""
        critiques = [
            "推理过程逻辑清晰，但可以更详细地说明中间步骤。",
            "答案正确，但缺少关键假设的说明。",
            "计算过程有误，第三步的推导需要重新检查。",
            "无需修改，答案正确且完整。",
            "建议增加验证步骤以确保答案的准确性。",
            "推理链中有逻辑跳跃，需要补充连接步骤。"
        ]
        return random.choice(critiques)


# ==================== 47.6.5 演示和测试代码 ====================

def demo_test_time_compute():
    """测试时计算演示"""
    print("=" * 60)
    print("测试时计算推理引擎演示")
    print("=" * 60)
    
    # 创建组件
    verifier = Verifier(hidden_size=256)
    prm = ProcessRewardModel(hidden_size=256, vocab_size=1000)
    config = SearchConfig(num_samples=8, beam_width=4)
    
    # 创建引擎
    engine = TestTimeInferenceEngine(
        model=None,
        verifier=verifier,
        prm=prm,
        config=config
    )
    
    # 测试问题
    question = "一个农场有鸡和兔共35只，脚共94只。鸡兔各几只？"
    
    print(f"\n问题: {question}\n")
    
    # 测试不同策略
    strategies = ["best_of_n", "iterative_refinement"]
    
    for strategy in strategies:
        print(f"\n{'='*40}")
        print(f"策略: {strategy}")
        print('='*40)
        
        try:
            answer = engine.generate(question, strategy=strategy)
            print(f"最终答案: {answer[:100]}...")
        except Exception as e:
            print(f"错误: {e}")


def demo_ttt_layer():
    """TTT层演示"""
    print("\n" + "=" * 60)
    print("TTT-Linear层演示")
    print("=" * 60)
    
    # 创建TTT层
    ttt_layer = TTTLinnerLayer(hidden_size=128)
    
    # 模拟输入
    batch_size = 2
    seq_len = 32
    hidden_size = 128
    
    x = torch.randn(batch_size, seq_len, hidden_size)
    
    print(f"输入形状: {x.shape}")
    
    # 前向传播
    with torch.no_grad():
        output = ttt_layer(x, use_ttt=True)
    
    print(f"输出形状: {output.shape}")
    
    # 对比标准前向
    with torch.no_grad():
        output_standard = ttt_layer(x, use_ttt=False)
    print(f"标准输出形状: {output_standard.shape}")
    
    # 内存使用对比
    print("\n内存使用对比:")
    
    # TTT模型
    ttt_model = TTTSequentialModel(
        vocab_size=10000,
        hidden_size=128,
        num_layers=4,
        use_ttt=True
    )
    
    ttt_mem = ttt_model.estimate_memory(seq_len=1024, batch_size=1)
    print(f"TTT模型: {ttt_mem['total_mb']:.2f} MB (参数: {ttt_mem['param_memory_mb']:.2f} MB)")
    
    # 标准Transformer模型
    trans_model = TTTSequentialModel(
        vocab_size=10000,
        hidden_size=128,
        num_layers=4,
        use_ttt=False
    )
    
    trans_param_mem = trans_model.count_parameters() * 4 / (1024**2)
    # Transformer有O(n^2)的注意力矩阵
    trans_attn_mem = 1024 * 1024 * 4 / (1024**2)  # O(n^2)注意力
    print(f"Transformer模型: {trans_param_mem + trans_attn_mem:.2f} MB (参数: {trans_param_mem:.2f} MB, 注意力: {trans_attn_mem:.2f} MB)")


def demo_chain_of_thought():
    """思维链生成演示"""
    print("\n" + "=" * 60)
    print("思维链生成器演示")
    print("=" * 60)
    
    # 创建生成器
    generator = ChainOfThoughtGenerator(model=None)
    
    questions = [
        "25乘以16等于多少？",
        "一个长方形长12米，宽8米，周长是多少？",
        "小明比小红大3岁，5年后小明15岁，小红现在几岁？"
    ]
    
    for question in questions:
        print(f"\n问题: {question}")
        print("-" * 40)
        
        # 标准CoT
        result = generator.generate_cot(question)
        print(f"思维链 ({len(result['reasoning_chain'])} 步):")
        for i, step in enumerate(result['reasoning_chain'], 1):
            print(f"  步骤{i}: {step}")
        print(f"最终答案: {result['final_answer']}")
        print(f"置信度: {result['confidence']:.2f}")


def demo_prm_training():
    """PRM训练演示"""
    print("\n" + "=" * 60)
    print("过程奖励模型训练演示")
    print("=" * 60)
    
    # 创建PRM
    device = "cuda" if torch.cuda.is_available() else "cpu"
    prm = ProcessRewardModel(hidden_size=128, vocab_size=1000)
    trainer = PRMTrainer(prm, lr=1e-3, device=device)
    
    # 模拟数据
    batch_size = 4
    seq_len = 50
    num_steps = 5
    
    batch = {
        'input_ids': torch.randint(0, 1000, (batch_size, seq_len)),
        'step_boundaries': [[10, 20, 30, 40, 50] for _ in range(batch_size)],
        'step_labels': torch.rand(batch_size, num_steps)
    }
    
    print(f"批次大小: {batch_size}")
    print(f"序列长度: {seq_len}")
    print(f"推理步数: {num_steps}")
    print(f"设备: {device}")
    
    # 训练步骤
    losses = []
    for step in range(10):
        loss = trainer.train_step(batch)
        losses.append(loss)
        if step % 3 == 0:
            print(f"步骤 {step}: 损失 = {loss:.4f}")
    
    print(f"最终损失: {losses[-1]:.4f}")
    print(f"平均损失: {sum(losses)/len(losses):.4f}")


def run_all_demos():
    """运行所有演示"""
    print("开始测试时计算与推理优化演示\n")
    
    demo_test_time_compute()
    demo_ttt_layer()
    demo_chain_of_thought()
    demo_prm_training()
    
    print("\n" + "=" * 60)
    print("所有演示完成！")
    print("=" * 60)


if __name__ == "__main__":
    run_all_demos()
