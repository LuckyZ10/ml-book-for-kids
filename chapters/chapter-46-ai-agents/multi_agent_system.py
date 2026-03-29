"""
第四十六章: 多智能体系统实现 (Multi-Agent System)
===============================================

费曼比喻：多智能体系统就像一支交响乐队 🎵
- 每个乐手（Agent）都有自己的专长和乐器
- 指挥（Coordinator）确保大家协调一致
- 乐谱（SOP）规定了演奏的顺序和配合方式
- 只有大家各司其职、密切配合，才能演奏出美妙的音乐

本章实现MetaGPT风格的多智能体软件开发框架，包含：
- 产品经理、架构师、工程师、QA工程师等角色
- 黑板系统和消息队列两种通信机制
- 标准操作程序（SOP）流程管理
- 完整的软件开发流程模拟

参考论文：Hong et al. (2023) - MetaGPT: Meta Programming for A Multi-Agent Collaborative Framework
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Callable, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
import json
import time
import random
from collections import defaultdict, deque


# ==================== 消息系统 ====================

class MessageType(Enum):
    """消息类型枚举"""
    TASK_ASSIGN = auto()      # 任务分配
    TASK_COMPLETE = auto()    # 任务完成
    QUESTION = auto()         # 提问
    ANSWER = auto()           # 回答
    UPDATE = auto()           # 状态更新
    DECISION = auto()         # 决策通知
    FEEDBACK = auto()         # 反馈


@dataclass
class Message:
    """智能体间传递的消息
    
    费曼比喻：消息就像办公室里的便签条
    - from: 谁写的（发件人）
    - to: 给谁（收件人，None表示广播）
    - content: 便签上写的具体内容
    - msg_type: 便签的类型（紧急/普通/询问等）
    """
    from_agent: str
    to_agent: Optional[str]  # None表示广播给所有人
    content: str
    msg_type: MessageType
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __str__(self) -> str:
        to_str = self.to_agent if self.to_agent else "ALL"
        return f"[{self.from_agent} -> {to_str}] {self.msg_type.name}: {self.content[:50]}..."


# ==================== 黑板系统 ====================

class Blackboard:
    """黑板系统 - 共享工作空间
    
    费曼比喻：黑板系统就像厨房的共享白板
    - 所有厨师都可以看到白板上的内容
    - 主厨写下今日特餐（任务）
    - 配菜师标记蔬菜已备好（状态更新）
    - 任何人都可以查看当前进度
    - 信息透明，大家随时同步
    
    与消息队列的区别：
    - 消息队列：一对一或一对多的直接通信（私聊/群聊）
    - 黑板系统：公开的信息板，所有人可见（公告栏）
    """
    
    def __init__(self):
        self.data: Dict[str, Any] = {}  # 共享数据
        self.updates: List[Dict[str, Any]] = []  # 更新历史
        self.subscribers: Dict[str, Set[str]] = defaultdict(set)  # 主题订阅者
    
    def write(self, key: str, value: Any, agent_name: str = "system"):
        """写入数据到黑板"""
        old_value = self.data.get(key)
        self.data[key] = value
        
        # 记录更新
        update_record = {
            "key": key,
            "old_value": old_value,
            "new_value": value,
            "agent": agent_name,
            "timestamp": time.time()
        }
        self.updates.append(update_record)
        
        # 通知订阅者（这里简化处理）
        if key in self.subscribers:
            # 实际系统中会主动推送通知
            pass
    
    def read(self, key: str) -> Optional[Any]:
        """从黑板读取数据"""
        return self.data.get(key)
    
    def subscribe(self, agent_name: str, key_pattern: str):
        """订阅特定键的更新"""
        self.subscribers[key_pattern].add(agent_name)
    
    def get_history(self, key: Optional[str] = None) -> List[Dict[str, Any]]:
        """获取更新历史"""
        if key:
            return [u for u in self.updates if u["key"] == key]
        return self.updates
    
    def get_all(self) -> Dict[str, Any]:
        """获取所有数据"""
        return self.data.copy()
    
    def clear(self):
        """清空黑板"""
        self.data.clear()
        self.updates.clear()


# ==================== 消息队列 ====================

class MessageQueue:
    """消息队列 - 智能体间通信基础设施
    
    费曼比喻：消息队列就像公司的内部邮件系统
    - 每个员工（Agent）有自己的收件箱
    - 可以发送邮件给特定同事（点对点）
    - 可以发送到部门群组（广播/多播）
    - 邮件按时间排序，保证送达
    """
    
    def __init__(self):
        self.queues: Dict[str, deque] = defaultdict(deque)  # 每个Agent的队列
        self.broadcast_queue: deque = deque()  # 广播消息队列
        self.message_history: List[Message] = []  # 消息历史（用于调试）
    
    def send(self, message: Message):
        """发送消息"""
        self.message_history.append(message)
        
        if message.to_agent is None:
            # 广播消息 - 放入广播队列
            self.broadcast_queue.append(message)
        else:
            # 点对点消息
            self.queues[message.to_agent].append(message)
    
    def receive(self, agent_name: str, block: bool = False) -> Optional[Message]:
        """接收消息（非阻塞默认）"""
        # 先检查个人队列
        if self.queues[agent_name]:
            return self.queues[agent_name].popleft()
        
        # 再检查广播队列（简化：所有Agent都能看到所有广播）
        # 实际系统需要跟踪每个Agent已读的广播
        return None
    
    def receive_all(self, agent_name: str) -> List[Message]:
        """接收所有待处理消息"""
        messages = []
        
        # 获取个人消息
        while self.queues[agent_name]:
            messages.append(self.queues[agent_name].popleft())
        
        return messages
    
    def peek_broadcast(self, since: Optional[float] = None) -> List[Message]:
        """查看广播消息（不删除）"""
        if since is None:
            return list(self.broadcast_queue)
        return [m for m in self.broadcast_queue if m.timestamp > since]
    
    def get_history(self, from_agent: Optional[str] = None, 
                    to_agent: Optional[str] = None) -> List[Message]:
        """获取消息历史"""
        result = self.message_history
        if from_agent:
            result = [m for m in result if m.from_agent == from_agent]
        if to_agent:
            result = [m for m in result if m.to_agent == to_agent]
        return result
    
    def clear(self):
        """清空所有队列"""
        self.queues.clear()
        self.broadcast_queue.clear()
        self.message_history.clear()


# ==================== 基础智能体 ====================

@dataclass
class Task:
    """任务定义"""
    task_id: str
    description: str
    assigned_to: Optional[str] = None
    status: str = "pending"  # pending, in_progress, completed, failed
    priority: int = 1
    dependencies: List[str] = field(default_factory=list)
    deliverable: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None


class BaseAgent(ABC):
    """多智能体系统中的基础智能体
    
    费曼比喻：基础智能体就像餐厅里的员工岗位说明书
    - 每个岗位（Agent）有明确的职责（role）
    - 有固定的汇报对象（report_to）
    - 可以接收任务（inbox）
    - 能产出工作成果（deliverables）
    """
    
    def __init__(self, name: str, role: str, 
                 blackboard: Blackboard,
                 message_queue: MessageQueue,
                 report_to: Optional[str] = None):
        self.name = name
        self.role = role
        self.blackboard = blackboard
        self.message_queue = message_queue
        self.report_to = report_to
        
        self.inbox: deque = deque()  # 个人任务队列
        self.current_task: Optional[Task] = None
        self.deliverables: List[Dict[str, Any]] = []
        self.skills: Set[str] = set()
        
        # 模拟LLM接口
        self.llm: Optional[Callable] = None
    
    def set_llm(self, llm_fn: Callable):
        """设置LLM接口"""
        self.llm = llm_fn
    
    def send_message(self, to: Optional[str], content: str, 
                     msg_type: MessageType = MessageType.UPDATE,
                     metadata: Dict[str, Any] = None):
        """发送消息"""
        msg = Message(
            from_agent=self.name,
            to_agent=to,
            content=content,
            msg_type=msg_type,
            metadata=metadata or {}
        )
        self.message_queue.send(msg)
        print(f"📨 {msg}")
    
    def receive_messages(self) -> List[Message]:
        """接收消息"""
        return self.message_queue.receive_all(self.name)
    
    def assign_task(self, task: Task):
        """分配任务"""
        task.assigned_to = self.name
        self.inbox.append(task)
        print(f"📋 [{self.name}] 收到任务: {task.description}")
    
    def has_pending_task(self) -> bool:
        """是否有待处理任务"""
        return len(self.inbox) > 0 or self.current_task is not None
    
    def get_task(self) -> Optional[Task]:
        """获取下一个任务"""
        if self.current_task:
            return self.current_task
        if self.inbox:
            self.current_task = self.inbox.popleft()
            self.current_task.status = "in_progress"
            return self.current_task
        return None
    
    def complete_task(self, deliverable: str):
        """完成任务"""
        if self.current_task:
            self.current_task.status = "completed"
            self.current_task.completed_at = time.time()
            self.current_task.deliverable = deliverable
            
            # 存储交付物
            record = {
                "task": self.current_task,
                "deliverable": deliverable,
                "agent": self.name
            }
            self.deliverables.append(record)
            
            # 写入黑板
            self.blackboard.write(
                f"deliverable_{self.current_task.task_id}",
                record,
                self.name
            )
            
            # 通知上级
            if self.report_to:
                self.send_message(
                    to=self.report_to,
                    content=f"任务 {self.current_task.task_id} 已完成: {deliverable[:100]}",
                    msg_type=MessageType.TASK_COMPLETE,
                    metadata={"task_id": self.current_task.task_id}
                )
            
            print(f"✅ [{self.name}] 完成任务: {self.current_task.task_id}")
            self.current_task = None
    
    @abstractmethod
    def execute(self) -> bool:
        """执行一步工作，返回是否还有工作要做"""
        pass
    
    def run(self) -> str:
        """运行直到没有任务"""
        while self.execute():
            pass
        return f"[{self.name}] 所有任务已完成"


# ==================== 角色定义 ====================

class ProductManagerAgent(BaseAgent):
    """产品经理智能体
    
    职责：
    - 接收用户需求
    - 编写产品需求文档（PRD）
    - 定义功能列表和优先级
    - 协调各方需求
    
    费曼比喻：产品经理就像餐厅的主厨
    - 决定菜单上有哪些菜（功能）
    - 确定每道菜的口味（需求规格）
    - 确保菜品满足顾客的期望
    - 协调前后厨的工作
    """
    
    def __init__(self, name: str, blackboard: Blackboard, 
                 message_queue: MessageQueue):
        super().__init__(name, "Product Manager", blackboard, message_queue)
        self.skills = {"requirement_analysis", "prd_writing", "prioritization"}
        self.user_requirements: List[str] = []
    
    def add_requirement(self, requirement: str):
        """添加用户需求"""
        self.user_requirements.append(requirement)
        print(f"📝 [{self.name}] 收到需求: {requirement}")
    
    def execute(self) -> bool:
        """执行产品经理工作"""
        if not self.user_requirements:
            # 检查消息
            messages = self.receive_messages()
            for msg in messages:
                if msg.msg_type == MessageType.QUESTION:
                    # 回答关于需求的问题
                    self._answer_question(msg)
            return False
        
        # 有需求待处理，编写PRD
        requirement = self.user_requirements.pop(0)
        print(f"\n{'='*60}")
        print(f"📊 [{self.name}] 开始编写PRD")
        print(f"需求: {requirement}")
        print(f"{'='*60}\n")
        
        # 模拟编写PRD（实际应调用LLM）
        prd = self._write_prd(requirement)
        
        # 发布到黑板
        self.blackboard.write("prd", prd, self.name)
        self.blackboard.write("requirement", requirement, self.name)
        
        print(f"\n📄 PRD编写完成，已发布到黑板")
        
        # 通知架构师
        self.send_message(
            to="Architect",
            content="PRD已准备就绪，请开始系统设计",
            msg_type=MessageType.TASK_ASSIGN
        )
        
        return len(self.user_requirements) > 0
    
    def _write_prd(self, requirement: str) -> Dict[str, Any]:
        """编写产品需求文档"""
        # 模拟PRD内容
        prd = {
            "title": f"产品需求文档: {requirement[:30]}...",
            "version": "1.0",
            "overview": requirement,
            "user_stories": [
                f"作为用户，我想要{requirement}，以便提高效率",
                "作为管理员，我想要监控功能，以便了解系统状态"
            ],
            "functional_requirements": [
                {"id": "FR-001", "desc": "核心功能实现", "priority": "P0"},
                {"id": "FR-002", "desc": "用户界面设计", "priority": "P1"},
                {"id": "FR-003", "desc": "数据存储功能", "priority": "P1"},
                {"id": "FR-004", "desc": "API接口", "priority": "P2"}
            ],
            "non_functional_requirements": [
                {"id": "NFR-001", "desc": "响应时间 < 1秒", "priority": "P0"},
                {"id": "NFR-002", "desc": "支持1000并发用户", "priority": "P1"}
            ],
            "acceptance_criteria": [
                "所有功能测试通过",
                "性能测试达标",
                "代码审查通过"
            ]
        }
        return prd
    
    def _answer_question(self, msg: Message):
        """回答需求相关问题"""
        answer = f"关于'{msg.content}'，根据PRD要求..."
        self.send_message(
            to=msg.from_agent,
            content=answer,
            msg_type=MessageType.ANSWER
        )


class ArchitectAgent(BaseAgent):
    """架构师智能体
    
    职责：
    - 设计系统架构
    - 选择技术栈
    - 定义接口规范
    - 输出设计文档
    
    费曼比喻：架构师就像建筑设计师
    - 设计房子的整体结构（系统架构）
    - 选择建筑材料（技术栈）
    - 绘制施工图纸（接口文档）
    - 确保房子稳固安全
    """
    
    def __init__(self, name: str, blackboard: Blackboard,
                 message_queue: MessageQueue):
        super().__init__(name, "Architect", blackboard, message_queue, 
                        report_to="ProductManager")
        self.skills = {"system_design", "tech_selection", "api_design"}
        self.design_doc: Optional[Dict[str, Any]] = None
    
    def execute(self) -> bool:
        """执行架构设计工作"""
        # 检查是否有PRD
        prd = self.blackboard.read("prd")
        if not prd:
            print(f"⏳ [{self.name}] 等待PRD...")
            return False
        
        # 检查是否已设计过
        if self.blackboard.read("design_doc"):
            # 处理后续任务
            messages = self.receive_messages()
            return False
        
        print(f"\n{'='*60}")
        print(f"🏗️  [{self.name}] 开始系统设计")
        print(f"基于PRD: {prd['title']}")
        print(f"{'='*60}\n")
        
        # 设计系统架构
        self.design_doc = self._create_design(prd)
        
        # 发布到黑板
        self.blackboard.write("design_doc", self.design_doc, self.name)
        self.blackboard.write("api_spec", self.design_doc["api_design"], self.name)
        
        print(f"\n📐 架构设计完成:")
        print(f"  - 架构模式: {self.design_doc['architecture_pattern']}")
        print(f"  - 技术栈: {', '.join(self.design_doc['tech_stack'])}")
        print(f"  - 模块数: {len(self.design_doc['modules'])}")
        
        # 通知工程师
        self.send_message(
            to=None,  # 广播给所有工程师
            content="架构设计完成，请按设计文档开始开发",
            msg_type=MessageType.TASK_ASSIGN,
            metadata={"design_ready": True}
        )
        
        return False
    
    def _create_design(self, prd: Dict[str, Any]) -> Dict[str, Any]:
        """创建系统设计文档"""
        design = {
            "architecture_pattern": "微服务架构",
            "tech_stack": ["Python/FastAPI", "PostgreSQL", "Redis", "Docker"],
            "modules": [
                {
                    "name": "api_gateway",
                    "desc": "API网关，处理路由和认证",
                    "endpoints": ["/api/v1/auth", "/api/v1/data"]
                },
                {
                    "name": "core_service",
                    "desc": "核心业务逻辑服务",
                    "interfaces": ["UserService", "DataService"]
                },
                {
                    "name": "data_layer",
                    "desc": "数据访问层",
                    "components": ["Repository", "Cache"]
                }
            ],
            "data_model": {
                "User": {"id": "int", "name": "str", "email": "str"},
                "Data": {"id": "int", "content": "str", "created_at": "datetime"}
            },
            "api_design": {
                "GET /api/v1/users": "获取用户列表",
                "POST /api/v1/users": "创建用户",
                "GET /api/v1/data/{id}": "获取数据",
                "POST /api/v1/data": "创建数据"
            },
            "deployment": {
                "containerization": "Docker",
                "orchestration": "Docker Compose",
                "scaling": "水平扩展支持"
            }
        }
        return design


class EngineerAgent(BaseAgent):
    """工程师智能体
    
    职责：
    - 根据设计文档编写代码
    - 实现功能模块
    - 编写单元测试
    - 代码审查
    
    费曼比喻：工程师就像建筑工人
    - 按照设计师的图纸施工（按设计文档编码）
    - 搭建房子的框架和墙体（实现核心功能）
    - 铺设水电管线（接口实现）
    - 进行质量自检（单元测试）
    """
    
    def __init__(self, name: str, blackboard: Blackboard,
                 message_queue: MessageQueue, specialty: str = "backend"):
        super().__init__(name, f"Engineer ({specialty})", blackboard, message_queue,
                        report_to="Architect")
        self.specialty = specialty
        self.skills = {f"{specialty}_dev", "unit_testing", "code_review"}
        self.code_modules: Dict[str, str] = {}
    
    def execute(self) -> bool:
        """执行开发工作"""
        # 检查设计文档
        design_doc = self.blackboard.read("design_doc")
        if not design_doc:
            print(f"⏳ [{self.name}] 等待设计文档...")
            return False
        
        # 检查是否已有任务
        task = self.get_task()
        if not task:
            # 自主创建开发任务
            task = self._create_development_task(design_doc)
            self.assign_task(task)
            task = self.get_task()
        
        print(f"\n{'─'*60}")
        print(f"💻 [{self.name}] 开始开发: {task.description}")
        print(f"{'─'*60}\n")
        
        # 模拟编码过程
        code = self._write_code(task, design_doc)
        self.code_modules[task.task_id] = code
        
        # 完成任务
        self.complete_task(code)
        
        # 检查是否还有更多模块需要开发
        completed = len(self.code_modules)
        total = len(design_doc["modules"])
        
        if completed >= total:
            # 所有模块开发完成
            self._finalize_implementation()
            return False
        
        return True
    
    def _create_development_task(self, design_doc: Dict[str, Any]) -> Task:
        """创建开发任务"""
        module_idx = len(self.code_modules)
        if module_idx < len(design_doc["modules"]):
            module = design_doc["modules"][module_idx]
            return Task(
                task_id=f"dev_{module['name']}",
                description=f"实现模块: {module['name']} - {module['desc']}",
                priority=1
            )
        return Task(
            task_id="dev_cleanup",
            description="代码清理和优化",
            priority=2
        )
    
    def _write_code(self, task: Task, design_doc: Dict[str, Any]) -> str:
        """编写代码（模拟）"""
        # 模拟生成代码
        code_template = f'''"""
{task.description}
Generated by {self.name}
"""

class {task.task_id.title().replace("_", "")}:
    """模块实现"""
    
    def __init__(self):
        self.initialized = True
    
    def process(self, data):
        """处理数据"""
        # TODO: 实现具体逻辑
        return {{"status": "success", "data": data}}
    
    def validate(self, input_data):
        """验证输入"""
        return input_data is not None

# 单元测试
if __name__ == "__main__":
    module = {task.task_id.title().replace("_", "")}()
    result = module.process({{"test": "data"}})
    print(result)
    assert result["status"] == "success"
    print("✓ 单元测试通过")
'''
        return code_template
    
    def _finalize_implementation(self):
        """完成实现，提交代码"""
        all_code = "\n\n".join([
            f"# Module: {name}\n{code}" 
            for name, code in self.code_modules.items()
        ])
        
        self.blackboard.write("implementation", all_code, self.name)
        self.blackboard.write("implementation_status", "completed", self.name)
        
        print(f"\n🚀 [{self.name}] 所有模块开发完成!")
        
        # 通知QA
        self.send_message(
            to="QAEngineer",
            content="开发完成，代码已提交，请开始测试",
            msg_type=MessageType.TASK_ASSIGN
        )


class QAEngineerAgent(BaseAgent):
    """QA工程师智能体
    
    职责：
    - 编写测试用例
    - 执行功能测试
    - 进行代码审查
    - 报告和跟踪缺陷
    
    费曼比喻：QA工程师就像质检员
    - 制定检测标准（测试用例）
    - 检查产品是否符合规格（功能测试）
    - 发现质量问题（Bug报告）
    - 确保出厂产品质量（发布把关）
    """
    
    def __init__(self, name: str, blackboard: Blackboard,
                 message_queue: MessageQueue):
        super().__init__(name, "QA Engineer", blackboard, message_queue,
                        report_to="ProductManager")
        self.skills = {"test_design", "test_execution", "bug_reporting"}
        self.test_cases: List[Dict[str, Any]] = []
        self.test_results: List[Dict[str, Any]] = []
        self.bugs: List[Dict[str, Any]] = []
    
    def execute(self) -> bool:
        """执行QA工作"""
        # 检查是否有可测试的实现
        impl = self.blackboard.read("implementation")
        if not impl:
            print(f"⏳ [{self.name}] 等待代码实现...")
            return False
        
        # 检查是否已有测试用例
        if not self.test_cases:
            # 第一阶段：编写测试用例
            return self._create_test_cases()
        
        # 检查是否已测试
        if not self.test_results:
            # 第二阶段：执行测试
            return self._run_tests(impl)
        
        # 第三阶段：总结报告
        return self._generate_report()
    
    def _create_test_cases(self) -> bool:
        """创建测试用例"""
        print(f"\n{'='*60}")
        print(f"🧪 [{self.name}] 编写测试用例")
        print(f"{'='*60}\n")
        
        # 基于PRD创建测试用例
        prd = self.blackboard.read("prd")
        
        self.test_cases = [
            {
                "id": "TC-001",
                "name": "核心功能测试",
                "steps": ["准备测试数据", "执行功能调用", "验证结果"],
                "expected": "功能正常执行，返回预期结果"
            },
            {
                "id": "TC-002", 
                "name": "边界条件测试",
                "steps": ["输入空值", "输入超大值", "输入特殊字符"],
                "expected": "系统优雅处理，不崩溃"
            },
            {
                "id": "TC-003",
                "name": "性能测试",
                "steps": ["模拟100并发请求"],
                "expected": "响应时间 < 1秒"
            }
        ]
        
        print(f"✓ 创建了 {len(self.test_cases)} 个测试用例")
        for tc in self.test_cases:
            print(f"  - {tc['id']}: {tc['name']}")
        
        self.blackboard.write("test_cases", self.test_cases, self.name)
        return True
    
    def _run_tests(self, impl: str) -> bool:
        """执行测试"""
        print(f"\n{'='*60}")
        print(f"🔍 [{self.name}] 执行测试")
        print(f"{'='*60}\n")
        
        for tc in self.test_cases:
            print(f"  运行: {tc['id']} - {tc['name']}")
            
            # 模拟测试结果（随机产生一些Bug）
            if random.random() > 0.8:  # 20%概率发现Bug
                bug = {
                    "id": f"BUG-{len(self.bugs)+1:03d}",
                    "test_case": tc['id'],
                    "severity": random.choice(["高", "中", "低"]),
                    "description": f"在{tc['name']}中发现异常行为"
                }
                self.bugs.append(bug)
                self.test_results.append({
                    "test_case": tc['id'],
                    "status": "FAILED",
                    "bug_id": bug['id']
                })
                print(f"    ❌ 失败 - 发现Bug: {bug['id']}")
            else:
                self.test_results.append({
                    "test_case": tc['id'],
                    "status": "PASSED"
                })
                print(f"    ✓ 通过")
        
        self.blackboard.write("test_results", self.test_results, self.name)
        self.blackboard.write("bugs", self.bugs, self.name)
        
        return True
    
    def _generate_report(self) -> bool:
        """生成测试报告"""
        print(f"\n{'='*60}")
        print(f"📊 [{self.name}] 生成测试报告")
        print(f"{'='*60}\n")
        
        passed = sum(1 for r in self.test_results if r['status'] == 'PASSED')
        failed = len(self.test_results) - passed
        
        report = {
            "total_tests": len(self.test_results),
            "passed": passed,
            "failed": failed,
            "pass_rate": f"{passed/len(self.test_results)*100:.1f}%",
            "bugs_found": len(self.bugs),
            "conclusion": "测试通过，可以发布" if failed == 0 else f"发现{failed}个问题，需要修复",
            "recommendations": [
                "继续监控系统性能" if failed == 0 else "优先修复高优先级Bug"
            ]
        }
        
        self.blackboard.write("qa_report", report, self.name)
        
        print(f"测试结果:")
        print(f"  - 总测试数: {report['total_tests']}")
        print(f"  - 通过: {report['passed']} ✓")
        print(f"  - 失败: {report['failed']} {'✓' if failed == 0 else '❌'}")
        print(f"  - 通过率: {report['pass_rate']}")
        print(f"  - Bug数: {report['bugs_found']}")
        print(f"\n结论: {report['conclusion']}")
        
        # 通知产品经理
        self.send_message(
            to="ProductManager",
            content=f"测试完成: {report['conclusion']}",
            msg_type=MessageType.UPDATE,
            metadata={"report": report}
        )
        
        return False


# ==================== 多智能体系统协调器 ====================

class MultiAgentSystem:
    """多智能体系统协调器
    
    费曼比喻：多智能体系统协调器就像交响乐团的指挥
    - 确定演出的曲目（系统目标）
    - 协调各个声部的进入时机（任务调度）
    - 确保整体和谐（冲突解决）
    - 掌控演出节奏（流程控制）
    
    MetaGPT风格SOP流程：
    1. 需求分析（产品经理）
    2. 架构设计（架构师）
    3. 代码实现（工程师）
    4. 测试验收（QA）
    """
    
    def __init__(self, name: str = "MetaGPT-Style System"):
        self.name = name
        self.blackboard = Blackboard()
        self.message_queue = MessageQueue()
        self.agents: Dict[str, BaseAgent] = {}
        
        # 统计信息
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
    
    def register_agent(self, agent: BaseAgent):
        """注册智能体"""
        self.agents[agent.name] = agent
        print(f"✅ 注册Agent: {agent.name} ({agent.role})")
    
    def create_software_team(self) -> Dict[str, BaseAgent]:
        """创建软件开发团队（MetaGPT风格）"""
        print(f"\n{'='*60}")
        print(f"🚀 创建软件开发团队")
        print(f"{'='*60}\n")
        
        # 创建各角色
        pm = ProductManagerAgent("ProductManager", self.blackboard, self.message_queue)
        architect = ArchitectAgent("Architect", self.blackboard, self.message_queue)
        engineer = EngineerAgent("Engineer", self.blackboard, self.message_queue, "fullstack")
        qa = QAEngineerAgent("QAEngineer", self.blackboard, self.message_queue)
        
        # 注册到系统
        self.register_agent(pm)
        self.register_agent(architect)
        self.register_agent(engineer)
        self.register_agent(qa)
        
        return {
            "pm": pm,
            "architect": architect,
            "engineer": engineer,
            "qa": qa
        }
    
    def run_software_project(self, requirement: str) -> Dict[str, Any]:
        """运行完整的软件开发项目
        
        SOP流程：
        1. PM编写PRD
        2. 架构师设计系统
        3. 工程师开发代码
        4. QA测试验收
        """
        print(f"\n{'='*70}")
        print(f"🎯 开始软件开发项目: {requirement[:50]}...")
        print(f"{'='*70}\n")
        
        self.start_time = time.time()
        
        # 获取团队
        pm = self.agents.get("ProductManager")
        architect = self.agents.get("Architect")
        engineer = self.agents.get("Engineer")
        qa = self.agents.get("QAEngineer")
        
        if not all([pm, architect, engineer, qa]):
            raise ValueError("请先调用create_software_team()创建团队")
        
        # ===== 阶段1: 需求分析 =====
        print(f"\n📌 阶段1: 需求分析")
        print(f"{'─'*50}")
        pm.add_requirement(requirement)
        pm.run()
        
        # ===== 阶段2: 架构设计 =====
        print(f"\n📌 阶段2: 架构设计")
        print(f"{'─'*50}")
        architect.run()
        
        # ===== 阶段3: 代码开发 =====
        print(f"\n📌 阶段3: 代码开发")
        print(f"{'─'*50}")
        engineer.run()
        
        # ===== 阶段4: 测试验收 =====
        print(f"\n📌 阶段4: 测试验收")
        print(f"{'─'*50}")
        qa.run()
        
        self.end_time = time.time()
        
        # 生成项目报告
        return self._generate_project_report()
    
    def _generate_project_report(self) -> Dict[str, Any]:
        """生成项目报告"""
        duration = self.end_time - self.start_time if self.end_time and self.start_time else 0
        
        report = {
            "project_name": self.name,
            "duration_seconds": round(duration, 2),
            "artifacts": {
                "prd": self.blackboard.read("prd"),
                "design_doc": self.blackboard.read("design_doc"),
                "implementation": self.blackboard.read("implementation"),
                "qa_report": self.blackboard.read("qa_report")
            },
            "statistics": {
                "agent_count": len(self.agents),
                "message_count": len(self.message_queue.message_history),
                "blackboard_updates": len(self.blackboard.updates)
            },
            "status": "completed"
        }
        
        print(f"\n{'='*70}")
        print(f"📋 项目完成报告")
        print(f"{'='*70}")
        print(f"耗时: {duration:.2f}秒")
        print(f"参与Agent: {report['statistics']['agent_count']}个")
        print(f"消息交换: {report['statistics']['message_count']}条")
        print(f"黑板更新: {report['statistics']['blackboard_updates']}次")
        
        if report['artifacts']['qa_report']:
            qa_report = report['artifacts']['qa_report']
            print(f"\n质量报告:")
            print(f"  - 通过率: {qa_report.get('pass_rate', 'N/A')}")
            print(f"  - 发现Bug: {qa_report.get('bugs_found', 0)}个")
            print(f"  - 结论: {qa_report.get('conclusion', 'N/A')}")
        
        return report
    
    def get_blackboard_summary(self) -> str:
        """获取黑板内容摘要"""
        data = self.blackboard.get_all()
        summary = "黑板内容:\n"
        for key, value in data.items():
            if isinstance(value, dict):
                summary += f"  {key}: {value.get('title', str(value)[:50])}...\n"
            else:
                summary += f"  {key}: {str(value)[:50]}...\n"
        return summary
    
    def get_message_log(self) -> str:
        """获取消息日志"""
        messages = self.message_queue.message_history
        log = f"消息日志 ({len(messages)}条):\n"
        for msg in messages[:10]:  # 只显示前10条
            log += f"  {msg}\n"
        if len(messages) > 10:
            log += f"  ... 还有 {len(messages)-10} 条消息\n"
        return log


# ==================== 演示 ====================

def demo_blackboard():
    """演示黑板系统"""
    print("\n" + "="*60)
    print("演示: 黑板系统")
    print("="*60)
    
    bb = Blackboard()
    
    # Agent A写入数据
    bb.write("task_status", "in_progress", "Agent_A")
    bb.write("partial_result", {"value": 42}, "Agent_A")
    
    # Agent B读取数据
    status = bb.read("task_status")
    result = bb.read("partial_result")
    
    print(f"Agent_B 读取到: status={status}, result={result}")
    
    # Agent B更新数据
    bb.write("task_status", "completed", "Agent_B")
    
    # 查看历史
    history = bb.get_history("task_status")
    print(f"\n状态变更历史:")
    for h in history:
        print(f"  {h['agent']}: {h['old_value']} -> {h['new_value']}")


def demo_message_queue():
    """演示消息队列"""
    print("\n" + "="*60)
    print("演示: 消息队列")
    print("="*60)
    
    mq = MessageQueue()
    
    # 发送点对点消息
    msg1 = Message("Alice", "Bob", "你能帮我检查代码吗？", MessageType.QUESTION)
    mq.send(msg1)
    
    # 发送广播消息
    msg2 = Message("Manager", None, "项目启动会议在下午2点", MessageType.UPDATE)
    mq.send(msg2)
    
    # Bob接收消息
    bob_msgs = mq.receive_all("Bob")
    print(f"Bob收到 {len(bob_msgs)} 条消息:")
    for m in bob_msgs:
        print(f"  - {m.from_agent}: {m.content}")


def demo_multi_agent_system():
    """演示完整的多智能体系统"""
    print("\n" + "="*70)
    print("演示: MetaGPT风格多智能体软件开发")
    print("="*70)
    
    # 创建系统
    system = MultiAgentSystem("智能任务管理系统")
    
    # 创建团队
    team = system.create_software_team()
    
    # 运行项目
    requirement = "开发一个智能任务管理系统，支持任务创建、分配、跟踪和报告功能"
    report = system.run_software_project(requirement)
    
    print(f"\n{'='*70}")
    print("系统状态摘要:")
    print(system.get_blackboard_summary())
    print(f"\n{system.get_message_log()}")
    
    return report


if __name__ == "__main__":
    # 运行演示
    demo_blackboard()
    demo_message_queue()
    demo_multi_agent_system()
