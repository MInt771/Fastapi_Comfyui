from datetime import datetime
from enum import Enum
from typing import Optional

from sqlmodel import SQLModel, Field, create_engine


DATABASE_URL = "sqlite:///./data.db"

engine = create_engine(DATABASE_URL, echo=False)


class GenerationRecord(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)

    text: str
    style: str
    prompt_text: str

    # 使用 JSON 字符串存多图信息，简单直观
    image_urls: str
    md5_list: str

    created_at: datetime = Field(default_factory=datetime.utcnow)
    duration_ms: int


class TaskStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"


class GenerationTask(SQLModel, table=True):
    """
    异步任务表，用于排队执行生成请求。
    """

    id: Optional[int] = Field(default=None, primary_key=True)

    text: str
    style: str
    count: int

    # LLM 配置：兼容 /generate 的单模型与协同模式
    use_llm: bool = Field(default=False)
    llm_name: Optional[str] = None
    use_collab: bool = Field(default=False)
    planner_llm_name: Optional[str] = None
    reviewer_llm_name: Optional[str] = None

    status: TaskStatus = Field(default=TaskStatus.PENDING)
    error_message: Optional[str] = None

    # 对应的生成记录 ID（成功时填充）
    record_id: Optional[int] = None

    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


def init_db() -> None:
    """
    初始化数据库：如果表不存在则自动创建。
    """
    SQLModel.metadata.create_all(engine)


