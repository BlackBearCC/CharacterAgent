# 导入模块
import datetime
import json
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Union

from sqlalchemy import Column, Integer, Text, Float, create_engine, text

try:
    from sqlalchemy.orm import declarative_base, sessionmaker
except ImportError:
    from sqlalchemy.ext.declarative import declarative_base

logger = logging.getLogger(__name__)

# Opinion 类：代表一个观点实体
class Opinion:
    """Represents an opinion entity."""

    def __init__(self, opinion_id: int, opinion: str, score: float, reason: str):
        self.opinion_id = opinion_id
        self.opinion = opinion
        self.score = score
        self.reason = reason

# BaseOpinionConverter 类：将 Opinion 转换为 SQLAlchemy 模型的基类
class BaseOpinionConverter(ABC):
    """Convert Opinion to the SQLAlchemy model."""

    @abstractmethod
    def from_sql_model(self, sql_opinion: Any) -> Opinion:
        """Convert a SQLAlchemy model to an Opinion instance."""
        raise NotImplementedError

    @abstractmethod
    def to_sql_model(self, opinion: Opinion) -> Any:
        """Convert an Opinion instance to a SQLAlchemy model."""
        raise NotImplementedError

    @abstractmethod
    def get_sql_model_class(self) -> Any:
        """Get the SQLAlchemy model class."""
        raise NotImplementedError

# 创建 Opinion 模型类
def create_opinion_model(table_name: str, DynamicBase: Any) -> Any:
    class OpinionModel(DynamicBase):
        __tablename__ = table_name
        opinion_id = Column(Integer, primary_key=True)
        opinion = Column(Text)
        score = Column(Float)
        reason = Column(Text)

    return OpinionModel

# 默认的 Opinion 转换器，用于 OpinionHistory
class DefaultOpinionConverter(BaseOpinionConverter):
    """The default opinion converter for OpinionHistory."""

    def __init__(self, table_name: str):
        self.model_class = create_opinion_model(table_name, declarative_base())

    def from_sql_model(self, sql_opinion: Any) -> Opinion:
        return Opinion(
            opinion_id=sql_opinion.opinion_id,
            opinion=sql_opinion.opinion,
            score=sql_opinion.score,
            reason=sql_opinion.reason
        )

    def to_sql_model(self, opinion: Opinion) -> Any:
        return self.model_class(
            opinion_id=opinion.opinion_id,
            opinion=opinion.opinion,
            score=opinion.score,
            reason=opinion.reason
        )

    def get_sql_model_class(self) -> Any:
        return self.model_class

# OpinionHistory 类：存储在 SQL 数据库中的观点历史
class OpinionMemory:
    """Opinion history stored in an SQL database."""

    def __init__(
        self,
        connection_string: str,
        table_name: str = "opinion_store",
        _create_table_if_not_exists: bool = True
    ):
        self.connection_string = connection_string
        self.engine = create_engine(connection_string, echo=False)
        self.table_name = table_name
        self.Session = sessionmaker(bind=self.engine)
        self.OpinionModel = self._create_opinion_model()
        if _create_table_if_not_exists:
            self._create_table_if_not_exists()

    def _create_opinion_model(self) -> Any:
        # 创建 Opinion 模型
        Base = declarative_base()

        class OpinionModel(Base):
            __tablename__ = self.table_name
            opinion_id = Column(Integer, primary_key=True)
            opinion = Column(Text)
            score = Column(Float)
            reason = Column(Text)

        return OpinionModel

    def _create_table_if_not_exists(self) -> None:
        # 如果不存在则创建表
        self.OpinionModel.metadata.create_all(self.engine)

    def get_opinions(self, count: int = 100) -> List[Opinion]:
        # 获取观点列表
        with self.Session() as session:
            query = session.query(self.OpinionModel).order_by(self.OpinionModel.opinion_id.desc()).limit(count)
            opinions = []
            for row in query.all():
                opinions.append(Opinion(opinion_id=row.opinion_id, opinion=row.opinion, score=row.score, reason=row.reason))
            return opinions

    def add_opinion(self, data: Union[Opinion, str]) -> None:
        # 如果输入参数是 Opinion 对象，则直接添加到数据库
        if isinstance(data, Opinion):
            opinion = data
        # 如果输入参数是 JSON 字符串，则将其转换为 Opinion 对象
        elif isinstance(data, str):
            json_data = json.loads(data)
            opinion = Opinion(
                opinion_id=None,  # 如果 opinion_id 是自增的数据库字段，可以设置为 None
                opinion=json_data.get("opinion"),
                score=float(json_data.get("score")),  # 将字符串类型的 score 转换为浮点数
                reason=json_data.get("reason")
            )
        else:
            raise ValueError("Invalid input type. Input must be either an Opinion object or a JSON string.")

        # 添加观点
        with self.Session() as session:
            new_opinion = self.OpinionModel(opinion_id=opinion.opinion_id, opinion=opinion.opinion, score=opinion.score,
                                            reason=opinion.reason)
            session.add(new_opinion)
            session.commit()

    def update_opinion(self, opinion_id: int, new_score: float, new_reason: str) -> None:
        # 更新观点
        with self.Session() as session:
            opinion_to_update = session.query(self.OpinionModel).filter_by(opinion_id=opinion_id).first()
            if opinion_to_update:
                opinion_to_update.score = new_score
                opinion_to_update.reason = new_reason
                session.commit()

    def buffer(self, count: int = 100) -> str:
        # 获取并返回观点历史的缓冲区
        try:
            _opinions = self.get_opinions(count)
            history_buffer = ""

            for opinion in reversed(_opinions):
                history_buffer += f"ID: {opinion.opinion_id}, Opinion: {opinion.opinion}, Score: {opinion.score}, Reason: {opinion.reason}\n"

            return history_buffer.strip()
        except Exception as e:
            logger.error(f"Error occurred while fetching opinions: {str(e)}")
            return "No opinions found"
