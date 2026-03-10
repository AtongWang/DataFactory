import datetime
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Boolean, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()

class ChatSession(Base):
    """聊天会话模型"""
    __tablename__ = 'chat_sessions'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False, default="新会话")
    database_name = Column(String(255))
    table_name = Column(String(255))
    model_name = Column(String(255))
    temperature = Column(Float, default=0.7)
    is_table_locked = Column(Boolean, default=False)  # 标记数据表是否已锁定
    auto_train = Column(Boolean, default=True) # 标记是否启用自动训练
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    
    # 关系
    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan")
    
    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "database_name": self.database_name,
            "table_name": self.table_name,
            "model_name": self.model_name,
            "temperature": self.temperature,
            "is_table_locked": self.is_table_locked,
            "auto_train": self.auto_train,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "message_count": len(self.messages) if self.messages else 0
        }

class ChatMessage(Base):
    """聊天消息模型"""
    __tablename__ = 'chat_messages'
    
    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey('chat_sessions.id'), nullable=False)
    role = Column(String(50), nullable=False)  # 'user' 或 'assistant'
    content = Column(Text, nullable=False)
    sql = Column(Text)  # 生成的SQL（对于assistant消息）
    result = Column(Text)  # SQL执行结果（对于assistant消息，JSON格式存储）
    visualization = Column(Text)  # 可视化配置（对于assistant消息，JSON格式存储）
    reasoning = Column(Text)  # 推理过程（对于assistant消息）
    thinking = Column(Text)  # 思考过程（对于assistant消息，存储<think>标签内容）
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    # 关系
    session = relationship("ChatSession", back_populates="messages")
    
    def to_dict(self):
        return {
            "id": self.id,
            "session_id": self.session_id,
            "role": self.role,
            "content": self.content,
            "sql": self.sql,
            "result": self.result,
            "visualization": self.visualization,
            "reasoning": self.reasoning,
            "thinking": self.thinking,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }

class SavedQuery(Base):
    """保存的查询模型"""
    __tablename__ = 'saved_queries'
    
    id = Column(Integer, primary_key=True)
    title = Column(String(255), nullable=False)
    description = Column(Text)
    question = Column(Text, nullable=False)
    sql = Column(Text, nullable=False)
    result = Column(Text)  # JSON格式存储
    visualization = Column(Text)  # JSON格式存储
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    
    def to_dict(self):
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "question": self.question,
            "sql": self.sql,
            "result": self.result,
            "visualization": self.visualization,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        } 