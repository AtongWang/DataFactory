import datetime
import json
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Boolean, Float, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()

class KnowledgeGraph(Base):
    """知识图谱模型"""
    __tablename__ = 'knowledge_graphs'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    database = Column(String(255), nullable=False)
    table = Column(String(255), nullable=False)
    node_count = Column(Integer, default=0)
    relationship_count = Column(Integer, default=0)
    config = Column(Text)  # 存储图谱构建配置的JSON
    status = Column(String(50), default='created')  # created, building, completed, failed
    error_message = Column(Text)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    
    # 关系
    node_types = relationship("NodeType", back_populates="knowledge_graph", cascade="all, delete-orphan")
    relationship_types = relationship("RelationshipType", back_populates="knowledge_graph", cascade="all, delete-orphan")
    
    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "database": self.database,
            "table": self.table,
            "node_count": self.node_count,
            "relationship_count": self.relationship_count,
            "config": json.loads(self.config) if self.config else None,
            "status": self.status,
            "error_message": self.error_message,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "node_types": [nt.to_dict() for nt in self.node_types] if self.node_types else [],
            "relationship_types": [rt.to_dict() for rt in self.relationship_types] if self.relationship_types else []
        }

class NodeType(Base):
    """实体类型模型"""
    __tablename__ = 'node_types'
    
    id = Column(Integer, primary_key=True)
    knowledge_graph_id = Column(Integer, ForeignKey('knowledge_graphs.id', ondelete='CASCADE'), nullable=False)
    name = Column(String(255), nullable=False)
    identifier_columns = Column(Text, nullable=False)  # JSON string list
    attribute_columns = Column(Text)  # JSON string list, optional
    split_config = Column(Text) # JSON string like {"enabled": bool, "delimiter": str}, optional
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    # 关系
    knowledge_graph = relationship("KnowledgeGraph", back_populates="node_types")
    source_relationships = relationship("RelationshipType", foreign_keys="RelationshipType.source_node_type_id", back_populates="source_node_type")
    target_relationships = relationship("RelationshipType", foreign_keys="RelationshipType.target_node_type_id", back_populates="target_node_type")
    
    def to_dict(self):
        return {
            "id": self.id,
            "knowledge_graph_id": self.knowledge_graph_id,
            "name": self.name,
            "identifier_columns": json.loads(self.identifier_columns) if self.identifier_columns else [],
            "attribute_columns": json.loads(self.attribute_columns) if self.attribute_columns else [],
            "split_config": json.loads(self.split_config) if self.split_config else {'enabled': False, 'delimiter': None},
            "created_at": self.created_at.isoformat() if self.created_at else None
        }

class RelationshipType(Base):
    """关系类型模型"""
    __tablename__ = 'relationship_types'
    
    id = Column(Integer, primary_key=True)
    knowledge_graph_id = Column(Integer, ForeignKey('knowledge_graphs.id'), nullable=False)
    source_node_type_id = Column(Integer, ForeignKey('node_types.id'), nullable=False)
    target_node_type_id = Column(Integer, ForeignKey('node_types.id'), nullable=False)
    type = Column(String(255), nullable=False) 
    direction = Column(String(50), default='uni')  # uni 或 bi
    matching_mode = Column(String(50), default='intra-row')  # intra-row 或 inter-row
    inter_row_config = Column(Text)  # 跨行匹配配置的JSON
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    # 关系
    knowledge_graph = relationship("KnowledgeGraph", back_populates="relationship_types")
    source_node_type = relationship("NodeType", foreign_keys=[source_node_type_id], back_populates="source_relationships")
    target_node_type = relationship("NodeType", foreign_keys=[target_node_type_id], back_populates="target_relationships")
    
    def to_dict(self):
        return {
            "id": self.id,
            "knowledge_graph_id": self.knowledge_graph_id,
            "source_node_type_id": self.source_node_type_id,
            "target_node_type_id": self.target_node_type_id,
            "type": self.type,
            "direction": self.direction,
            "matching_mode": self.matching_mode,
            "inter_row_config": json.loads(self.inter_row_config) if self.inter_row_config else None,
            "created_at": self.created_at.isoformat() if self.created_at else None
        } 