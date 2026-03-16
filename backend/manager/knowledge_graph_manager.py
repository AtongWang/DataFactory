import os
import json
import logging
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, scoped_session
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, date
from neo4j import GraphDatabase
from neo4j.exceptions import ClientError
import time  # 用于 GDS 临时图名称
import re

from backend.models.knowledge_graph_models import (
    KnowledgeGraph,
    NodeType,
    RelationshipType,
    Base,
)
from backend.manager.vanna_manager import VannaManager
from backend.utils.db_utils import get_database_connection_string
from backend.utils.openai_compat import strip_reasoning_content_tags

logger = logging.getLogger(__name__)

# 不再使用环境变量, 改为从配置读取
# NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
# NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
# NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "12345678")


def safe_create_completion_for_kg(client, completion_params, model_name="unknown"):
    """
    安全的创建completion请求，包含provider参数错误的自动回退机制
    专门用于知识图谱管理器中的LLM调用
    """
    try:
        # 首先尝试原始请求
        response = client.chat.completions.create(**completion_params)
        return response
    except Exception as e:
        error_msg = str(e).lower()

        # 检查是否是provider参数相关的错误
        if "provider" in error_msg and (
            "unexpected" in error_msg or "invalid" in error_msg
        ):
            logger.warning(
                f"知识图谱管理器: 模型 {model_name} 不支持provider参数，尝试移除provider参数后重试"
            )

            # 创建不包含provider参数的副本
            fallback_params = completion_params.copy()
            if "provider" in fallback_params:
                del fallback_params["provider"]

            try:
                # 重试不带provider参数的请求
                response = client.chat.completions.create(**fallback_params)
                logger.info(
                    f"知识图谱管理器: 模型 {model_name} 成功使用fallback请求（无provider参数）"
                )
                return response
            except Exception as fallback_e:
                logger.error(
                    f"知识图谱管理器: 模型 {model_name} fallback请求也失败: {str(fallback_e)}"
                )
                raise fallback_e
        else:
            # 其他类型的错误，直接抛出
            raise e


class KnowledgeGraphManager:
    def __init__(self, db_path="db_file/kg.db", multi_graph_strategy="namespace"):
        """初始化知识图谱管理器

        Args:
            db_path: SQLite数据库文件路径，用于存储图谱元数据
            multi_graph_strategy: 多图谱共存策略
                - 'namespace': 使用命名空间（graph_id前缀），推荐
                - 'label': 使用标签隔离
                - 'legacy': 传统模式（可能有冲突）
        """
        self.db_path = db_path
        self.multi_graph_strategy = multi_graph_strategy
        self.engine = create_engine(f"sqlite:///{db_path}")
        Base.metadata.create_all(self.engine)
        self.Session = scoped_session(sessionmaker(bind=self.engine))

        # 用于执行长时间任务的线程池
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.building_tasks = {}  # 存储正在构建的任务

        # 使用全局VannaManager实例而不是创建新实例
        from backend.services.vanna_service import vanna_manager

        self.vanna_manager = vanna_manager

        # 初始化Neo4j连接
        self._init_neo4j_connection()

    def _init_neo4j_connection(self):
        """初始化Neo4j连接"""
        # 从配置中获取Neo4j连接信息
        config = self.vanna_manager.get_config()
        # 处理配置可能是dict类型的情况，并确保当配置中没有neo4j部分时使用默认值
        if isinstance(config, dict):
            neo4j_config = config.get("neo4j", {})
            neo4j_uri = neo4j_config.get("uri", "bolt://localhost:7687")
            neo4j_user = neo4j_config.get("user", "neo4j")
            neo4j_password = neo4j_config.get("password", "12345678")
        else:
            # 处理配置是AppConfig对象的情况
            try:
                neo4j_uri = config.neo4j.uri
                neo4j_user = config.neo4j.user
                neo4j_password = config.neo4j.password
            except AttributeError:
                # 如果配置对象没有neo4j属性，使用默认值
                neo4j_uri = "bolt://localhost:7687"
                neo4j_user = "neo4j"
                neo4j_password = "12345678"

        # Initialize Neo4j Driver
        try:
            # 如果之前有连接，先关闭
            if hasattr(self, "neo4j_driver") and self.neo4j_driver:
                self.neo4j_driver.close()

            self.neo4j_driver = GraphDatabase.driver(
                neo4j_uri, auth=(neo4j_user, neo4j_password)
            )
            self.neo4j_driver.verify_connectivity()
            logger.info("成功连接到 Neo4j")
        except Exception as e:
            logger.error(f"连接 Neo4j 失败: {e}")
            self.neo4j_driver = None  # Set driver to None if connection fails

    def reload_config(self, new_config_dict):
        """重新加载配置并更新相关组件

        Args:
            new_config_dict: 新的配置字典
        """
        try:
            logger.info("知识图谱管理器开始重新加载配置...")

            # 重新初始化Neo4j连接（使用最新配置）
            self._init_neo4j_connection()

            logger.info("知识图谱管理器配置重新加载完成")
        except Exception as e:
            logger.error(f"知识图谱管理器配置重新加载失败: {str(e)}")

    def get_database_tables(self, database_config):
        """获取指定数据库中的表列表"""
        try:
            conn = self._get_database_connection(database_config)
            if not conn:
                return []

            cursor = conn.cursor()
            db_name = database_config.get("database", "")

            # 根据数据库类型获取表
            db_type = database_config.get("type", "mysql").lower()

            if db_type == "mysql":
                cursor.execute(f"SHOW TABLES FROM `{db_name}`")
            else:
                # 提供其他数据库类型的支持
                raise NotImplementedError(f"不支持的数据库类型: {db_type}")

            tables = cursor.fetchall()
            cursor.close()
            conn.close()

            # 处理结果 - 增强健壮性，检查返回的列名
            result = []
            if tables:
                # 确保第一行数据可访问
                if not tables or len(tables) == 0 or not tables[0]:
                    return []

                # 简单方法：直接使用第一列作为表名
                # MySQL通常返回表名作为结果集的第一列
                for row in tables:
                    if row and len(row) > 0:
                        result.append(row[0])

            return result
        except Exception as e:
            logger.error(f"获取数据库表失败: {str(e)}")
            return []

    def get_table_schema(self, table_name, database_config=None):
        """获取表结构"""
        try:
            # 如果提供了数据库配置，使用直接SQL查询
            if database_config:
                conn = self._get_database_connection(database_config)
                if not conn:
                    raise Exception("无法连接到数据库")

                # pymysql使用cursors.DictCursor获取字典结果
                import pymysql.cursors

                cursor = conn.cursor(pymysql.cursors.DictCursor)
                db_type = database_config.get("type", "mysql").lower()

                if db_type == "mysql":
                    # 使用INFORMATION_SCHEMA获取列信息
                    query = f"""
                    SELECT COLUMN_NAME as name, DATA_TYPE as type, 
                        CASE WHEN COLUMN_KEY = 'PRI' THEN 1 ELSE 0 END as primary_key
                    FROM INFORMATION_SCHEMA.COLUMNS 
                    WHERE TABLE_NAME = %s AND TABLE_SCHEMA = %s
                    """
                    cursor.execute(
                        query, (table_name, database_config.get("database", ""))
                    )
                    result = cursor.fetchall()
                    cursor.close()
                    conn.close()

                    # 转换为前端需要的格式
                    for row in result:
                        row["primary_key"] = bool(row["primary_key"])

                    return result
                else:
                    cursor.close()
                    conn.close()
                    raise Exception(f"不支持的数据库类型: {db_type}")

            # 如果没有提供数据库配置，使用vanna_manager (原来的方法)
            schema_query = f"""
            SELECT COLUMN_NAME as name, DATA_TYPE as type, 
                CASE WHEN COLUMN_KEY = 'PRI' THEN true ELSE false END as primary_key
            FROM INFORMATION_SCHEMA.COLUMNS 
            WHERE TABLE_NAME = '{table_name}' AND TABLE_SCHEMA = DATABASE()
            """
            schema_df = self.vanna_manager.vn.run_sql(schema_query)

            # 转换为前端需要的格式
            return schema_df.to_dict(orient="records")
        except Exception as e:
            logger.error(f"获取表结构失败: {str(e)}")
            # 备用方法：DESCRIBE table
            try:
                describe_df = self.vanna_manager.vn.run_sql(f"DESCRIBE {table_name}")
                # 转换DESCRIBE结果为所需格式
                result = []
                for _, row in describe_df.iterrows():
                    result.append(
                        {
                            "name": row["Field"],
                            "type": row["Type"],
                            "primary_key": row["Key"] == "PRI",
                        }
                    )
                return result
            except Exception as e2:
                logger.error(f"获取表结构失败(备用方法): {str(e2)}")
                return []

    def get_all_knowledge_graphs(self):
        """获取所有知识图谱"""
        session = self.Session()
        try:
            graphs = session.query(KnowledgeGraph).all()
            return [graph.to_dict() for graph in graphs]
        except Exception as e:
            logger.error(f"获取知识图谱列表失败: {str(e)}")
            return []
        finally:
            session.close()

    def get_knowledge_graph(self, graph_id):
        """获取指定ID的知识图谱"""
        session = self.Session()
        try:
            graph = (
                session.query(KnowledgeGraph)
                .filter(KnowledgeGraph.id == graph_id)
                .first()
            )
            return graph.to_dict() if graph else None
        except Exception as e:
            logger.error(f"获取知识图谱失败: {str(e)}")
            return None
        finally:
            session.close()

    def create_knowledge_graph(self, graph_data):
        """创建一个新的知识图谱定义"""
        session = self.Session()
        try:
            # 创建知识图谱记录
            # 构建完整的config，包含node_types和relationships
            config_data = {
                "node_types": graph_data.get("node_types", []),
                "relationships": graph_data.get("relationships", []),
                "database": graph_data["database"],  # 也保存在config中作为备用
                "table": graph_data["table"],  # 也保存在config中作为备用
            }

            graph = KnowledgeGraph(
                name=graph_data["name"],
                database=graph_data["database"],
                table=graph_data["table"],
                config=json.dumps(config_data),
                status="pending",
            )
            session.add(graph)
            session.flush()  # 获取ID，但不提交事务
            graph_id = graph.id

            # 添加节点类型
            node_types_created = []
            for node_type_data in graph_data.get("node_types", []):
                # Ensure identifier_columns is always a list, even if only one is selected
                identifier_cols = node_type_data.get("identifier_columns", [])
                if not isinstance(identifier_cols, list):
                    identifier_cols = (
                        [identifier_cols] if identifier_cols else []
                    )  # Ensure it's a list

                # Prepare split_config
                split_config = node_type_data.get(
                    "split_config", {"enabled": False, "delimiter": None}
                )
                split_config_json = (
                    json.dumps(split_config)
                    if split_config and split_config.get("enabled")
                    else None
                )

                node_type = NodeType(
                    knowledge_graph_id=graph_id,
                    name=node_type_data["name"],
                    identifier_columns=json.dumps(identifier_cols),  # Save as list
                    attribute_columns=json.dumps(
                        node_type_data.get("attribute_columns", [])
                    ),
                    split_config=split_config_json,  # Save split config
                )
                session.add(node_type)
                node_types_created.append(node_type)

            session.flush()  # 确保所有节点类型都有ID，但不提交事务

            # 验证节点类型是否正确添加
            if not node_types_created:
                raise Exception(f"图谱 {graph_id} 没有创建任何节点类型")

            # 获取所有节点类型的字典，用于关系映射
            node_types = {}
            for nt in node_types_created:
                session.refresh(nt)  # 确保获取最新的ID
                node_types[nt.name] = nt

            # 添加关系类型
            relationships_created = []
            for rel_data in graph_data.get("relationships", []):
                # 查找源节点和目标节点的ID
                source_node = node_types.get(rel_data["source"])
                target_node = node_types.get(rel_data["target"])

                if not source_node or not target_node:
                    logger.warning(
                        f"跳过关系 {rel_data['type']}: 源节点或目标节点不存在"
                    )
                    continue

                # 存储跨行匹配配置
                inter_row_config = None
                if rel_data.get("matching_mode") == "inter-row" and rel_data.get(
                    "inter_row_options"
                ):
                    inter_row_config = json.dumps(rel_data["inter_row_options"])

                relationship = RelationshipType(
                    knowledge_graph_id=graph_id,
                    source_node_type_id=source_node.id,
                    target_node_type_id=target_node.id,
                    type=rel_data["type"],
                    direction=rel_data.get("direction", "uni"),
                    matching_mode=rel_data.get("matching_mode", "intra-row"),
                    inter_row_config=inter_row_config,
                )
                session.add(relationship)
                relationships_created.append(relationship)

            # 最终验证和提交
            session.flush()  # 确保所有关系类型也有ID

            # 验证数据完整性
            logger.info(
                f"图谱 {graph_id} 创建了 {len(node_types_created)} 个节点类型，{len(relationships_created)} 个关系类型"
            )
            if len(node_types_created) == 0:
                raise Exception(f"图谱 {graph_id} 节点类型创建失败")

            # 提交事务
            session.commit()

            # 记录成功信息
            logger.info(f"知识图谱 {graph_id} 及其节点/关系类型已成功提交到数据库")

            # 关闭会话
            session.close()

            # 开始异步构建 - 在会话关闭后调用，避免会话绑定问题
            # 移除自动构建，让调用方决定何时构建
            # self.start_graph_building(graph_id)

            # 返回新创建的图谱信息
            return self.get_knowledge_graph(graph_id)
        except Exception as e:
            logger.error(f"创建知识图谱失败: {str(e)}")
            if "session" in locals():
                session.rollback()
                session.close()
            raise

    def delete_knowledge_graph(self, graph_id):
        """删除知识图谱"""
        session = self.Session()
        try:
            graph = (
                session.query(KnowledgeGraph)
                .filter(KnowledgeGraph.id == graph_id)
                .first()
            )
            if not graph:
                return False

            # 如果图谱正在构建中，取消构建任务
            if graph.status == "building" and graph_id in self.building_tasks:
                # 这里暂时无法取消ThreadPoolExecutor中的任务，只能将其状态标记为已取消
                self.building_tasks[graph_id]["cancelled"] = True

            session.delete(graph)  # 级联删除将处理关联的节点类型和关系类型
            session.commit()

            # 从 Neo4j 删除图数据 (异步或同步)
            self._delete_graph_data_from_neo4j(graph_id)

            return True
        except Exception as e:
            session.rollback()
            logger.error(f"删除知识图谱失败: {str(e)}")
            return False
        finally:
            session.close()

    def _delete_graph_data_from_neo4j(self, graph_id):
        """从 Neo4j 中删除指定 graph_id 的所有节点和关系"""
        if not self.neo4j_driver:
            logger.error("Neo4j 驱动未初始化，无法删除数据")
            return

        try:
            with self.neo4j_driver.session() as neo4j_session:
                # 分批删除关系和节点以避免内存问题
                while True:
                    result = neo4j_session.run(
                        """
                        MATCH (n {graph_id: $graph_id})
                        WITH n LIMIT 10000
                        DETACH DELETE n
                        RETURN count(n) as deleted_count
                    """,
                        graph_id=graph_id,
                    )
                    deleted_count = result.single()["deleted_count"]
                    logger.info(
                        f"从 Neo4j 中删除了 {deleted_count} 个 graph_id={graph_id} 的节点和它们的关系"
                    )
                    if deleted_count == 0:
                        break
            logger.info(f"成功从 Neo4j 删除 graph_id={graph_id} 的所有数据")
        except Exception as e:
            logger.error(f"从 Neo4j 删除 graph_id={graph_id} 数据失败: {e}")

    def start_graph_building(self, graph_id):
        """开始异步构建知识图谱"""
        # 更新图谱状态为构建中
        session = self.Session()
        try:
            graph = (
                session.query(KnowledgeGraph)
                .filter(KnowledgeGraph.id == graph_id)
                .first()
            )
            if not graph:
                logger.error(f"无法启动构建: 图谱ID {graph_id} 不存在")
                return {"status": "error", "message": f"图谱ID {graph_id} 不存在"}

            # 如果已经在构建中或已完成，则不重新启动
            if graph.status in ("building", "completed"):
                logger.warning(f"图谱 {graph_id} 状态为 {graph.status}，跳过构建")
                return {
                    "status": "skipped",
                    "message": f"图谱状态为 {graph.status}，跳过构建",
                }

            # 更新状态
            graph.status = "building"
            graph.error_message = None
            session.commit()

            # 提交到线程池异步执行
            future = self.executor.submit(self._build_knowledge_graph, graph_id)

            # 存储任务信息
            self.building_tasks[graph_id] = {"future": future, "cancelled": False}

            return {"status": "success", "message": f"图谱 {graph_id} 构建已启动"}
        except Exception as e:
            session.rollback()
            logger.error(f"启动图谱构建失败: {str(e)}")
            return {"status": "error", "message": f"启动构建失败: {str(e)}"}
        finally:
            session.close()

    def _build_knowledge_graph(self, graph_id):
        """实际执行知识图谱构建的方法（在线程池中运行）"""
        # 在异步任务中创建新的会话
        session = self.Session()
        try:
            # 获取图谱及其配置
            graph = (
                session.query(KnowledgeGraph)
                .filter(KnowledgeGraph.id == graph_id)
                .first()
            )
            if not graph:
                logger.error(f"构建失败: 图谱 {graph_id} 不存在")
                session.close()  # Ensure session is closed on early exit
                return

            if graph_id in self.building_tasks and self.building_tasks[graph_id].get(
                "cancelled"
            ):
                logger.info(f"图谱 {graph_id} 构建已取消")
                graph.status = "failed"
                graph.error_message = "任务已取消"
                session.commit()
                session.close()  # Ensure session is closed
                # Also clean up any partially created data in Neo4j if needed
                self._delete_graph_data_from_neo4j(graph_id)
                return

            config = json.loads(graph.config)

            # 获取表数据 - 从graph对象的独立字段获取，而不是从config中获取
            database_name = graph.database
            table_name = graph.table
            table_data = self._fetch_table_data(database_name, table_name)
            if table_data is None or table_data.empty:
                raise Exception(f"无法获取表 {table_name} 的数据")

            # 加载节点类型和关系类型 - 确保在当前会话中加载
            node_types_list = (
                session.query(NodeType)
                .filter(NodeType.knowledge_graph_id == graph_id)
                .all()
            )
            relationship_types = (
                session.query(RelationshipType)
                .filter(RelationshipType.knowledge_graph_id == graph_id)
                .all()
            )

            # 确保已经获取了所有必要的数据
            if not node_types_list:
                raise Exception(f"未找到图谱 {graph_id} 的节点类型定义")

            # 创建节点类型 ID 到对象的映射，方便查找
            node_types_by_id = {nt.id: nt for nt in node_types_list}

            # 创建节点和关系所需的所有数据
            # 在提交完图谱状态更新前，先收集所有需要的数据
            node_count = 0
            relationship_count = 0
            entity_store = {}  # 存储所有实体
            relationships_data = []

            # 创建和存储实体
            # 使用 node_types_list 替代旧的 node_types 查询结果
            for node_type in node_types_list:
                entity_type = node_type.name
                id_columns = json.loads(node_type.identifier_columns)
                attr_columns = (
                    json.loads(node_type.attribute_columns)
                    if node_type.attribute_columns
                    else []
                )
                # Load split_config
                split_config = (
                    json.loads(node_type.split_config)
                    if node_type.split_config
                    else None
                )

                # 创建该类型的实体, 传递 split_config 和 graph_id
                entities = self._create_entities(
                    table_data,
                    entity_type,
                    id_columns,
                    attr_columns,
                    split_config,
                    graph_id,
                )
                entity_store[entity_type] = entities
                node_count += len(entities)

            # 创建和存储关系
            for rel_type in relationship_types:
                # 获取源和目标节点类型对象 (从字典中查找)
                source_node_type = node_types_by_id.get(rel_type.source_node_type_id)
                target_node_type = node_types_by_id.get(rel_type.target_node_type_id)

                if not source_node_type or not target_node_type:
                    logger.warning(
                        f"跳过关系 {rel_type.type}: 找不到源节点类型ID {rel_type.source_node_type_id} 或目标节点类型ID {rel_type.target_node_type_id} 对应的对象"
                    )
                    continue

                source_entities = entity_store.get(source_node_type.name, {})
                target_entities = entity_store.get(target_node_type.name, {})

                if not source_entities or not target_entities:
                    logger.warning(f"跳过关系 {rel_type.type}: 源实体或目标实体为空")
                    continue

                # 根据匹配模式创建关系
                new_relationships = []

                if rel_type.matching_mode == "intra-row":
                    # 同行匹配
                    new_relationships = self._create_intra_row_relationships(
                        table_data,
                        source_entities,
                        target_entities,
                        source_node_type,
                        target_node_type,
                        rel_type,
                    )
                else:
                    # 跨行匹配
                    if not rel_type.inter_row_config:
                        logger.warning(f"关系 {rel_type.id} 缺少跨行匹配配置，跳过")
                        continue

                    try:
                        inter_row_config = json.loads(rel_type.inter_row_config)
                        # 确保配置有效
                        if not inter_row_config:
                            logger.warning(
                                f"关系 {rel_type.id} 跨行匹配配置为空或无效，跳过"
                            )
                            continue

                        new_relationships = self._create_inter_row_relationships(
                            table_data,
                            source_entities,
                            target_entities,
                            source_node_type,
                            target_node_type,
                            rel_type,  # Pass NodeType objects
                            inter_row_config,
                        )
                    except Exception as e:
                        logger.error(
                            f"处理关系 {rel_type.type} 跨行匹配配置时出错: {str(e)}"
                        )
                        continue

                relationships_data.extend(new_relationships)
                relationship_count += len(new_relationships)

            # Update graph status BEFORE writing to Neo4j
            graph.node_count = node_count
            graph.relationship_count = relationship_count
            graph.status = "saving"  # Indicate saving to Neo4j
            session.commit()

            # Export data to Neo4j instead of JSON files
            success = self._export_graph_data_to_neo4j(
                graph_id, entity_store, relationships_data
            )

            if success:
                graph.status = "completed"
                logger.info(
                    f"图谱 {graph_id} 构建并保存到 Neo4j 完成: {node_count} 个节点, {relationship_count} 个关系"
                )

                # **关键修复**: 图谱构建完成后确保Neo4j schema刷新
                # 双重保险确保后续schema获取不会为空
                try:
                    logger.info(f"图谱 {graph_id} 构建完成，执行最终schema刷新...")
                    if self.neo4j_driver:
                        with self.neo4j_driver.session(
                            database="neo4j"
                        ) as final_session:
                            # 强制schema刷新
                            final_session.run("CALL db.schema.nodeTypeProperties()")
                            final_session.run("CALL db.schema.relTypeProperties()")
                        logger.info(f"图谱 {graph_id} 最终schema刷新完成")
                except Exception as final_refresh_error:
                    logger.warning(
                        f"图谱 {graph_id} 最终schema刷新失败，但不影响图谱构建: {final_refresh_error}"
                    )
            else:
                graph.status = "failed"
                graph.error_message = "保存到 Neo4j 失败"
                logger.error(f"图谱 {graph_id} 构建完成但保存到 Neo4j 失败")

            session.commit()  # Commit final status

        except Exception as e:
            logger.error(f"构建图谱 {graph_id} 失败: {str(e)}")
            try:
                if session and session.is_active:
                    graph = (
                        session.query(KnowledgeGraph)
                        .filter(KnowledgeGraph.id == graph_id)
                        .first()
                    )
                    if graph:
                        graph.status = "failed"
                        graph.error_message = str(e)
                        session.commit()
                    # Attempt to clean up partial data in Neo4j on failure
                    self._delete_graph_data_from_neo4j(graph_id)
            except Exception as inner_e:
                logger.error(f"更新图谱状态或清理Neo4j失败: {str(inner_e)}")
        finally:
            if session:
                session.close()
            if graph_id in self.building_tasks:
                del self.building_tasks[graph_id]

    def _export_graph_data_to_neo4j(self, graph_id, entity_store, relationships):
        """导出图谱数据到 Neo4j"""
        if not self.neo4j_driver:
            logger.error("Neo4j 驱动未初始化，无法导出数据")
            return False

        # 检查APOC插件可用性
        apoc_available = self.check_apoc_availability()
        logger.info(f"APOC 插件可用性: {apoc_available}")

        try:
            with self.neo4j_driver.session(
                database="neo4j"
            ) as neo4j_session:  # Specify database if needed
                # 0. 先删除该 graph_id 的旧数据 - 更彻底的清理
                logger.info(f"开始清理 Neo4j 中 graph_id={graph_id} 的旧数据...")
                self._delete_graph_data_from_neo4j_thorough(
                    graph_id, neo4j_session
                )  # 使用更彻底的清理方法

                # 1. 创建唯一性约束 (如果不存在) - 分离到单独的事务
                all_entity_types = list(entity_store.keys())
                logger.info(f"开始创建约束...")
                for entity_type in all_entity_types:
                    # Escape label name if it contains special characters (though unlikely here)
                    escaped_label = f"`{entity_type}`"
                    try:
                        neo4j_session.run(
                            f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{escaped_label}) REQUIRE n.unique_id IS UNIQUE"
                        )
                        logger.info(f"确保 Neo4j 约束存在: {escaped_label}.unique_id")
                    except Exception as constraint_e:
                        # Log constraint error but continue, might fail if label has tricky chars
                        logger.warning(
                            f"创建 Neo4j 约束时出错 ({escaped_label}): {constraint_e}. 导入仍将继续。"
                        )

                # 2. 检查并去重实体数据
                logger.info(f"开始数据预处理和去重...")
                entities_flat = []
                unique_ids_seen = set()

                for entity_type, entities in entity_store.items():
                    for entity_id, entity_data in entities.items():
                        # 检查是否有重复的unique_id
                        unique_id = entity_data[
                            "id"
                        ]  # Use the 'EntityType:Identifier' string
                        if unique_id in unique_ids_seen:
                            logger.warning(f"发现重复的实体ID: {unique_id}，跳过重复项")
                            continue
                        unique_ids_seen.add(unique_id)

                        # Prepare properties, ensuring graph_id and a unique ID
                        properties = entity_data.get("attributes", {})
                        properties["graph_id"] = graph_id
                        properties["unique_id"] = unique_id
                        properties["name"] = unique_id.split(":", 1)[
                            -1
                        ]  # Extract name after type:
                        # Convert non-primitive types like lists/dicts if necessary, though attributes seem basic
                        for k, v in properties.items():
                            if isinstance(v, (list, dict)):
                                properties[k] = json.dumps(
                                    v
                                )  # Store complex types as JSON strings

                        entities_flat.append(
                            {
                                "labels": [entity_type],  # Use entity type as label
                                "properties": properties,
                            }
                        )

                logger.info(
                    f"数据去重完成，原始实体数: {sum(len(entities) for entities in entity_store.values())}，去重后: {len(entities_flat)}"
                )

                # 3. 批量写入节点 - 使用MERGE避免冲突
                logger.info(f"开始向 Neo4j 写入 graph_id={graph_id} 的节点...")
                node_creation_batch_size = 500  # 减小批次大小以降低事务冲突风险

                # Write nodes in batches
                for i in range(0, len(entities_flat), node_creation_batch_size):
                    batch = entities_flat[i : i + node_creation_batch_size]

                    if apoc_available:
                        # 使用APOC方法
                        try:
                            # 使用 MERGE 替代 CREATE 来避免重复创建
                            neo4j_session.run(
                                """
                                UNWIND $batch as node_data
                                MERGE (n {unique_id: node_data.properties.unique_id, graph_id: $graph_id})
                                SET n = node_data.properties
                                WITH n, node_data.labels as labels
                                CALL apoc.create.addLabels(n, labels) YIELD node
                                RETURN count(node)
                            """,
                                batch=batch,
                                graph_id=graph_id,
                            )
                            logger.info(
                                f"写入了 {len(batch)} 个节点到 Neo4j (APOC，批次 {i // node_creation_batch_size + 1})"
                            )
                            continue  # 成功则继续下一批次
                        except Exception as apoc_e:
                            logger.error(f"APOC 方法失败，切换到备用方案: {apoc_e}")
                            apoc_available = (
                                False  # 标记APOC不可用，后续批次直接使用备用方案
                            )

                    # 备用方案：不使用APOC，按标签分组创建
                    logger.info(
                        f"使用备用方案 (非APOC) 处理批次 {i // node_creation_batch_size + 1}"
                    )
                    try:
                        # 按标签分组
                        nodes_by_label = {}
                        for node in batch:
                            label = (
                                node["labels"][0] if node["labels"] else "DefaultLabel"
                            )
                            if label not in nodes_by_label:
                                nodes_by_label[label] = []
                            nodes_by_label[label].append(node["properties"])

                        # 为每个标签分别创建节点
                        for label, properties_list in nodes_by_label.items():
                            escaped_label = f"`{label}`"
                            neo4j_session.run(
                                f"""
                                UNWIND $properties_list as props
                                MERGE (n:{escaped_label} {{unique_id: props.unique_id, graph_id: $graph_id}})
                                SET n = props
                            """,
                                properties_list=properties_list,
                                graph_id=graph_id,
                            )
                            logger.info(
                                f"备用方案写入了 {len(properties_list)} 个 {label} 节点"
                            )

                    except Exception as fallback_e:
                        logger.error(f"备用方案也失败: {fallback_e}")
                        # 尝试逐个插入以找出问题节点
                        logger.info("尝试逐个节点插入...")
                        for j, node in enumerate(batch):
                            try:
                                label = (
                                    node["labels"][0]
                                    if node["labels"]
                                    else "DefaultLabel"
                                )
                                escaped_label = f"`{label}`"
                                neo4j_session.run(
                                    f"""
                                    MERGE (n:{escaped_label} {{unique_id: $unique_id, graph_id: $graph_id}})
                                    SET n = $properties
                                """,
                                    unique_id=node["properties"]["unique_id"],
                                    properties=node["properties"],
                                    graph_id=graph_id,
                                )
                            except Exception as single_e:
                                logger.error(
                                    f"单个节点写入失败 {node['properties']['unique_id']}: {single_e}"
                                )
                                # 继续处理其他节点

                logger.info(f"节点写入完成，共 {len(entities_flat)} 个。")

                # 4. 批量写入关系 - 使用MERGE避免重复关系
                logger.info(f"开始向 Neo4j 写入 graph_id={graph_id} 的关系...")
                rel_creation_batch_size = 500  # 减小批次大小

                # Prepare relationships data for batching
                rels_to_create = []
                rel_signatures_seen = set()

                for rel in relationships:
                    # 检查重复关系
                    rel_signature = f"{rel['source']}-{rel['target']}-{rel['type']}"
                    if rel_signature in rel_signatures_seen:
                        logger.debug(f"跳过重复关系: {rel_signature}")
                        continue
                    rel_signatures_seen.add(rel_signature)

                    # Ensure properties is a dict, add graph_id
                    properties = rel.get("properties", {})
                    if not isinstance(properties, dict):
                        properties = {}  # Ensure it's a dict
                    properties["graph_id"] = graph_id
                    # Convert complex property values to JSON strings if needed
                    for k, v in properties.items():
                        if isinstance(v, (list, dict)):
                            properties[k] = json.dumps(v)

                    rels_to_create.append(
                        {
                            "source_id": rel["source"],
                            "target_id": rel["target"],
                            "type": rel["type"]
                            .replace("-", "_")
                            .replace(" ", "_"),  # Sanitize relationship type for Neo4j
                            "properties": properties,
                        }
                    )

                logger.info(
                    f"关系去重完成，原始关系数: {len(relationships)}，去重后: {len(rels_to_create)}"
                )

                # Write relationships in batches using UNWIND and MERGE
                for i in range(0, len(rels_to_create), rel_creation_batch_size):
                    batch = rels_to_create[i : i + rel_creation_batch_size]

                    if apoc_available:
                        # 使用APOC方法
                        try:
                            # 使用 MERGE 避免重复关系
                            neo4j_session.run(
                                f"""
                                 UNWIND $batch as rel_data
                                 MATCH (source {{unique_id: rel_data.source_id, graph_id: $graph_id}})
                                 MATCH (target {{unique_id: rel_data.target_id, graph_id: $graph_id}})
                                 CALL apoc.create.relationship(source, rel_data.type, rel_data.properties, target) YIELD rel 
                                 RETURN count(rel)
                             """,
                                batch=batch,
                                graph_id=graph_id,
                            )
                            logger.info(
                                f"写入了 {len(batch)} 个关系到 Neo4j (APOC，批次 {i // rel_creation_batch_size + 1})"
                            )
                            continue  # 成功则继续下一批次
                        except Exception as apoc_e:
                            logger.error(
                                f"关系 APOC 方法失败，切换到备用方案: {apoc_e}"
                            )
                            apoc_available = False  # 标记APOC不可用

                    # 备用方案：不使用APOC，直接创建关系
                    logger.info(
                        f"使用备用方案 (非APOC) 处理关系批次 {i // rel_creation_batch_size + 1}"
                    )
                    try:
                        # 按关系类型分组以提高效率
                        rels_by_type = {}
                        for rel in batch:
                            rel_type = rel["type"]
                            if rel_type not in rels_by_type:
                                rels_by_type[rel_type] = []
                            rels_by_type[rel_type].append(rel)

                        # 为每种关系类型分别创建
                        for rel_type, rels in rels_by_type.items():
                            # 使用动态Cypher构建关系类型
                            escaped_rel_type = f"`{rel_type}`"
                            neo4j_session.run(
                                f"""
                                 UNWIND $rels as rel_data
                                 MATCH (source {{unique_id: rel_data.source_id, graph_id: $graph_id}})
                                 MATCH (target {{unique_id: rel_data.target_id, graph_id: $graph_id}})
                                 MERGE (source)-[r:{escaped_rel_type}]->(target)
                                 SET r = rel_data.properties
                             """,
                                rels=rels,
                                graph_id=graph_id,
                            )
                            logger.info(
                                f"备用方案写入了 {len(rels)} 个 {rel_type} 关系"
                            )

                    except Exception as fallback_e:
                        logger.error(f"关系备用方案也失败: {fallback_e}")
                        # 尝试逐个创建关系
                        logger.info("尝试逐个关系创建...")
                        for rel in batch:
                            try:
                                escaped_rel_type = f"`{rel['type']}`"
                                neo4j_session.run(
                                    f"""
                                     MATCH (source {{unique_id: $source_id, graph_id: $graph_id}})
                                     MATCH (target {{unique_id: $target_id, graph_id: $graph_id}})
                                     MERGE (source)-[r:{escaped_rel_type}]->(target)
                                     SET r = $properties
                                 """,
                                    source_id=rel["source_id"],
                                    target_id=rel["target_id"],
                                    properties=rel["properties"],
                                    graph_id=graph_id,
                                )
                            except Exception as single_rel_e:
                                logger.error(
                                    f"单个关系创建失败 {rel['source_id']}->{rel['target_id']}: {single_rel_e}"
                                )
                                # 继续处理其他关系

                logger.info(f"关系写入完成，共 {len(rels_to_create)} 个。")
                logger.info(f"图谱数据 (graph_id={graph_id}) 已成功导出到 Neo4j")

                # **关键修复**: 数据写入完成后立即刷新Neo4j schema缓存
                # 这解决了schema获取为空的问题
                try:
                    logger.info(f"开始刷新Neo4j schema缓存...")
                    # 通过执行简单的schema相关查询来强制刷新缓存
                    neo4j_session.run("CALL db.schema.visualization()")
                    logger.info(f"Neo4j schema缓存刷新完成")
                except Exception as refresh_error:
                    logger.warning(
                        f"刷新Neo4j schema缓存失败，但不影响数据写入: {refresh_error}"
                    )

                return True
        except Exception as e:
            logger.error(f"导出图谱数据到 Neo4j 失败 (graph_id={graph_id}): {str(e)}")
            # Attempt to clean up potentially partial data in Neo4j
            self._delete_graph_data_from_neo4j(graph_id)
            return False
        # No finally block needed for session as 'with' handles it

    def _delete_graph_data_from_neo4j_thorough(self, graph_id, neo4j_session=None):
        """更彻底地从 Neo4j 中删除指定 graph_id 的所有节点和关系"""
        if not self.neo4j_driver:
            logger.error("Neo4j 驱动未初始化，无法删除数据")
            return

        # 如果没有提供会话，创建新的会话
        if neo4j_session is None:
            with self.neo4j_driver.session() as session:
                self._delete_graph_data_from_neo4j_thorough(graph_id, session)
            return

        try:
            # 1. 首先删除所有相关约束 (可选，如果需要完全清理)
            # 注意：这可能影响其他图谱，所以暂时注释掉
            # all_constraints = neo4j_session.run("SHOW CONSTRAINTS").data()
            # for constraint in all_constraints:
            #     if 'unique_id' in constraint.get('description', ''):
            #         neo4j_session.run(f"DROP CONSTRAINT {constraint['name']} IF EXISTS")

            # 2. 分批删除关系和节点，更大的批次
            deleted_total = 0
            while True:
                result = neo4j_session.run(
                    """
                    MATCH (n {graph_id: $graph_id})
                    WITH n LIMIT 50000
                    DETACH DELETE n
                    RETURN count(n) as deleted_count
                """,
                    graph_id=graph_id,
                )
                deleted_count = result.single()["deleted_count"]
                deleted_total += deleted_count
                logger.info(
                    f"从 Neo4j 中删除了 {deleted_count} 个 graph_id={graph_id} 的节点和它们的关系"
                )
                if deleted_count == 0:
                    break

            # 3. 确保清理完毕 - 检查是否还有残留数据
            remaining_result = neo4j_session.run(
                """
                MATCH (n {graph_id: $graph_id})
                RETURN count(n) as remaining_count
            """,
                graph_id=graph_id,
            )
            remaining_count = remaining_result.single()["remaining_count"]

            if remaining_count > 0:
                logger.warning(
                    f"仍有 {remaining_count} 个节点未被删除，尝试强制清理..."
                )
                # 尝试强制删除
                neo4j_session.run(
                    """
                    MATCH (n {graph_id: $graph_id})
                    DETACH DELETE n
                """,
                    graph_id=graph_id,
                )

            logger.info(
                f"彻底清理完成，共删除 {deleted_total} 个 graph_id={graph_id} 的节点"
            )
        except Exception as e:
            logger.error(f"彻底清理 Neo4j graph_id={graph_id} 数据失败: {e}")

    def _fetch_table_data(self, database_name, table_name):
        """获取表数据"""
        try:
            # 检测数据库类型 - 如果database_name看起来像SQLite文件路径，则使用SQLite
            if database_name and (
                "/" in database_name
                or "\\" in database_name
                or database_name.endswith(".sqlite")
                or database_name.endswith(".db")
            ):
                # 这是一个SQLite文件路径
                database_config = {"type": "sqlite", "database": database_name}
            else:
                # 使用Vanna manager的配置
                database_config = {
                    "type": self.vanna_manager.config.database.type,
                    "host": self.vanna_manager.config.database.host,
                    "port": self.vanna_manager.config.database.port,
                    "username": self.vanna_manager.config.database.username,
                    "password": self.vanna_manager.config.database.password,
                    "database": database_name
                    or self.vanna_manager.config.database.database_name,
                }

            conn = self._get_database_connection(database_config)
            if not conn:
                logger.error(f"无法连接到数据库获取表数据")
                return None

            cursor = conn.cursor()
            query = f"SELECT * FROM {table_name}"
            cursor.execute(query)
            rows = cursor.fetchall()

            # 获取列名
            column_names = [column[0] for column in cursor.description]

            # 转换为DataFrame
            import pandas as pd

            df = pd.DataFrame(rows, columns=column_names)

            cursor.close()
            conn.close()

            return df
        except Exception as e:
            logger.error(f"获取表 {table_name} 数据失败: {str(e)}")

            # 备用方法：使用vanna_manager
            try:
                # 使用vanna_manager运行查询
                query = f"SELECT * FROM {table_name}"
                return self.vanna_manager.vn.run_sql(query)
            except Exception as e2:
                logger.error(f"备用方法获取表数据失败: {str(e2)}")
                return None

    def _create_entities(
        self,
        table_data,
        entity_type,
        id_columns,
        attr_columns,
        split_config=None,
        graph_id=None,
    ):
        """
        创建实体并返回实体字典

        Args:
            table_data: 表数据
            entity_type: 实体类型名称
            id_columns: 标识符列名列表
            attr_columns: 属性列名列表
            split_config: 拆分配置 {enabled: bool, delimiter: str}
            graph_id: 图谱ID，用于生成唯一的namespace

        Returns:
            实体字典 {entity_id: {id, type, attributes, row_indices}}
        """
        entities = {}
        duplicate_count = 0
        empty_identifier_count = 0

        # 解析分拆配置
        split_enabled = False
        delimiter = None
        split_column = None
        if split_config and isinstance(split_config, dict):
            split_enabled = split_config.get("enabled", False)
            delimiter = split_config.get("delimiter", ",")
            # Determine which identifier column to split on
            # 优先使用第一个标识符列作为分拆列
            if split_enabled and id_columns:
                split_column = id_columns[0]  # Use the first ID column for splitting

        # 验证分拆配置
        if split_enabled and not delimiter:
            logger.warning(
                f"实体类型 '{entity_type}' 启用了拆分但未提供分隔符，将禁用拆分。"
            )
            split_enabled = False
        if split_enabled and not split_column:
            logger.warning(
                f"实体类型 '{entity_type}' 启用了拆分但未提供有效的标识符列，将禁用拆分。"
            )
            split_enabled = False

        logger.info(
            f"开始为实体类型 '{entity_type}' 创建实体，分拆模式: {split_enabled}"
        )

        for index, row in table_data.iterrows():
            entity_identifiers = []

            if split_enabled:
                # Split the value from the designated column
                raw_value = row[split_column]
                if raw_value and isinstance(raw_value, str):
                    # Split and strip whitespace from each part
                    if delimiter in raw_value:
                        split_values = [
                            val.strip()
                            for val in raw_value.split(delimiter)
                            if val.strip()
                        ]
                    else:
                        split_values = [raw_value.strip()]
                    if split_values:
                        entity_identifiers.extend(split_values)
                    else:
                        # Handle cases where splitting results in nothing (e.g., just delimiter)
                        # Treat the original non-empty value as a single identifier if splitting fails
                        if raw_value.strip():
                            entity_identifiers.append(raw_value.strip())
                elif (
                    raw_value
                ):  # Handle non-string but non-empty values (e.g., numbers)
                    entity_identifiers.append(str(raw_value))  # Convert to string
            else:
                # No splitting, use the combined values from id_columns as before (or single if only one)
                id_values = [
                    str(row[col])
                    for col in id_columns
                    if col in row and pd.notna(row[col])
                ]
                if id_values:  # Only proceed if we have some identifier values
                    # Use ':' as separator only if multiple columns are used.
                    # If only one column, the identifier is just the value itself.
                    identifier = (
                        ":".join(id_values) if len(id_columns) > 1 else id_values[0]
                    )
                    entity_identifiers.append(identifier)

            # Skip if no valid identifiers found
            if not entity_identifiers:
                empty_identifier_count += 1
                logger.debug(f"行 {index} 没有有效的标识符值，跳过实体创建")
                continue

            # Create or update entities for each identifier found
            for identifier_value in entity_identifiers:
                # 过滤空白或无效的标识符
                if (
                    not identifier_value
                    or str(identifier_value).strip() == ""
                    or str(identifier_value) == "nan"
                ):
                    empty_identifier_count += 1
                    continue

                # Construct the full entity ID with graph namespace
                # 根据多图谱策略生成不同格式的entity_id
                if graph_id is not None and self.multi_graph_strategy == "namespace":
                    # 命名空间策略：graph_{graph_id}_{entity_type}:{identifier_value}
                    entity_id = f"graph_{graph_id}_{entity_type}:{str(identifier_value).strip()}"
                elif graph_id is not None and self.multi_graph_strategy == "label":
                    # 标签策略：保持原有格式，但会在Neo4j中添加图谱标签
                    entity_id = f"{entity_type}:{str(identifier_value).strip()}"
                else:
                    # 传统模式或未指定图谱ID
                    entity_id = f"{entity_type}:{str(identifier_value).strip()}"

                # Collect attributes (always based on the original row)
                attributes = {}
                # Include the original identifier columns' values
                for col in id_columns:
                    if col in row and pd.notna(row[col]):
                        attributes[col] = row[col]
                # Include attribute columns' values
                for col in attr_columns:
                    if col in row and pd.notna(row[col]):
                        attributes[col] = row[col]

                # Check if entity already exists
                if entity_id in entities:
                    duplicate_count += 1
                    # Add the current row index if not already present for this entity
                    if index not in entities[entity_id]["row_indices"]:
                        entities[entity_id]["row_indices"].append(index)
                        # 合并属性信息（如果有新的非空属性）
                        for k, v in attributes.items():
                            if k not in entities[entity_id]["attributes"] or pd.isna(
                                entities[entity_id]["attributes"][k]
                            ):
                                entities[entity_id]["attributes"][k] = v
                    else:
                        logger.debug(
                            f"实体 {entity_id} 在同一行 {index} 重复创建，跳过"
                        )
                else:
                    # Create new entity
                    entities[entity_id] = {
                        "id": entity_id,
                        "type": entity_type,
                        "attributes": attributes,
                        "row_indices": [index],  # Start list with current row index
                    }

        # 记录统计信息
        logger.info(f"实体类型 '{entity_type}' 创建完成:")
        logger.info(f"  - 唯一实体数: {len(entities)}")
        logger.info(f"  - 重复实体次数: {duplicate_count}")
        logger.info(f"  - 空标识符数: {empty_identifier_count}")
        logger.info(f"  - 处理的数据行数: {len(table_data)}")

        # 检查是否有异常高的重复率
        if duplicate_count > len(entities) * 2:
            logger.warning(
                f"实体类型 '{entity_type}' 的重复率异常高 ({duplicate_count}/{len(entities)})，请检查标识符列配置"
            )

        return entities

    def _create_intra_row_relationships(
        self,
        table_data,
        source_entities,
        target_entities,
        source_node_type,
        target_node_type,
        rel_type,
    ):
        """创建同行匹配的关系"""
        relationships = []

        # 用于跟踪已创建的关系
        relationship_set = set()

        # 反向查找映射：从行索引到实体ID
        source_by_row = self._create_row_entity_map(source_entities)
        target_by_row = self._create_row_entity_map(target_entities)

        # 对每一行，如果同时存在源实体和目标实体，则创建关系
        for idx in range(len(table_data)):
            if idx in source_by_row.keys() and idx in target_by_row.keys():
                for source_id in source_by_row[idx]:
                    for target_id in target_by_row[idx]:
                        # 创建关系唯一标识
                        rel_key = f"{source_id}|{target_id}|{rel_type.type}"

                        # 检查关系是否已存在
                        if rel_key not in relationship_set:
                            relationship = {
                                "source": source_id,
                                "target": target_id,
                                "type": rel_type.type,
                            }
                            relationships.append(relationship)
                            relationship_set.add(rel_key)

                            # 如果是双向关系，添加反向关系
                            if rel_type.direction == "bi":
                                reverse_rel_key = (
                                    f"{target_id}|{source_id}|{rel_type.type}"
                                )
                                if reverse_rel_key not in relationship_set:
                                    reverse_relationship = {
                                        "source": target_id,
                                        "target": source_id,
                                        "type": rel_type.type,
                                    }
                                    relationships.append(reverse_relationship)
                                    relationship_set.add(reverse_rel_key)

        return relationships

    def _create_inter_row_relationships(
        self,
        table_data,
        source_entities,
        target_entities,
        source_node_type,
        target_node_type,
        rel_type,
        inter_row_config,
    ):
        """创建跨行匹配的关系"""
        relationships = []

        # 获取分组列
        grouping_columns = inter_row_config.get("grouping_columns", [])

        # 修改：如果没有分组列，将所有行作为一个组处理
        if not grouping_columns:
            logger.info(f"关系 {rel_type.id} 没有指定分组列，将在全表范围内匹配")
            # 将所有行作为一个组
            all_row_indices = list(table_data.index)
            groups = {"all_rows": all_row_indices}
        else:
            # 按分组列对数据分组
            groups = {}
            for idx, row in table_data.iterrows():
                # 创建分组键
                group_values = tuple(str(row[col]) for col in grouping_columns)
                if group_values not in groups:
                    groups[group_values] = []
                groups[group_values].append(idx)

        # 反向查找映射：从行索引到实体ID
        source_by_row = self._create_row_entity_map(source_entities)
        target_by_row = self._create_row_entity_map(target_entities)

        # 处理每个组
        for group_key, row_indices in groups.items():
            # 跳过只有一行的组
            if len(row_indices) <= 1:
                continue

            # 获取该组中的源实体和目标实体
            group_sources = []
            for idx in row_indices:
                if idx in source_by_row:
                    group_sources.append({"idx": idx, "entities": source_by_row[idx]})

            group_targets = []
            for idx in row_indices:
                if idx in target_by_row:
                    group_targets.append({"idx": idx, "entities": target_by_row[idx]})

            # 如果组内没有足够的实体，跳过
            if not group_sources or not group_targets:
                continue

            # 根据匹配方法创建关系 - *** 修改：移除条件方法选择，总是使用规则评估 ***
            # condition_method = inter_row_config.get('condition_method', 'rule')

            # if condition_method == 'rule':
            # 基于规则匹配 (现在包含语义规则)
            # 需要在 _apply_rule_conditions 内部处理嵌入缓存
            embedding_cache = {}  # 初始化嵌入缓存
            rule_relationships = self._apply_rule_conditions(
                table_data,
                row_indices,
                group_sources,
                group_targets,
                rel_type,
                inter_row_config.get("rules", []),
                source_node_type,
                target_node_type,
                embedding_cache,  # 传递缓存
            )
            relationships.extend(rule_relationships)

            # elif condition_method == 'semantic':
            #     # 基于语义匹配 - *** 移除此分支 ***
            #     semantic_relationships = self._apply_semantic_matching(
            #         table_data, row_indices, group_sources, group_targets,
            #         rel_type, inter_row_config.get('semantic_config', {}), source_node_type, target_node_type
            #     )
            #     relationships.extend(semantic_relationships)

        return relationships

    def _create_row_entity_map(self, entities):
        """创建行索引到实体ID的映射"""
        row_to_entities = {}
        for entity_id, entity in entities.items():
            row_idxs = entity.get("row_indices")
            if len(row_idxs) > 0:
                for row_idx in row_idxs:
                    if row_idx not in row_to_entities:
                        row_to_entities[row_idx] = []
                    row_to_entities[row_idx].append(entity_id)
        return row_to_entities

    def _apply_rule_conditions(
        self,
        table_data,
        row_indices,
        source_entities,
        target_entities,
        rel_type,
        rules,
        source_node_type,
        target_node_type,
        embedding_cache,
    ):  # 添加 embedding_cache 参数
        """应用规则条件创建关系 (现在包含处理语义规则)"""
        if not rules:
            return []

        relationships = []
        relationship_set = set()
        rows_by_id = {idx: table_data.iloc[idx] for idx in row_indices}

        # 对每个源目标对，检查是否满足规则
        # source_data 和 target_data 现在是 complex objects: {'idx': ..., 'entities': [str_id1, str_id2, ...]}
        for source_data in source_entities:
            for target_data in target_entities:
                source_row_idx = source_data["idx"]
                target_row_idx = target_data["idx"]

                # 跳过同一行的实体对 (跨行匹配的核心逻辑)
                if source_row_idx == target_row_idx:
                    continue

                # 获取源和目标实体的行
                if source_row_idx not in rows_by_id or target_row_idx not in rows_by_id:
                    continue

                source_row = rows_by_id[source_row_idx]
                target_row = rows_by_id[target_row_idx]

                # 检查是否满足复杂规则条件 (传入 embedding_cache)
                match_result = self._evaluate_rule_items(
                    rules,
                    source_row,
                    target_row,
                    embedding_cache,
                    source_row_idx,
                    target_row_idx,
                )

                # 如果满足条件，为 complex object 内的每个 string ID 对创建关系
                if match_result["matched"]:
                    for s_entity_id_str in source_data[
                        "entities"
                    ]:  # Iterate string IDs
                        for t_entity_id_str in target_data[
                            "entities"
                        ]:  # Iterate string IDs
                            # 添加自环检查 (修正语法 - 使用括号)
                            if (
                                source_node_type
                                and target_node_type
                                and source_node_type.name == target_node_type.name
                                and s_entity_id_str == t_entity_id_str
                            ):
                                continue

                            # 构建关系对象，包含语义相似度信息
                            relationship = {
                                "source": s_entity_id_str,  # *** Use the string ID ***
                                "target": t_entity_id_str,  # *** Use the string ID ***
                                "type": rel_type.type,
                            }

                            # 如果有语义匹配信息，添加到关系属性中
                            if match_result["semantic_info"]:
                                properties = {}

                                # 处理多个语义条件的情况
                                if len(match_result["semantic_info"]) == 1:
                                    # 单个语义条件
                                    semantic = match_result["semantic_info"][0]
                                    properties["similarity_score"] = semantic[
                                        "similarity_score"
                                    ]
                                    properties["semantic_columns"] = ",".join(
                                        semantic["columns"]
                                    )
                                    properties["similarity_threshold"] = semantic[
                                        "threshold"
                                    ]
                                else:
                                    # 多个语义条件，取最高相似度
                                    max_similarity = max(
                                        s["similarity_score"]
                                        for s in match_result["semantic_info"]
                                    )
                                    all_columns = []
                                    all_thresholds = []
                                    for s in match_result["semantic_info"]:
                                        all_columns.extend(s["columns"])
                                        all_thresholds.append(s["threshold"])

                                    properties["similarity_score"] = max_similarity
                                    properties["semantic_columns"] = ",".join(
                                        set(all_columns)
                                    )  # 去重
                                    properties["similarity_threshold"] = max(
                                        all_thresholds
                                    )
                                    properties["semantic_count"] = len(
                                        match_result["semantic_info"]
                                    )

                                relationship["properties"] = properties

                                # 记录语义匹配日志
                                logger.info(
                                    f"创建语义匹配关系: {s_entity_id_str} -> {t_entity_id_str}, 相似度: {properties['similarity_score']}"
                                )

                            rel_key = (
                                f"{s_entity_id_str}|{t_entity_id_str}|{rel_type.type}"
                            )
                            if rel_key not in relationship_set:  # 检查关系是否已存在
                                relationships.append(relationship)
                                relationship_set.add(rel_key)

                            # 如果是双向关系，添加反向关系
                            if rel_type.direction == "bi":
                                reverse_relationship = {
                                    "source": t_entity_id_str,  # Use string ID
                                    "target": s_entity_id_str,  # Use string ID
                                    "type": rel_type.type,
                                }

                                # 如果有语义匹配信息，也添加到反向关系中
                                if match_result["semantic_info"]:
                                    reverse_relationship["properties"] = (
                                        relationship.get("properties", {}).copy()
                                    )

                                reverse_rel_key = f"{t_entity_id_str}|{s_entity_id_str}|{rel_type.type}"
                                if (
                                    reverse_rel_key not in relationship_set
                                ):  # 检查反向关系是否已存在
                                    relationships.append(reverse_relationship)
                                    relationship_set.add(reverse_rel_key)

        return relationships

    def _evaluate_rule_items(
        self,
        rule_items,
        source_row,
        target_row,
        embedding_cache,
        source_row_idx,
        target_row_idx,
    ):  # 添加 cache 和 idx 参数
        """评估规则条件项目（可能是单个规则、语义条件或规则组）

        Args:
            rule_items: 规则项目列表，格式如:
                        [
                          {"type": "rule", ...},
                          {"type": "semantic", "columns": [...], "threshold": 0.8, "logic_operator": "OR"},
                          {"type": "group", "logic_operator": "AND", "items": [...]},
                        ]
            source_row: 源实体行数据
            target_row: 目标实体行数据
            embedding_cache: 用于缓存行嵌入向量的字典 {row_idx: {col_combo_key: embedding}}
            source_row_idx: 源实体行索引
            target_row_idx: 目标实体行索引

        Returns:
            dict: {
                'matched': bool,                    # 是否满足规则条件
                'semantic_info': list              # 语义匹配信息列表
            }
        """
        if not rule_items:
            return {"matched": True, "semantic_info": []}  # 空规则列表视为 True

        semantic_info = []  # 收集所有语义匹配信息

        # 计算第一个项目的结果
        first_item = rule_items[0]
        result = False  # Initialize result
        if first_item["type"] == "rule":
            result = self._evaluate_single_rule(first_item, source_row, target_row)
        elif first_item["type"] == "semantic":  # 处理语义条件
            semantic_result = self._evaluate_single_semantic_condition(
                first_item,
                source_row,
                target_row,
                embedding_cache,
                source_row_idx,
                target_row_idx,
            )
            if isinstance(semantic_result, dict):
                result = semantic_result["matched"]
                if semantic_result["matched"]:  # 只保存成功匹配的语义信息
                    semantic_info.append(semantic_result)
            else:
                result = bool(semantic_result)
        elif first_item["type"] == "inter_entity_compare":  # 处理实体间比较条件
            result = self._evaluate_single_rule(first_item, source_row, target_row)
        elif first_item["type"] == "group":
            group_result = self._evaluate_rule_items(
                first_item["items"],
                source_row,
                target_row,
                embedding_cache,
                source_row_idx,
                target_row_idx,
            )
            result = group_result["matched"]
            semantic_info.extend(group_result["semantic_info"])
        else:
            logger.warning(f"未知的规则项类型: {first_item.get('type')}")
            result = False  # Treat unknown type as false

        # 处理后续项目
        for i in range(1, len(rule_items)):
            item = rule_items[i]
            logic_op = item.get(
                "logic_operator", "AND"
            )  # 获取与前一项的连接符，默认为AND

            # 依据上一个结果和逻辑运算符决定是否需要继续计算
            if logic_op == "AND" and not result:
                return {"matched": False, "semantic_info": semantic_info}  # 短路AND
            if logic_op == "OR" and result:
                return {"matched": True, "semantic_info": semantic_info}  # 短路OR

            # 计算当前项目的结果
            current_result = False
            if item["type"] == "rule":
                current_result = self._evaluate_single_rule(
                    item, source_row, target_row
                )
            elif item["type"] == "semantic":  # 处理语义条件
                semantic_result = self._evaluate_single_semantic_condition(
                    item,
                    source_row,
                    target_row,
                    embedding_cache,
                    source_row_idx,
                    target_row_idx,
                )
                if isinstance(semantic_result, dict):
                    current_result = semantic_result["matched"]
                    if semantic_result["matched"]:  # 只保存成功匹配的语义信息
                        semantic_info.append(semantic_result)
                else:
                    current_result = bool(semantic_result)
            elif item["type"] == "inter_entity_compare":  # 处理实体间比较条件
                current_result = self._evaluate_single_rule(
                    item, source_row, target_row
                )
            elif item["type"] == "group":
                group_result = self._evaluate_rule_items(
                    item["items"],
                    source_row,
                    target_row,
                    embedding_cache,
                    source_row_idx,
                    target_row_idx,
                )
                current_result = group_result["matched"]
                semantic_info.extend(group_result["semantic_info"])
            else:
                logger.warning(f"未知的规则项类型: {item.get('type')}")
                current_result = False

            # 根据逻辑运算符更新结果
            if logic_op == "AND":
                result = result and current_result
            else:  # OR
                result = result or current_result

        return {"matched": result, "semantic_info": semantic_info}

    def _evaluate_single_semantic_condition(
        self,
        condition,
        source_row,
        target_row,
        embedding_cache,
        source_row_idx,
        target_row_idx,
    ):
        """评估单个语义条件

        Returns:
            dict: {
                'matched': bool,           # 是否匹配成功
                'similarity_score': float, # 相似度数值 (0-1)
                'columns': list,          # 参与匹配的列
                'threshold': float        # 使用的阈值
            } 或 False (当无法计算时)
        """
        columns = condition.get("columns")
        threshold = condition.get("threshold")

        if not columns or threshold is None:
            logger.warning(f"语义条件缺少列或阈值: {condition}")
            return False

        # --- 使用缓存计算或获取嵌入 ---
        # 为列组合创建一个唯一的键 (排序以确保顺序无关)
        col_combo_key = tuple(sorted(columns))

        # 获取或计算源嵌入
        source_embedding = None
        if (
            source_row_idx in embedding_cache
            and col_combo_key in embedding_cache[source_row_idx]
        ):
            source_embedding = embedding_cache[source_row_idx][col_combo_key]
        else:
            source_text = self._generate_text_for_embedding(source_row, columns)
            if source_text:
                embeddings_list = self._compute_embeddings([source_text])
                if embeddings_list and len(embeddings_list) > 0:
                    source_embedding = embeddings_list[0]
                    # 存入缓存
                    if source_row_idx not in embedding_cache:
                        embedding_cache[source_row_idx] = {}
                    embedding_cache[source_row_idx][col_combo_key] = source_embedding
                else:
                    logger.warning(f"无法计算源行 {source_row_idx} 的嵌入 ({columns})")
            else:
                logger.warning(f"无法为源行 {source_row_idx} 生成文本 ({columns})")

        # 获取或计算目标嵌入
        target_embedding = None
        if (
            target_row_idx in embedding_cache
            and col_combo_key in embedding_cache[target_row_idx]
        ):
            target_embedding = embedding_cache[target_row_idx][col_combo_key]
        else:
            target_text = self._generate_text_for_embedding(target_row, columns)
            if target_text:
                embeddings_list = self._compute_embeddings([target_text])
                if embeddings_list and len(embeddings_list) > 0:
                    target_embedding = embeddings_list[0]
                    # 存入缓存
                    if target_row_idx not in embedding_cache:
                        embedding_cache[target_row_idx] = {}
                    embedding_cache[target_row_idx][col_combo_key] = target_embedding
                else:
                    logger.warning(
                        f"无法计算目标行 {target_row_idx} 的嵌入 ({columns})"
                    )
            else:
                logger.warning(f"无法为目标行 {target_row_idx} 生成文本 ({columns})")

        # 如果任何一个嵌入为空，则无法比较
        if source_embedding is None or target_embedding is None:
            return False

        # 计算相似度
        similarity = self._compute_similarity(source_embedding, target_embedding)

        # 返回详细的语义匹配信息
        return {
            "matched": similarity >= threshold,
            "similarity_score": round(similarity, 4),  # 保留4位小数
            "columns": columns,
            "threshold": threshold,
        }

    def _evaluate_single_rule(self, rule, source_row, target_row):
        """评估单个规则条件

        Args:
            rule: 单个规则条件
            source_row: 源实体行数据
            target_row: 目标实体行数据

        Returns:
            bool: 是否满足规则条件
        """
        # 检查是否是实体间比较规则
        if rule.get("type") == "inter_entity_compare":
            source_column = rule.get("source_column")
            target_column = rule.get("target_column")
            operator = rule.get("operator")

            if not source_column or not target_column or not operator:
                logger.warning(f"实体间比较规则缺少必要字段: {rule}")
                return False

            # 获取两个实体的列值
            source_value = (
                source_row[source_column] if source_column in source_row else None
            )
            target_value = (
                target_row[target_column] if target_column in target_row else None
            )

            # 使用实体间比较方法
            return self._check_inter_entity_condition(
                source_value, target_value, operator
            )

        # 原有的单实体规则逻辑
        entity_type = rule.get("entity_type", "source")
        column = rule.get("column")
        operator = rule.get("operator")
        compare_value = rule.get("value")

        if not column or not operator:
            logger.warning(f"规则缺少列或操作符: {rule}")
            return False

        # 选择要检查的行
        row = source_row if entity_type == "source" else target_row
        # 获取该行列的值
        value = row[column] if column in row else None

        # 检查规则条件
        return self._check_rule_condition(value, None, operator, compare_value)

    def _check_rule_condition(
        self, source_value, target_value, operator, compare_value
    ):
        """检查规则条件是否满足

        改进版本：使用智能数据类型处理
        """
        # 使用source_value作为要检查的值
        value = source_value

        # 处理null检查
        if operator == "is_null":
            return value is None or pd.isna(value)
        elif operator == "is_not_null":
            return value is not None and not pd.isna(value)

        # 处理其他操作符，需要比较值
        if value is None or pd.isna(value):
            return False

        # 检查compare_value是否为None（除了null检查操作符）
        if compare_value is None and operator not in ["is_null", "is_not_null"]:
            logger.warning(f"规则条件比较值为None，操作符: {operator}")
            return False

        try:
            # 根据不同操作符进行比较，使用新的智能比较方法
            if operator == "==":
                return self._compare_values_equal(value, compare_value)
            elif operator == "!=":
                return not self._compare_values_equal(value, compare_value)
            elif operator in [">", "<", ">=", "<="]:
                return self._compare_values_numeric(value, compare_value, operator)
            elif operator == "in":
                if compare_value is None:
                    return False
                values = [v.strip() for v in str(compare_value).split(",")]
                return str(value) in values
            elif operator == "not_in":
                if compare_value is None:
                    return False
                values = [v.strip() for v in str(compare_value).split(",")]
                return str(value) not in values
            elif operator == "contains":
                if compare_value is None:
                    return False
                return str(compare_value) in str(value)
            elif operator == "starts_with":
                if compare_value is None:
                    return False
                return str(value).startswith(str(compare_value))
            elif operator == "ends_with":
                if compare_value is None:
                    return False
                return str(value).endswith(str(compare_value))
        except (ValueError, TypeError) as e:
            logger.warning(
                f"规则条件比较错误: {str(e)}，值: {value}，比较值: {compare_value}，操作符: {operator}"
            )
            return False

        return False

    def _check_inter_entity_condition(self, source_value, target_value, operator):
        """检查实体间比较条件是否满足

        改进版本：智能处理各种数据类型的比较
        """
        if source_value is None or target_value is None:
            return False

        # 检查pandas的NaN值
        if pd.isna(source_value) or pd.isna(target_value):
            return False

        try:
            # 根据不同操作符进行比较
            if operator == "==":
                return self._compare_values_equal(source_value, target_value)
            elif operator == "!=":
                return not self._compare_values_equal(source_value, target_value)
            elif operator in [">", "<", ">=", "<="]:
                return self._compare_values_numeric(
                    source_value, target_value, operator
                )
            elif operator == "in":
                values = [v.strip() for v in str(target_value).split(",")]
                return str(source_value) in values
            elif operator == "not_in":
                values = [v.strip() for v in str(target_value).split(",")]
                return str(source_value) not in values
            elif operator == "contains":
                return str(target_value) in str(source_value)
            elif operator == "starts_with":
                return str(source_value).startswith(str(target_value))
            elif operator == "ends_with":
                return str(source_value).endswith(str(target_value))
        except (ValueError, TypeError) as e:
            logger.warning(
                f"实体间比较条件比较错误: {str(e)}，源值: {source_value}，目标值: {target_value}，操作符: {operator}"
            )
            return False

        return False

    def _compare_values_equal(self, value1, value2):
        """智能的相等性比较"""
        try:
            # 1. 直接相等检查
            if value1 == value2:
                return True

            # 2. 字符串化比较（处理不同类型但值相同的情况）
            if str(value1) == str(value2):
                return True

            # 3. 数值比较（如果两者都可以转换为数值）
            try:
                if float(value1) == float(value2):
                    return True
            except (ValueError, TypeError):
                pass

            # 4. 日期时间比较
            datetime1 = self._try_parse_datetime(value1)
            datetime2 = self._try_parse_datetime(value2)
            if datetime1 is not None and datetime2 is not None:
                return datetime1 == datetime2

            return False
        except Exception:
            return False

    def _compare_values_numeric(self, source_value, target_value, operator):
        """智能的数值比较（包括日期时间）"""
        try:
            # 1. 尝试日期时间比较
            source_dt = self._try_parse_datetime(source_value)
            target_dt = self._try_parse_datetime(target_value)

            if source_dt is not None and target_dt is not None:
                # 两者都是日期时间类型
                if operator == ">":
                    return source_dt > target_dt
                elif operator == "<":
                    return source_dt < target_dt
                elif operator == ">=":
                    return source_dt >= target_dt
                elif operator == "<=":
                    return source_dt <= target_dt

            # 2. 尝试数值比较
            try:
                source_num = self._try_parse_number(source_value)
                target_num = self._try_parse_number(target_value)

                if source_num is not None and target_num is not None:
                    if operator == ">":
                        return source_num > target_num
                    elif operator == "<":
                        return source_num < target_num
                    elif operator == ">=":
                        return source_num >= target_num
                    elif operator == "<=":
                        return source_num <= target_num
            except (ValueError, TypeError):
                pass

            # 3. 如果都无法解析，尝试字符串比较（按字典序）
            try:
                source_str = str(source_value)
                target_str = str(target_value)

                logger.info(
                    f"回退到字符串比较: '{source_str}' {operator} '{target_str}'"
                )

                if operator == ">":
                    return source_str > target_str
                elif operator == "<":
                    return source_str < target_str
                elif operator == ">=":
                    return source_str >= target_str
                elif operator == "<=":
                    return source_str <= target_str
            except Exception:
                pass

            # 4. 最后的回退：无法比较
            logger.warning(
                f"无法比较值: {source_value} ({type(source_value)}) {operator} {target_value} ({type(target_value)})"
            )
            return False

        except Exception as e:
            logger.warning(f"数值比较失败: {str(e)}")
            return False

    def _try_parse_datetime(self, value):
        """尝试将值解析为日期时间对象"""
        try:
            # 1. 已经是pandas Timestamp
            if hasattr(value, "to_pydatetime"):
                return value.to_pydatetime()

            # 2. 已经是 datetime 或 date 对象
            if isinstance(value, (datetime, date)):
                if isinstance(value, date) and not isinstance(value, datetime):
                    # 将 date 转换为 datetime
                    return datetime.combine(value, datetime.min.time())
                return value

            # 3. 尝试解析字符串格式的日期
            if isinstance(value, str):
                value = value.strip()
                if not value:
                    return None

                # 尝试常见的日期格式
                date_formats = [
                    "%Y-%m-%d %H:%M:%S",  # 2023-01-01 12:00:00
                    "%Y-%m-%d %H:%M:%S.%f",  # 2023-01-01 12:00:00.123456
                    "%Y-%m-%d",  # 2023-01-01
                    "%Y/%m/%d",  # 2023/01/01
                    "%d/%m/%Y",  # 01/01/2023
                    "%d-%m-%Y",  # 01-01-2023
                    "%Y%m%d",  # 20230101
                    "%d.%m.%Y",  # 01.01.2023
                    "%m/%d/%Y",  # 01/01/2023 (美式)
                    "%Y-%m-%dT%H:%M:%S",  # ISO 8601 without microseconds
                    "%Y-%m-%dT%H:%M:%S.%f",  # ISO 8601 with microseconds
                    "%Y-%m-%dT%H:%M:%SZ",  # ISO 8601 with Z
                ]

                for fmt in date_formats:
                    try:
                        return datetime.strptime(value, fmt)
                    except ValueError:
                        continue

                # 尝试使用 dateutil 作为最后的备选方案（如果可用）
                try:
                    import dateutil.parser

                    return dateutil.parser.parse(value)
                except (ImportError, ValueError, TypeError):
                    pass

            # 4. 尝试解析数值时间戳
            if isinstance(value, (int, float)):
                try:
                    # 假设是Unix时间戳（秒）
                    if 1000000000 <= value <= 9999999999:  # 合理的时间戳范围
                        return datetime.fromtimestamp(value)
                    # 假设是毫秒时间戳
                    elif 1000000000000 <= value <= 9999999999999:
                        return datetime.fromtimestamp(value / 1000)
                except (ValueError, OSError):
                    pass

            return None
        except Exception:
            return None

    def _try_parse_number(self, value):
        """尝试将值解析为数值"""
        try:
            # 1. 已经是数值类型
            if isinstance(value, (int, float)):
                return float(value)

            # 2. 布尔值
            if isinstance(value, bool):
                return float(value)  # True -> 1.0, False -> 0.0

            # 3. 字符串数值
            if isinstance(value, str):
                # 移除空白字符
                value = value.strip()

                # 处理空字符串
                if not value:
                    return None

                # 处理百分号
                if value.endswith("%"):
                    return float(value[:-1]) / 100

                # 处理逗号分隔的数字（如 1,234.56）
                if "," in value:
                    value = value.replace(",", "")

                # 尝试转换为float
                return float(value)

            # 4. 其他类型，尝试直接转换
            return float(value)

        except (ValueError, TypeError):
            return None

    def _generate_text_for_embedding(self, row, columns):
        """为嵌入生成文本"""
        text_parts = []
        for col in columns:
            if col in row:
                text_parts.append(f"{col}: {row[col]}")
        return " ".join(text_parts)

    def _compute_embeddings(self, texts):
        """计算文本的嵌入向量"""
        try:
            # 使用OllamaEmbeddingFunction计算嵌入
            embedding_function = self.vanna_manager.vn.embedding_function
            if embedding_function:
                return embedding_function(texts)

            # 备用方法：如果vanna的embedding_function不可用，直接使用Ollama API
            model_name = self.vanna_manager.config.store_database.embedding_function
            ollama_url = self.vanna_manager.config.store_database.embedding_ollama_url

            import requests

            embeddings = []

            for text in texts:
                response = requests.post(
                    f"{ollama_url}/api/embeddings",
                    json={"model": model_name, "prompt": text},
                )

                if response.status_code == 200:
                    embedding = response.json().get("embedding", [])
                    embeddings.append(embedding)
                else:
                    logger.error(f"获取嵌入失败: {response.text}")
                    embeddings.append([0] * 1536)  # 使用零向量作为备用

            return embeddings
        except Exception as e:
            logger.error(f"计算嵌入失败: {str(e)}")
            # 返回一些空向量，不中断处理
            return [[0] * 1536 for _ in range(len(texts))]

    def _compute_similarity(self, vec1, vec2):
        """计算余弦相似度"""
        try:
            vec1 = np.array(vec1)
            vec2 = np.array(vec2)

            # 归一化向量
            vec1_norm = np.linalg.norm(vec1)
            vec2_norm = np.linalg.norm(vec2)

            if vec1_norm == 0 or vec2_norm == 0:
                return 0

            vec1_normalized = vec1 / vec1_norm
            vec2_normalized = vec2 / vec2_norm

            # 计算余弦相似度
            similarity = np.dot(vec1_normalized, vec2_normalized)
            return similarity
        except Exception as e:
            logger.error(f"计算相似度失败: {str(e)}")
            return 0

    def _get_database_connection(self, database_config):
        """创建数据库连接

        Args:
            database_config: 数据库配置信息字典

        Returns:
            连接对象
        """
        try:
            db_type = database_config.get("type", "mysql").lower()

            if db_type == "mysql":
                # 使用pymysql替代mysql.connector
                import pymysql

                conn = pymysql.connect(
                    host=database_config.get("host", "localhost"),
                    port=int(database_config.get("port", 3306)),
                    user=database_config.get("username", "root"),
                    password=database_config.get("password", ""),
                    database=database_config.get("database", ""),
                )
                return conn
            elif db_type == "sqlite":
                # 添加对SQLite的支持
                import sqlite3

                database_name = database_config.get("database", "")
                # 如果database_name看起来像一个路径，直接使用
                if (
                    "/" in database_name
                    or "\\" in database_name
                    or database_name.endswith(".sqlite")
                    or database_name.endswith(".db")
                ):
                    db_path = database_name
                else:
                    # 否则可能是相对路径或内存数据库
                    db_path = database_name if database_name else ":memory:"

                conn = sqlite3.connect(db_path)
                return conn
            elif db_type == "postgres" or db_type == "postgresql":
                # 添加对PostgreSQL的支持
                import psycopg2

                conn = psycopg2.connect(
                    host=database_config.get("host", "localhost"),
                    port=int(database_config.get("port", 5432)),
                    user=database_config.get("username", "postgres"),
                    password=database_config.get("password", ""),
                    dbname=database_config.get("database", ""),
                )
                return conn
            else:
                logger.error(f"不支持的数据库类型: {db_type}")
                return None
        except Exception as e:
            logger.error(f"创建数据库连接失败: {str(e)}")
            return None

    def close_neo4j_driver(self):
        """关闭 Neo4j 驱动连接"""
        if self.neo4j_driver:
            self.neo4j_driver.close()
            logger.info("Neo4j 驱动已关闭")

    def check_gds_availability(self):
        """检查 Neo4j GDS 插件是否可用"""
        if not self.neo4j_driver:
            return False
        try:
            with self.neo4j_driver.session(database="neo4j") as session:
                # 尝试调用一个简单的 GDS 函数
                session.run("RETURN gds.version()")
            logger.info("Neo4j GDS 插件可用")
            return True
        except ClientError as e:
            # CypherSyntaxError or ClientError indicates GDS procedures might not exist
            logger.warning(f"Neo4j GDS 插件似乎不可用或配置不正确: {e}")
            return False
        except Exception as e:
            logger.error(f"检查 GDS 可用性时出错: {e}")
            return False

    def check_apoc_availability(self):
        """检查 Neo4j APOC 插件是否可用"""
        if not self.neo4j_driver:
            return False
        try:
            with self.neo4j_driver.session(database="neo4j") as session:
                # 尝试调用一个简单的 APOC 函数
                session.run("RETURN apoc.version()")
            logger.info("Neo4j APOC 插件可用")
            return True
        except ClientError as e:
            # CypherSyntaxError or ClientError indicates APOC procedures might not exist
            logger.warning(f"Neo4j APOC 插件似乎不可用或配置不正确: {e}")
            return False
        except Exception as e:
            logger.error(f"检查 APOC 可用性时出错: {e}")
            return False

    def calculate_graph_metrics(self, graph_id):
        """计算指定知识图谱的各项指标 (不再自动进行 LLM 分析)"""
        if not self.neo4j_driver:
            raise ConnectionError("Neo4j 驱动未初始化，无法计算指标")

        metrics = {"graph_id": graph_id}
        gds_available = self.check_gds_availability()
        error_messages = []

        try:
            with self.neo4j_driver.session(database="neo4j") as session:
                # --- 基础统计指标 ---
                logger.info(f"开始计算图谱 {graph_id} 的基础指标...")
                node_count_res = session.run(
                    "MATCH (n {graph_id: $graph_id}) RETURN count(n) as count",
                    graph_id=graph_id,
                ).single()
                metrics["node_count"] = node_count_res["count"] if node_count_res else 0

                rel_count_res = session.run(
                    "MATCH (n {graph_id: $graph_id})-[r]->(m {graph_id: $graph_id}) RETURN count(r) as count",
                    graph_id=graph_id,
                ).single()
                metrics["relationship_count"] = (
                    rel_count_res["count"] if rel_count_res else 0
                )

                node_type_dist_res = session.run(
                    """
                    MATCH (n {graph_id: $graph_id})
                    WHERE size(labels(n)) > 0
                    RETURN labels(n)[0] as type, count(n) as count
                    ORDER BY count DESC
                """,
                    graph_id=graph_id,
                )
                metrics["node_type_distribution"] = {
                    record["type"]: record["count"] for record in node_type_dist_res
                }

                rel_type_dist_res = session.run(
                    """
                    MATCH (n {graph_id: $graph_id})-[r]->(m {graph_id: $graph_id})
                    RETURN type(r) as type, count(r) as count
                    ORDER BY count DESC
                """,
                    graph_id=graph_id,
                )
                metrics["relationship_type_distribution"] = {
                    record["type"]: record["count"] for record in rel_type_dist_res
                }
                logger.info("基础指标计算完成。")

                # --- 结构与连通性指标 ---
                logger.info("开始计算结构与连通性指标...")
                if metrics["node_count"] > 0:
                    avg_degree_res = session.run(
                        """
                        MATCH (n {graph_id: $graph_id})
                        RETURN avg(COUNT {(n)--()}) as avg_degree
                    """,
                        graph_id=graph_id,
                    ).single()
                    # avg_degree might be null if no relationships exist
                    metrics["average_degree"] = (
                        round(avg_degree_res["avg_degree"], 2)
                        if avg_degree_res and avg_degree_res["avg_degree"] is not None
                        else 0.0
                    )

                    # 度分布 (获取原始数据，前端处理)
                    degree_dist_res = session.run(
                        """
                        MATCH (n {graph_id: $graph_id})
                        WITH COUNT {(n)--()} as degree
                        RETURN degree, count(*) as node_count
                        ORDER BY degree ASC
                    """,
                        graph_id=graph_id,
                    )
                    metrics["degree_distribution"] = {
                        record["degree"]: record["node_count"]
                        for record in degree_dist_res
                    }

                    # 图密度 (有向图近似)
                    if metrics["node_count"] > 1:
                        max_edges = metrics["node_count"] * (metrics["node_count"] - 1)
                        metrics["graph_density"] = (
                            round(metrics["relationship_count"] / max_edges, 5)
                            if max_edges > 0
                            else 0
                        )
                    else:
                        metrics["graph_density"] = (
                            0.0  # Density is 0 for single node or no nodes
                        )
                else:
                    metrics["average_degree"] = 0.0
                    metrics["degree_distribution"] = {}
                    metrics["graph_density"] = 0.0
                logger.info("结构与连通性指标计算完成。")

                # --- 数据质量指标 ---
                logger.info("开始计算数据质量指标...")
                orphan_nodes_res = session.run(
                    """
                     MATCH (n {graph_id: $graph_id})
                     WHERE NOT (n)--()
                     RETURN count(n) as count
                """,
                    graph_id=graph_id,
                ).single()
                metrics["orphan_node_count"] = (
                    orphan_nodes_res["count"] if orphan_nodes_res else 0
                )
                logger.info("数据质量指标计算完成。")

                # --- 初始化 GDS 相关指标 ---
                metrics["community_detection"] = {
                    "available": gds_available,
                    "calculated": False,
                    "error": None,
                }
                metrics["degree_centrality"] = {
                    "available": gds_available,
                    "calculated": False,
                    "error": None,
                    "top_nodes": [],
                    "min_score": None,
                    "max_score": None,
                }
                metrics["betweenness_centrality"] = {
                    "available": gds_available,
                    "calculated": False,
                    "error": None,
                    "top_nodes": [],
                    "min_score": None,
                    "max_score": None,
                }
                metrics["closeness_centrality"] = {
                    "available": gds_available,
                    "calculated": False,
                    "error": None,
                    "top_nodes": [],
                    "min_score": None,
                    "max_score": None,
                }

                # --- GDS 指标计算 ---
                if (
                    gds_available and metrics["node_count"] > 0
                ):  # 仅当 GDS 可用且图不为空时计算
                    logger.info("开始使用 GDS 计算图指标...")
                    projected_graph_name = (
                        f"kg_metrics_graph_{graph_id}_{int(time.time())}"
                    )
                    gds_graph_projected = (
                        False  # Flag to track if projection was successful
                    )

                    try:
                        # 1. 投射图
                        projection_query = """
                        CALL gds.graph.project.cypher(
                          $graph_name,
                          'MATCH (n {graph_id: $graph_id}) RETURN id(n) AS id, labels(n) as labels',
                          'MATCH (n1 {graph_id: $graph_id})-[r]->(n2 {graph_id: $graph_id}) RETURN id(n1) AS source, id(n2) AS target, type(r) as type',
                          { parameters: { graph_id: $graph_id } }
                        ) YIELD graphName, nodeCount, relationshipCount
                        RETURN graphName, nodeCount, relationshipCount
                        """
                        project_result = session.run(
                            projection_query,
                            graph_id=graph_id,
                            graph_name=projected_graph_name,
                        ).single()

                        if project_result and project_result["nodeCount"] > 0:
                            gds_graph_projected = True
                            logger.info(
                                f"GDS 图 '{projected_graph_name}' 创建成功 ({project_result['nodeCount']} nodes, {project_result['relationshipCount']} relationships)."
                            )

                            # 2. 社区发现 (Louvain)
                            if metrics["relationship_count"] > 0:  # 社区发现需要关系
                                logger.info("开始 GDS 社区发现 (Louvain)...")
                                try:
                                    louvain_query = """
                                    CALL gds.louvain.stream($graph_name)
                                    YIELD nodeId, communityId
                                    RETURN communityId, count(nodeId) as communitySize
                                    ORDER BY communitySize DESC
                                    """
                                    community_results = session.run(
                                        louvain_query, graph_name=projected_graph_name
                                    )
                                    community_stats = {}
                                    total_communities = 0
                                    community_sizes = []
                                    for record in community_results:
                                        community_id = record["communityId"]
                                        size = record["communitySize"]
                                        community_stats[community_id] = size
                                        community_sizes.append(size)
                                        total_communities += 1

                                    if total_communities > 0:
                                        metrics["community_detection"].update(
                                            {
                                                "calculated": True,
                                                "algorithm": "Louvain",
                                                "community_count": total_communities,
                                                "community_size_distribution": community_sizes,
                                                "min_community_size": min(
                                                    community_sizes
                                                )
                                                if community_sizes
                                                else 0,
                                                "max_community_size": max(
                                                    community_sizes
                                                )
                                                if community_sizes
                                                else 0,
                                                "avg_community_size": round(
                                                    np.mean(community_sizes), 2
                                                )
                                                if community_sizes
                                                else 0,
                                                "median_community_size": int(
                                                    np.median(community_sizes)
                                                )
                                                if community_sizes
                                                else 0,
                                            }
                                        )
                                        logger.info(
                                            f"GDS 社区发现完成，发现 {total_communities} 个社区。"
                                        )
                                    else:
                                        metrics["community_detection"]["error"] = (
                                            "Louvain 算法未返回社区结果。"
                                        )
                                        logger.warning(
                                            f"图谱 {graph_id} 的 Louvain 算法未返回社区结果。"
                                        )
                                except ClientError as ce:
                                    metrics["community_detection"]["error"] = (
                                        f"GDS Louvain 执行错误: {ce}"
                                    )
                                    logger.error(
                                        f"图谱 {graph_id} GDS Louvain 失败: {ce}"
                                    )
                                except Exception as e:
                                    metrics["community_detection"]["error"] = (
                                        f"社区发现时发生未知错误: {e}"
                                    )
                                    logger.error(
                                        f"图谱 {graph_id} 社区发现时发生未知错误: {e}"
                                    )
                            else:
                                metrics["community_detection"]["error"] = (
                                    "图谱没有关系，无法进行社区发现。"
                                )
                                logger.info("图谱没有关系，跳过社区发现。")

                            # 3. 度中心性 (Degree Centrality)
                            logger.info("开始 GDS 度中心性计算...")
                            try:
                                # Get Global Top 10
                                degree_global_query = """
                                    CALL gds.degree.stream($graph_name)
                                    YIELD nodeId, score
                                    WITH gds.util.asNode(nodeId) AS node, score
                                    RETURN node.unique_id AS unique_id, COALESCE(node.name, node.unique_id) AS node_name, labels(node)[0] AS node_type, score
                                    ORDER BY score DESC
                                    LIMIT 10
                                """
                                global_top_degree_nodes = [
                                    record.data()
                                    for record in session.run(
                                        degree_global_query,
                                        graph_name=projected_graph_name,
                                    )
                                ]

                                # Get larger pool for per-type ranking
                                degree_pool_query = """
                                    CALL gds.degree.stream($graph_name)
                                    YIELD nodeId, score
                                    WITH gds.util.asNode(nodeId) AS node, score
                                    RETURN node.unique_id AS unique_id, COALESCE(node.name, node.unique_id) AS node_name, labels(node)[0] AS node_type, score
                                    ORDER BY score DESC
                                    LIMIT 5000 // Pool size for processing
                                """
                                degree_pool_nodes = [
                                    record.data()
                                    for record in session.run(
                                        degree_pool_query,
                                        graph_name=projected_graph_name,
                                    )
                                ]

                                # Process pool for per-type top 10
                                top_degree_nodes_by_type = {}
                                nodes_grouped_by_type = {}
                                for node in degree_pool_nodes:
                                    node_type = node.get("node_type")
                                    if node_type:
                                        if node_type not in nodes_grouped_by_type:
                                            nodes_grouped_by_type[node_type] = []
                                        nodes_grouped_by_type[node_type].append(node)

                                for node_type, nodes in nodes_grouped_by_type.items():
                                    # Already sorted by score due to the query
                                    top_degree_nodes_by_type[node_type] = nodes[:10]

                                # Get Min/Max scores
                                degree_stats_res = session.run(
                                    """
                                    CALL gds.degree.stats($graph_name)
                                    YIELD centralityDistribution
                                    RETURN centralityDistribution.min as min_score, centralityDistribution.max as max_score
                                """,
                                    graph_name=projected_graph_name,
                                ).single()
                                degree_min_max = (
                                    (
                                        degree_stats_res["min_score"],
                                        degree_stats_res["max_score"],
                                    )
                                    if degree_stats_res
                                    else (None, None)
                                )

                                metrics["degree_centrality"].update(
                                    {
                                        "calculated": True,
                                        "top_nodes_global": global_top_degree_nodes,  # Store global top 10
                                        "top_nodes_by_type": top_degree_nodes_by_type,  # Store per-type top 10
                                        "min_score": degree_min_max[0],
                                        "max_score": degree_min_max[1],
                                    }
                                )
                                logger.info(
                                    f"GDS 度中心性计算完成。Min: {degree_min_max[0]}, Max: {degree_min_max[1]}, Global/Per-Type Top fetched."
                                )
                            except ClientError as ce:
                                metrics["degree_centrality"]["error"] = (
                                    f"GDS Degree Centrality 执行错误: {ce}"
                                )
                                logger.error(
                                    f"图谱 {graph_id} GDS Degree Centrality 失败: {ce}"
                                )
                            except Exception as e:
                                metrics["degree_centrality"]["error"] = (
                                    f"度中心性计算时发生未知错误: {e}"
                                )
                                logger.error(
                                    f"图谱 {graph_id} 度中心性计算时发生未知错误: {e}"
                                )

                            # 4. 中间中心性 (Betweenness Centrality)
                            logger.info("开始 GDS 中间中心性计算...")
                            try:
                                # Get Global Top 10
                                betweenness_global_query = """
                                    CALL gds.betweenness.stream($graph_name)
                                    YIELD nodeId, score
                                    WITH gds.util.asNode(nodeId) AS node, score
                                    RETURN node.unique_id AS unique_id, COALESCE(node.name, node.unique_id) AS node_name, labels(node)[0] AS node_type, score
                                    ORDER BY score DESC
                                    LIMIT 10
                                """
                                global_top_betweenness_nodes = [
                                    record.data()
                                    for record in session.run(
                                        betweenness_global_query,
                                        graph_name=projected_graph_name,
                                    )
                                ]

                                # Get larger pool for per-type ranking
                                betweenness_pool_query = """
                                    CALL gds.betweenness.stream($graph_name)
                                    YIELD nodeId, score
                                    WITH gds.util.asNode(nodeId) AS node, score
                                    RETURN node.unique_id AS unique_id, COALESCE(node.name, node.unique_id) AS node_name, labels(node)[0] AS node_type, score
                                    ORDER BY score DESC
                                    LIMIT 5000 // Pool size for processing
                                """
                                betweenness_pool_nodes = [
                                    record.data()
                                    for record in session.run(
                                        betweenness_pool_query,
                                        graph_name=projected_graph_name,
                                    )
                                ]

                                # Process pool for per-type top 10
                                top_betweenness_nodes_by_type = {}
                                nodes_grouped_by_type = {}
                                for node in betweenness_pool_nodes:
                                    node_type = node.get("node_type")
                                    if node_type:
                                        if node_type not in nodes_grouped_by_type:
                                            nodes_grouped_by_type[node_type] = []
                                        nodes_grouped_by_type[node_type].append(node)

                                for node_type, nodes in nodes_grouped_by_type.items():
                                    # Already sorted by score
                                    top_betweenness_nodes_by_type[node_type] = nodes[
                                        :10
                                    ]

                                # Get Min/Max scores
                                betweenness_stats_res = session.run(
                                    """
                                    CALL gds.betweenness.stats($graph_name)
                                    YIELD centralityDistribution
                                    RETURN centralityDistribution.min as min_score, centralityDistribution.max as max_score
                                """,
                                    graph_name=projected_graph_name,
                                ).single()
                                betweenness_min_max = (
                                    (
                                        betweenness_stats_res["min_score"],
                                        betweenness_stats_res["max_score"],
                                    )
                                    if betweenness_stats_res
                                    else (None, None)
                                )

                                metrics["betweenness_centrality"].update(
                                    {
                                        "calculated": True,
                                        "top_nodes_global": global_top_betweenness_nodes,  # Store global top 10
                                        "top_nodes_by_type": top_betweenness_nodes_by_type,  # Store per-type top 10
                                        "min_score": betweenness_min_max[0],
                                        "max_score": betweenness_min_max[1],
                                    }
                                )
                                logger.info(
                                    f"GDS 中间中心性计算完成。Min: {betweenness_min_max[0]}, Max: {betweenness_min_max[1]}, Global/Per-Type Top fetched."
                                )
                            except ClientError as ce:
                                metrics["betweenness_centrality"]["error"] = (
                                    f"GDS Betweenness Centrality 执行错误: {ce}"
                                )
                                logger.error(
                                    f"图谱 {graph_id} GDS Betweenness Centrality 失败: {ce}"
                                )
                            except Exception as e:
                                metrics["betweenness_centrality"]["error"] = (
                                    f"中间中心性计算时发生未知错误: {e}"
                                )
                                logger.error(
                                    f"图谱 {graph_id} 中间中心性计算时发生未知错误: {e}"
                                )

                            # 5. 接近中心性 (Closeness Centrality)
                            # Closeness centrality might not be suitable for disconnected graphs.
                            # Check component count first? Or let GDS handle it (it might return 0 or NaN for nodes in disconnected components).
                            logger.info("开始 GDS 接近中心性计算...")
                            try:
                                # Get Global Top 10
                                closeness_global_query = """
                                    CALL gds.closeness.stream($graph_name)
                                    YIELD nodeId, score
                                    WITH gds.util.asNode(nodeId) AS node, score
                                    WHERE NOT isNaN(score)
                                    RETURN node.unique_id AS unique_id, COALESCE(node.name, node.unique_id) AS node_name, labels(node)[0] AS node_type, score
                                    ORDER BY score DESC
                                    LIMIT 10
                                """
                                global_top_closeness_nodes = [
                                    record.data()
                                    for record in session.run(
                                        closeness_global_query,
                                        graph_name=projected_graph_name,
                                    )
                                ]

                                # Get larger pool for per-type ranking
                                closeness_pool_query = """
                                    CALL gds.closeness.stream($graph_name)
                                    YIELD nodeId, score
                                    WITH gds.util.asNode(nodeId) AS node, score
                                    WHERE NOT isNaN(score)
                                    RETURN node.unique_id AS unique_id, COALESCE(node.name, node.unique_id) AS node_name, labels(node)[0] AS node_type, score
                                    ORDER BY score DESC
                                    LIMIT 5000 // Pool size for processing
                                """
                                closeness_pool_nodes = [
                                    record.data()
                                    for record in session.run(
                                        closeness_pool_query,
                                        graph_name=projected_graph_name,
                                    )
                                ]

                                # Process pool for per-type top 10
                                top_closeness_nodes_by_type = {}
                                nodes_grouped_by_type = {}
                                for node in closeness_pool_nodes:
                                    node_type = node.get("node_type")
                                    if node_type:
                                        if node_type not in nodes_grouped_by_type:
                                            nodes_grouped_by_type[node_type] = []
                                        nodes_grouped_by_type[node_type].append(node)

                                for node_type, nodes in nodes_grouped_by_type.items():
                                    # Already sorted by score
                                    top_closeness_nodes_by_type[node_type] = nodes[:10]

                                # Get Min/Max scores
                                closeness_stats_res = session.run(
                                    """
                                    CALL gds.closeness.stats($graph_name)
                                    YIELD centralityDistribution
                                    RETURN centralityDistribution.min as min_score, centralityDistribution.max as max_score
                                """,
                                    graph_name=projected_graph_name,
                                ).single()
                                closeness_min_max = (
                                    (
                                        closeness_stats_res["min_score"],
                                        closeness_stats_res["max_score"],
                                    )
                                    if closeness_stats_res
                                    else (None, None)
                                )
                                # Adjust NaN Min/Max if necessary
                                if closeness_min_max[0] is not None and np.isnan(
                                    closeness_min_max[0]
                                ):
                                    closeness_min_max = (0.0, closeness_min_max[1])
                                if closeness_min_max[1] is not None and np.isnan(
                                    closeness_min_max[1]
                                ):
                                    closeness_min_max = (closeness_min_max[0], 0.0)

                                metrics["closeness_centrality"].update(
                                    {
                                        "calculated": True,
                                        "top_nodes_global": global_top_closeness_nodes,  # Store global top 10
                                        "top_nodes_by_type": top_closeness_nodes_by_type,  # Store per-type top 10
                                        "min_score": closeness_min_max[0],
                                        "max_score": closeness_min_max[1],
                                    }
                                )
                                logger.info(
                                    f"GDS 接近中心性计算完成。Min: {closeness_min_max[0]}, Max: {closeness_min_max[1]}, Global/Per-Type Top fetched."
                                )
                            except ClientError as ce:
                                # Check if error is related to disconnected graph if needed
                                metrics["closeness_centrality"]["error"] = (
                                    f"GDS Closeness Centrality 执行错误: {ce}"
                                )
                                logger.error(
                                    f"图谱 {graph_id} GDS Closeness Centrality 失败: {ce}"
                                )
                            except Exception as e:
                                metrics["closeness_centrality"]["error"] = (
                                    f"接近中心性计算时发生未知错误: {e}"
                                )
                                logger.error(
                                    f"图谱 {graph_id} 接近中心性计算时发生未知错误: {e}"
                                )

                        else:
                            # GDS projection failed or graph was empty
                            error_msg = "GDS 图投影失败或图为空。"
                            logger.warning(f"图谱 {graph_id}: {error_msg}")
                            metrics["community_detection"]["error"] = error_msg
                            metrics["degree_centrality"]["error"] = error_msg
                            metrics["betweenness_centrality"]["error"] = error_msg
                            metrics["closeness_centrality"]["error"] = error_msg

                    except ClientError as e:
                        # Catch projection errors or other general GDS errors
                        error_msg = f"GDS 执行错误: {e}"
                        logger.error(f"图谱 {graph_id} GDS 操作失败: {e}")
                        metrics["community_detection"]["error"] = error_msg
                        metrics["degree_centrality"]["error"] = error_msg
                        metrics["betweenness_centrality"]["error"] = error_msg
                        metrics["closeness_centrality"]["error"] = error_msg
                    except Exception as e:
                        error_msg = f"GDS 计算时发生未知错误: {e}"
                        logger.error(f"图谱 {graph_id} GDS 计算时发生未知错误: {e}")
                        metrics["community_detection"]["error"] = error_msg
                        metrics["degree_centrality"]["error"] = error_msg
                        metrics["betweenness_centrality"]["error"] = error_msg
                        metrics["closeness_centrality"]["error"] = error_msg
                    finally:
                        # 清理 GDS 临时图
                        if (
                            gds_graph_projected
                        ):  # Only drop if projection succeeded initially
                            try:
                                session.run(
                                    "CALL gds.graph.drop($graph_name, false)",
                                    graph_name=projected_graph_name,
                                )
                                logger.info(
                                    f"GDS 临时图 '{projected_graph_name}' 已删除。"
                                )
                            except Exception as drop_e:
                                logger.warning(
                                    f"删除 GDS 临时图 '{projected_graph_name}' 失败: {drop_e}"
                                )

                elif not gds_available:
                    error_msg = "Neo4j GDS 插件不可用。"
                    metrics["community_detection"]["error"] = error_msg
                    metrics["degree_centrality"]["error"] = error_msg
                    metrics["betweenness_centrality"]["error"] = error_msg
                    metrics["closeness_centrality"]["error"] = error_msg
                else:  # GDS available but graph is empty
                    error_msg = "图谱为空，无法计算 GDS 指标。"
                    metrics["community_detection"]["error"] = error_msg
                    metrics["degree_centrality"]["error"] = error_msg
                    metrics["betweenness_centrality"]["error"] = error_msg
                    metrics["closeness_centrality"]["error"] = error_msg

                # --- 不再在此处调用 LLM 分析 ---
                # logger.info("开始生成大模型分析...")
                # try:
                #     llm_analysis = self.generate_llm_analysis(metrics) # 移除这行
                #     metrics['llm_analysis'] = llm_analysis             # 移除这行
                #     logger.info("大模型分析生成完成。")
                # except Exception as e:
                #     logger.error(f"生成 LLM 分析失败: {str(e)}")
                #     metrics['llm_analysis'] = "生成指标分析时出错。"    # 移除这行
                #     error_messages.append("LLM 分析失败")

        except ConnectionError as e:
            logger.error(f"连接 Neo4j 失败: {e}")
            raise  # Re-raise connection errors
        except Exception as e:
            logger.error(f"计算图谱 {graph_id} 指标时发生未知错误: {str(e)}")
            error_messages.append(f"计算指标时出错: {str(e)}")
            # Return partial metrics if possible, or raise error
            # For now, let's return what we have with an error flag

        if error_messages:
            metrics["errors"] = error_messages

        return metrics

    def generate_llm_analysis(self, metrics):
        """使用 LLM 分析计算出的图谱指标"""
        # 获取当前语言设置
        config = self.vanna_manager.get_config()
        language = config.get("language", {}).get("language", "zh-CN")

        if language == "zh-CN":
            # 准备给 LLM 的输入（中文）
            prompt = f"""
            请分析以下知识图谱的指标数据，并提供一份简洁的总结报告。
            报告应包括对图谱规模、结构、连通性、数据质量和社区结构（如果可用）的解读，
            指出任何显著的模式、潜在的优势或问题，并可以提出简要的改进建议。

            知识图谱 ID: {metrics.get("graph_id")}
            指标数据:
            - 节点总数: {metrics.get("node_count", "N/A")}
            - 关系总数: {metrics.get("relationship_count", "N/A")}
            - 节点类型分布: {json.dumps(metrics.get("node_type_distribution", {}), ensure_ascii=False, indent=2)}
            - 关系类型分布: {json.dumps(metrics.get("relationship_type_distribution", {}), ensure_ascii=False, indent=2)}
            - 平均度数: {metrics.get("average_degree", "N/A")}
            - 图密度: {metrics.get("graph_density", "N/A")}
            - 孤立节点数: {metrics.get("orphan_node_count", "N/A")}
            """

            # 添加度分布摘要 (避免发送整个字典，可能太大)
            degree_dist = metrics.get("degree_distribution", {})
            if degree_dist:
                min_degree = min(degree_dist.keys()) if degree_dist else 0
                max_degree = max(degree_dist.keys()) if degree_dist else 0
                prompt += f"- 度数分布: 最小度数={min_degree}, 最大度数={max_degree}\n"  # 可以在这里添加更多统计信息

            # 添加社区发现结果
            community_info = metrics.get("community_detection", {})
            if community_info.get("calculated"):
                prompt += f"""
            - 社区发现 (使用 {community_info.get("algorithm", "未知")} 算法):
                - 社区数量: {community_info.get("community_count", "N/A")}
                - 最小社区规模: {community_info.get("min_community_size", "N/A")}
                - 最大社区规模: {community_info.get("max_community_size", "N/A")}
                - 平均社区规模: {community_info.get("avg_community_size", "N/A")}
                - 中位数社区规模: {community_info.get("median_community_size", "N/A")}
                """
            elif community_info.get("available") == False:
                prompt += "- 社区发现: 未执行 (GDS 插件不可用)\n"
            elif community_info.get("error"):
                prompt += f"- 社区发现: 计算失败 ({community_info.get('error')})\n"
            else:
                prompt += f"- 社区发现: 未计算\n"

            prompt += "\n请根据以上数据生成分析报告:"
        else:
            # 英文提示词
            prompt = f"""
            Please analyze the following knowledge graph metrics data and provide a concise summary report.
            The report should include interpretation of graph scale, structure, connectivity, data quality, and community structure (if available),
            point out any significant patterns, potential advantages or issues, and can propose brief improvement suggestions.

            Knowledge Graph ID: {metrics.get("graph_id")}
            Metrics data:
            - Total nodes: {metrics.get("node_count", "N/A")}
            - Total relationships: {metrics.get("relationship_count", "N/A")}
            - Node type distribution: {json.dumps(metrics.get("node_type_distribution", {}), ensure_ascii=False, indent=2)}
            - Relationship type distribution: {json.dumps(metrics.get("relationship_type_distribution", {}), ensure_ascii=False, indent=2)}
            - Average degree: {metrics.get("average_degree", "N/A")}
            - Graph density: {metrics.get("graph_density", "N/A")}
            - Orphan node count: {metrics.get("orphan_node_count", "N/A")}
            """

            # 添加度分布摘要
            degree_dist = metrics.get("degree_distribution", {})
            if degree_dist:
                min_degree = min(degree_dist.keys()) if degree_dist else 0
                max_degree = max(degree_dist.keys()) if degree_dist else 0
                prompt += f"- Degree distribution: Min degree={min_degree}, Max degree={max_degree}\n"

            # 添加社区发现结果
            community_info = metrics.get("community_detection", {})
            if community_info.get("calculated"):
                prompt += f"""
            - Community detection (using {community_info.get("algorithm", "unknown")} algorithm):
                - Community count: {community_info.get("community_count", "N/A")}
                - Min community size: {community_info.get("min_community_size", "N/A")}
                - Max community size: {community_info.get("max_community_size", "N/A")}
                - Average community size: {community_info.get("avg_community_size", "N/A")}
                - Median community size: {community_info.get("median_community_size", "N/A")}
                """
            elif community_info.get("available") == False:
                prompt += (
                    "- Community detection: Not executed (GDS plugin not available)\n"
                )
            elif community_info.get("error"):
                prompt += f"- Community detection: Calculation failed ({community_info.get('error')})\n"
            else:
                prompt += f"- Community detection: Not calculated\n"

            prompt += "\nPlease generate an analysis report based on the above data:"

        # --- 调用 LLM ---
        try:
            # 首先尝试使用VannaManager的通用generate_text方法
            try:
                response = self.vanna_manager.generate_text(prompt)
                return response.strip()
            except Exception as vanna_e:
                logger.warning(
                    f"使用VannaManager.generate_text失败: {str(vanna_e)}，尝试直接调用"
                )

            # 备用方案：根据配置的模型类型直接调用API
            if hasattr(self.vanna_manager, "config") and hasattr(
                self.vanna_manager.config, "model"
            ):
                model_type = self.vanna_manager.config.model.type

                if model_type == "openai":
                    # 使用OpenAI模型
                    try:
                        # 检查VannaManager中是否有OpenAI client实例
                        if hasattr(self.vanna_manager, "vn") and hasattr(
                            self.vanna_manager.vn, "client"
                        ):
                            client = self.vanna_manager.vn.client
                            model_name = (
                                self.vanna_manager.config.model.model_name
                                or "gpt-3.5-turbo"
                            )

                            if client:
                                completion_params = {
                                    "model": model_name,
                                    "messages": [{"role": "user", "content": prompt}],
                                    "temperature": 0.7,
                                    "max_tokens": 2000,
                                }
                                response = safe_create_completion_for_kg(
                                    client, completion_params, model_name
                                )
                                analysis = response.choices[0].message.content.strip()
                                analysis = strip_reasoning_content_tags(
                                    analysis
                                ).strip()
                                return analysis
                            else:
                                logger.error("OpenAI客户端未初始化")
                                if language == "zh-CN":
                                    return "OpenAI客户端未初始化，无法生成分析。"
                                else:
                                    return "OpenAI client not initialized, cannot generate analysis."
                        else:
                            # 如果VannaManager没有client，尝试直接创建OpenAI客户端
                            from openai import OpenAI

                            client = OpenAI(
                                api_key=self.vanna_manager.config.model.api_key,
                                base_url=self.vanna_manager.config.model.api_base,
                            )
                            model_name = (
                                self.vanna_manager.config.model.model_name
                                or "gpt-3.5-turbo"
                            )

                            completion_params = {
                                "model": model_name,
                                "messages": [{"role": "user", "content": prompt}],
                                "temperature": 0.7,
                                "max_tokens": 2000,
                            }
                            response = safe_create_completion_for_kg(
                                client, completion_params, model_name
                            )
                            analysis = response.choices[0].message.content.strip()
                            analysis = strip_reasoning_content_tags(analysis).strip()
                            return analysis
                    except Exception as openai_e:
                        logger.error(f"使用OpenAI生成分析失败: {str(openai_e)}")
                        raise ConnectionError(f"调用OpenAI API失败: {openai_e}")

                elif model_type == "ollama":
                    # 使用Ollama模型
                    try:
                        import requests

                        ollama_url = self.vanna_manager.config.model.ollama_url
                        model_name = self.vanna_manager.config.model.ollama_model

                        response = requests.post(
                            f"{ollama_url}/api/generate",
                            json={
                                "model": model_name,
                                "prompt": prompt,
                                "stream": False,
                            },
                            timeout=120,
                        )
                        response.raise_for_status()
                        if language == "zh-CN":
                            default_response = "无法从Ollama模型获取分析结果。"
                        else:
                            default_response = (
                                "Unable to get analysis results from Ollama model."
                            )
                        analysis = response.json().get("response", default_response)
                        analysis = strip_reasoning_content_tags(analysis).strip()
                        return analysis.strip()
                    except requests.exceptions.RequestException as e:
                        logger.error(f"调用Ollama API出错: {e}")
                        raise ConnectionError(f"调用Ollama API失败: {e}")

                else:
                    # 其他模型类型
                    logger.warning(f"不支持的模型类型: {model_type}")
                    if language == "zh-CN":
                        return f"不支持的模型类型 '{model_type}'，无法生成分析。"
                    else:
                        return f"Unsupported model type '{model_type}', cannot generate analysis."

            # 如果所有方法都失败，返回占位符
            logger.warning("未找到合适的 LLM 调用方法，将返回占位符分析。")
            if language == "zh-CN":
                return "分析功能暂不可用，请检查模型配置或网络连接。"
            else:
                return "Analysis function is temporarily unavailable, please check model configuration or network connection."

        except Exception as e:
            logger.error(f"LLM 分析时发生未知错误: {str(e)}")
            raise

    def generate_kg_construction_suggestions(
        self, table_name, schema, database_config=None
    ):
        """使用大模型根据表结构生成知识图谱构建条件建议"""
        try:
            # 当表结构为空时，返回友好提示
            if not schema or len(schema) == 0:
                # 获取当前语言设置
                config = self.vanna_manager.get_config()
                language = config.get("language", {}).get("language", "zh-CN")

                if language == "zh-CN":
                    return {
                        "graph_name": f"{table_name} 知识图谱",
                        "error": "无法基于空的表结构生成建议。请确保表存在并且可以访问其结构。",
                    }
                else:
                    return {
                        "graph_name": f"{table_name} Knowledge Graph",
                        "error": "Unable to generate suggestions based on empty table structure. Please ensure the table exists and its structure is accessible.",
                    }

            # 获取表格的代表性数据（前10行）
            sample_data = None
            try:
                # 如果提供了数据库配置，使用它来获取数据
                if database_config:
                    conn = self._get_database_connection(database_config)
                    if conn:
                        cursor = conn.cursor()
                        # 限制获取前10行数据
                        sample_query = f"SELECT * FROM {table_name} LIMIT 10"
                        cursor.execute(sample_query)
                        rows = cursor.fetchall()

                        # 获取列名
                        column_names = [column[0] for column in cursor.description]

                        # 转换为更易读的格式
                        sample_data = []
                        for row in rows:
                            row_dict = {}
                            for i, value in enumerate(row):
                                if i < len(column_names):
                                    # 处理None值和长文本
                                    if value is None:
                                        row_dict[column_names[i]] = "NULL"
                                    elif isinstance(value, str) and len(value) > 50:
                                        row_dict[column_names[i]] = value[:50] + "..."
                                    else:
                                        row_dict[column_names[i]] = value
                            sample_data.append(row_dict)

                        cursor.close()
                        conn.close()
                else:
                    # 使用vanna_manager作为备用方法
                    try:
                        sample_df = self.vanna_manager.vn.run_sql(
                            f"SELECT * FROM {table_name} LIMIT 10"
                        )
                        if not sample_df.empty:
                            sample_data = []
                            for _, row in sample_df.iterrows():
                                row_dict = {}
                                for col in sample_df.columns:
                                    value = row[col]
                                    if pd.isna(value):
                                        row_dict[col] = "NULL"
                                    elif isinstance(value, str) and len(value) > 50:
                                        row_dict[col] = value[:50] + "..."
                                    else:
                                        row_dict[col] = value
                                sample_data.append(row_dict)
                    except Exception as vanna_e:
                        logger.warning(
                            f"使用vanna_manager获取样本数据失败: {str(vanna_e)}"
                        )

            except Exception as data_e:
                logger.warning(f"获取表 {table_name} 的样本数据失败: {str(data_e)}")
                # 继续处理，即使没有样本数据

            # 构建用于LLM的提示文本（包含样本数据）
            prompt = self._build_llm_construction_prompt(
                table_name, schema, sample_data
            )
            # 获取当前语言设置
            config = self.vanna_manager.get_config()
            language = config.get("language", {}).get("language", "zh-CN")

            if language == "zh-CN":
                logger.info(f"知识图谱构建，LLM提示文本: {prompt}")
            else:
                logger.info(f"Knowledge graph construction, LLM prompt text: {prompt}")

            # 获取LLM分析结果
            llm_response = self.vanna_manager.generate_text(prompt)

            if language == "zh-CN":
                logger.info(f"知识图谱构建，LLM回复: {llm_response}")
            else:
                logger.info(
                    f"Knowledge graph construction, LLM response: {llm_response}"
                )
            # 解析LLM回复内容
            return self._parse_kg_construction_suggestions(
                llm_response, table_name, schema
            )

        except Exception as e:
            logger.error(f"生成知识图谱构建条件建议失败: {str(e)}")
            # 获取当前语言设置
            config = self.vanna_manager.get_config()
            language = config.get("language", {}).get("language", "zh-CN")

            if language == "zh-CN":
                return {
                    "graph_name": f"{table_name} 知识图谱",
                    "error": f"生成建议时发生错误: {str(e)}",
                }
            else:
                return {
                    "graph_name": f"{table_name} Knowledge Graph",
                    "error": f"Error occurred while generating suggestions: {str(e)}",
                }

    def _parse_kg_construction_suggestions(self, llm_response, table_name, schema):
        """解析LLM返回的知识图谱构建条件建议

        处理各种可能的回复格式，确保返回有效的JSON结构
        """
        try:
            # 尝试提取JSON部分
            json_str = llm_response

            # 如果响应包含反引号围绕的代码块，尝试提取
            if "```json" in llm_response:
                pattern = r"```json\s*([\s\S]*?)\s*```"
                matches = re.findall(pattern, llm_response)
                if matches and len(matches) > 0:
                    json_str = matches[0].strip()
            elif "```" in llm_response:
                pattern = r"```\s*([\s\S]*?)\s*```"
                matches = re.findall(pattern, llm_response)
                if matches and len(matches) > 0:
                    json_str = matches[0].strip()

            # 尝试解析为JSON
            suggestions = json.loads(json_str)

            # 验证必要字段是否存在，如果不存在则添加默认值
            if not isinstance(suggestions, dict):
                # 获取当前语言设置
                config = self.vanna_manager.get_config()
                language = config.get("language", {}).get("language", "zh-CN")

                if language == "zh-CN":
                    raise ValueError("解析结果不是有效的字典对象")
                else:
                    raise ValueError("Parsing result is not a valid dictionary object")

            # 获取当前语言设置
            config = self.vanna_manager.get_config()
            language = config.get("language", {}).get("language", "zh-CN")

            # 设置默认图谱名称
            if "graph_name" not in suggestions or not suggestions["graph_name"]:
                if language == "zh-CN":
                    suggestions["graph_name"] = f"{table_name} 知识图谱"
                else:
                    suggestions["graph_name"] = f"{table_name} Knowledge Graph"

            # 确保node_types字段存在且为列表
            if "node_types" not in suggestions or not isinstance(
                suggestions["node_types"], list
            ):
                suggestions["node_types"] = []

            # 确保relationships字段存在且为列表
            if "relationships" not in suggestions or not isinstance(
                suggestions["relationships"], list
            ):
                suggestions["relationships"] = []

            # 对每个节点类型进行验证和标准化
            for i, node_type in enumerate(suggestions["node_types"]):
                # 确保必要字段存在
                if "name" not in node_type or not node_type["name"]:
                    if language == "zh-CN":
                        node_type["name"] = f"实体类型{i + 1}"
                    else:
                        node_type["name"] = f"EntityType{i + 1}"

                # 确保identifier_columns为列表
                if "identifier_columns" not in node_type:
                    node_type["identifier_columns"] = []
                elif not isinstance(node_type["identifier_columns"], list):
                    node_type["identifier_columns"] = [node_type["identifier_columns"]]

                # 确保attribute_columns为列表
                if "attribute_columns" not in node_type:
                    node_type["attribute_columns"] = []
                elif not isinstance(node_type["attribute_columns"], list):
                    node_type["attribute_columns"] = [node_type["attribute_columns"]]

                # 确保split_config存在并格式正确
                if "split_config" not in node_type or not isinstance(
                    node_type["split_config"], dict
                ):
                    node_type["split_config"] = {"enabled": False, "delimiter": None}
                else:
                    if "enabled" not in node_type["split_config"]:
                        node_type["split_config"]["enabled"] = False
                    if "delimiter" not in node_type["split_config"]:
                        node_type["split_config"]["delimiter"] = None

            # 对每个关系进行验证和标准化
            for i, relationship in enumerate(suggestions["relationships"]):
                # 确保必要字段存在
                if (
                    "source_node_type" not in relationship
                    or not relationship["source_node_type"]
                ):
                    # 查找第一个节点类型作为默认值
                    if suggestions["node_types"]:
                        relationship["source_node_type"] = suggestions["node_types"][0][
                            "name"
                        ]
                    else:
                        if language == "zh-CN":
                            relationship["source_node_type"] = "默认源类型"
                        else:
                            relationship["source_node_type"] = "DefaultSourceType"

                if (
                    "target_node_type" not in relationship
                    or not relationship["target_node_type"]
                ):
                    # 查找第二个节点类型作为默认值，如果没有则使用第一个
                    if len(suggestions["node_types"]) > 1:
                        relationship["target_node_type"] = suggestions["node_types"][1][
                            "name"
                        ]
                    elif suggestions["node_types"]:
                        relationship["target_node_type"] = suggestions["node_types"][0][
                            "name"
                        ]
                    else:
                        if language == "zh-CN":
                            relationship["target_node_type"] = "默认目标类型"
                        else:
                            relationship["target_node_type"] = "DefaultTargetType"

                if "type" not in relationship or not relationship["type"]:
                    if language == "zh-CN":
                        relationship["type"] = f"关系类型{i + 1}"
                    else:
                        relationship["type"] = f"RelationshipType{i + 1}"

                # 设置默认方向和匹配模式
                if "direction" not in relationship:
                    relationship["direction"] = "uni"

                if "matching_mode" not in relationship:
                    relationship["matching_mode"] = "intra-row"

                # 如果是跨行匹配但没有配置，添加默认配置
                if relationship["matching_mode"] == "inter-row" and (
                    "inter_row_options" not in relationship
                    or not isinstance(relationship["inter_row_options"], dict)
                ):
                    relationship["inter_row_options"] = {"grouping_columns": []}

                # 验证和标准化跨行匹配规则
                if (
                    relationship["matching_mode"] == "inter-row"
                    and "inter_row_options" in relationship
                ):
                    inter_row_options = relationship["inter_row_options"]

                    # 确保grouping_columns存在且为列表
                    if "grouping_columns" not in inter_row_options:
                        inter_row_options["grouping_columns"] = []
                    elif not isinstance(inter_row_options["grouping_columns"], list):
                        inter_row_options["grouping_columns"] = [
                            inter_row_options["grouping_columns"]
                        ]

                    # 分组列现在是可选的，为空是正常行为，不需要智能推荐
                    # 移除原有的智能推荐逻辑
                    if not inter_row_options["grouping_columns"]:
                        if language == "zh-CN":
                            logger.info(
                                f"关系 '{relationship['type']}' 是跨行匹配且无分组列，将在全表范围内匹配"
                            )
                        else:
                            logger.info(
                                f"Relationship '{relationship['type']}' is cross-row matching with no grouping columns, will match across the entire table"
                            )
                    else:
                        if language == "zh-CN":
                            logger.info(
                                f"关系 '{relationship['type']}' 是跨行匹配，分组列: {inter_row_options['grouping_columns']}"
                            )
                        else:
                            logger.info(
                                f"Relationship '{relationship['type']}' is cross-row matching, grouping columns: {inter_row_options['grouping_columns']}"
                            )

                    # 验证和标准化rules
                    if "rules" in inter_row_options and isinstance(
                        inter_row_options["rules"], list
                    ):
                        # **新增：智能修正错误的规则配置**
                        corrected_rules = self._smart_correct_rules(
                            inter_row_options["rules"], relationship["type"]
                        )
                        inter_row_options["rules"] = corrected_rules

                        for j, rule in enumerate(inter_row_options["rules"]):
                            if not isinstance(rule, dict):
                                continue

                            # 确保规则类型存在
                            if "type" not in rule:
                                rule["type"] = "rule"  # 默认为规则条件

                            if rule["type"] == "rule":
                                # 验证规则条件必要字段
                                if "entity_type" not in rule:
                                    rule["entity_type"] = "source"
                                if "column" not in rule:
                                    rule["column"] = ""
                                if "operator" not in rule:
                                    rule["operator"] = "=="
                                if "value" not in rule and rule["operator"] not in [
                                    "is_null",
                                    "is_not_null",
                                ]:
                                    rule["value"] = ""

                            elif rule["type"] == "semantic":
                                # 验证语义条件必要字段
                                if "columns" not in rule:
                                    rule["columns"] = []
                                elif not isinstance(rule["columns"], list):
                                    rule["columns"] = [rule["columns"]]
                                if "threshold" not in rule:
                                    rule["threshold"] = 0.7
                                elif not isinstance(rule["threshold"], (int, float)):
                                    try:
                                        rule["threshold"] = float(rule["threshold"])
                                    except (ValueError, TypeError):
                                        rule["threshold"] = 0.7

                            elif rule["type"] == "inter_entity_compare":
                                # 验证实体间比较条件必要字段
                                if "source_column" not in rule:
                                    rule["source_column"] = ""
                                if "target_column" not in rule:
                                    rule["target_column"] = ""
                                if "operator" not in rule:
                                    rule["operator"] = "=="

                            # 确保logic_operator存在(除了最后一个规则)
                            if (
                                j < len(inter_row_options["rules"]) - 1
                                and "logic_operator" not in rule
                            ):
                                rule["logic_operator"] = "AND"
                    else:
                        # 如果没有rules或格式不正确，添加空数组
                        inter_row_options["rules"] = []
            # 获取当前语言设置
            config = self.vanna_manager.get_config()
            language = config.get("language", {}).get("language", "zh-CN")

            if language == "zh-CN":
                logger.info(f"知识图谱构建，LLM回复规范后: {suggestions}")
            else:
                logger.info(
                    f"Knowledge graph construction, LLM response after normalization: {suggestions}"
                )
            # 验证和规范化完成
            return suggestions

        except json.JSONDecodeError as e:
            logger.error(
                f"无法解析LLM返回的JSON: {str(e)}\n响应内容: {llm_response[:500]}..."
            )
            # 获取当前语言设置
            config = self.vanna_manager.get_config()
            language = config.get("language", {}).get("language", "zh-CN")

            # 如果JSON解析失败，返回基本结构
            if language == "zh-CN":
                return {
                    "graph_name": f"{table_name} 知识图谱",
                    "node_types": [],
                    "relationships": [],
                    "error": "大模型返回的内容格式不正确，无法解析为有效的知识图谱构建条件。",
                    "raw_response": llm_response,
                }
            else:
                return {
                    "graph_name": f"{table_name} Knowledge Graph",
                    "node_types": [],
                    "relationships": [],
                    "error": "The content returned by the large model is not in the correct format and cannot be parsed as valid knowledge graph construction conditions.",
                    "raw_response": llm_response,
                }
        except Exception as e:
            logger.error(f"解析知识图谱构建条件建议时出错: {str(e)}")
            # 获取当前语言设置
            config = self.vanna_manager.get_config()
            language = config.get("language", {}).get("language", "zh-CN")

            if language == "zh-CN":
                return {
                    "graph_name": f"{table_name} 知识图谱",
                    "node_types": [],
                    "relationships": [],
                    "error": f"解析建议时发生错误: {str(e)}",
                    "raw_response": llm_response,
                }
            else:
                return {
                    "graph_name": f"{table_name} Knowledge Graph",
                    "node_types": [],
                    "relationships": [],
                    "error": f"Error occurred while parsing suggestions: {str(e)}",
                    "raw_response": llm_response,
                }

    def _build_llm_construction_prompt(self, table_name, schema, sample_data=None):
        """构建用于生成知识图谱构建条件的提示文本"""
        # 获取当前语言设置
        config = self.vanna_manager.get_config()
        language = config.get("language", {}).get("language", "zh-CN")

        # 将表结构转换为文本格式
        schema_text = ""
        for column in schema:
            if language == "zh-CN":
                primary_key = "是主键" if column.get("primary_key", False) else "非主键"
            else:
                primary_key = (
                    "Primary Key"
                    if column.get("primary_key", False)
                    else "Not Primary Key"
                )
            schema_text += f"- {column.get('name', '')}: {column.get('type', '')} ({primary_key})\n"

        # 构建样本数据文本
        sample_data_text = ""
        if sample_data and len(sample_data) > 0:
            if language == "zh-CN":
                sample_data_text = "\n表格代表性数据（前5行）:\n"
            else:
                sample_data_text = "\nTable representative data (first 5 rows):\n"
            # 添加表头
            columns = list(sample_data[0].keys()) if sample_data else []
            if columns:
                sample_data_text += "| " + " | ".join(columns) + " |\n"
                sample_data_text += "| " + " | ".join(["---"] * len(columns)) + " |\n"

                # 添加数据行
                for i, row in enumerate(sample_data[:5]):  # 只显示前5行
                    row_values = []
                    for col in columns:
                        value = row.get(col, "")
                        # 处理过长的值
                        if isinstance(value, str) and len(str(value)) > 30:
                            value = str(value)[:30] + "..."
                        row_values.append(str(value))
                    sample_data_text += "| " + " | ".join(row_values) + " |\n"

                if len(sample_data) > 5:
                    if language == "zh-CN":
                        sample_data_text += f"... (共 {len(sample_data)} 行样本数据)\n"
                    else:
                        sample_data_text += (
                            f"... (Total {len(sample_data)} rows of sample data)\n"
                        )
        else:
            if language == "zh-CN":
                sample_data_text = (
                    "\n注意：未能获取到表格的样本数据，请仅根据表结构进行分析。\n"
                )
            else:
                sample_data_text = "\nNote: Unable to obtain sample data for the table, please analyze based on table structure only.\n"

        # 根据语言设置构建不同的提示词
        if language == "zh-CN":
            prompt = f"""作为知识图谱构建专家，请根据以下表结构和样本数据为"{table_name}"表设计知识图谱的构建条件。

表名: {table_name}
表结构:
{schema_text}
{sample_data_text}

请详细分析这个表结构和样本数据，识别潜在的实体类型和关系，然后提供严格按照以下JSON格式的知识图谱构建配置：

```json
{{
  "graph_name": "描述性的图谱名称",
  "node_types": [
    {{
      "name": "实体类型名称",
      "identifier_columns": ["用于唯一标识实体的列名"],
      "attribute_columns": ["作为实体属性的列名"],
      "split_config": {{
        "enabled": false,
        "delimiter": null
      }}
    }}
  ],
  "relationships": [
    {{
      "source_node_type": "源实体类型名称",
      "target_node_type": "目标实体类型名称",
      "type": "关系类型名称",
      "direction": "uni",
      "matching_mode": "intra-row",
      "inter_row_options": {{
        "grouping_columns": ["分组列名（可选）"],
        "rules": [
          {{
            "type": "rule",
            "entity_type": "source",
            "column": "列名",
            "operator": "==",
            "value": "比较值",
            "logic_operator": "AND"
          }},
          {{
            "type": "semantic",
            "columns": ["用于语义匹配的列名"],
            "threshold": 0.7,
            "logic_operator": "OR"
          }},
          {{
            "type": "inter_entity_compare",
            "source_column": "源实体列名",
            "target_column": "目标实体列名",
            "operator": "=="
          }},
          {{
            "type": "group",
            "items": [
              {{"type": "rule", "entity_type": "source", "column": "列名", "operator": "==", "value": "值1", "logic_operator": "OR"}},
              {{"type": "rule", "entity_type": "source", "column": "列名", "operator": "==", "value": "值2"}}
            ],
            "logic_operator": "AND"
          }}
        ]
      }}
    }}
  ]
}}
```

分析指导原则：

1. **基于样本数据的实体识别**：
   - 观察样本数据中的实际值，理解数据的含义和模式
   - 识别哪些列包含实体标识符（如ID、名称、编码等）
   - 注意数据中的分隔符模式（如逗号分隔的标签、分号分隔的类别等）
   - 主键列通常是标识实体的好选择，但也要考虑业务含义
   - 检查样本数据中是否存在分隔符分离的多值字段
   - 观察数据的层次结构或分类模式
   - 识别可能的枚举值或类别字段
   - 注意时间、地理、数值等特殊数据类型

2. **实体拆分配置**：
   - 对于包含分隔符(如逗号、分号、管道符)的文本列，可以启用split_config
   - 根据样本数据确定实际使用的分隔符
   - 例如：样本数据中看到 "AI,机器学习,数据科学" 可以拆分为多个标签实体
   - 配置示例: "split_config": {{"enabled": true, "delimiter": ","}}

3. 关系设计
   - 分析列名和数据类型，识别可能的实体之间的关系
   - 方向(direction)：单向(uni)或双向(bi)
   - 匹配模式(matching_mode)必须选择以下之一：
     * "intra-row": 同行匹配，关系在同一行数据中建立
     * "inter-row": 跨行匹配，关系在不同行之间建立(适用于自引用关系等)

4. 跨行匹配详细配置(当matching_mode为"inter-row"时)
   - grouping_columns: 可选的分组列，用于在这些列值相同的行之间查找关系。如果为空，则在全表范围内匹配
   - rules: 跨行匹配的规则数组，支持以下两种类型：
   
   **分组列选择策略**：
   - 分组列完全可选，可以为空数组 []
   - 当分组列为空时，系统将在整个表的所有行之间查找关系
   - 当指定分组列时，只在这些列值相同的行之间查找关系
   - 常见分组列类型：
     * 类别列：部门、类型、分类等
     * 时间列：日期、年份、季度等  
     * 地理列：地区、城市、区域等
     * 项目列：项目ID、任务ID、批次号等
     * 状态列：阶段、状态、版本等
   
   a) 规则条件(rule类型)：
   - entity_type: "source"或"target"，指定比较哪个实体的列
   - column: 要比较的列名
   - operator: 操作符，可选："==", "!=", ">", "<", ">=", "<=", "in", "not_in", "contains", "starts_with", "ends_with", "is_null", "is_not_null"
   - value: 比较值(对于is_null和is_not_null可省略)
   - logic_operator: 与下一个条件的逻辑关系，"AND"或"OR"
   
   b) 语义条件(semantic类型)：
   - columns: 用于语义相似度计算的列名数组
   - threshold: 相似度阈值(0-1之间的浮点数)
   - logic_operator: 与下一个条件的逻辑关系，"AND"或"OR"
   
   c) 实体间比较条件(inter_entity_compare类型)：
   - source_column: 源实体的列名
   - target_column: 目标实体的列名
   - operator: 操作符，可选："==", "!=", ">", "<", ">=", "<=", "in", "not_in", "contains", "starts_with", "ends_with"
   
   d) 分组条件(group类型)：
   - type: "group"
   - items: 包含其他规则条件的数组，可以是rule、semantic、inter_entity_compare或嵌套的group
   - logic_operator: 与下一个条件的逻辑关系，"AND"或"OR"
   
   **复杂逻辑表达式支持**：
   - 使用group类型可以创建复杂的括号分组逻辑
   - 例如：(A AND B) OR (C AND D) 的结构
   - group类型相当于添加括号，其items内的条件会被当作一个整体
   
   规则示例：
   - 同类型实体关系: [{{"type": "rule", "entity_type": "source", "column": "类型", "operator": "==", "value": "设备", "logic_operator": "AND"}}, {{"type": "rule", "entity_type": "target", "column": "类型", "operator": "==", "value": "设备"}}]
   - 层级关系: [{{"type": "inter_entity_compare", "source_column": "级别", "target_column": "级别", "operator": "<"}}]
   - 排名关系: [{{"type": "inter_entity_compare", "source_column": "排名", "target_column": "排名", "operator": "<"}}]
   - 语义相似关系: [{{"type": "semantic", "columns": ["名称", "描述"], "threshold": 0.8}}]
   - 实体间比较: [{{"type": "inter_entity_compare", "source_column": "源列", "target_column": "目标列", "operator": "=="}}]
   - 复杂分组条件: [
     {{
       "type": "group",
       "items": [
         {{"type": "rule", "entity_type": "source", "column": "状态", "operator": "==", "value": "活跃", "logic_operator": "AND"}},
         {{"type": "rule", "entity_type": "target", "column": "状态", "operator": "==", "value": "活跃"}}
       ],
       "logic_operator": "OR"
     }},
     {{
       "type": "group", 
       "items": [
         {{"type": "semantic", "columns": ["名称"], "threshold": 0.9, "logic_operator": "AND"}},
         {{"type": "rule", "entity_type": "source", "column": "类型", "operator": "==", "value": "重要设备"}}
       ]
     }}
   ]

5. 常见跨行关系场景与分组策略
   - 全表自引用关系：不设分组列，直接在所有行之间查找关系
   - 组织层级关系：上级部门-下级部门 (分组列：组织、部门类型)
   - 设备依赖关系：主设备-从属设备 (分组列：系统、位置、类型)
   - 类别继承关系：父类别-子类别 (分组列：分类体系、领域)
   - 空间位置关系：包含位置-被包含位置 (分组列：区域、地理范围)
   - 时间序列关系：前一状态-后一状态 (分组列：对象ID、流程ID)
   - 供应链关系：供应商-采购商 (分组列：行业、地区、产品类型)
   - 学术关系：导师-学生、合作者 (分组列：机构、学科、项目)

6. 跨行匹配配置示例
   以设备管理表为例，包含列：设备ID、设备名称、类型、级别、系统、位置
   - 全表设备关联（无分组）：
     * grouping_columns: []
     * rules: [{{"type": "semantic", "columns": ["设备名称"], "threshold": 0.8}}]
   - 同系统设备关联：
     * grouping_columns: ["系统"]
     * rules: [{{"type": "rule", "entity_type": "source", "column": "类型", "operator": "==", "value": "主设备", "logic_operator": "AND"}}, {{"type": "rule", "entity_type": "target", "column": "类型", "operator": "==", "value": "从设备"}}]
   - 同位置设备层级：
     * grouping_columns: ["位置", "系统"]  
     * rules: [{{"type": "inter_entity_compare", "source_column": "级别", "target_column": "级别", "operator": ">"}}]

7. 注意事项
   - 每个实体类型必须有唯一的名称
   - 关系的源实体和目标实体必须是已定义的实体类型
   - 对于复杂表结构，可能需要定义多个实体类型和多种关系
   - 跨行匹配的分组列完全可选，为空时在全表范围内匹配
   - 关系类型 名称 和 实体名称 必须使用中文 

请确保生成的JSON是有效且完整的，不要包含任何额外的解释文本，只返回严格符合要求的JSON对象。
"""
        else:
            # 英文提示词
            prompt = f"""As a knowledge graph construction expert, please design knowledge graph construction conditions for the "{table_name}" table based on the following table structure and sample data.

Table name: {table_name}
Table structure:
{schema_text}
{sample_data_text}

Please analyze this table structure and sample data in detail, identify potential entity types and relationships, then provide knowledge graph construction configuration strictly following the JSON format below:

```json
{{
  "graph_name": "Descriptive graph name",
  "node_types": [
    {{
      "name": "Entity type name",
      "identifier_columns": ["Column names used to uniquely identify entities"],
      "attribute_columns": ["Column names as entity attributes"],
      "split_config": {{
        "enabled": false,
        "delimiter": null
      }}
    }}
  ],
  "relationships": [
    {{
      "source_node_type": "Source entity type name",
      "target_node_type": "Target entity type name",
      "type": "Relationship type name",
      "direction": "uni",
      "matching_mode": "intra-row",
      "inter_row_options": {{
        "grouping_columns": ["Grouping column names (optional)"],
        "rules": [
          {{
            "type": "rule",
            "entity_type": "source",
            "column": "Column name",
            "operator": "==",
            "value": "Comparison value",
            "logic_operator": "AND"
          }},
          {{
            "type": "semantic",
            "columns": ["Column names for semantic matching"],
            "threshold": 0.7,
            "logic_operator": "OR"
          }},
          {{
            "type": "inter_entity_compare",
            "source_column": "Source entity column name",
            "target_column": "Target entity column name",
            "operator": "=="
          }},
          {{
            "type": "group",
            "items": [
              {{"type": "rule", "entity_type": "source", "column": "Column name", "operator": "==", "value": "Value1", "logic_operator": "OR"}},
              {{"type": "rule", "entity_type": "source", "column": "Column name", "operator": "==", "value": "Value2"}}
            ],
            "logic_operator": "AND"
          }}
        ]
      }}
    }}
  ]
}}
```

Analysis Guidelines:

1. **Entity Identification Based on Sample Data**:
   - Observe actual values in sample data to understand data meaning and patterns
   - Identify columns containing entity identifiers (such as IDs, names, codes, etc.)
   - Pay attention to delimiter patterns in data (such as comma-separated tags, semicolon-separated categories, etc.)
   - Primary key columns are usually good choices for identifying entities, but also consider business meaning
   - Check if there are delimiter-separated multi-value fields in sample data
   - Observe hierarchical structure or classification patterns in data
   - Identify possible enumeration values or category fields
   - Pay attention to special data types such as time, geography, numerical values, etc.

2. **Entity Split Configuration**:
   - For text columns containing delimiters (such as commas, semicolons, pipe symbols), you can enable split_config
   - Determine the actual delimiter used based on sample data
   - For example: If sample data shows "AI,Machine Learning,Data Science", it can be split into multiple tag entities
   - Configuration example: "split_config": {{"enabled": true, "delimiter": ","}}

3. Relationship Design
   - Analyze column names and data types to identify possible relationships between entities
   - Direction: unidirectional (uni) or bidirectional (bi)
   - Matching mode must be one of the following:
     * "intra-row": Same-row matching, relationships established within the same row of data
     * "inter-row": Cross-row matching, relationships established between different rows (suitable for self-referencing relationships, etc.)

4. Cross-row Matching Detailed Configuration (when matching_mode is "inter-row")
   - grouping_columns: Optional grouping columns for finding relationships between rows with the same values in these columns. If empty, match across the entire table
   - rules: Array of cross-row matching rules, supporting the following types:
   
   **Grouping Column Selection Strategy**:
   - Grouping columns are completely optional and can be an empty array []
   - When grouping columns are empty, the system will search for relationships between all rows in the entire table
   - When grouping columns are specified, only search for relationships between rows with the same values in these columns
   - Common grouping column types:
     * Category columns: department, type, classification, etc.
     * Time columns: date, year, quarter, etc.
     * Geographic columns: region, city, area, etc.
     * Project columns: project ID, task ID, batch number, etc.
     * Status columns: stage, status, version, etc.
   
   a) Rule conditions (rule type):
   - entity_type: "source" or "target", specifying which entity's column to compare
   - column: Column name to compare
   - operator: Operator, options: "==", "!=", ">", "<", ">=", "<=", "in", "not_in", "contains", "starts_with", "ends_with", "is_null", "is_not_null"
   - value: Comparison value (can be omitted for is_null and is_not_null)
   - logic_operator: Logical relationship with the next condition, "AND" or "OR"
   
   b) Semantic conditions (semantic type):
   - columns: Array of column names for semantic similarity calculation
   - threshold: Similarity threshold (between 0-1)
   - logic_operator: Logical relationship with the next condition, "AND" or "OR"
   
   c) Inter-entity comparison conditions (inter_entity_compare type):
   - source_column: Source entity column name
   - target_column: Target entity column name
   - operator: Comparison operator
   
   d) Group conditions (group type):
   - items: Array containing multiple sub-conditions
   - logic_operator: Logical relationship between conditions within the group
   
   **Complex Logic Expression Support**:
   - Use group type to create complex parenthetical grouping logic
   - For example: (A AND B) OR (C AND D) structure
   - Group type is equivalent to adding parentheses, conditions within items are treated as a whole
   
   Rule examples:
   - Same type entity relationship: [{{"type": "rule", "entity_type": "source", "column": "type", "operator": "==", "value": "device", "logic_operator": "AND"}}, {{"type": "rule", "entity_type": "target", "column": "type", "operator": "==", "value": "device"}}]
   - Hierarchical relationship: [{{"type": "inter_entity_compare", "source_column": "level", "target_column": "level", "operator": "<"}}]
   - Ranking relationship: [{{"type": "inter_entity_compare", "source_column": "rank", "target_column": "rank", "operator": "<"}}]
   - Semantic similarity relationship: [{{"type": "semantic", "columns": ["name", "description"], "threshold": 0.8}}]
   - Inter-entity comparison: [{{"type": "inter_entity_compare", "source_column": "source_column", "target_column": "target_column", "operator": "=="}}]
   - Complex grouping conditions: [
     {{
       "type": "group",
       "items": [
         {{"type": "rule", "entity_type": "source", "column": "status", "operator": "==", "value": "active", "logic_operator": "AND"}},
         {{"type": "rule", "entity_type": "target", "column": "status", "operator": "==", "value": "active"}}
       ],
       "logic_operator": "OR"
     }},
     {{
       "type": "group", 
       "items": [
         {{"type": "semantic", "columns": ["name"], "threshold": 0.9, "logic_operator": "AND"}},
         {{"type": "rule", "entity_type": "source", "column": "type", "operator": "==", "value": "important_device"}}
       ]
     }}
   ]

5. Common Cross-row Relationship Scenarios and Grouping Strategies
   - Full table self-reference relationship: No grouping columns, directly search for relationships between all rows
   - Organizational hierarchy relationship: Superior department - Subordinate department (grouping columns: organization, department type)
   - Device dependency relationship: Main device - Subordinate device (grouping columns: system, location, type)
   - Category inheritance relationship: Parent category - Child category (grouping columns: classification system, domain)
   - Spatial location relationship: Containing location - Contained location (grouping columns: region, geographic scope)
   - Temporal sequence relationship: Previous state - Next state (grouping columns: object ID, process ID)
   - Supply chain relationship: Supplier - Purchaser (grouping columns: industry, region, product type)
   - Academic relationship: Mentor - Student, Collaborator (grouping columns: institution, discipline, project)

6. Cross-row Matching Configuration Examples
   Taking device management table as example, containing columns: Device ID, Device Name, Type, Level, System, Location
   - Full table device association (no grouping):
     * grouping_columns: []
     * rules: [{{"type": "semantic", "columns": ["device_name"], "threshold": 0.8}}]
   - Same system device association:
     * grouping_columns: ["system"]
     * rules: [{{"type": "rule", "entity_type": "source", "column": "type", "operator": "==", "value": "main_device", "logic_operator": "AND"}}, {{"type": "rule", "entity_type": "target", "column": "type", "operator": "==", "value": "sub_device"}}]
   - Same location device hierarchy:
     * grouping_columns: ["location", "system"]  
     * rules: [{{"type": "inter_entity_compare", "source_column": "level", "target_column": "level", "operator": ">"}}]

7. Important Notes
   - Each entity type must have a unique name
   - Source and target entities of relationships must be defined entity types
   - For complex table structures, multiple entity types and relationship types may need to be defined
   - Cross-row matching grouping columns are completely optional, when empty, match across the entire table

Please ensure the returned JSON format is completely correct, with all field names and values enclosed in double quotes."""

        return prompt

    def _create_entities_with_graph_label(
        self,
        table_data,
        entity_type,
        id_columns,
        attr_columns,
        split_config=None,
        graph_id=None,
        graph_name=None,
    ):
        """
        使用图谱标签隔离的实体创建方法（备选方案）

        每个实体会添加额外的图谱标签，如：Graph_10, ProjectKnowledgeGraph
        """
        entities = self._create_entities(
            table_data, entity_type, id_columns, attr_columns, split_config
        )

        # 为每个实体添加图谱标签信息
        for entity_id, entity_data in entities.items():
            entity_data["graph_labels"] = []
            if graph_id is not None:
                entity_data["graph_labels"].append(f"Graph_{graph_id}")
            if graph_name:
                # 清理图谱名称作为标签
                clean_graph_name = re.sub(
                    r"[^a-zA-Z0-9_]", "", graph_name.replace(" ", "_")
                )
                entity_data["graph_labels"].append(clean_graph_name)

        return entities

    def _smart_correct_rules(self, rules, relationship_type):
        """智能修正LLM生成的错误规则配置

        Args:
            rules: 原始规则列表
            relationship_type: 关系类型名称，用于日志记录

        Returns:
            修正后的规则列表
        """
        if not rules or len(rules) < 2:
            return rules

        corrected_rules = []
        i = 0

        while i < len(rules):
            current_rule = rules[i]

            # 检查是否是连续的两个rule类型且都涉及value为None的比较
            if (
                i + 1 < len(rules)
                and current_rule.get("type") == "rule"
                and rules[i + 1].get("type") == "rule"
                and current_rule.get("value") is None
                and rules[i + 1].get("value") is None
            ):
                next_rule = rules[i + 1]

                # 检查是否符合实体间比较的模式
                if self._is_inter_entity_comparison_pattern(current_rule, next_rule):
                    # 转换为inter_entity_compare规则
                    corrected_rule = self._convert_to_inter_entity_compare(
                        current_rule, next_rule
                    )
                    corrected_rules.append(corrected_rule)
                    # 获取当前语言设置
                    config = self.vanna_manager.get_config()
                    language = config.get("language", {}).get("language", "zh-CN")

                    if language == "zh-CN":
                        logger.info(
                            f"关系 '{relationship_type}' 的规则已智能修正：将两个无效的单实体规则转换为实体间比较规则"
                        )
                    else:
                        logger.info(
                            f"Rules for relationship '{relationship_type}' have been intelligently corrected: converted two invalid single-entity rules to inter-entity comparison rules"
                        )
                    i += 2  # 跳过下一个规则，因为已经合并处理
                    continue

            # 如果不符合修正模式，保持原样
            corrected_rules.append(current_rule)
            i += 1

        return corrected_rules

    def _is_inter_entity_comparison_pattern(self, rule1, rule2):
        """检查两个规则是否符合实体间比较的模式

        Args:
            rule1, rule2: 要检查的两个规则

        Returns:
            bool: 是否符合实体间比较模式
        """
        # 检查基本条件
        if (
            rule1.get("type") != "rule"
            or rule2.get("type") != "rule"
            or rule1.get("value") is not None
            or rule2.get("value") is not None
        ):
            return False

        # 检查是否是相同列的比较
        column1 = rule1.get("column")
        column2 = rule2.get("column")
        if not column1 or not column2 or column1 != column2:
            return False

        # 检查entity_type是否不同
        entity_type1 = rule1.get("entity_type")
        entity_type2 = rule2.get("entity_type")
        if entity_type1 == entity_type2:
            return False

        # 检查操作符是否是对应的比较操作符
        op1 = rule1.get("operator")
        op2 = rule2.get("operator")

        # 常见的对应模式
        comparison_pairs = [
            ("<", ">"),
            (">", "<"),
            ("<=", ">="),
            (">=", "<="),
            ("==", "=="),
            ("!=", "!="),
        ]

        for pair in comparison_pairs:
            if (op1, op2) == pair or (op2, op1) == pair:
                return True

        return False

    def _convert_to_inter_entity_compare(self, rule1, rule2):
        """将两个单实体规则转换为一个实体间比较规则

        Args:
            rule1, rule2: 要转换的两个规则

        Returns:
            dict: 转换后的inter_entity_compare规则
        """
        # 确定源列和目标列
        column = rule1.get("column")  # 两个规则应该是同一列

        # 确定哪个是源实体，哪个是目标实体
        if rule1.get("entity_type") == "source":
            source_rule = rule1
            target_rule = rule2
        else:
            source_rule = rule2
            target_rule = rule1

        # 确定比较操作符
        # 我们需要构建 source_column operator target_column 的形式
        source_op = source_rule.get("operator")
        target_op = target_rule.get("operator")

        # 根据源实体的操作符来确定最终的比较操作符
        # 例如：source.rank < None 和 target.rank > None
        # 应该转换为 source.rank < target.rank
        final_operator = source_op

        # 保留逻辑操作符（如果第二个规则有的话）
        logic_operator = target_rule.get("logic_operator")

        inter_entity_rule = {
            "type": "inter_entity_compare",
            "source_column": column,
            "target_column": column,
            "operator": final_operator,
        }

        # 如果有逻辑操作符，保留它
        if logic_operator:
            inter_entity_rule["logic_operator"] = logic_operator

        return inter_entity_rule


# 配置更新回调函数
def _on_kg_config_update(new_config_dict):
    """知识图谱管理器配置更新回调"""
    try:
        # 重新加载配置
        kg_manager.reload_config(new_config_dict)
        logger.info("知识图谱管理器配置已更新")
    except Exception as e:
        logger.error(f"知识图谱管理器配置更新失败: {str(e)}")


# Update the global instance - consider how the application lifecycle manages closing the driver
kg_manager = KnowledgeGraphManager()

# 注册配置更新回调
from backend.services.vanna_service import vanna_manager

vanna_manager.register_config_callback(_on_kg_config_update)

# Example of closing the driver when the app shuts down (depends on your framework)
# atexit.register(kg_manager.close_neo4j_driver)
