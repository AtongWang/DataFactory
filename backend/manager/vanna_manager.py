import os
import json
import threading
import requests
import re
import plotly.io as pio
from vanna.remote import VannaDefault
from vanna.ollama import Ollama
from vanna.openai import OpenAI_Chat
from flask import current_app as app
from vanna.chromadb import ChromaDB_VectorStore
from chromadb.api.types import EmbeddingFunction
from chromadb.utils.embedding_functions.ollama_embedding_function import (
    OllamaEmbeddingFunction,
)
from backend.config.config_templates import AppConfig
from backend.services.enhanced_vanna_models import EnhancedOllama, EnhancedOpenAI_Chat
from vanna.utils import deterministic_uuid
import logging

logger = logging.getLogger(__name__)
# 优化的HNSW配置
optimized_hnsw_config = {
    "hnsw:space": "cosine",  # 余弦相似度（适合文本）
    "hnsw:construction_ef": 200,  # 构建时更多候选（提高质量）
    "hnsw:M": 64,  # 更多邻居连接（提高精度）
    "hnsw:search_ef": 100,  # 搜索时更多候选（提高召回）
    "hnsw:num_threads": 4,  # 多线程加速
    "hnsw:resize_factor": 1.2,  # 合理的增长率
    "hnsw:batch_size": 1000,  # 大批次处理
    "hnsw:sync_threshold": 5000,  # 适合大数据集的同步阈值
}


# 自定义嵌入函数包装器，过滤无效参数
class FilteredOllamaEmbeddingFunction:
    """
    过滤无效参数的 Ollama 嵌入函数包装器
    """

    def __init__(self, url: str, model_name: str, embedding_options: dict = None):
        self.url = url
        self.model_name = model_name
        # 只保留嵌入模型支持的参数
        self.embedding_options = embedding_options or {}
        # 过滤掉嵌入模型不支持的参数
        self.valid_embedding_params = {"temperature"}  # 嵌入模型支持的参数白名单
        self.filtered_options = {
            k: v
            for k, v in self.embedding_options.items()
            if k in self.valid_embedding_params
        }

        # 创建底层的 OllamaEmbeddingFunction
        self.underlying_function = OllamaEmbeddingFunction(
            url=url, model_name=model_name
        )

    def __call__(self, input):
        """
        调用底层的嵌入函数，不传递无效的参数
        """
        return self.underlying_function(input)


def safe_create_completion_for_vanna(client, completion_params, model_name="unknown"):
    """
    安全的创建completion请求，包含provider参数错误的自动回退机制
    专门用于Vanna管理器中的LLM调用
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
                f"Vanna管理器: 模型 {model_name} 不支持provider参数，尝试移除provider参数后重试"
            )

            # 创建不包含provider参数的副本
            fallback_params = completion_params.copy()
            if "provider" in fallback_params:
                del fallback_params["provider"]

            try:
                # 重试不带provider参数的请求
                response = client.chat.completions.create(**fallback_params)
                logger.info(
                    f"Vanna管理器: 模型 {model_name} 成功使用fallback请求（无provider参数）"
                )
                return response
            except Exception as fallback_e:
                logger.error(
                    f"Vanna管理器: 模型 {model_name} fallback请求也失败: {str(fallback_e)}"
                )
                raise fallback_e
        else:
            # 其他类型的错误，直接抛出
            raise e


class VannaOllama(ChromaDB_VectorStore, EnhancedOllama):
    question_table_name = None

    def __init__(self, config=None):
        if config and "embedding_model" in config:
            # 使用自定义的过滤嵌入函数包装器
            embedding_ollama_url = config.get(
                "embedding_ollama_url", "http://localhost:11434"
            )
            config["embedding_function"] = FilteredOllamaEmbeddingFunction(
                url=f"{embedding_ollama_url}/api/embeddings",
                model_name=config.get("embedding_model", "llama2"),
                embedding_options=config.get("embedding_options", {}),
            )
            config["collection_metadata"] = optimized_hnsw_config
        ChromaDB_VectorStore.__init__(self, config=config)
        EnhancedOllama.__init__(self, config=config)

    @staticmethod
    def get_available_models(ollama_url="http://localhost:11434"):
        """获取可用的Ollama模型列表"""
        try:
            response = requests.get(f"{ollama_url}/api/tags")
            if response.status_code == 200:
                return [model["name"] for model in response.json()["models"]]
            return []
        except Exception as e:
            logger.info(f"获取Ollama模型列表失败: {str(e)}")
            return []

    def get_related_ddl(self, question: str, **kwargs) -> list[str]:
        """
        Retrieves DDL statements relevant to the question by matching table names.
        First tries exact matching via metadata, then falls back to text-based matching.

        Args:
            question: The natural language question
            **kwargs: Additional parameters, can include explicit table_name

        Returns:
            List of DDL statements related to the question
        """
        try:
            logger.info(f"[DEBUG] get_related_ddl 开始执行: question='{question}'")

            # 1. 检查是否有明确指定的表名 - 优先使用显式传入的参数
            # 修复: 显式传入的 table_name 应该优先于实例状态
            explicit_table_name = kwargs.get("table_name") or self.question_table_name

            logger.info(
                f"DEBUG: get_related_ddl called with question='{question}', explicit_table_name='{explicit_table_name}'"
            )
            logger.info(
                f"DEBUG: kwargs.get('table_name')='{kwargs.get('table_name')}', self.question_table_name='{self.question_table_name}'"
            )

            # 2. 如果有明确指定表名，优先使用metadata精确匹配
            if explicit_table_name:
                try:
                    logger.info(
                        f"DEBUG: 尝试使用metadata精确匹配表名: '{explicit_table_name}'"
                    )
                    logger.info(
                        f"[DEBUG] 检查 ddl_collection 是否存在: {hasattr(self, 'ddl_collection')}"
                    )
                    if not hasattr(self, "ddl_collection"):
                        logger.info(f"[ERROR] ddl_collection 不存在!")
                        return []

                    # 使用metadata精确匹配 - 尝试多种格式
                    metadata_queries = [
                        {"table_name": explicit_table_name},
                        {"table_name": explicit_table_name.lower()},
                        {"table_name": explicit_table_name.upper()},
                    ]

                    for i, query in enumerate(metadata_queries):
                        logger.info(
                            f"[DEBUG] 执行metadata查询 {i + 1}/{len(metadata_queries)}: {query}"
                        )
                        metadata_results = self.ddl_collection.get(where=query)
                        logger.info(
                            f"DEBUG: Metadata查询 {query} 结果: {len(metadata_results.get('documents', []))} 个文档"
                        )
                        # 如果找到匹配的DDL，直接返回
                        if metadata_results and metadata_results.get("documents", []):
                            logger.info(f"DEBUG: 通过metadata找到匹配的DDL")
                            return metadata_results.get("documents", [])

                    logger.info(f"DEBUG: 所有metadata查询都失败，继续尝试其他方法")
                except Exception as e:
                    logger.info(f"[ERROR] Metadata查询失败: {e}")
                    import traceback

                    traceback.print_exc()

            logger.info(f"[DEBUG] 开始从问题中提取表名...")
            # 3. 如果没有明确表名或精确匹配失败，从问题中提取可能的表名
            question_table_names = extract_table_names_from_question(question)
            logger.info(f"DEBUG: 从问题中提取的表名: {question_table_names}")

            if not question_table_names and not explicit_table_name:
                logger.info(f"DEBUG: 无法提取表名且没有明确指定，返回空列表")
                return []

            logger.info(f"[DEBUG] 开始获取所有DDL文档...")
            # 4. 尝试获取所有DDL
            try:
                all_ddls_data = self.ddl_collection.get()
                all_ddl_statements = all_ddls_data.get("documents", [])
                all_metadatas = all_ddls_data.get("metadatas", [])
                logger.info(f"DEBUG: 获取到 {len(all_ddl_statements)} 个DDL文档")

                # 打印所有存储的表名用于调试
                if all_metadatas:
                    stored_table_names = [
                        meta.get("table_name", "N/A") for meta in all_metadatas if meta
                    ]
                    logger.info(f"DEBUG: 存储的表名列表: {stored_table_names}")

            except Exception as e:
                logger.info(f"[ERROR] Error fetching DDLs from ChromaDB: {e}")
                import traceback

                traceback.print_exc()
                return []

            if not all_ddl_statements:
                logger.info(f"DEBUG: 没有找到任何DDL文档")
                return []

            logger.info(f"[DEBUG] 开始匹配表名...")
            # 5. 通过metadata或DDL文本内容匹配表名
            related_ddl = []

            # 合并显式指定的表名和从问题提取的表名
            all_possible_table_names = set()
            if question_table_names:
                all_possible_table_names.update(
                    [name.lower() for name in question_table_names]
                )
            if explicit_table_name:
                all_possible_table_names.add(explicit_table_name.lower())

            logger.info(f"DEBUG: 要匹配的表名集合: {all_possible_table_names}")

            for i, ddl in enumerate(all_ddl_statements):
                if not isinstance(ddl, str):
                    continue

                # 检查metadata (如果有)
                if all_metadatas and i < len(all_metadatas):
                    metadata_table_name = all_metadatas[i].get("table_name", "")
                    logger.info(
                        f"DEBUG: 检查metadata[{i}]: '{metadata_table_name}' vs 目标表名集合"
                    )

                    if metadata_table_name:
                        # 严格匹配和大小写不敏感匹配
                        if (
                            metadata_table_name in all_possible_table_names
                            or metadata_table_name.lower() in all_possible_table_names
                        ):
                            logger.info(
                                f"DEBUG: 通过metadata匹配到表名: '{metadata_table_name}'"
                            )
                            related_ddl.append(ddl)
                            continue

                # 如果metadata没有匹配，尝试从DDL内容中提取表名
                ddl_table_name = extract_table_name_from_ddl(ddl)
                logger.info(f"DEBUG: 从DDL[{i}]提取的表名: '{ddl_table_name}'")

                if ddl_table_name and ddl_table_name in all_possible_table_names:
                    logger.info(f"DEBUG: 通过DDL内容匹配到表名: '{ddl_table_name}'")
                    related_ddl.append(ddl)

            logger.info(f"DEBUG: 最终匹配到 {len(related_ddl)} 个相关DDL")
            logger.info(f"[DEBUG] get_related_ddl 执行完成")
            return related_ddl
        except Exception as e:
            logger.info(f"[ERROR] 获取相关DDL时出错: {e}")
            import traceback

            traceback.print_exc()
            return []

    def add_ddl(self, ddl: str, **kwargs) -> str:
        id = deterministic_uuid(ddl) + "-ddl"
        self.ddl_collection.add(
            documents=ddl,
            embeddings=self.generate_embedding(ddl),
            ids=id,
            metadatas={"table_name": kwargs.get("table_name", "")}
            if "table_name" in kwargs
            else None,
        )
        return id


class VannaOpenAI(ChromaDB_VectorStore, EnhancedOpenAI_Chat):
    question_table_name = None

    def __init__(self, config=None):
        if config and "embedding_model" in config:
            # 使用自定义的过滤嵌入函数包装器
            embedding_ollama_url = config.get(
                "embedding_ollama_url", "http://localhost:11434"
            )
            config["embedding_function"] = FilteredOllamaEmbeddingFunction(
                url=f"{embedding_ollama_url}/api/embeddings",
                model_name=config.get("embedding_model", "llama2"),
                embedding_options=config.get("embedding_options", {}),
            )
        ChromaDB_VectorStore.__init__(self, config=config)
        EnhancedOpenAI_Chat.__init__(
            self, client=config.get("openai_client"), config=config
        )

    def get_related_ddl(self, question: str, **kwargs) -> list[str]:
        """
        Retrieves DDL statements relevant to the question by matching table names.
        First tries exact matching via metadata, then falls back to text-based matching.

        Args:
            question: The natural language question
            **kwargs: Additional parameters, can include explicit table_name

        Returns:
            List of DDL statements related to the question
        """
        try:
            logger.info(f"[DEBUG] get_related_ddl 开始执行: question='{question}'")

            # 1. 检查是否有明确指定的表名 - 优先使用显式传入的参数
            # 修复: 显式传入的 table_name 应该优先于实例状态
            explicit_table_name = kwargs.get("table_name") or self.question_table_name

            logger.info(
                f"DEBUG: get_related_ddl called with question='{question}', explicit_table_name='{explicit_table_name}'"
            )
            logger.info(
                f"DEBUG: kwargs.get('table_name')='{kwargs.get('table_name')}', self.question_table_name='{self.question_table_name}'"
            )

            # 2. 如果有明确指定表名，优先使用metadata精确匹配
            if explicit_table_name:
                try:
                    logger.info(
                        f"DEBUG: 尝试使用metadata精确匹配表名: '{explicit_table_name}'"
                    )
                    logger.info(
                        f"[DEBUG] 检查 ddl_collection 是否存在: {hasattr(self, 'ddl_collection')}"
                    )
                    if not hasattr(self, "ddl_collection"):
                        logger.info(f"[ERROR] ddl_collection 不存在!")
                        return []

                    # 使用metadata精确匹配 - 尝试多种格式
                    metadata_queries = [
                        {"table_name": explicit_table_name},
                        {"table_name": explicit_table_name.lower()},
                        {"table_name": explicit_table_name.upper()},
                    ]

                    for i, query in enumerate(metadata_queries):
                        logger.info(
                            f"[DEBUG] 执行metadata查询 {i + 1}/{len(metadata_queries)}: {query}"
                        )
                        metadata_results = self.ddl_collection.get(where=query)
                        logger.info(
                            f"DEBUG: Metadata查询 {query} 结果: {len(metadata_results.get('documents', []))} 个文档"
                        )
                        # 如果找到匹配的DDL，直接返回
                        if metadata_results and metadata_results.get("documents", []):
                            logger.info(f"DEBUG: 通过metadata找到匹配的DDL")
                            return metadata_results.get("documents", [])

                    logger.info(f"DEBUG: 所有metadata查询都失败，继续尝试其他方法")
                except Exception as e:
                    logger.info(f"[ERROR] Metadata查询失败: {e}")
                    import traceback

                    traceback.print_exc()

            logger.info(f"[DEBUG] 开始从问题中提取表名...")
            # 3. 如果没有明确表名或精确匹配失败，从问题中提取可能的表名
            question_table_names = extract_table_names_from_question(question)
            logger.info(f"DEBUG: 从问题中提取的表名: {question_table_names}")

            if not question_table_names and not explicit_table_name:
                logger.info(f"DEBUG: 无法提取表名且没有明确指定，返回空列表")
                return []

            logger.info(f"[DEBUG] 开始获取所有DDL文档...")
            # 4. 尝试获取所有DDL
            try:
                all_ddls_data = self.ddl_collection.get()
                all_ddl_statements = all_ddls_data.get("documents", [])
                all_metadatas = all_ddls_data.get("metadatas", [])
                logger.info(f"DEBUG: 获取到 {len(all_ddl_statements)} 个DDL文档")

                # 打印所有存储的表名用于调试
                if all_metadatas:
                    stored_table_names = [
                        meta.get("table_name", "N/A") for meta in all_metadatas if meta
                    ]
                    logger.info(f"DEBUG: 存储的表名列表: {stored_table_names}")

            except Exception as e:
                logger.info(f"[ERROR] Error fetching DDLs from ChromaDB: {e}")
                import traceback

                traceback.print_exc()
                return []

            if not all_ddl_statements:
                logger.info(f"DEBUG: 没有找到任何DDL文档")
                return []

            logger.info(f"[DEBUG] 开始匹配表名...")
            # 5. 通过metadata或DDL文本内容匹配表名
            related_ddl = []

            # 合并显式指定的表名和从问题提取的表名
            all_possible_table_names = set()
            if question_table_names:
                all_possible_table_names.update(
                    [name.lower() for name in question_table_names]
                )
            if explicit_table_name:
                all_possible_table_names.add(explicit_table_name.lower())

            logger.info(f"DEBUG: 要匹配的表名集合: {all_possible_table_names}")

            for i, ddl in enumerate(all_ddl_statements):
                if not isinstance(ddl, str):
                    continue

                # 检查metadata (如果有)
                if all_metadatas and i < len(all_metadatas):
                    metadata_table_name = all_metadatas[i].get("table_name", "")
                    logger.info(
                        f"DEBUG: 检查metadata[{i}]: '{metadata_table_name}' vs 目标表名集合"
                    )

                    if metadata_table_name:
                        # 严格匹配和大小写不敏感匹配
                        if (
                            metadata_table_name in all_possible_table_names
                            or metadata_table_name.lower() in all_possible_table_names
                        ):
                            logger.info(
                                f"DEBUG: 通过metadata匹配到表名: '{metadata_table_name}'"
                            )
                            related_ddl.append(ddl)
                            continue

                # 如果metadata没有匹配，尝试从DDL内容中提取表名
                ddl_table_name = extract_table_name_from_ddl(ddl)
                logger.info(f"DEBUG: 从DDL[{i}]提取的表名: '{ddl_table_name}'")

                if ddl_table_name and ddl_table_name in all_possible_table_names:
                    logger.info(f"DEBUG: 通过DDL内容匹配到表名: '{ddl_table_name}'")
                    related_ddl.append(ddl)

            logger.info(f"DEBUG: 最终匹配到 {len(related_ddl)} 个相关DDL")
            logger.info(f"[DEBUG] get_related_ddl 执行完成")
            return related_ddl
        except Exception as e:
            logger.info(f"[ERROR] 获取相关DDL时出错: {e}")
            import traceback

            traceback.print_exc()
            return []

    def add_ddl(self, ddl: str, **kwargs) -> str:
        id = deterministic_uuid(ddl) + "-ddl"
        self.ddl_collection.add(
            documents=ddl,
            embeddings=self.generate_embedding(ddl),
            ids=id,
            metadatas={"table_name": kwargs.get("table_name", "")}
            if "table_name" in kwargs
            else None,
        )
        return id


# Helper function to extract table names from question (simple version)
def extract_table_names_from_question(question: str) -> set[str]:
    """
    从问题中提取可能的表名。

    使用多种策略：
    1. 单词分割和简单处理
    2. 识别特定模式，如"表XX"，"XX表"，"in the XX"等

    Args:
        question: 用户的问题文本

    Returns:
        可能的表名集合，全部转为小写
    """
    # 基本的单词提取（保留原有逻辑作为基础）
    words = re.findall(r"\b\w+\b", question.lower())
    potential_tables = set(words)

    # 识别更复杂的表名模式
    # 中文表名模式: "表XXX"或"XXX表"
    cn_table_patterns = [
        r"表\s*([a-zA-Z0-9_]+)",  # 匹配"表user", "表 orders"等
        r"([a-zA-Z0-9_]+)\s*表",  # 匹配"user表", "orders 表"等
        r"从\s*([a-zA-Z0-9_]+)",  # 匹配"从users", "从 orders"等
    ]

    # 英文表名模式
    en_table_patterns = [
        r"from\s+([a-zA-Z0-9_]+)",  # 匹配"from users", "from orders"等
        r"in\s+(?:the\s+)?([a-zA-Z0-9_]+)",  # 匹配"in users", "in the orders"等
        r"table\s+([a-zA-Z0-9_]+)",  # 匹配"table users", "table orders"等
        r"(?:tables|relations)\s+([a-zA-Z0-9_]+)",  # 匹配"tables users", "relations orders"等
    ]

    # 组合所有模式
    all_patterns = cn_table_patterns + en_table_patterns

    # 应用所有模式
    for pattern in all_patterns:
        matches = re.findall(pattern, question.lower())
        potential_tables.update(matches)

    # 过滤掉常见的SQL关键字，避免误判
    sql_keywords = {
        "select",
        "from",
        "where",
        "and",
        "or",
        "join",
        "inner",
        "outer",
        "left",
        "right",
        "group",
        "by",
        "having",
        "order",
        "limit",
        "offset",
        "case",
        "when",
        "then",
        "else",
        "union",
        "all",
        "insert",
        "update",
        "delete",
        "count",
        "sum",
        "avg",
        "min",
        "max",
    }

    # 移除短词和SQL关键字
    filtered_tables = {
        table
        for table in potential_tables
        if len(table) > 2 and table not in sql_keywords
    }

    return filtered_tables


# 增强从DDL提取表名的函数
def extract_table_name_from_ddl(ddl: str) -> str | None:
    """
    从DDL语句中提取表名。
    支持多种格式：
    - 有无引号/反引号
    - 可选的IF NOT EXISTS
    - 支持模式名.表名格式
    - 支持非ASCII字符（如中文表名）

    Args:
        ddl: CREATE TABLE DDL语句

    Returns:
        提取的表名（小写），如果无法提取则返回None
    """
    if not ddl:
        return None

    # 标准化DDL（删除多余空格、换行符）以简化处理
    normalized_ddl = re.sub(r"\s+", " ", ddl).strip()

    # 1. 匹配带反引号的表名: CREATE TABLE [IF NOT EXISTS] `schema`.`table` 或 `table`
    backtick_patterns = [
        r"CREATE\s+TABLE(?:\s+IF\s+NOT\s+EXISTS)?\s+`(?:[^`]+\.)?([^`]+)`",  # `schema`.`table` 或 `table`
        r"CREATE\s+TABLE(?:\s+IF\s+NOT\s+EXISTS)?\s+(?:[^`]+\.)?`([^`]+)`",  # schema.`table` 或 `table`
    ]

    # 2. 匹配带双引号的表名: CREATE TABLE [IF NOT EXISTS] "schema"."table" 或 "table"
    quote_patterns = [
        r'CREATE\s+TABLE(?:\s+IF\s+NOT\s+EXISTS)?\s+"(?:[^"]+\.)?"([^"]+)"',  # "schema"."table" 或 "table"
        r'CREATE\s+TABLE(?:\s+IF\s+NOT\s+EXISTS)?\s+(?:[^"]+\.)?"([^"]+)"',  # schema."table" 或 "table"
    ]

    # 3. 匹配不带引号的表名: CREATE TABLE [IF NOT EXISTS] schema.table 或 table
    no_quote_pattern = (
        r"CREATE\s+TABLE(?:\s+IF\s+NOT\s+EXISTS)?\s+(?:[a-zA-Z0-9_]+\.)?([a-zA-Z0-9_]+)"
    )

    # 4. 匹配支持Unicode字符（如中文）的模式
    unicode_pattern = (
        r'CREATE\s+TABLE(?:\s+IF\s+NOT\s+EXISTS)?\s+(?:`|")?([^\s`"]+)(?:`|")?'
    )

    # 依次尝试所有模式
    for pattern in (
        backtick_patterns + quote_patterns + [no_quote_pattern, unicode_pattern]
    ):
        match = re.search(pattern, normalized_ddl, re.IGNORECASE)
        if match:
            return match.group(1).lower()

    # 如果上述所有模式都未匹配成功，尝试一个更宽松的模式
    fallback_pattern = (
        r'CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?(?:`|")?([^\s`"(),]+)(?:`|")?'
    )
    match = re.search(fallback_pattern, normalized_ddl, re.IGNORECASE)
    if match:
        table_name = match.group(1).lower()
        # 移除可能的模式名部分
        if "." in table_name:
            table_name = table_name.split(".")[-1]
        return table_name

    return None


class VannaManager:
    # 优化的SQL生成提示词
    ENHANCED_SQL_PROMPT = """
You are an expert SQL analyst focused on systematic data exploration and discovery.

=== SYSTEMATIC APPROACH ===
1. **UNDERSTAND THE REQUEST**: Analyze what the user is really asking for
2. **EXPLORE BEFORE ASSUME**: Discover what data actually exists rather than assuming based on schema
3. **PROGRESSIVE QUERYING**: Start broad, then narrow down based on findings
4. **ADAPTIVE STRATEGIES**: If one approach fails, try alternative methods

=== CORE PRINCIPLES ===
1. **Schema Skepticism**: DDL shows structure samples, not complete data reality
   - Column examples may not represent all possible values
   - Use exploratory queries to discover actual data patterns
   - Don't assume specific values exist without checking

2. **Data Discovery First**: Before answering complex questions
   - Sample actual data: SELECT DISTINCT column, COUNT(*) FROM table GROUP BY column
   - Explore value patterns: SELECT column, COUNT(*) FROM table WHERE column IS NOT NULL GROUP BY column
   - Check data availability: SELECT COUNT(*) FROM table WHERE condition

3. **Progressive Query Strategy**:
   Phase 1 - EXPLORATION: "What types of data exist?"
   Phase 2 - DISCOVERY: "What are the actual values and patterns?"
   Phase 3 - TARGETED: "Query specific information based on discoveries"
   Phase 4 - REFINEMENT: "Adjust approach based on results"

=== MANDATORY EXPLORATION SEQUENCE ===
Step 1: Discover Available Data Types
- Query: SELECT COLUMN_NAME, DATA_TYPE FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = 'target_table'
- Alternative: DESCRIBE table_name OR SHOW COLUMNS FROM table_name

Step 2: Sample Actual Values in Relevant Fields
- Query: SELECT DISTINCT target_column FROM table_name WHERE target_column IS NOT NULL LIMIT 50
- Pattern analysis: SELECT target_column, COUNT(*) FROM table_name GROUP BY target_column ORDER BY COUNT(*) DESC

Step 3: Pattern Analysis Based on Discovered Values
- Look for naming patterns, value formats, data distributions
- Check for partial matches, similar spellings, related terms
- Understand the actual data landscape before making assumptions

Step 4: Strategic Targeting Using Confirmed Values Only
- Based on discovered values, design specific queries
- Use flexible matching for entities that might exist in different formats
- Only query for values that were confirmed in previous steps

=== CRITICAL EXPLORATION RULES ===
1. **EXPLORATION BEFORE ASSUMPTION**: Never search for specific entities without first sampling
2. **EVIDENCE-BASED QUERIES**: Only use values confirmed through discovery queries
3. **ITERATIVE REFINEMENT**: Use exploration results to guide each subsequent query
4. **NO EMPTY ASSUMPTIONS**: Always verify data existence through sampling

=== EXPLORATION-FIRST GUIDELINES ===
For questions about specific entities (people, products, events):
✅ CORRECT Approach:
1. First discover what data types exist: SELECT DISTINCT column_name FROM table_name LIMIT 20
2. Then sample actual values: SELECT DISTINCT target_column FROM table_name WHERE target_column IS NOT NULL LIMIT 50
3. Finally target based on discoveries: Use confirmed values for specific queries

❌ WRONG Approach:
- Directly searching for assumed values without discovery
- Assuming specific names/IDs exist based only on schema examples

=== QUERY TECHNIQUES ===
1. **Flexible Matching for Robustness**:
   - Use UPPER() or LOWER() for case-insensitive searches
   - Use LIKE '%keyword%' for partial matching when exact terms might not exist
   - Use OR conditions to try multiple variations: (field LIKE '%term1%' OR field LIKE '%term2%')

2. **Data Exploration Queries**:
   - Value discovery: SELECT DISTINCT column_name FROM table LIMIT 20
   - Pattern analysis: SELECT column_name, COUNT(*) FROM table GROUP BY column_name ORDER BY COUNT(*) DESC
   - Data sampling: SELECT * FROM table LIMIT 5

3. **Null-Safe Operations**:
   - Always consider NULL values: WHERE column IS NOT NULL
   - Use COALESCE or ISNULL for handling missing data
   - Include NULL checks in filtering conditions

=== EXPLORATION EXAMPLES ===
For questions about specific entities:
```sql
-- STEP 1: First explore what exists
SELECT DISTINCT name FROM table WHERE name IS NOT NULL LIMIT 20;
-- STEP 2: Then search with flexibility based on discoveries
SELECT * FROM table WHERE UPPER(name) LIKE UPPER('%search_term%');
```

For ranking/ordering questions:
```sql
-- STEP 1: First check what ranking/ordering fields exist  
SELECT DISTINCT rank_field FROM table WHERE rank_field IS NOT NULL LIMIT 10;
-- STEP 2: Then get ordered results based on confirmed fields
SELECT * FROM table WHERE rank_field IS NOT NULL ORDER BY rank_field DESC;
```

For aggregation questions:
```sql
-- STEP 1: First understand data distribution
SELECT status, COUNT(*) FROM table GROUP BY status;
-- STEP 2: Then aggregate based on findings
SELECT category, COUNT(*) as count FROM table 
WHERE status IS NOT NULL GROUP BY category ORDER BY count DESC;
```

Generate queries that systematically explore and discover data to provide robust, reliable answers.
"""

    # 配置更新回调管理
    _config_update_callbacks = []

    def __init__(self, config_file="config.json"):
        self.config_file = config_file
        self.vn = None
        try:
            pio.renderers.default = "json"
        except Exception:
            pass
        self.config = self.load_config()
        self.store_database_config = self.config.store_database
        self._config_apply_lock = threading.Lock()
        self.init_vanna()

    @classmethod
    def register_config_callback(cls, callback):
        """注册配置更新回调函数"""
        if callback not in cls._config_update_callbacks:
            cls._config_update_callbacks.append(callback)
            logger.info(f"已注册配置更新回调: {callback.__name__}")

    @classmethod
    def unregister_config_callback(cls, callback):
        """取消注册配置更新回调函数"""
        if callback in cls._config_update_callbacks:
            cls._config_update_callbacks.remove(callback)
            logger.info(f"已取消注册配置更新回调: {callback.__name__}")

    def _notify_config_update(self, new_config_dict):
        """通知所有注册的组件配置已更新"""
        logger.info(f"通知 {len(self._config_update_callbacks)} 个组件配置已更新")
        for callback in self._config_update_callbacks:
            try:
                callback(new_config_dict)
                logger.info(f"成功通知组件: {callback.__name__}")
            except Exception as e:
                logger.error(f"通知组件配置更新失败 {callback.__name__}: {str(e)}")

    def _apply_runtime_config(self, new_config_dict):
        with self._config_apply_lock:
            self._notify_config_update(new_config_dict)
            self.init_vanna()

    def load_config(self):
        """加载配置文件"""
        if os.path.exists(self.config_file):
            with open(self.config_file, "r") as f:
                data = json.load(f)
                return AppConfig.from_dict(data)
        return AppConfig()

    def save_config(self):
        """保存配置到文件"""
        with open(self.config_file, "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)

    def init_vanna(self):
        """初始化Vanna实例"""
        # 确保 ChromaDB 数据库目录存在
        if not os.path.exists(self.store_database_config.path):
            try:
                os.makedirs(self.store_database_config.path, exist_ok=True)
            except Exception as e:
                app.logger.error(f"创建 ChromaDB 数据库目录失败: {str(e)}")
                raise

        if self.config.model.type == "vanna":
            self.vn = VannaDefault(
                model=self.config.model.model_name, api_key=self.config.model.api_key
            )
        elif self.config.model.type == "ollama":
            self.vn = VannaOllama(
                config={
                    "ollama_host": self.config.model.ollama_url,
                    "model": self.config.model.ollama_model,
                    "path": self.store_database_config.path,
                    "embedding_model": self.store_database_config.embedding_function,
                    "embedding_ollama_url": self.store_database_config.embedding_ollama_url,
                    "options": {
                        "temperature": self.config.model.temperature,
                        "num_ctx": self.config.model.num_ctx,
                    },
                    "embedding_options": {
                        "temperature": self.config.model.temperature
                    },  # 只保留嵌入模型需要的参数
                    "system_prompt": self.config.model.system_prompt,
                    "initial_prompt": self.ENHANCED_SQL_PROMPT,
                    "n_results_ddl": 3,
                    "n_results_documentation": 3,
                    "n_results_sql": 5,
                    "collection_metadata": optimized_hnsw_config,
                }
            )
        elif self.config.model.type == "openai":
            from openai import OpenAI

            # 根据是否启用Provider Routing决定使用哪个模型
            if self.config.model.use_provider_routing:
                # 启用Provider Routing时使用OpenAI Router
                openai_client = OpenAI(
                    api_key=self.config.model.api_key,
                    base_url=self.config.model.api_base,
                )
                logger.info("已启用OpenAI Provider Routing")
            else:
                # 禁用Provider Routing时直接使用OpenAI
                openai_client = OpenAI(
                    api_key=self.config.model.api_key,
                    base_url=self.config.model.api_base,
                )
                logger.info("已禁用OpenAI Provider Routing")

            self.vn = VannaOpenAI(
                config={
                    "openai_client": openai_client,
                    "path": self.store_database_config.path,
                    "embedding_model": self.store_database_config.embedding_function,
                    "embedding_ollama_url": self.store_database_config.embedding_ollama_url,
                    "model": self.config.model.model_name,
                    "temperature": self.config.model.temperature,
                    "system_prompt": self.config.model.system_prompt,
                    "initial_prompt": self.ENHANCED_SQL_PROMPT,
                    "n_results_ddl": 3,
                    "n_results_documentation": 3,
                    "n_results_sql": 5,
                }
            )
        else:
            # 默认使用Vanna
            self.vn = VannaDefault(
                model=self.config.model.model_name, api_key=self.config.model.api_key
            )

        self.connect_database()

    def connect_database(self):
        """连接数据库"""
        try:
            if self.config.database.type == "mysql":
                self.vn.connect_to_mysql(
                    host=self.config.database.host,
                    user=self.config.database.username,
                    password=self.config.database.password,
                    dbname=self.config.database.database_name,
                    port=int(self.config.database.port),
                )
            elif self.config.database.type == "postgres":
                self.vn.connect_to_postgres(
                    host=self.config.database.host,
                    user=self.config.database.username,
                    password=self.config.database.password,
                    dbname=self.config.database.database_name,
                    port=self.config.database.port,
                )
            elif self.config.database.type == "sqlite":
                self.vn.connect_to_sqlite(url=self.config.database.url)
        except Exception as e:
            logger.info(f"数据库连接错误: {str(e)}")
            raise

    def update_config(self, new_config_dict, apply_async=False):
        self.config = AppConfig.from_dict(new_config_dict)
        self.store_database_config = self.config.store_database
        self.save_config()
        if apply_async:
            threading.Thread(
                target=self._apply_runtime_config,
                args=(new_config_dict,),
                daemon=True,
            ).start()
            return

        self._apply_runtime_config(new_config_dict)

    def get_config(self):
        """获取当前配置"""
        return self.config.to_dict()

    def ask(self, question):
        """向Vanna发送问题"""
        return self.vn.ask(
            question=question,
            allow_llm_to_see_data=True,
            print_results=False,
        )

    def train(self, **kwargs):
        """训练Vanna"""
        if "sql" in kwargs and "question" in kwargs:
            self.vn.train(sql=kwargs["sql"], question=kwargs["question"])
        elif "documentation" in kwargs:
            self.vn.train(documentation=kwargs["documentation"])
        elif "ddl" in kwargs:
            table_name = kwargs.get("table_name", "")
            self.vn.add_ddl(ddl=kwargs["ddl"], table_name=table_name)

    def get_training_data(self):
        """获取训练数据"""
        return self.vn.get_training_data()

    def remove_training_data(self, id):
        """删除训练数据"""
        try:
            self.vn.remove_training_data(id=id)
            return True
        except Exception as e:
            logger.info(f"删除训练数据失败: {str(e)}")
            return False

    def generate_text(self, prompt):
        """使用大模型生成文本响应"""
        try:
            # 检查使用的模型类型
            if isinstance(self.vn, VannaOllama):
                # 使用Ollama模型生成文本
                try:
                    # 直接使用Ollama API
                    model_name = self.config.model.ollama_model
                    ollama_url = (
                        self.config.model.ollama_url or "http://localhost:11434"
                    )

                    response = requests.post(
                        f"{ollama_url}/api/generate",
                        json={"model": model_name, "prompt": prompt, "stream": False},
                        timeout=120,
                    )
                    response.raise_for_status()
                    return response.json().get("response", "无法从模型获取响应。")
                except Exception as e:
                    logger.info(f"使用Ollama生成文本失败: {str(e)}")
                    raise

            elif isinstance(self.vn, VannaOpenAI):
                # 使用OpenAI模型生成文本
                try:
                    # 获取模型配置
                    model_name = self.config.model.model_name or "gpt-3.5-turbo"

                    # 使用OpenAI客户端实例直接调用
                    client = self.vn.client  # 使用VannaOpenAI中已经初始化的client
                    if client:
                        completion_params = {
                            "model": model_name,
                            "messages": [{"role": "user", "content": prompt}],
                            "temperature": 0.7,
                            "max_tokens": 2000,
                        }
                        response = safe_create_completion_for_vanna(
                            client, completion_params, model_name
                        )
                        return response.choices[0].message.content.strip()
                    else:
                        logger.info("OpenAI客户端未初始化")
                        return "OpenAI客户端未初始化，无法生成响应。"
                except Exception as e:
                    logger.info(f"使用OpenAI生成文本失败: {str(e)}")
                    raise

            else:
                # 对于其他模型类型，尝试使用通用的调用方式
                try:
                    # 尝试使用vn的内部方法调用LLM，如果存在
                    if hasattr(self.vn, "generate_text"):
                        return self.vn.generate_text(prompt)
                    elif hasattr(self.vn, "_generate_text"):
                        return self.vn._generate_text(prompt)
                    else:
                        return "不支持的模型类型，无法生成文本响应。"
                except Exception as e:
                    logger.info(f"使用通用方法生成文本失败: {str(e)}")
                    raise

        except Exception as e:
            logger.info(f"生成文本失败: {str(e)}")
            raise
