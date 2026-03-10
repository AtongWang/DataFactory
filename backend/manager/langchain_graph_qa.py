from typing import Any, Dict, Iterator, List, Optional, Tuple, Union
from pydantic import Field, BaseModel
from langchain_core.language_models import BaseLanguageModel
from neo4j_graphrag.schema import format_schema
from langchain_neo4j.chains.graph_qa.cypher import GraphCypherQAChain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.callbacks import CallbackManagerForChainRun
from langchain_core.example_selectors import (
    BaseExampleSelector,
    SemanticSimilarityExampleSelector,
)
from langchain_neo4j.graphs.graph_store import GraphStore
from langchain_chroma import Chroma
import os
import json
from datetime import datetime
from neo4j_graphrag.retrievers.text2cypher import extract_cypher
import uuid
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class CypherExample(BaseModel):
    """用于存储Cypher查询示例的模型"""

    question: str
    query: str
    result: Optional[List[Dict[str, Any]]] = None
    answer: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)


class GraphFewShotQAChain(GraphCypherQAChain):
    """
    支持few-shot学习和文档检索的图数据库问答链

    此链通过存储和检索Cypher查询示例来改进问答能力，
    并支持从文档中检索相关信息以增强回答。
    """

    cypher_example_selector: Optional[BaseExampleSelector] = None
    doc_example_selector: Optional[BaseExampleSelector] = None
    embeddings: Any = None
    cypher_vector_store: Optional[Any] = None
    doc_vector_store: Optional[Any] = None
    persist_directory: str = "cypher_examples_db"
    num_cypher_examples: int = 5
    num_doc_examples: int = 3
    autotrain_enabled: bool = False
    system_prompt: str = None
    target_graph_id_prop: Optional[str] = Field(
        None, alias="target_graph_id_prop_alias"
    )

    @staticmethod
    def _construct_filtered_schema(
        structured_schema: Dict[str, Any],
        target_prop: Optional[str],
        include_types: List[str],
        exclude_types: List[str],
        is_enhanced: bool,
    ) -> str:
        """
        构建 schema 字符串，可选地根据 target_prop 的存在性进行过滤。
        同时应用 include_types 和 exclude_types 过滤器。
        """

        # 基础的类型过滤函数
        def base_filter_func(type_name: str) -> bool:
            return (
                type_name in include_types
                if include_types
                else type_name not in exclude_types
            )

        # 如果没有指定 target_prop，则只应用 include/exclude 过滤
        if not target_prop:
            filtered_schema_dict: Dict[str, Any] = {
                "node_props": {
                    k: v
                    for k, v in structured_schema.get("node_props", {}).items()
                    if base_filter_func(k)
                },
                "rel_props": {
                    k: v
                    for k, v in structured_schema.get("rel_props", {}).items()
                    if base_filter_func(k)
                },
                "relationships": [
                    r
                    for r in structured_schema.get("relationships", [])
                    # 确保关系的 start, end, type 都满足 include/exclude 条件
                    if all(
                        base_filter_func(r.get(t, "")) for t in ["start", "end", "type"]
                    )
                ],
                "metadata": structured_schema.get("metadata", {}),  # 保留元数据
            }
            return format_schema(filtered_schema_dict, is_enhanced)

        # --- 当提供了 target_prop 时的过滤逻辑 ---

        # 1. 过滤 node_props: 保留具有 target_prop 且满足 include/exclude 的节点类型
        filtered_node_props = {}
        valid_node_labels = set()
        for label, props in structured_schema.get("node_props", {}).items():
            for p in props:
                if p.get("property") == "graph_id":
                    if "min" in p and "max" in p:
                        if int(p["min"]) <= int(target_prop) <= int(p["max"]):
                            filtered_node_props[label] = props
                            valid_node_labels.add(label)

        # 2. 过滤 rel_props: 保留具有 target_prop 且满足 include/exclude 的关系类型
        filtered_rel_props = {}
        valid_rel_types = set()
        for rel_type, props in structured_schema.get("rel_props", {}).items():
            for p in props:
                if p.get("property") == "graph_id":
                    if "min" in p and "max" in p:
                        if int(p["min"]) <= int(target_prop) <= int(p["max"]):
                            filtered_rel_props[rel_type] = props
                            valid_rel_types.add(rel_type)

        # 3. 过滤 relationships: 保留 start/end 节点类型和关系类型都有效的关系
        filtered_relationships = []
        for r in structured_schema.get("relationships", []):
            start_node_valid = r.get("start") in valid_node_labels
            end_node_valid = r.get("end") in valid_node_labels
            rel_type_valid = r.get("type") in valid_rel_types

            # 确保关系的 start, end, type 也满足 include/exclude (虽然可能冗余，但更安全)
            relationship_types_valid = all(
                base_filter_func(r.get(t, "")) for t in ["start", "end", "type"]
            )

            if (
                start_node_valid
                and end_node_valid
                and rel_type_valid
                and relationship_types_valid
            ):
                filtered_relationships.append(r)

        # 组装过滤后的 schema 字典
        filtered_schema_dict = {
            "node_props": filtered_node_props,
            "rel_props": filtered_rel_props,
            "relationships": filtered_relationships,
            "metadata": structured_schema.get("metadata", {}),  # 保留元数据
        }

        # 使用导入的 format_schema 格式化过滤后的 schema
        return format_schema(filtered_schema_dict, is_enhanced)

    def create_graph_id_indexes(self):
        """
        为graph_id属性创建索引以提高查询性能
        这将显著加快schema获取和查询的速度
        """
        logger.info("开始创建graph_id索引...")

        try:
            # 首先获取所有节点标签
            labels_query = "CALL db.labels()"
            labels_result = self.graph.query(labels_query)
            node_labels = [record["label"] for record in labels_result]

            logger.info(f"发现 {len(node_labels)} 个节点标签: {node_labels}")

            # 为每个节点标签创建graph_id索引
            created_indexes = []
            for label in node_labels:
                try:
                    # 检查索引是否已存在
                    check_index_query = f"""
                    SHOW INDEXES 
                    WHERE labelsOrTypes = ['{label}'] AND properties = ['graph_id']
                    """
                    existing_indexes = self.graph.query(check_index_query)

                    if not existing_indexes:
                        # 创建索引
                        create_index_query = f"CREATE INDEX graph_id_idx_{label.lower().replace(' ', '_')} FOR (n:`{label}`) ON (n.graph_id)"
                        self.graph.query(create_index_query)
                        created_indexes.append(label)
                        logger.debug(f"为标签 '{label}' 创建了graph_id索引")
                    else:
                        logger.debug(f"标签 '{label}' 的graph_id索引已存在")

                except Exception as e:
                    logger.warning(f"为标签 '{label}' 创建索引失败: {str(e)}")

            logger.info(f"索引创建完成，共为 {len(created_indexes)} 个标签创建了新索引")
            return created_indexes

        except Exception as e:
            logger.error(f"创建graph_id索引失败: {str(e)}")
            return []

    def _get_graph_schema(self, target_prop, include_types=[], exclude_types=[]):
        """
        直接通过Cypher查询获取指定graph_id的schema信息
        这比获取全部schema再过滤要高效得多
        """
        if not target_prop:
            # 如果没有指定target_prop，使用原有逻辑
            try:
                raw_structured_schema = self.graph.get_structured_schema
                logger.info(f"使用全局schema (无target_prop指定)")
                if not isinstance(raw_structured_schema, dict):
                    raise TypeError(
                        "graph.get_structured_schema did not return a dictionary."
                    )
            except AttributeError:
                raise ValueError(
                    "The provided graph object does not have a 'get_structured_schema' attribute."
                )
            except Exception as e:
                raise RuntimeError(f"Failed to get structured schema from graph: {e}")

            # 应用include/exclude过滤
            try:
                is_enhanced = getattr(self.graph, "_enhanced_schema", False)
            except AttributeError:
                is_enhanced = False

            return self._construct_filtered_schema(
                structured_schema=raw_structured_schema,
                target_prop=None,
                include_types=include_types,
                exclude_types=exclude_types,
                is_enhanced=is_enhanced,
            )

        # 直接查询指定graph_id的schema信息
        logger.info(f"开始获取graph_id={target_prop}的schema信息")

        try:
            # 构建高效的schema查询结果
            structured_schema = {
                "node_props": {},
                "rel_props": {},
                "relationships": [],
                "metadata": {"constraint": [], "index": []},
            }

            # 1. 获取节点标签和属性
            node_query = """
            MATCH (n) WHERE n.graph_id = $graph_id
            WITH labels(n) as node_labels, keys(n) as node_keys
            UNWIND node_labels as label
            RETURN label, 
                   collect(DISTINCT node_keys) as all_keys,
                   count(*) as node_count
            """

            node_results = self.graph.query(node_query, {"graph_id": int(target_prop)})
            logger.info(f"节点查询返回 {len(node_results)} 个标签")

            for record in node_results:
                label = record["label"]

                # 应用include/exclude过滤
                if include_types and label not in include_types:
                    continue
                if exclude_types and label in exclude_types:
                    continue

                all_keys_sets = record["all_keys"]
                node_count = record["node_count"]

                # 收集所有唯一的属性键
                unique_keys = set()
                for key_set in all_keys_sets:
                    unique_keys.update(key_set)

                # 为每个属性构建详细信息
                properties = []
                for key in sorted(unique_keys):
                    if key == "graph_id":
                        # 对于graph_id属性，设置min/max为target_prop值
                        properties.append(
                            {
                                "property": key,
                                "type": "INTEGER",
                                "min": int(target_prop),
                                "max": int(target_prop),
                            }
                        )
                    else:
                        # 使用Neo4j内置的类型检测方法（不依赖APOC）
                        prop_query = f"""
                        MATCH (n:{label}) WHERE n.graph_id = $graph_id AND n.{key} IS NOT NULL
                        WITH n.{key} as value
                        RETURN 
                            CASE 
                                WHEN value IS NULL THEN 'NULL'
                                WHEN value = true OR value = false THEN 'BOOLEAN'
                                WHEN toString(value) =~ '^-?[0-9]+$' AND toInteger(toString(value)) IS NOT NULL THEN 'INTEGER'
                                WHEN toString(value) =~ '^-?[0-9]*\\.[0-9]+([eE][+-]?[0-9]+)?$' AND toFloat(toString(value)) IS NOT NULL THEN 'FLOAT'
                                WHEN toString(value) STARTS WITH '[' AND toString(value) ENDS WITH ']' THEN 'LIST'
                                ELSE 'STRING'
                            END as type,
                            min(value) as min_val,
                            max(value) as max_val,
                            count(*) as count
                        LIMIT 1
                        """

                        try:
                            prop_results = self.graph.query(
                                prop_query, {"graph_id": int(target_prop)}
                            )
                            if prop_results:
                                prop_info = prop_results[0]
                                prop_dict = {
                                    "property": key,
                                    "type": prop_info.get("type", "STRING"),
                                }

                                # 添加min/max值
                                if prop_info.get("min_val") is not None:
                                    prop_dict["min"] = prop_info["min_val"]
                                if prop_info.get("max_val") is not None:
                                    prop_dict["max"] = prop_info["max_val"]

                                # 为STRING类型属性获取示例值
                                if prop_info.get("type") == "STRING":
                                    try:
                                        sample_query = f"""
                                        MATCH (n:{label}) WHERE n.graph_id = $graph_id AND n.{key} IS NOT NULL
                                        WITH DISTINCT substring(toString(n.{key}), 0, 50) as sample_value
                                        RETURN collect(sample_value)[..5] as sample_values,
                                               size(collect(sample_value)) as distinct_count
                                        """
                                        sample_results = self.graph.query(
                                            sample_query, {"graph_id": int(target_prop)}
                                        )
                                        if sample_results and sample_results[0].get(
                                            "sample_values"
                                        ):
                                            sample_values = sample_results[0][
                                                "sample_values"
                                            ]
                                            distinct_count = sample_results[0].get(
                                                "distinct_count", len(sample_values)
                                            )
                                            if sample_values:
                                                prop_dict["values"] = sample_values
                                                prop_dict["distinct_count"] = (
                                                    distinct_count
                                                )
                                    except Exception as sample_e:
                                        logger.warning(
                                            f"获取属性 {key} 的示例值失败: {sample_e}"
                                        )

                                properties.append(prop_dict)
                            else:
                                # 对于查询失败的属性，尝试获取STRING类型的示例值
                                prop_dict = {"property": key, "type": "STRING"}
                                try:
                                    sample_query = f"""
                                    MATCH (n:{label}) WHERE n.graph_id = $graph_id AND n.{key} IS NOT NULL
                                    WITH DISTINCT substring(toString(n.{key}), 0, 50) as sample_value
                                    RETURN collect(sample_value)[..5] as sample_values,
                                           size(collect(sample_value)) as distinct_count
                                    """
                                    sample_results = self.graph.query(
                                        sample_query, {"graph_id": int(target_prop)}
                                    )
                                    if sample_results and sample_results[0].get(
                                        "sample_values"
                                    ):
                                        sample_values = sample_results[0][
                                            "sample_values"
                                        ]
                                        distinct_count = sample_results[0].get(
                                            "distinct_count", len(sample_values)
                                        )
                                        if sample_values:
                                            prop_dict["values"] = sample_values
                                            prop_dict["distinct_count"] = distinct_count
                                except Exception as sample_e:
                                    logger.warning(
                                        f"获取属性 {key} 的示例值失败: {sample_e}"
                                    )

                                properties.append(prop_dict)
                        except Exception as e:
                            logger.warning(f"查询属性 {key} 的详细信息失败: {e}")
                            # 对于查询失败的属性，也尝试获取示例值
                            prop_dict = {"property": key, "type": "STRING"}
                            try:
                                sample_query = f"""
                                MATCH (n:{label}) WHERE n.graph_id = $graph_id AND n.{key} IS NOT NULL
                                WITH DISTINCT substring(toString(n.{key}), 0, 50) as sample_value
                                RETURN collect(sample_value)[..5] as sample_values,
                                       size(collect(sample_value)) as distinct_count
                                """
                                sample_results = self.graph.query(
                                    sample_query, {"graph_id": int(target_prop)}
                                )
                                if sample_results and sample_results[0].get(
                                    "sample_values"
                                ):
                                    sample_values = sample_results[0]["sample_values"]
                                    distinct_count = sample_results[0].get(
                                        "distinct_count", len(sample_values)
                                    )
                                    if sample_values:
                                        prop_dict["values"] = sample_values
                                        prop_dict["distinct_count"] = distinct_count
                            except Exception as sample_e:
                                logger.warning(
                                    f"获取属性 {key} 的示例值失败: {sample_e}"
                                )

                            properties.append(prop_dict)

                structured_schema["node_props"][label] = properties

            # 2. 获取关系类型和属性
            rel_query = """
            MATCH (a)-[r]->(b) 
            WHERE a.graph_id = $graph_id AND b.graph_id = $graph_id
            WITH type(r) as rel_type, keys(r) as rel_keys, labels(a) as start_labels, labels(b) as end_labels
            RETURN rel_type,
                   collect(DISTINCT rel_keys) as all_keys,
                   collect(DISTINCT start_labels) as start_label_sets,
                   collect(DISTINCT end_labels) as end_label_sets,
                   count(*) as rel_count
            """

            rel_results = self.graph.query(rel_query, {"graph_id": int(target_prop)})
            logger.info(f"关系查询返回 {len(rel_results)} 个关系类型")

            for record in rel_results:
                rel_type = record["rel_type"]

                # 应用include/exclude过滤
                if include_types and rel_type not in include_types:
                    continue
                if exclude_types and rel_type in exclude_types:
                    continue

                all_keys_sets = record["all_keys"]
                start_label_sets = record["start_label_sets"]
                end_label_sets = record["end_label_sets"]

                # 收集所有唯一的属性键
                unique_keys = set()
                for key_set in all_keys_sets:
                    unique_keys.update(key_set)

                # 为每个属性构建详细信息
                properties = []
                for key in sorted(unique_keys):
                    if key == "graph_id":
                        properties.append(
                            {
                                "property": key,
                                "type": "INTEGER",
                                "min": int(target_prop),
                                "max": int(target_prop),
                            }
                        )
                    else:
                        # 为关系属性获取示例值
                        prop_dict = {
                            "property": key,
                            "type": "STRING",  # 简化处理，关系属性通常较少
                        }
                        try:
                            sample_query = f"""
                            MATCH (a)-[r:`{rel_type}`]->(b) 
                            WHERE a.graph_id = $graph_id AND b.graph_id = $graph_id AND r.{key} IS NOT NULL
                            WITH DISTINCT substring(toString(r.{key}), 0, 50) as sample_value
                            RETURN collect(sample_value)[..5] as sample_values,
                                   size(collect(sample_value)) as distinct_count
                            """
                            sample_results = self.graph.query(
                                sample_query, {"graph_id": int(target_prop)}
                            )
                            if sample_results and sample_results[0].get(
                                "sample_values"
                            ):
                                sample_values = sample_results[0]["sample_values"]
                                distinct_count = sample_results[0].get(
                                    "distinct_count", len(sample_values)
                                )
                                if sample_values:
                                    prop_dict["values"] = sample_values
                                    prop_dict["distinct_count"] = distinct_count
                        except Exception as sample_e:
                            logger.warning(
                                f"获取关系属性 {key} 的示例值失败: {sample_e}"
                            )

                        properties.append(prop_dict)

                structured_schema["rel_props"][rel_type] = properties

                # 构建关系模式
                for start_label_set in start_label_sets:
                    for end_label_set in end_label_sets:
                        for start_label in start_label_set:
                            for end_label in end_label_set:
                                # 应用节点标签的include/exclude过滤
                                start_filtered = (
                                    not include_types or start_label in include_types
                                )
                                start_filtered = start_filtered and (
                                    not exclude_types
                                    or start_label not in exclude_types
                                )
                                end_filtered = (
                                    not include_types or end_label in include_types
                                )
                                end_filtered = end_filtered and (
                                    not exclude_types or end_label not in exclude_types
                                )

                                if start_filtered and end_filtered:
                                    relationship = {
                                        "start": start_label,
                                        "type": rel_type,
                                        "end": end_label,
                                    }

                                    # 避免重复添加相同的关系
                                    if (
                                        relationship
                                        not in structured_schema["relationships"]
                                    ):
                                        structured_schema["relationships"].append(
                                            relationship
                                        )

            logger.info(
                f"成功获取graph_id={target_prop}的schema: "
                f"{len(structured_schema['node_props'])} 节点标签, "
                f"{len(structured_schema['rel_props'])} 关系类型, "
                f"{len(structured_schema['relationships'])} 关系模式"
            )

            # 使用neo4j_graphrag的format_schema格式化
            try:
                is_enhanced = getattr(self.graph, "_enhanced_schema", False)
            except AttributeError:
                is_enhanced = False

            return format_schema(structured_schema, is_enhanced)

        except Exception as e:
            logger.error(f"直接查询graph_id={target_prop}的schema失败: {str(e)}")
            # 回退到原有逻辑
            logger.info("回退到原有schema获取逻辑")
            try:
                raw_structured_schema = self.graph.get_structured_schema
                if not isinstance(raw_structured_schema, dict):
                    raise TypeError(
                        "graph.get_structured_schema did not return a dictionary."
                    )
            except AttributeError:
                raise ValueError(
                    "The provided graph object does not have a 'get_structured_schema' attribute."
                )
            except Exception as e:
                raise RuntimeError(f"Failed to get structured schema from graph: {e}")

            try:
                is_enhanced = getattr(self.graph, "_enhanced_schema", False)
            except AttributeError:
                is_enhanced = False

            return self._construct_filtered_schema(
                structured_schema=raw_structured_schema,
                target_prop=target_prop,
                include_types=include_types,
                exclude_types=exclude_types,
                is_enhanced=is_enhanced,
            )

    @classmethod
    def from_llm(
        cls,
        llm: Optional[BaseLanguageModel] = None,
        *,
        target_graph_id_prop: Optional[str] = None,
        embeddings: Optional[Any] = None,
        persist_directory: Optional[str] = None,
        num_cypher_examples: int = 5,
        num_doc_examples: int = 3,
        qa_prompt: Optional[Any] = None,
        cypher_prompt: Optional[Any] = None,
        cypher_llm: Optional[BaseLanguageModel] = None,
        qa_llm: Optional[BaseLanguageModel] = None,
        exclude_types: List[str] = [],
        include_types: List[str] = [],
        validate_cypher: bool = False,
        qa_llm_kwargs: Optional[Dict[str, Any]] = None,
        cypher_llm_kwargs: Optional[Dict[str, Any]] = None,
        use_function_response: bool = False,
        autotrain_enabled: bool = False,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> "GraphFewShotQAChain":
        """
        从LLM初始化，支持few-shot学习和文档检索

        Args:
            llm: 基础语言模型
            examples: Cypher查询示例列表，每个示例包含"question"和"query"
            embeddings: 用于示例检索的嵌入模型
            persist_directory: 向量存储的持久化目录
            num_examples: 在few-shot提示中使用的示例数量
            以及GraphCypherQAChain的所有参数
        """
        # 初始化向量存储
        if persist_directory:
            persist_dir = persist_directory
        else:
            persist_dir = cls.__name__.lower() + "_examples_db"

        cypher_example_selector = None
        doc_example_selector = None
        cypher_vector_store = None
        doc_vector_store = None
        if embeddings:
            if os.path.exists(persist_dir):
                # 如果目录已存在，尝试加载现有向量存储
                cypher_vector_store = Chroma(
                    collection_name="cypher-examples",
                    embedding_function=embeddings,
                    persist_directory=persist_dir,
                )
                cypher_example_selector = SemanticSimilarityExampleSelector(
                    vectorstore=cypher_vector_store,
                    k=num_cypher_examples,
                    input_keys=["question"],
                )
                doc_vector_store = Chroma(
                    collection_name="doc-examples",
                    embedding_function=embeddings,
                    persist_directory=persist_dir,
                )
                doc_example_selector = SemanticSimilarityExampleSelector(
                    vectorstore=doc_vector_store,
                    k=num_doc_examples,
                    input_keys=["question"],
                )

        # 修改Cypher生成提示模板以支持few-shot学习
        if cypher_example_selector and doc_example_selector and not cypher_prompt:
            cypher_prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        (
                            "You are an expert knowledge graph analyst focused on systematic data exploration and discovery.\n\n"
                            "SYSTEMATIC EXPLORATION APPROACH:\n"
                            "1. **UNDERSTAND THE REQUEST**: Analyze what the user is really seeking\n"
                            "2. **EXPLORE BEFORE ASSUME**: Discover what entities and relationships actually exist\n"
                            "3. **PROGRESSIVE QUERYING**: Start broad, then narrow down based on discoveries\n"
                            "4. **ADAPTIVE STRATEGIES**: If one approach fails, try alternative entity/relationship patterns\n\n"
                            "CORE PRINCIPLES:\n"
                            "- **Schema Skepticism**: Schema shows structure examples, not all actual entities\n"
                            "- **Entity Discovery First**: Sample real entities before making assumptions\n"
                            "- **Pattern Exploration**: Understand relationship patterns in actual data\n"
                            "- **Flexible Matching**: Use partial matching when exact entities might not exist\n\n"
                            "MANDATORY EXPLORATION SEQUENCE:\n"
                            "Step 1: Discover Available Entity Types\n"
                            "- Query: MATCH (n) WHERE n.graph_id = X RETURN DISTINCT labels(n), COUNT(n) ORDER BY COUNT(n) DESC\n"
                            "Step 2: Sample Actual Entity Values\n"
                            "- Query: MATCH (n:NodeType) WHERE n.graph_id = X RETURN DISTINCT n.property_name LIMIT 50\n"
                            "Step 3: Pattern Analysis Based on Discovered Values\n"
                            "- Look for naming patterns, value formats, entity distributions\n"
                            "- Check for partial matches, similar spellings, related terms\n"
                            "Step 4: Strategic Targeting Using Confirmed Entities Only\n"
                            "- Based on discovered entities, design specific queries\n"
                            "- Use flexible matching for entities that might exist in different formats\n\n"
                            "CRITICAL EXPLORATION RULES:\n"
                            "1. **EXPLORATION BEFORE ASSUMPTION**: Never search for specific entities without first sampling\n"
                            "2. **EVIDENCE-BASED QUERIES**: Only use entities confirmed through discovery queries\n"
                            "3. **ITERATIVE REFINEMENT**: Use exploration results to guide each subsequent query\n"
                            "4. **NO EMPTY RESPONSES**: Always generate valid Cypher queries, never return empty results\n\n"
                            "QUERY GENERATION RULES:\n"
                            "- ALWAYS generate valid Cypher queries, never return empty responses\n"
                            "- Generate exactly ONE Cypher query with ONE RETURN statement only\n"
                            "- Use UNION ALL carefully: All subqueries must have identical column names and count\n"
                            "- When using UNION ALL, ensure all parts return the same column structure\n"
                            "- Prefer single focused queries over complex UNION ALL combinations\n"
                            "- Use CONTAINS, STARTS WITH, or regex for flexible text matching\n"
                            "- When uncertain about entities, start with exploratory queries\n"
                            "- Include graph_id filters when available for scoped queries\n"
                            "- Return Cypher query only - no explanations or markdown formatting\n"
                        ),
                    ),
                    (
                        "human",
                        (
                            """Based on the knowledge graph schema below, generate a Cypher query using systematic exploration principles.

SCHEMA INFORMATION (structural hints only - real entities may vary):
{schema}

SYSTEMATIC QUERY APPROACH:

EXPLORATION STRATEGIES:
- For entity discovery: MATCH (n:NodeType) WHERE n.graph_id = X RETURN DISTINCT n.property LIMIT 10
- For relationship patterns: MATCH (a)-[r]-(b) WHERE a.graph_id = X RETURN type(r), labels(a), labels(b) LIMIT 10  
- For value sampling: MATCH (n:NodeType) WHERE n.graph_id = X AND n.property IS NOT NULL RETURN n.property LIMIT 20

PROGRESSIVE QUERYING:
Phase 1 - BROAD EXPLORATION: Discover what entity types and values actually exist
Phase 2 - PATTERN DISCOVERY: Understand relationship structures and data patterns
Phase 3 - TARGETED SEARCH: Query specific information based on discoveries
Phase 4 - FLEXIBLE MATCHING: Use partial matching when exact matches fail

EXPLORATION-FIRST GUIDELINES:
For questions about specific entities (people, places, events):
✅ CORRECT Approach:
1. First discover what entity types exist: MATCH (n) WHERE n.graph_id = X RETURN DISTINCT labels(n)
2. Then sample actual values: MATCH (n:EntityType) WHERE n.graph_id = X RETURN DISTINCT n.name LIMIT 20
3. Finally target based on discoveries: Use confirmed entities for specific queries

❌ WRONG Approach:
- Directly searching for assumed entities without discovery
- Assuming specific names/values exist based only on schema

FLEXIBLE MATCHING TECHNIQUES:
- Text searches: WHERE n.name CONTAINS 'keyword' OR n.name =~ '(?i).*keyword.*'
- Multiple variations: WHERE n.name CONTAINS 'term1' OR n.name CONTAINS 'term2' 
- Pattern exploration: WHERE n.property IS NOT NULL AND n.property <> ''
- Relationship discovery: MATCH (a)-[r]-(b) WHERE ... RETURN DISTINCT type(r)

EXAMPLE QUERY PATTERNS:

For entity exploration:
```cypher
MATCH (n) WHERE n.graph_id = [TARGET_GRAPH_ID] 
RETURN DISTINCT labels(n) as entity_types, COUNT(n) as count 
ORDER BY count DESC LIMIT 10
```

For specific entity discovery:
```cypher  
MATCH (n:EntityType) WHERE n.graph_id = [TARGET_GRAPH_ID] 
AND (n.name CONTAINS 'search_term' OR n.property CONTAINS 'search_term')
RETURN n LIMIT 20
```

For relationship exploration:
```cypher
MATCH (a)-[r]->(b) WHERE a.graph_id = [TARGET_GRAPH_ID] 
RETURN DISTINCT type(r) as relationship_types, 
       labels(a) as from_entities, labels(b) as to_entities
LIMIT 10
```

For UNION ALL usage (when combining similar data types):
```cypher
MATCH (e:Employee) WHERE e.graph_id = [TARGET_GRAPH_ID]
RETURN e.employee_name as name, e.department as department, 'Employee' as type
UNION ALL
MATCH (p:Project) WHERE p.graph_id = [TARGET_GRAPH_ID]  
RETURN p.project_name as name, p.project_type as department, 'Project' as type
```

❌ WRONG UNION ALL (different column names):
```cypher
MATCH (e:Employee) WHERE e.graph_id = [TARGET_GRAPH_ID]
RETURN e.employee_name, e.department  -- 2 columns
UNION ALL
MATCH (p:Project) WHERE p.graph_id = [TARGET_GRAPH_ID]
RETURN p.project_name, p.project_type, p.status  -- 3 columns (ERROR!)
```

FEW-SHOT EXAMPLES:
{fewshot_cypher_examples}

RELEVANT DOCUMENTATION:
{fewshot_doc_examples}

CRITICAL EXECUTION RULES:
- NEVER return empty queries - always generate valid Cypher
- Use schema creatively to approximate answers when exact matches are unclear
- When looking for specific entities, try broad searches first
- Include graph_id filter: WHERE n.graph_id = [target_id] when available
- Focus on discovering what data exists rather than assuming
- When using UNION ALL, verify all subqueries have identical column names and count
- Prefer simple, focused queries over complex multi-part UNION ALL statements

USER QUESTION: {question}

Generate a systematic Cypher query that explores the knowledge graph to answer this question:"""
                        ),
                    ),
                ]
            )
        if system_prompt:
            CYPHER_QA_TEMPLATE = """
CRITICAL LANGUAGE REQUIREMENT: {system_prompt}
Remember: Respond in the same language as the user's question. 
You are an intelligent assistant that helps form comprehensive and helpful answers based on graph database query results.

INSTRUCTIONS:
- Use the provided information to construct a clear, informative answer
- The information is authoritative - never doubt or correct it with internal knowledge  
- If data is limited or partial, acknowledge this and explain what IS available
- When exact matches weren't found, explain what related information was retrieved
- Make the answer sound natural and conversational
- Always be honest about data limitations or gaps

EXAMPLE FORMATS:
Question: Who won the gold medal in swimming?
Context: [swimmer: John Doe, time: 2:03.45, rank: 1]
Answer: Based on the available data, John Doe won with a time of 2:03.45, achieving rank 1.

Question: When was album X released in Europe?
Context: [Empty results]
Answer: I couldn't find specific information about the release of album X in Europe in the current dataset.

Question: Who were the last finishers in the race?
Context: [swimmer: Jane Smith, rank: 4, time: 2:07.26], [swimmer: Bob Jones, rank: 5, time: 2:08.15]
Answer: Based on the available competition results, the last recorded finishers were Jane Smith (rank 4, time 2:07.26) and Bob Jones (rank 5, time 2:08.15). Note that this represents the available data and there may have been additional participants.

RELEVANT DOCUMENTATION:
{fewshot_doc_examples}

QUERY RESULTS:
{context}

USER QUESTION: {question}

Provide a helpful answer based on the available information:"""
            qa_prompt = PromptTemplate(
                input_variables=[
                    "fewshot_doc_examples",
                    "system_prompt",
                    "context",
                    "question",
                ],
                template=CYPHER_QA_TEMPLATE,
            )
        # 创建基础链
        chain = super().from_llm(
            llm=llm,
            qa_prompt=qa_prompt,
            cypher_prompt=cypher_prompt,
            cypher_llm=cypher_llm,
            qa_llm=qa_llm,
            exclude_types=exclude_types,
            include_types=include_types,
            validate_cypher=validate_cypher,
            qa_llm_kwargs=qa_llm_kwargs,
            cypher_llm_kwargs=cypher_llm_kwargs,
            use_function_response=use_function_response,
            **kwargs,
        )

        # 将few-shot功能添加到链中
        chain.cypher_example_selector = cypher_example_selector
        chain.doc_example_selector = doc_example_selector
        chain.embeddings = embeddings
        chain.cypher_vector_store = cypher_vector_store
        chain.doc_vector_store = doc_vector_store
        chain.persist_directory = persist_dir
        chain.num_cypher_examples = num_cypher_examples
        chain.num_doc_examples = num_doc_examples
        chain.autotrain_enabled = autotrain_enabled
        chain.system_prompt = system_prompt
        if target_graph_id_prop:
            chain.target_graph_id_prop = target_graph_id_prop
            chain.graph_schema = chain._get_graph_schema(
                target_graph_id_prop, include_types, exclude_types
            )
            logger.info(f"图谱Schema更新: {chain.graph_schema}")
        return chain

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """
        使用few-shot学习和文档检索生成Cypher语句，查询数据库并回答问题
        """
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        callbacks = _run_manager.get_child()
        question = inputs[self.input_key]
        normal_question = inputs.get("normal_query", question)
        # 准备参数
        args = {
            "question": question,
            "schema": self.graph_schema,
        }
        args.update(inputs)

        # 确保graph_id变量可用（用于提示模板中的引用）
        if hasattr(self, "target_graph_id_prop") and self.target_graph_id_prop:
            args["graph_id"] = self.target_graph_id_prop
        elif "target_graph_id" in args:
            args["graph_id"] = args["target_graph_id"]
        else:
            args["graph_id"] = "861"  # 默认值

        # 如果有示例选择器，添加few-shot示例
        if self.cypher_example_selector:
            NL = "\n"
            fewshot_examples = (NL * 2).join(
                [
                    f"Question: {el['question']}{NL}Cypher:{el['query']}"
                    for el in self.cypher_example_selector.select_examples(
                        {"question": question}
                    )
                ]
            )
            args["fewshot_cypher_examples"] = fewshot_examples
        else:
            args["fewshot_cypher_examples"] = ""

        if self.doc_example_selector:
            NL = "\n"
            fewshot_examples = (NL * 2).join(
                [
                    f"Document:{el['documentation']}"
                    for el in self.doc_example_selector.select_examples(
                        {"question": question}
                    )
                ]
            )
            args["fewshot_doc_examples"] = fewshot_examples
        else:
            args["fewshot_doc_examples"] = ""
        # 其余流程与原始链相同
        intermediate_steps: List = []

        generated_cypher = self.cypher_generation_chain.invoke(
            args, callbacks=callbacks
        )

        # 提取Cypher代码
        generated_cypher = getattr(generated_cypher, "content", generated_cypher)

        # 从字符串中提取Cypher代码（如果被包裹在引号或代码块内）
        generated_cypher = extract_cypher(generated_cypher)

        # **新增：如果生成的Cypher为空，尝试生成后备查询**
        if not generated_cypher or generated_cypher.strip() == "":
            _run_manager.on_text(
                "Primary Cypher generation failed, attempting fallback query...",
                end="\n",
                verbose=self.verbose,
            )

            # 生成一个更简单的后备查询来获取相关数据
            fallback_args = args.copy()
            fallback_question = f"""
Based on the available schema, generate a broad query to retrieve any relevant data that might help answer: {question}
Focus on finding entities, relationships, or properties that could be related to the question.
Use partial matching and don't worry about exact specificity.
Generate a simple MATCH query that returns potentially relevant nodes and relationships.
"""
            fallback_args["question"] = fallback_question

            try:
                fallback_cypher = self.cypher_generation_chain.invoke(
                    fallback_args, callbacks=callbacks
                )
                fallback_cypher = getattr(fallback_cypher, "content", fallback_cypher)
                fallback_cypher = extract_cypher(fallback_cypher)

                if fallback_cypher and fallback_cypher.strip():
                    generated_cypher = fallback_cypher
                    _run_manager.on_text(
                        "Using fallback query:", end="\n", verbose=self.verbose
                    )
                else:
                    # 最后的后备方案：生成一个基本的探索性查询
                    target_id = getattr(self, "target_graph_id_prop", None) or args.get(
                        "target_graph_id", "861"
                    )
                    generated_cypher = f"""
MATCH (n)
WHERE n.graph_id = {target_id}
RETURN n, labels(n) as node_type
LIMIT 20
"""
                    _run_manager.on_text(
                        "Using basic exploration query as last resort:",
                        end="\n",
                        verbose=self.verbose,
                    )
            except Exception as e:
                _run_manager.on_text(
                    f"Fallback generation failed: {str(e)}",
                    end="\n",
                    verbose=self.verbose,
                )
                # 最基本的查询作为最后手段
                target_id = getattr(self, "target_graph_id_prop", None) or args.get(
                    "target_graph_id", "861"
                )
                generated_cypher = f"""
MATCH (n)
WHERE n.graph_id = {target_id}
RETURN n
LIMIT 10
"""

        # 校正Cypher查询（如果启用）
        if self.cypher_query_corrector:
            original_cypher = generated_cypher
            generated_cypher = self.cypher_query_corrector(generated_cypher)
            logger.info(
                f"Cypher corrector: Original='{original_cypher}' -> Corrected='{generated_cypher}'"
            )
            if not generated_cypher and original_cypher:
                logger.warning(
                    f"Cypher corrector returned empty for valid query: {original_cypher}"
                )
                # 可选：回退到原始查询
                generated_cypher = original_cypher

        _run_manager.on_text("Generated Cypher:", end="\n", verbose=self.verbose)
        _run_manager.on_text(
            generated_cypher, color="green", end="\n", verbose=self.verbose
        )

        intermediate_steps.append({"query": generated_cypher})

        # 检索并限制结果数量
        if generated_cypher:
            context = self.graph.query(generated_cypher)[: self.top_k]
        else:
            context = []

        final_result: Union[List[Dict[str, Any]], str]
        if self.return_direct:
            final_result = context
        else:
            _run_manager.on_text("Full Context:", end="\n", verbose=self.verbose)
            _run_manager.on_text(
                str(context), color="green", end="\n", verbose=self.verbose
            )

            intermediate_steps.append({"context": context})
            if self.use_function_response:
                from langchain_neo4j.chains.graph_qa.cypher import get_function_response

                function_response = get_function_response(question, context)
                final_result = self.qa_chain.invoke(
                    {"question": question, "function_response": function_response},
                )
            else:
                if self.system_prompt:
                    final_result = self.qa_chain.invoke(
                        {
                            "question": question,
                            "context": context,
                            "system_prompt": self.system_prompt,
                            "fewshot_doc_examples": args["fewshot_doc_examples"]
                            if "fewshot_doc_examples" in args
                            else "",
                        },
                        callbacks=callbacks,
                    )
                else:
                    final_result = self.qa_chain.invoke(
                        {
                            "question": question,
                            "context": context,
                            "fewshot_doc_examples": args["fewshot_doc_examples"]
                            if "fewshot_doc_examples" in args
                            else "",
                        },
                        callbacks=callbacks,
                    )

        chain_result: Dict[str, Any] = {self.output_key: final_result}
        if self.return_intermediate_steps:
            chain_result["intermediate_steps"] = intermediate_steps

        # 如果启用了自动训练，保存此次查询作为示例
        if self.autotrain_enabled and generated_cypher and context:
            self.add_cypher_question(
                normal_question, generated_cypher, context, str(final_result)
            )

        return chain_result

    @staticmethod
    def _chunk_to_text(chunk: Any) -> str:
        if chunk is None:
            return ""
        if isinstance(chunk, str):
            return chunk
        if hasattr(chunk, "content") and isinstance(chunk.content, str):
            return chunk.content
        if isinstance(chunk, dict):
            for key in ("text", "content", "result"):
                value = chunk.get(key)
                if isinstance(value, str):
                    return value
        return str(chunk)

    def stream_answer(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Iterator[Dict[str, Any]]:
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        callbacks = _run_manager.get_child()
        question = inputs[self.input_key]
        normal_question = inputs.get("normal_query", question)

        args = {
            "question": question,
            "schema": self.graph_schema,
        }
        args.update(inputs)

        if hasattr(self, "target_graph_id_prop") and self.target_graph_id_prop:
            args["graph_id"] = self.target_graph_id_prop
        elif "target_graph_id" in args:
            args["graph_id"] = args["target_graph_id"]
        else:
            args["graph_id"] = "861"

        if self.cypher_example_selector:
            nl = "\n"
            fewshot_examples = (nl * 2).join(
                [
                    f"Question: {el['question']}{nl}Cypher:{el['query']}"
                    for el in self.cypher_example_selector.select_examples(
                        {"question": question}
                    )
                ]
            )
            args["fewshot_cypher_examples"] = fewshot_examples
        else:
            args["fewshot_cypher_examples"] = ""

        if self.doc_example_selector:
            nl = "\n"
            fewshot_examples = (nl * 2).join(
                [
                    f"Document:{el['documentation']}"
                    for el in self.doc_example_selector.select_examples(
                        {"question": question}
                    )
                ]
            )
            args["fewshot_doc_examples"] = fewshot_examples
        else:
            args["fewshot_doc_examples"] = ""

        intermediate_steps: List = []
        generated_cypher = self.cypher_generation_chain.invoke(
            args, callbacks=callbacks
        )
        generated_cypher = getattr(generated_cypher, "content", generated_cypher)
        generated_cypher = extract_cypher(generated_cypher)

        if not generated_cypher or generated_cypher.strip() == "":
            fallback_args = args.copy()
            fallback_question = f"""
Based on the available schema, generate a broad query to retrieve any relevant data that might help answer: {question}
Focus on finding entities, relationships, or properties that could be related to the question.
Use partial matching and don't worry about exact specificity.
Generate a simple MATCH query that returns potentially relevant nodes and relationships.
"""
            fallback_args["question"] = fallback_question
            try:
                fallback_cypher = self.cypher_generation_chain.invoke(
                    fallback_args, callbacks=callbacks
                )
                fallback_cypher = getattr(fallback_cypher, "content", fallback_cypher)
                fallback_cypher = extract_cypher(fallback_cypher)
                if fallback_cypher and fallback_cypher.strip():
                    generated_cypher = fallback_cypher
                else:
                    target_id = getattr(self, "target_graph_id_prop", None) or args.get(
                        "target_graph_id", "861"
                    )
                    generated_cypher = f"""
MATCH (n)
WHERE n.graph_id = {target_id}
RETURN n, labels(n) as node_type
LIMIT 20
"""
            except Exception:
                target_id = getattr(self, "target_graph_id_prop", None) or args.get(
                    "target_graph_id", "861"
                )
                generated_cypher = f"""
MATCH (n)
WHERE n.graph_id = {target_id}
RETURN n
LIMIT 10
"""

        if self.cypher_query_corrector:
            original_cypher = generated_cypher
            generated_cypher = self.cypher_query_corrector(generated_cypher)
            if not generated_cypher and original_cypher:
                generated_cypher = original_cypher

        intermediate_steps.append({"query": generated_cypher})

        if generated_cypher:
            context = self.graph.query(generated_cypher)[: self.top_k]
        else:
            context = []

        intermediate_steps.append({"context": context})
        yield {"event": "query", "cypher": generated_cypher, "context": context}

        if self.return_direct:
            final_result = context
            yield {
                "event": "final",
                "result": final_result,
                "intermediate_steps": intermediate_steps,
            }
            return

        token_parts: List[str] = []
        if self.use_function_response:
            from langchain_neo4j.chains.graph_qa.cypher import get_function_response

            function_response = get_function_response(question, context)
            stream_iter = self.qa_chain.stream(
                {"question": question, "function_response": function_response},
                callbacks=callbacks,
            )
        else:
            if self.system_prompt:
                qa_inputs = {
                    "question": question,
                    "context": context,
                    "system_prompt": self.system_prompt,
                    "fewshot_doc_examples": args.get("fewshot_doc_examples", ""),
                }
            else:
                qa_inputs = {
                    "question": question,
                    "context": context,
                    "fewshot_doc_examples": args.get("fewshot_doc_examples", ""),
                }
            stream_iter = self.qa_chain.stream(qa_inputs, callbacks=callbacks)

        for chunk in stream_iter:
            delta = self._chunk_to_text(chunk)
            if delta:
                token_parts.append(delta)
                yield {"event": "token", "delta": delta}

        final_result = "".join(token_parts).strip()
        if not final_result:
            if self.use_function_response:
                from langchain_neo4j.chains.graph_qa.cypher import get_function_response

                function_response = get_function_response(question, context)
                final_result = self.qa_chain.invoke(
                    {"question": question, "function_response": function_response}
                )
            else:
                if self.system_prompt:
                    final_result = self.qa_chain.invoke(
                        {
                            "question": question,
                            "context": context,
                            "system_prompt": self.system_prompt,
                            "fewshot_doc_examples": args.get(
                                "fewshot_doc_examples", ""
                            ),
                        },
                        callbacks=callbacks,
                    )
                else:
                    final_result = self.qa_chain.invoke(
                        {
                            "question": question,
                            "context": context,
                            "fewshot_doc_examples": args.get(
                                "fewshot_doc_examples", ""
                            ),
                        },
                        callbacks=callbacks,
                    )

        if self.autotrain_enabled and generated_cypher and context:
            self.add_cypher_question(
                normal_question, generated_cypher, context, final_result
            )

        yield {
            "event": "final",
            "result": final_result,
            "intermediate_steps": intermediate_steps,
        }

    def add_cypher_question(
        self,
        question: str,
        query: str,
        result: Optional[List[Dict[str, Any]]] = None,
        answer: Optional[str] = None,
    ) -> None:
        """
        添加一个新的Cypher查询示例到向量存储

        Args:
            question: 用户问题
            query: 对应的Cypher查询
            result: 可选的查询结果
            answer: 可选的生成答案
        """
        # 如果有向量存储，更新向量存储
        if self.cypher_vector_store and self.embeddings:
            self.cypher_vector_store.add_texts(
                texts=[question],
                metadatas=[{"question": question, "query": query}],
                ids=[f"example-{uuid.uuid4()}"],
            )

    def add_documentation(self, documentation: List[Dict[str, str]]) -> None:
        """
        添加文档到向量存储以增强检索能力

        Args:
            documentation: 文档列表，每个文档包含"text"和"metadata"
        """
        if not self.doc_vector_store or not self.embeddings:
            raise ValueError("向量存储和嵌入模型必须已初始化才能添加文档")

        texts = documentation

        self.doc_vector_store.add_texts(
            texts=[texts],
            metadatas=[{"documentation": texts}],
            ids=[f"doc-{uuid.uuid4()}" for _ in range(len(texts))],
        )

    def set_autotrain(self, enabled: bool = True) -> None:
        """
        启用或禁用自动训练

        Args:
            enabled: 是否启用自动训练
        """
        self.autotrain_enabled = enabled

    def get_training_data(self):  # 获取训练数据
        """获取训练数据"""
        df = pd.DataFrame()
        if self.cypher_vector_store:
            cypher_data = self.cypher_vector_store._collection.get()
            if cypher_data is not None and len(cypher_data["metadatas"]) > 0:
                documents = cypher_data["metadatas"]
                ids = cypher_data["ids"]
                df_cypher = pd.DataFrame(
                    {
                        "id": ids,
                        "question": [doc["question"] for doc in documents],
                        "content": [doc["query"] for doc in documents],
                    }
                )
                df_cypher["training_data_type"] = "cypher"
                df = pd.concat([df, df_cypher])
        if self.doc_vector_store:
            doc_data = self.doc_vector_store._collection.get()
            if doc_data is not None and len(doc_data["documents"]) > 0:
                documents = doc_data["documents"]
                ids = doc_data["ids"]
                df_doc = pd.DataFrame(
                    {
                        "id": ids,
                        "question": [None for doc in documents],
                        "content": [doc for doc in documents],
                    }
                )
                df_doc["training_data_type"] = "documentation"
                df = pd.concat([df, df_doc])
        return df

    def delete_training_data(self, id: str) -> None:
        """删除指定的训练数据"""
        if "example-" in id:
            if self.cypher_vector_store:
                self.cypher_vector_store.delete(ids=[id])
                return True
        elif "doc-" in id:
            if self.doc_vector_store:
                self.doc_vector_store.delete(ids=[id])
                return True
        return False
