import json
import logging
import pandas as pd
import numpy as np
import re
import os
from backend.services.vanna_service import vanna_manager
from backend.utils.token_tracking import create_token_callback
from backend.manager.langchain_graph_qa import GraphFewShotQAChain
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_neo4j import Neo4jGraph
from backend.utils.db_utils import (
    create_kgqa_chat_session,
    update_kgqa_chat_session,
    get_kgqa_chat_session,
    get_kgqa_chat_sessions,
    delete_kgqa_chat_session,
    get_kgqa_chat_messages,
    add_kgqa_chat_message,
    delete_kgqa_chat_message,
    save_kgqa_query,
    get_saved_kgqa_queries,
    get_saved_kgqa_query,
    delete_saved_kgqa_query,
)
from datetime import datetime
from neo4j.time import DateTime, Date, Time, Duration
from decimal import Decimal

logger = logging.getLogger(__name__)


# 添加辅助函数，用于处理NumPy和时间类型的JSON序列化
def convert_numpy_types(obj):
    """递归转换字典中的NumPy和时间类型为Python原生或字符串类型"""
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return convert_numpy_types(obj.tolist())
    # 处理 Neo4j 和 Python 的日期时间类型
    elif isinstance(obj, (DateTime, Date, Time, datetime)):
        try:
            # DateTime, Date, Time 有 to_iso_format 或 isoformat
            if hasattr(obj, "isoformat"):
                return obj.isoformat()
            elif hasattr(obj, "to_iso_format"):  # 兼容旧版neo4j驱动或不同方法名
                return obj.to_iso_format()
        except AttributeError:
            logger.warning(f"无法序列化日期/时间对象类型 {type(obj)}，将转换为字符串。")
            return str(obj)  # Fallback to string conversion
    elif isinstance(obj, Duration):
        # Duration 可以转换为总秒数或字符串表示
        return str(obj)  # Convert Duration to string representation
    elif pd.isna(obj):
        return None
    elif hasattr(obj, "item"):  # 处理 numpy scalar types
        return obj.item()
    elif isinstance(obj, Decimal):
        # 处理Decimal类型 - 转换为字符串或浮点数
        return float(obj)
    else:
        return obj


class KGQAManager:
    """数据库问答管理器"""

    def __init__(self):
        self.vanna_manager = vanna_manager
        self.graph_qa_chain = None
        self.neo4j_graph = None
        self.embeddings = None
        self.llm = None

    def _get_current_language(self):
        """获取当前语言设置"""
        try:
            config = self.vanna_manager.get_config()
            return config.get("language", {}).get("language", "zh-CN")
        except Exception as e:
            logger.warning(f"获取语言设置失败: {str(e)}，使用默认中文")
            return "zh-CN"

    def _get_localized_message(self, zh_message, en_message):
        """根据当前语言设置返回对应的消息"""
        language = self._get_current_language()
        if language == "en-US":
            return en_message
        else:
            return zh_message

    def init_graph_qa_chain(
        self, config=None, target_graph_id_prop=None, include_types=[], exclude_types=[]
    ):
        """初始化图数据库问答链"""
        if not config:
            config = self.vanna_manager.get_config()

        # 获取Neo4j连接信息
        neo4j_config = config.get("neo4j", {})
        neo4j_uri = neo4j_config.get("uri", "bolt://localhost:7687")
        neo4j_username = neo4j_config.get("username", "neo4j")
        neo4j_password = neo4j_config.get("password", "12345678")

        # 初始化Neo4j图数据库连接
        self.neo4j_graph = Neo4jGraph(
            url=neo4j_uri,
            username=neo4j_username,
            password=neo4j_password,
            enhanced_schema=True,
            refresh_schema=False,
        )
        self.neo4j_driver = self.neo4j_graph._driver
        # 获取LLM配置
        model_config = config.get("model", {})
        model_type = model_config.get("type", "ollama")
        temperature = model_config.get("temperature", 0.7)
        system_prompt = model_config.get("system_prompt", "")
        # 初始化嵌入模型
        embedding_ollama_url = config.get("store_database", {}).get(
            "embedding_ollama_url", "http://localhost:11434"
        )
        embedding_model = config.get("store_database", {}).get(
            "embedding_function", "bge-m3"
        )
        self.embeddings = OllamaEmbeddings(
            model=embedding_model, base_url=embedding_ollama_url
        )

        # 创建token跟踪callback
        token_callback = None
        try:
            token_callback = create_token_callback("knowledge_graph")
            logger.info("知识图谱token跟踪callback已创建")
        except Exception:
            logger.info("无法导入token跟踪callback，将跳过token跟踪")

        # 初始化LLM，并添加callback
        callbacks = [token_callback] if token_callback else []

        if model_type == "ollama":
            # 设置HTTP客户端参数，提升连接稳定性
            import httpx

            try:
                # 尝试启用HTTP/2，如果失败则回退到HTTP/1.1
                http_client = httpx.AsyncClient(
                    timeout=httpx.Timeout(
                        connect=10.0,  # 连接超时
                        read=60.0,  # 读取超时
                        write=10.0,  # 写入超时
                        pool=5.0,  # 连接池超时
                    ),
                    limits=httpx.Limits(
                        max_keepalive_connections=5,
                        max_connections=10,
                        keepalive_expiry=30.0,
                    ),
                    http2=True,  # 使用HTTP/2提高性能
                )
                logger.info("KGQAManager: HTTP/2 client created successfully")
            except ImportError as e:
                # 如果没有h2包，回退到HTTP/1.1
                logger.warning(
                    f"KGQAManager: HTTP/2 not available ({str(e)}), falling back to HTTP/1.1"
                )
                http_client = httpx.AsyncClient(
                    timeout=httpx.Timeout(
                        connect=10.0, read=60.0, write=10.0, pool=5.0
                    ),
                    limits=httpx.Limits(
                        max_keepalive_connections=5,
                        max_connections=10,
                        keepalive_expiry=30.0,
                    ),
                    # http2参数默认为False
                )

            self.llm = ChatOllama(
                base_url=model_config.get("ollama_url", "http://localhost:11434"),
                model=model_config.get("ollama_model", "gemma3:27b-it-q8_0"),
                temperature=temperature,
                extract_reasoning=True,
                num_ctx=model_config.get("num_ctx", 25600),
                client=http_client,  # 使用自定义HTTP客户端
                request_timeout=60.0,  # 总体请求超时
                num_retries=2,  # 添加自动重试
                callbacks=callbacks,
            )
        elif model_type == "openai":
            # 如果需要其他类型的LLM，可以在这里添加
            from langchain_openai import ChatOpenAI

            openai_model = model_config.get("model_name", "gpt-3.5-turbo")
            self.llm = ChatOpenAI(
                model=openai_model,
                temperature=temperature,
                api_key=model_config.get("api_key"),
                base_url=model_config.get("api_base", "https://api.openai.com/v1"),
                max_tokens=model_config.get("num_ctx", 25600),
                request_timeout=60.0,  # 请求超时
                max_retries=2,
                callbacks=callbacks,
            )
            logger.info(f"初始化OpenAI模型: {openai_model}")
        else:
            logger.warning(f"不支持的模型类型: {model_type}，使用默认的Ollama模型")
            # 为fallback情况也设置HTTP客户端
            import httpx

            try:
                http_client = httpx.AsyncClient(
                    timeout=httpx.Timeout(
                        connect=10.0, read=60.0, write=10.0, pool=5.0
                    ),
                    limits=httpx.Limits(
                        max_keepalive_connections=5,
                        max_connections=10,
                        keepalive_expiry=30.0,
                    ),
                    http2=True,
                )
            except ImportError:
                http_client = httpx.AsyncClient(
                    timeout=httpx.Timeout(
                        connect=10.0, read=60.0, write=10.0, pool=5.0
                    ),
                    limits=httpx.Limits(
                        max_keepalive_connections=5,
                        max_connections=10,
                        keepalive_expiry=30.0,
                    ),
                )

            self.llm = ChatOllama(
                base_url="http://localhost:11434",
                model="gemma3:27b-it-q8_0",
                extract_reasoning=True,
                client=http_client,
                request_timeout=60.0,
                num_retries=2,
                callbacks=callbacks,
            )
        # 初始化GraphFewShotQAChain

        chroma_path = config.get("store_database", {}).get("path", "./chroma_db")
        chroma_path_kgqa = os.path.join(chroma_path, "kgqa")
        os.makedirs(chroma_path_kgqa, exist_ok=True)
        # 创建问答链
        self.graph_qa_chain = GraphFewShotQAChain.from_llm(
            llm=self.llm,
            graph=self.neo4j_graph,
            embeddings=self.embeddings,
            persist_directory=chroma_path_kgqa,
            num_cypher_examples=5,
            num_doc_examples=3,
            validate_cypher=False,  # 禁用验证以避免corrector返回空查询
            autotrain_enabled=True,
            top_k=100,  # 增加返回结果数量上限
            return_intermediate_steps=True,
            allow_dangerous_requests=True,  # 注意：生产环境中应谨慎使用
            system_prompt=system_prompt,
            verbose=True,
            target_graph_id_prop=target_graph_id_prop,
            include_types=include_types,
            exclude_types=exclude_types,
        )

        # 自动创建graph_id索引以提高查询性能
        logger.info("检查并创建graph_id索引以提高性能...")
        try:
            created_indexes = self.graph_qa_chain.create_graph_id_indexes()
            if created_indexes:
                logger.info(f"成功为以下标签创建了graph_id索引: {created_indexes}")
            else:
                logger.info("所有必要的graph_id索引都已存在或创建完成")
        except Exception as e:
            logger.warning(
                f"创建graph_id索引时出现问题: {str(e)}，但不影响功能正常使用"
            )

        return self.graph_qa_chain

    def create_session(
        self,
        name=None,
        database_name=None,
        table_name=None,
        model_name=None,
        temperature=0.7,
    ):
        """创建新的聊天会话"""
        try:
            # 如果没有提供名称，使用默认名称
            if not name:
                name = f"新对话 {datetime.now().strftime('%Y-%m-%d %H:%M')}"

            session_id = create_kgqa_chat_session(
                name=name,
                database_name=database_name,
                table_name=table_name,
                model_name=model_name,
                temperature=temperature,
            )

            return session_id
        except Exception as e:
            logger.error(f"创建聊天会话失败: {str(e)}")
            raise

    def get_session(self, session_id):
        """获取会话信息"""
        return get_kgqa_chat_session(session_id)

    def get_all_sessions(self):
        """获取所有会话"""
        return get_kgqa_chat_sessions()

    def update_session(self, session_id, **kwargs):
        """更新会话信息"""
        return update_kgqa_chat_session(session_id, **kwargs)

    def delete_session(self, session_id):
        """删除会话"""
        return delete_kgqa_chat_session(session_id)

    def get_messages(self, session_id):
        """获取会话的所有消息"""
        return get_kgqa_chat_messages(session_id)

    def add_user_message(self, session_id, content):
        """添加用户消息"""
        return add_kgqa_chat_message(session_id, "user", content)

    def add_assistant_message(
        self,
        session_id,
        content,
        cypher=None,
        result=None,
        visualization=None,
        reasoning=None,
        thinking=None,
    ):
        """添加助手消息"""
        return add_kgqa_chat_message(
            session_id=session_id,
            role="assistant",
            content=content,
            cypher=cypher,
            result=result,
            visualization=visualization,
            reasoning=reasoning,
            thinking=thinking,
        )

    def delete_message(self, message_id):
        """删除消息"""
        return delete_kgqa_chat_message(message_id)

    def generate_session_name(self, session_id, first_message=None):
        """为会话生成名称

        Args:
            session_id: 会话ID
            first_message: 用户的第一条消息，如果为None则从数据库获取

        Returns:
            生成的会话名称，如果生成失败则返回默认名称
        """
        try:
            logger.info(
                f"尝试为会话 {session_id} 生成名称，首条消息: {first_message and first_message[:30]}..."
            )

            # 获取配置
            config = self.vanna_manager.get_config()
            naming_model_config = config.get("naming_model", {})

            # 检查是否启用了自动命名
            if not naming_model_config.get("enabled", False):
                logger.info(f"会话 {session_id} 的自动命名未启用")
                return None

            # 获取会话的第一条用户消息
            if not first_message:
                messages = get_kgqa_chat_messages(session_id)
                user_messages = [msg for msg in messages if msg["role"] == "user"]
                if not user_messages:
                    logger.warning(f"会话 {session_id} 中没有用户消息，无法生成名称")
                    return None
                first_message = user_messages[0]["content"]

            # 获取提示词模板并填充
            prompt_template = naming_model_config.get(
                "prompt_template",
                "根据以下对话内容，为这个对话生成一个简短的标题（不超过20个字符）：{conversation}",
            )
            prompt = prompt_template.replace("{conversation}", first_message)

            logger.info(f"会话 {session_id} 命名提示词: {prompt[:50]}...")

            # 判断是否使用系统模型
            use_system_model = naming_model_config.get("use_system_model", True)
            logger.info(f"会话 {session_id} 使用系统模型: {use_system_model}")

            if use_system_model:
                # 使用系统当前配置的大模型生成名称
                if self.vanna_manager.vn is None:
                    logger.error(f"会话 {session_id} 命名失败: Vanna未初始化")
                    return None

                # 根据系统模型类型调用不同的API
                model_type = config.get("model", {}).get("type", "vanna")
                logger.info(f"会话 {session_id} 使用系统模型类型: {model_type}")

                session_name = None

                if model_type == "ollama":
                    try:
                        # 使用Ollama生成名称
                        ollama_url = config.get("model", {}).get(
                            "ollama_url", "http://localhost:11434"
                        )
                        ollama_model = config.get("model", {}).get(
                            "ollama_model", "llama2"
                        )

                        logger.info(
                            f"会话 {session_id} 使用Ollama模型 {ollama_model} 生成名称"
                        )

                        import requests

                        response = requests.post(
                            f"{ollama_url}/api/generate",
                            json={
                                "model": ollama_model,
                                "prompt": prompt,
                                "stream": False,
                            },
                            timeout=60,
                        )
                        if response.status_code == 200:
                            session_name = response.json().get("response", "").strip()
                            logger.info(
                                f"会话 {session_id} Ollama生成的名称: {session_name}"
                            )
                        else:
                            logger.error(
                                f"会话 {session_id} Ollama请求失败: {response.status_code} {response.text}"
                            )
                    except Exception as e:
                        logger.error(f"使用Ollama生成会话名称失败: {str(e)}")

                elif model_type == "openai":
                    try:
                        from openai import OpenAI

                        openai_client = OpenAI(
                            api_key=config.get("model", {}).get("api_key", ""),
                            base_url=config.get("model", {}).get(
                                "api_base", "https://api.openai.com/v1"
                            ),
                        )

                        logger.info(f"会话 {session_id} 使用OpenAI生成名称")

                        response = openai_client.chat.completions.create(
                            model=config.get("model", {}).get(
                                "model_name", "gpt-3.5-turbo"
                            ),
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0.7,
                            max_tokens=30,
                        )
                        session_name = (
                            response.choices[0].message.content or ""
                        ).strip()
                        logger.info(
                            f"会话 {session_id} OpenAI生成的名称: {session_name}"
                        )
                    except Exception as e:
                        logger.error(f"使用OpenAI生成会话名称失败: {str(e)}")

                else:
                    # 对于其他模型类型，可以使用Vanna的通用接口
                    try:
                        # 使用Vanna的ask方法，但只提取文本回答
                        logger.info(f"会话 {session_id} 使用Vanna生成名称")
                        response = self.vanna_manager.vn.ask(
                            question=prompt,
                            auto_train=False,
                            visualize=False,
                            print_results=False,
                        )
                        # 提取文本回答
                        if (
                            response
                            and isinstance(response, tuple)
                            and len(response) > 1
                        ):
                            # 如果返回的是DataFrame，尝试获取第一个值
                            df = response[1]
                            if df is not None and not df.empty:
                                session_name = str(df.iloc[0, 0])
                                logger.info(
                                    f"会话 {session_id} Vanna生成的名称: {session_name}"
                                )
                            else:
                                logger.warning(
                                    f"会话 {session_id} Vanna返回的DataFrame为空"
                                )
                        else:
                            logger.warning(
                                f"会话 {session_id} Vanna返回格式不正确: {response}"
                            )
                    except Exception as e:
                        logger.error(f"使用Vanna生成会话名称失败: {str(e)}")

            else:
                # 使用单独配置的命名模型
                model_type = naming_model_config.get("model_type", "ollama")
                logger.info(f"会话 {session_id} 使用独立命名模型类型: {model_type}")

                session_name = None

                if model_type == "ollama":
                    try:
                        # 使用Ollama生成名称
                        ollama_url = naming_model_config.get(
                            "ollama_url", "http://localhost:11434"
                        )
                        ollama_model = naming_model_config.get("ollama_model", "llama2")

                        logger.info(
                            f"会话 {session_id} 使用独立Ollama模型 {ollama_model} 生成名称"
                        )

                        import requests

                        response = requests.post(
                            f"{ollama_url}/api/generate",
                            json={
                                "model": ollama_model,
                                "prompt": prompt,
                                "stream": False,
                            },
                            timeout=60,
                        )
                        if response.status_code == 200:
                            session_name = response.json().get("response", "").strip()
                            logger.info(
                                f"会话 {session_id} 独立Ollama生成的名称: {session_name}"
                            )
                        else:
                            logger.error(
                                f"会话 {session_id} 独立Ollama请求失败: {response.status_code} {response.text}"
                            )
                    except Exception as e:
                        logger.error(f"使用命名配置的Ollama生成会话名称失败: {str(e)}")

                elif model_type == "openai":
                    try:
                        from openai import OpenAI

                        openai_client = OpenAI(
                            api_key=naming_model_config.get("openai_api_key", ""),
                            base_url=naming_model_config.get(
                                "openai_api_base", "https://api.openai.com/v1"
                            ),
                        )

                        logger.info(f"会话 {session_id} 使用独立OpenAI生成名称")

                        response = openai_client.chat.completions.create(
                            model=naming_model_config.get(
                                "openai_model", "gpt-3.5-turbo"
                            ),
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0.7,
                            max_tokens=30,
                        )
                        session_name = (
                            response.choices[0].message.content or ""
                        ).strip()
                        logger.info(
                            f"会话 {session_id} 独立OpenAI生成的名称: {session_name}"
                        )
                    except Exception as e:
                        logger.error(f"使用命名配置的OpenAI生成会话名称失败: {str(e)}")

                elif model_type == "vanna":
                    try:
                        # 使用临时Vanna实例生成名称
                        from vanna.remote import VannaDefault

                        vanna_api_key = naming_model_config.get("vanna_api_key", "")
                        vanna_model_name = naming_model_config.get(
                            "vanna_model_name", ""
                        )

                        logger.info(f"会话 {session_id} 使用独立Vanna模型生成名称")

                        temp_vn = VannaDefault(
                            model=vanna_model_name, api_key=vanna_api_key
                        )

                        # 使用Vanna的ask方法直接查询
                        response = temp_vn.ask(
                            question=prompt,
                            auto_train=False,
                            visualize=False,
                            print_results=False,
                        )
                        if (
                            response
                            and isinstance(response, tuple)
                            and len(response) > 1
                        ):
                            df = response[1]
                            if df is not None and not df.empty:
                                session_name = str(df.iloc[0, 0])
                                logger.info(
                                    f"会话 {session_id} 独立Vanna生成的名称: {session_name}"
                                )
                            else:
                                logger.warning(
                                    f"会话 {session_id} 独立Vanna返回的DataFrame为空"
                                )
                        else:
                            logger.warning(
                                f"会话 {session_id} 独立Vanna返回格式不正确: {response}"
                            )
                    except Exception as e:
                        logger.error(f"使用命名配置的Vanna生成会话名称失败: {str(e)}")

            # 如果成功生成名称，更新会话
            if session_name:
                # 首先清理<think>标签及其内容
                session_name = re.sub(
                    r"<think>.*?</think>", "", session_name, flags=re.DOTALL
                ).strip()

                # 限制长度
                if len(session_name) > 50:
                    session_name = session_name[:47] + "..."

                # 移除可能的引号（模型可能返回带引号的文本）
                session_name = session_name.strip("\"'")

                logger.info(f"会话 {session_id} 更新名称为: {session_name}")

                # 更新会话名称
                update_kgqa_chat_session(session_id, name=session_name)
                return session_name
            else:
                logger.warning(f"会话 {session_id} 未能生成有效的名称")

            return None
        except Exception as e:
            logger.error(f"生成会话名称失败: {str(e)}")
            return None

    def ask_question(self, session_id, question, database_name=None, table_name=None):
        """提问并获取回答"""
        try:
            # 获取会话信息
            session = get_kgqa_chat_session(session_id)
            if not session:
                raise ValueError(f"会话ID {session_id} 不存在")

            # 检查会话是否已锁定数据表
            is_table_locked = session.get("is_table_locked", False)

            # 用新传入的数据库和表名更新会话信息(如果提供了且会话未锁定)
            if (database_name or table_name) and not is_table_locked:
                update_data = {}
                if database_name:
                    update_data["database_name"] = database_name
                if table_name:
                    update_data["table_name"] = table_name

                # 如果设置了数据库或表名，自动锁定
                if update_data:
                    # 判断是否需要锁定：如果有数据库名和表名
                    current_db = session.get("database_name")
                    current_table = session.get("table_name")

                    # 如果更新后将同时存在数据库名和表名，则锁定
                    db_to_use = database_name or current_db
                    table_to_use = table_name or current_table

                    if db_to_use and table_to_use:
                        update_data["is_table_locked"] = True

                if update_data:
                    update_kgqa_chat_session(session_id, **update_data)
                    # 重新获取更新后的会话信息
                    session = get_kgqa_chat_session(session_id)

            # 添加用户消息
            message_id = self.add_user_message(session_id, question)

            messages = get_kgqa_chat_messages(session_id)

            if len(messages) == 1:  # 只有一条消息说明是第一次问答
                logger.info(f"会话 {session_id} 的第一条消息，将尝试生成会话名称")
                # 直接调用命名方法（而不是使用线程）以便于调试
                try:
                    new_name = self.generate_session_name(session_id, question)
                    if new_name:
                        logger.info(f"会话 {session_id} 自动命名成功: {new_name}")
                    else:
                        logger.warning(f"会话 {session_id} 自动命名失败")
                except Exception as e:
                    logger.error(f"会话 {session_id} 自动命名异常: {str(e)}")

            # 设置温度
            temperature = session.get("temperature", 0.7)

            # 获取当前系统配置
            current_config = self.vanna_manager.get_config()

            logger.info(f"会话 {session_id} 的当前系统配置为: {current_config}")
            # 确保GraphQAChain已初始化
            if self.graph_qa_chain is None:
                self.init_graph_qa_chain(current_config)
                logger.info(f"会话 {session_id} 的图谱QA链已初始化")
            # 检查会话中是否有设置模型名称
            model_name = session.get("model_name")
            if model_name:
                # 重新初始化LLM
                model_type = current_config.get("model", {}).get("type", "ollama")

                # 根据模型类型更新LLM
                if model_type == "ollama":
                    # 为会话特定的模型也设置HTTP客户端
                    import httpx

                    try:
                        http_client = httpx.AsyncClient(
                            timeout=httpx.Timeout(
                                connect=10.0, read=60.0, write=10.0, pool=5.0
                            ),
                            limits=httpx.Limits(
                                max_keepalive_connections=5,
                                max_connections=10,
                                keepalive_expiry=30.0,
                            ),
                            http2=True,
                        )
                    except ImportError:
                        http_client = httpx.AsyncClient(
                            timeout=httpx.Timeout(
                                connect=10.0, read=60.0, write=10.0, pool=5.0
                            ),
                            limits=httpx.Limits(
                                max_keepalive_connections=5,
                                max_connections=10,
                                keepalive_expiry=30.0,
                            ),
                        )

                    self.llm = ChatOllama(
                        base_url=current_config.get("model", {}).get(
                            "ollama_url", "http://localhost:11434"
                        ),
                        model=model_name,
                        temperature=temperature,
                        extract_reasoning=True,
                        num_ctx=current_config.get("model", {}).get("num_ctx", 25600),
                        client=http_client,
                        request_timeout=60.0,
                        num_retries=2,
                    )
                elif model_type == "openai":
                    from langchain_openai import ChatOpenAI

                    self.llm = ChatOpenAI(
                        model=model_name,
                        temperature=temperature,
                        api_key=current_config.get("model", {}).get("api_key"),
                        base_url=current_config.get("model", {}).get(
                            "api_base", "https://api.openai.com/v1"
                        ),
                        max_tokens=current_config.get("model", {}).get(
                            "num_ctx", 25600
                        ),
                    )
                    # 更新graph_qa_chain
                    self.graph_qa_chain.cypher_generation_chain.bind(llm=self.llm)
                    self.graph_qa_chain.qa_chain.bind(llm=self.llm)

            # 如果指定了表名，为问题添加上下文
            context = ""
            table_to_query = session.get("table_name")
            if table_to_query:
                graph_id, graph_name = table_to_query.split(":")
                target_graph_id_prop = graph_id
                logger.info(f"会话 {session_id} 的图谱ID为: {target_graph_id_prop}")
                self.graph_qa_chain.graph_schema = (
                    self.graph_qa_chain._get_graph_schema(target_graph_id_prop)
                )
                logger.info(
                    f"会话 {session_id} 的图谱Schema为: {self.graph_qa_chain.graph_schema}"
                )
                context = f"Query the knowledge graph {graph_name} with ID {graph_id}, corresponding to the Cypher query MATCH(n) WHERE n.graph_id = {graph_id} RETURN n. Please ensure to merge the WHERE condition in this Cypher query."
                enhanced_question = f"{context}{question}"
            else:
                enhanced_question = question

            # 获取会话的 auto_train 设置，默认为 True
            auto_train_setting = session.get("auto_train", True)
            logger.info(f"会话 {session_id} 的 auto_train 设置为: {auto_train_setting}")

            # 设置GraphQAChain的autotrain
            self.graph_qa_chain.set_autotrain(auto_train_setting)

            # 使用GraphFewShotQAChain进行问答
            try:
                response = self.graph_qa_chain.invoke({"query": enhanced_question})

                # 解析返回结果
                cypher = None
                result_data = None
                visualization = None
                reasoning = None
                thinking = None
                content = response.get("result", "")

                # 提取中间结果
                if "intermediate_steps" in response:
                    for step in response.get("intermediate_steps", []):
                        if "query" in step:
                            cypher = step["query"]
                        if "context" in step:
                            context_data = step["context"]
                            if (
                                context_data
                                and context_data
                                != "I couldn't find any relevant information in the database"
                            ):
                                # 不再转换为DataFrame，直接使用原始图数据
                                # 使用更新后的函数处理潜在的特殊类型
                                try:
                                    result_data = convert_numpy_types(context_data)
                                    logger.info(
                                        f"原始图数据结构类型: {type(result_data)}"
                                    )
                                    if isinstance(result_data, list) and result_data:
                                        # 检查第一个元素，现在应该是序列化后的
                                        logger.info(
                                            f"第一个元素类型: {type(result_data[0])}"
                                        )
                                    # 现在 result_data 应该是 JSON 可序列化的
                                    # 可以选择性地再次检查，但理论上不再需要
                                    # json.dumps(result_data) # 可选的验证步骤
                                except Exception as conv_err:
                                    logger.error(
                                        f"使用 convert_numpy_types 处理图数据时出错: {conv_err}"
                                    )
                                    result_data = {
                                        "error": "Error processing graph data after conversion",
                                        "details": str(conv_err),
                                    }

                # 尝试从content中提取思考过程
                if content:
                    # 检查是否包含<think>标签
                    think_match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
                    if think_match:
                        thinking = think_match.group(1).strip()
                        # 从content中移除思考过程
                        clean_content = re.sub(
                            r"<think>.*?</think>", "", content, flags=re.DOTALL
                        ).strip()
                        if clean_content:
                            content = clean_content

                # 构建可视化所需的知识图谱子图数据
                graph_data = self._build_visualization_subgraph(
                    session_id, cypher, result_data
                )
                graph_data_converted = convert_numpy_types(graph_data)
                # 添加助手消息
                self.add_assistant_message(
                    session_id=session_id,
                    content=content,
                    cypher=cypher,
                    result=result_data,
                    visualization=graph_data_converted
                    if not self.graph_data_is_empty(graph_data_converted)
                    else None,
                    reasoning=reasoning,
                    thinking=thinking,
                )

                return {
                    "status": "success",
                    "cypher": cypher,
                    "result": result_data,
                    "visualization": graph_data_converted
                    if not self.graph_data_is_empty(graph_data_converted)
                    else None,
                    "content": content,
                    "reasoning": reasoning,
                    "explanation": content,
                    "thinking": thinking,
                }

            except Exception as e:
                logger.error(f"GraphQAChain处理问题失败: {str(e)}")
                error_msg = f"处理问题时出错: {str(e)}"
                self.add_assistant_message(session_id=session_id, content=error_msg)
                return {"status": "error", "message": error_msg}

        except Exception as e:
            logger.error(f"问答处理失败: {str(e)}")
            return {"status": "error", "message": str(e)}

    def ask_question_stream(
        self, session_id, question, database_name=None, table_name=None
    ):
        try:
            session = get_kgqa_chat_session(session_id)
            if not session:
                yield {"type": "error", "message": f"会话ID {session_id} 不存在"}
                yield {"type": "end"}
                return

            is_table_locked = session.get("is_table_locked", False)
            if (database_name or table_name) and not is_table_locked:
                update_data = {}
                if database_name:
                    update_data["database_name"] = database_name
                if table_name:
                    update_data["table_name"] = table_name

                if update_data:
                    current_db = session.get("database_name")
                    current_table = session.get("table_name")
                    db_to_use = database_name or current_db
                    table_to_use = table_name or current_table
                    if db_to_use and table_to_use:
                        update_data["is_table_locked"] = True

                if update_data:
                    update_kgqa_chat_session(session_id, **update_data)
                    session = get_kgqa_chat_session(session_id)

            self.add_user_message(session_id, question)
            messages = get_kgqa_chat_messages(session_id)
            if len(messages) == 1:
                try:
                    self.generate_session_name(session_id, question)
                except Exception:
                    pass

            temperature = session.get("temperature", 0.7)
            current_config = self.vanna_manager.get_config()

            if self.graph_qa_chain is None:
                self.init_graph_qa_chain(current_config)

            model_name = session.get("model_name")
            if model_name:
                model_type = current_config.get("model", {}).get("type", "ollama")
                if model_type == "ollama":
                    import httpx

                    try:
                        http_client = httpx.AsyncClient(
                            timeout=httpx.Timeout(
                                connect=10.0, read=60.0, write=10.0, pool=5.0
                            ),
                            limits=httpx.Limits(
                                max_keepalive_connections=5,
                                max_connections=10,
                                keepalive_expiry=30.0,
                            ),
                            http2=True,
                        )
                    except ImportError:
                        http_client = httpx.AsyncClient(
                            timeout=httpx.Timeout(
                                connect=10.0, read=60.0, write=10.0, pool=5.0
                            ),
                            limits=httpx.Limits(
                                max_keepalive_connections=5,
                                max_connections=10,
                                keepalive_expiry=30.0,
                            ),
                        )

                    self.llm = ChatOllama(
                        base_url=current_config.get("model", {}).get(
                            "ollama_url", "http://localhost:11434"
                        ),
                        model=model_name,
                        temperature=temperature,
                        extract_reasoning=True,
                        num_ctx=current_config.get("model", {}).get("num_ctx", 25600),
                        client=http_client,
                        request_timeout=60.0,
                        num_retries=2,
                    )
                elif model_type == "openai":
                    from langchain_openai import ChatOpenAI

                    self.llm = ChatOpenAI(
                        model=model_name,
                        temperature=temperature,
                        api_key=current_config.get("model", {}).get("api_key"),
                        base_url=current_config.get("model", {}).get(
                            "api_base", "https://api.openai.com/v1"
                        ),
                        max_tokens=current_config.get("model", {}).get(
                            "num_ctx", 25600
                        ),
                    )
                    self.graph_qa_chain.cypher_generation_chain.bind(llm=self.llm)
                    self.graph_qa_chain.qa_chain.bind(llm=self.llm)

            context = ""
            table_to_query = session.get("table_name")
            if table_to_query:
                parts = table_to_query.split(":", 1)
                if len(parts) == 2:
                    graph_id, graph_name = parts
                else:
                    graph_id, graph_name = parts[0], parts[0]
                self.graph_qa_chain.graph_schema = (
                    self.graph_qa_chain._get_graph_schema(graph_id)
                )
                context = (
                    f"Query the knowledge graph {graph_name} with ID {graph_id}, corresponding to the Cypher query "
                    f"MATCH(n) WHERE n.graph_id = {graph_id} RETURN n. Please ensure to merge the WHERE condition in this Cypher query."
                )
                enhanced_question = f"{context}{question}"
            else:
                enhanced_question = question

            auto_train_setting = session.get("auto_train", True)
            self.graph_qa_chain.set_autotrain(auto_train_setting)

            yield {"type": "start"}
            yield {
                "type": "status",
                "stage": "prepare",
                "progress": 10,
                "message": self._get_localized_message(
                    "正在准备图谱查询上下文", "Preparing graph query context"
                ),
            }

            yield {
                "type": "status",
                "stage": "sql",
                "progress": 30,
                "message": self._get_localized_message(
                    "正在生成Cypher并执行查询", "Generating Cypher and running query"
                ),
            }

            cypher = None
            result_data = None
            content = ""
            thinking = None
            summary_started = False

            for stream_event in self.graph_qa_chain.stream_answer(
                {"query": enhanced_question, "normal_query": question}
            ):
                event_type = stream_event.get("event")

                if event_type == "query":
                    cypher = stream_event.get("cypher")
                    context_rows = stream_event.get("context")
                    if (
                        context_rows
                        and context_rows
                        != "I couldn't find any relevant information in the database"
                    ):
                        result_data = convert_numpy_types(context_rows)

                    yield {
                        "type": "status",
                        "stage": "process",
                        "progress": 60,
                        "message": self._get_localized_message(
                            "正在处理结果与子图", "Processing result and subgraph"
                        ),
                    }
                    continue

                if event_type == "token":
                    if not summary_started:
                        summary_started = True
                        yield {
                            "type": "status",
                            "stage": "summary",
                            "progress": 75,
                            "message": self._get_localized_message(
                                "LLM 正在生成分析", "LLM is generating analysis"
                            ),
                        }
                    delta = stream_event.get("delta") or ""
                    if delta:
                        content += str(delta)
                        yield {"type": "chunk", "delta": str(delta)}
                    continue

                if event_type == "final":
                    final_result = stream_event.get("result")
                    if (
                        isinstance(final_result, str)
                        and final_result.strip()
                        and not content.strip()
                    ):
                        content = final_result.strip()
                        yield {"type": "chunk", "delta": content}

            yield {
                "type": "status",
                "stage": "persist",
                "progress": 92,
                "message": self._get_localized_message(
                    "正在保存消息", "Saving message"
                ),
            }

            if content:
                think_match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
                if think_match:
                    thinking = think_match.group(1).strip()
                    clean_content = re.sub(
                        r"<think>.*?</think>", "", content, flags=re.DOTALL
                    ).strip()
                    if clean_content:
                        content = clean_content

            graph_data = self._build_visualization_subgraph(
                session_id, cypher, result_data
            )
            graph_data_converted = convert_numpy_types(graph_data)
            visualization = (
                graph_data_converted
                if not self.graph_data_is_empty(graph_data_converted)
                else None
            )

            self.add_assistant_message(
                session_id=session_id,
                content=content,
                cypher=cypher,
                result=result_data,
                visualization=visualization,
                reasoning=None,
                thinking=thinking,
            )

            payload = {
                "cypher": cypher,
                "result": result_data,
                "visualization": visualization,
                "thinking": thinking,
                "explanation": content,
            }
            yield {"type": "meta", "payload": payload}
            yield {
                "type": "status",
                "stage": "done",
                "progress": 100,
                "message": self._get_localized_message("已完成", "Completed"),
            }
            yield {"type": "end"}
        except Exception as e:
            logger.error(f"KG流式问答处理失败: {str(e)}")
            yield {"type": "error", "message": str(e)}
            yield {"type": "end"}

    def graph_data_is_empty(self, result_data):
        graph_data = {
            "nodes": [],
            "relationships": [],
            "statistics": {
                "node_types": {},
                "relationship_types": {},
                "node_count": 0,
                "relationship_count": 0,
            },
        }
        # 两个dict如何比较相同
        return graph_data == result_data

    def _build_visualization_subgraph(self, session_id, cypher, result_data):
        """构建用于可视化的子图数据

        Args:
            session_id: 图谱ID
            cypher: Cypher查询
            result_data: 查询结果数据

        Returns:
            子图数据对象
        """
        try:
            # 初始化返回结构
            graph_data = {
                "nodes": [],
                "relationships": [],
                "statistics": {
                    "node_types": {},
                    "relationship_types": {},
                    "node_count": 0,
                    "relationship_count": 0,
                },
            }

            # 检查是否有Neo4j驱动
            if not hasattr(self, "neo4j_driver") or not self.neo4j_driver:
                logger.error("Neo4j驱动未初始化，无法构建可视化子图")
                return graph_data

            # 直接从Neo4j执行Cypher查询获取节点和关系数据
            nodes_data = []
            relationships_data = []
            node_types_stats = {}
            relationship_types_stats = {}
            nodes_map = {}  # 使用字典存储唯一节点
            # --------------------- 新增：用于关系去重的集合 ---------------------
            added_relationship_keys = set()
            # -------------------------------------------------------------------

            with self.neo4j_driver.session() as neo4j_session:
                # 根据提供的Cypher查询构建可视化查询
                if cypher and cypher.strip():
                    try:
                        # 处理Cypher查询以保证返回可视化所需数据
                        visualization_cypher = cypher

                        # 1. 处理非查询语句 (CREATE, DELETE, etc.)
                        is_modifying_query = False
                        if re.search(
                            r"\b(CREATE|DELETE|SET|REMOVE|MERGE)\b",
                            visualization_cypher,
                            re.IGNORECASE,
                        ):
                            is_modifying_query = True
                            if not re.search(
                                r"\bRETURN\b", visualization_cypher, re.IGNORECASE
                            ):
                                visualization_cypher += (
                                    " RETURN count(*) as affected_count"
                                )

                        # 2. 确保有 RETURN 语句 (如果不是修改语句)
                        if not is_modifying_query and not re.search(
                            r"\bRETURN\b", visualization_cypher, re.IGNORECASE
                        ):
                            # 尝试从 MATCH 中提取变量并返回
                            match_vars = re.findall(
                                r"\bMATCH\s*(?:\([^)]*\)|\[[^\]]*\]|\{[^}]*\}|\s*)*\(\s*([a-zA-Z0-9_]+)\s*[:)]",
                                visualization_cypher,
                            )
                            if match_vars:
                                visualization_cypher += (
                                    f" RETURN {', '.join(set(match_vars))}"
                                )
                            else:
                                visualization_cypher += " RETURN *"  # 作为后备

                        # 3.【新增】检查并修改仅属性返回的 RETURN 子句
                        return_match = re.search(
                            r"\bRETURN\b\s+(.*)",
                            visualization_cypher,
                            re.IGNORECASE | re.DOTALL,
                        )
                        if return_match and not is_modifying_query:
                            return_clause = return_match.group(1).strip()
                            # 移除可能的 LIMIT, ORDER BY 等后缀
                            return_items_str = re.split(
                                r"\b(LIMIT|ORDER BY|SKIP)\b",
                                return_clause,
                                maxsplit=1,
                                flags=re.IGNORECASE,
                            )[0].strip()

                            # 分割返回项
                            items = [
                                item.strip() for item in return_items_str.split(",")
                            ]

                            # 【新增】检测是否包含聚合函数
                            has_aggregation = any(
                                re.search(
                                    r"\b(COUNT|SUM|AVG|MIN|MAX|COLLECT)\s*\(",
                                    item,
                                    re.IGNORECASE,
                                )
                                for item in items
                            )

                            if has_aggregation:
                                logger.info(f"检测到聚合函数，生成专门的可视化查询")
                                # 为聚合查询生成专门的可视化查询
                                # 从原始查询中提取所有变量
                                all_vars = set(
                                    re.findall(
                                        r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s*:", cypher
                                    )
                                )  # 节点变量
                                all_vars.update(
                                    re.findall(
                                        r"-\[\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*.*?]->",
                                        cypher,
                                    )
                                )  # 关系变量

                                # 构建简化的可视化查询
                                if all_vars:
                                    # 使用原始查询的MATCH部分，但替换RETURN部分
                                    match_part = re.sub(
                                        r"\bRETURN\b.*",
                                        "",
                                        cypher,
                                        flags=re.IGNORECASE | re.DOTALL,
                                    ).strip()
                                    visualization_cypher = f"{match_part}\nRETURN {', '.join(sorted(all_vars))} LIMIT 100"
                                    logger.info(
                                        f"生成聚合查询的可视化版本: {visualization_cypher}"
                                    )
                                else:
                                    logger.warning(
                                        "无法从聚合查询中提取变量，使用原始查询"
                                    )
                            else:
                                # 检查是否所有项都包含属性访问 ('.') 并且没有聚合函数或 '*'
                                is_property_only = True
                                base_variables = set()
                                if not items or "*" in items:
                                    is_property_only = False
                                else:
                                    for item in items:
                                        # 忽略函数调用如 count(a), properties(n) 等
                                        if "(" in item and ")" in item:
                                            is_property_only = False
                                            break
                                        # 检查是否包含点号且不是数字 (如 1.5)
                                        if (
                                            "." not in item
                                            or item.replace(".", "", 1).isdigit()
                                        ):
                                            # 如果有一项不是属性访问，则标记为False
                                            if not re.match(
                                                r"^[a-zA-Z_][a-zA-Z0-9_]*$", item
                                            ):  # 允许返回简单变量名
                                                is_property_only = False
                                                # break # 不立即break，允许混合返回，但优先提取基础变量
                                        else:
                                            # 提取基础变量名 (e.g., 'a' from 'a.name')
                                            var_match = re.match(
                                                r"^([a-zA-Z_][a-zA-Z0-9_]*)\.", item
                                            )
                                            if var_match:
                                                base_variables.add(var_match.group(1))
                                            else:  # 无法识别基础变量，可能不是简单的属性访问
                                                is_property_only = False
                                                # break

                                # 如果确定是仅属性返回，并且我们成功提取了基础变量
                                if is_property_only and base_variables:
                                    logger.info(
                                        f"检测到仅属性返回，提取基础变量: {base_variables}"
                                    )
                                    # 从原始查询 (MATCH/OPTIONAL MATCH/WITH) 中查找可能的关系变量
                                    rel_vars = set(
                                        re.findall(
                                            r"-\[\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*.*?]->",
                                            cypher,
                                        )
                                    )
                                    path_vars = set(
                                        re.findall(
                                            r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*\(\s*\)\s*-\[\*.*\]->\s*\(\s*\)",
                                            cypher,
                                        )
                                    )  # 查找路径变量 p = ()-[*]->()

                                    # 合并所有需要返回的变量
                                    return_vars = list(
                                        base_variables | rel_vars | path_vars
                                    )

                                    if return_vars:
                                        new_return_clause = (
                                            f"RETURN {', '.join(return_vars)}"
                                        )
                                        # 保留原始查询中的 LIMIT / ORDER BY 等后缀，但需要处理ORDER BY中的别名问题
                                        suffix_match = re.search(
                                            r"(\b(?:ORDER BY|SKIP|LIMIT)\b.*)",
                                            return_clause,
                                            re.IGNORECASE | re.DOTALL,
                                        )
                                        if suffix_match:
                                            suffix_clause = suffix_match.group(1)
                                            # 检查是否包含ORDER BY子句
                                            if re.search(
                                                r"\bORDER BY\b",
                                                suffix_clause,
                                                re.IGNORECASE,
                                            ):
                                                # 对于修改后的RETURN子句，移除可能无效的ORDER BY子句
                                                # 因为ORDER BY中的别名在新的RETURN中可能不存在
                                                logger.warning(
                                                    "检测到ORDER BY子句，但修改后的RETURN子句可能不包含相应的别名，将移除ORDER BY子句"
                                                )
                                                # 只保留LIMIT和SKIP子句
                                                limit_skip_match = re.search(
                                                    r"(\b(?:SKIP|LIMIT)\b.*)",
                                                    suffix_clause,
                                                    re.IGNORECASE | re.DOTALL,
                                                )
                                                if limit_skip_match:
                                                    new_return_clause += (
                                                        " " + limit_skip_match.group(1)
                                                    )
                                            else:
                                                # 如果没有ORDER BY，正常添加后缀
                                                new_return_clause += " " + suffix_clause

                                        logger.info(
                                            f"修改可视化查询 RETURN 子句为: {new_return_clause}"
                                        )
                                        # 替换原 RETURN 子句
                                        visualization_cypher = re.sub(
                                            r"\bRETURN\b\s+.*",
                                            new_return_clause,
                                            visualization_cypher,
                                            flags=re.IGNORECASE | re.DOTALL,
                                        )
                                    else:
                                        logger.warning(
                                            "无法从仅属性返回中可靠地提取基础变量，将尝试使用原始查询。"
                                        )
                                elif (
                                    base_variables and not is_property_only
                                ):  # 混合返回属性和变量
                                    logger.info(
                                        f"检测到混合返回，提取的基础变量: {base_variables}"
                                    )
                                    # 查找关系和路径变量
                                    rel_vars = set(
                                        re.findall(
                                            r"-\[\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*.*?]->",
                                            cypher,
                                        )
                                    )
                                    path_vars = set(
                                        re.findall(
                                            r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*\(\s*\)\s*-\[\*.*\]->\s*\(\s*\)",
                                            cypher,
                                        )
                                    )

                                    existing_vars = set(
                                        item
                                        for item in items
                                        if re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", item)
                                    )
                                    all_vars_to_return = list(
                                        base_variables
                                        | rel_vars
                                        | path_vars
                                        | existing_vars
                                    )

                                    if all_vars_to_return:
                                        # 构建新的RETURN，包含所有识别的变量和原始非属性项
                                        original_non_prop_items = [
                                            item
                                            for item in items
                                            if "." not in item or "(" in item
                                        ]
                                        final_return_items = list(
                                            dict.fromkeys(
                                                all_vars_to_return
                                                + original_non_prop_items
                                            )
                                        )  # 去重并保持一定顺序

                                        new_return_clause = (
                                            f"RETURN {', '.join(final_return_items)}"
                                        )
                                        # 保留原始查询中的 LIMIT / ORDER BY 等后缀，但需要处理ORDER BY中的别名问题
                                        suffix_match = re.search(
                                            r"(\b(?:ORDER BY|SKIP|LIMIT)\b.*)",
                                            return_clause,
                                            re.IGNORECASE | re.DOTALL,
                                        )
                                        if suffix_match:
                                            suffix_clause = suffix_match.group(1)
                                            # 检查是否包含ORDER BY子句
                                            if re.search(
                                                r"\bORDER BY\b",
                                                suffix_clause,
                                                re.IGNORECASE,
                                            ):
                                                # 对于修改后的RETURN子句，移除可能无效的ORDER BY子句
                                                # 因为ORDER BY中的别名在新的RETURN中可能不存在
                                                logger.warning(
                                                    "检测到ORDER BY子句，但修改后的RETURN子句可能不包含相应的别名，将移除ORDER BY子句"
                                                )
                                                # 只保留LIMIT和SKIP子句
                                                limit_skip_match = re.search(
                                                    r"(\b(?:SKIP|LIMIT)\b.*)",
                                                    suffix_clause,
                                                    re.IGNORECASE | re.DOTALL,
                                                )
                                                if limit_skip_match:
                                                    new_return_clause += (
                                                        " " + limit_skip_match.group(1)
                                                    )
                                            else:
                                                # 如果没有ORDER BY，正常添加后缀
                                                new_return_clause += " " + suffix_clause

                                        logger.info(
                                            f"修改混合返回的可视化查询 RETURN 子句为: {new_return_clause}"
                                        )
                                        visualization_cypher = re.sub(
                                            r"\bRETURN\b\s+.*",
                                            new_return_clause,
                                            visualization_cypher,
                                            flags=re.IGNORECASE | re.DOTALL,
                                        )

                        logger.info(f"执行可视化Cypher查询: {visualization_cypher}")

                        # 执行查询获取记录
                        result = neo4j_session.run(visualization_cypher)
                        records = list(result)

                        # 如果记录为空，并且原始查询不是修改语句，尝试查询所有与当前图谱相关的节点和关系
                        if not records or len(records) == 0 and not is_modifying_query:
                            logger.info(
                                "原始查询未返回可视化数据或返回为空，尝试查询图谱节点和关系"
                            )

                            # 查询节点
                            node_result = neo4j_session.run(
                                """
                                MATCH (n {graph_id: $graph_id})
                                RETURN n, labels(n) as types
                                LIMIT 1000 
                            """,
                                graph_id=session_id,
                            )

                            for record in node_result:
                                node = record["n"]
                                types = record["types"]

                                # 转换Neo4j节点为字典
                                node_props = dict(node)
                                node_id = str(node.id)
                                node_type = types[0] if types else "Unknown"

                                # 清理内部属性
                                node_props.pop("graph_id", None)

                                node_dict = {
                                    "id": node_id,
                                    "type": node_type,
                                    "attributes": node_props,
                                }

                                nodes_data.append(node_dict)

                                # 更新统计
                                node_types_stats[node_type] = (
                                    node_types_stats.get(node_type, 0) + 1
                                )

                            # 查询关系
                            rel_result = neo4j_session.run(
                                """
                                MATCH (source {graph_id: $graph_id})-[r]->(target {graph_id: $graph_id})
                                RETURN source, r, target, type(r) as rel_type
                                LIMIT 2000 
                            """,
                                graph_id=session_id,
                            )

                            for record in rel_result:
                                source = record["source"]
                                target = record["target"]
                                rel = record["r"]
                                rel_type = record["rel_type"]

                                # 获取关系属性
                                rel_props = dict(rel)
                                rel_props.pop("graph_id", None)

                                # 创建关系字典
                                rel_dict = {
                                    "source": str(source.id),
                                    "target": str(target.id),
                                    "type": rel_type,
                                    "properties": rel_props,
                                }

                                relationships_data.append(rel_dict)

                                # 更新统计
                                relationship_types_stats[rel_type] = (
                                    relationship_types_stats.get(rel_type, 0) + 1
                                )
                        else:
                            # 从记录中提取节点和关系
                            # nodes_map = {} # nodes_map 移到 with 外部初始化

                            for record in records:
                                # 处理每个记录中的所有字段
                                for key in record.keys():
                                    value = record[key]
                                    # --------------------- 修改：传递 added_relationship_keys ---------------------
                                    self._extract_graph_elements_from_result(
                                        value,
                                        nodes_map,
                                        relationships_data,
                                        added_relationship_keys,
                                    )
                                    # --------------------------------------------------------------------------

                            # 转换节点映射为列表
                            nodes_data = list(nodes_map.values())

                            # 检查是否只有节点但没有关系 - 如果是则执行补充查询获取关系
                            if len(nodes_data) > 0 and len(relationships_data) == 0:
                                logger.info(
                                    f"查询只返回了节点({len(nodes_data)}个)但没有关系，执行补充查询获取关系"
                                )

                                # 提取所有节点ID用于补充查询
                                node_ids = []
                                for node in nodes_data:
                                    # 尝试从不同位置获取unique_id
                                    unique_id = None
                                    # 1. 检查attributes中是否有unique_id
                                    if (
                                        "attributes" in node
                                        and "unique_id" in node["attributes"]
                                    ):
                                        unique_id = node["attributes"]["unique_id"]
                                    # 2. 或者使用id作为后备
                                    elif "id" in node:
                                        unique_id = node["id"]

                                    if unique_id:
                                        node_ids.append(unique_id)

                                # 当节点数量过多时，限制补充查询的节点数量以避免性能问题
                                max_nodes_for_query = 100  # 设置合理的上限
                                if len(node_ids) > max_nodes_for_query:
                                    logger.warning(
                                        f"节点数量({len(node_ids)})超过补充查询上限({max_nodes_for_query})，将只使用部分节点"
                                    )
                                    node_ids = node_ids[:max_nodes_for_query]

                                try:
                                    # 方法1：查询这些节点之间的所有关系
                                    if (
                                        len(node_ids) <= 20
                                    ):  # 当节点不多时可以直接查询它们之间的关系
                                        # 为节点ID添加引号，并用逗号连接
                                        node_ids_str = ", ".join(
                                            [f"'{node_id}'" for node_id in node_ids]
                                        )
                                        rel_query = f"""
                                            MATCH (source)-[r]->(target)
                                            WHERE source.unique_id IN [{node_ids_str}] AND target.unique_id IN [{node_ids_str}]
                                            RETURN source, r, target, type(r) as rel_type
                                            LIMIT 1000
                                        """
                                    else:
                                        # 方法2：对于较多节点，仍然只查询节点集合内部的关系
                                        # 为节点ID添加引号，并用逗号连接
                                        node_ids_str = ", ".join(
                                            [f"'{node_id}'" for node_id in node_ids]
                                        )
                                        rel_query = f"""
                                            MATCH (source)-[r]->(target)
                                            WHERE source.unique_id IN [{node_ids_str}] AND target.unique_id IN [{node_ids_str}]
                                            RETURN source, r, target, type(r) as rel_type
                                            LIMIT 5000
                                        """

                                    logger.info(
                                        f"执行关系补充查询，涉及{len(node_ids)}个节点"
                                    )
                                    rel_result = neo4j_session.run(rel_query)

                                    for record in rel_result:
                                        # --------------------- 修改：传递 added_relationship_keys ---------------------
                                        self._extract_graph_elements_from_result(
                                            record,
                                            nodes_map,
                                            relationships_data,
                                            added_relationship_keys,
                                        )
                                        # --------------------------------------------------------------------------
                                        # (移除原有的添加和统计逻辑，因为 _extract_graph_elements_from_result 会处理)

                                    # 重新从 nodes_map 获取最新的 nodes_data，因为补充查询可能添加了新节点
                                    nodes_data = list(nodes_map.values())
                                    logger.info(
                                        f"结果数据补充查询完成，当前关系数: {len(relationships_data)}"
                                    )

                                except Exception as e:
                                    logger.error(f"执行关系补充查询失败: {str(e)}")

                            # 计算节点类型统计
                            for node in nodes_data:
                                node_type = node.get("type", "Unknown")
                                node_types_stats[node_type] = (
                                    node_types_stats.get(node_type, 0) + 1
                                )

                            # 计算关系类型统计
                            for rel in relationships_data:
                                rel_type = rel.get("type", "Unknown")
                                relationship_types_stats[rel_type] = (
                                    relationship_types_stats.get(rel_type, 0) + 1
                                )

                    except Exception as e:
                        logger.error(f"执行可视化Cypher查询失败: {str(e)}")
                        # 如果有结果数据，尝试从中提取
                        if result_data:
                            logger.info("从结果数据中提取可视化信息")
                            # nodes_map = {} # nodes_map 移到 with 外部初始化

                            if isinstance(result_data, list):
                                for item in result_data:
                                    # --------------------- 修改：传递 added_relationship_keys ---------------------
                                    self._extract_graph_elements_from_result(
                                        item,
                                        nodes_map,
                                        relationships_data,
                                        added_relationship_keys,
                                    )
                                    # --------------------------------------------------------------------------
                            elif isinstance(result_data, dict):
                                # --------------------- 修改：传递 added_relationship_keys ---------------------
                                self._extract_graph_elements_from_result(
                                    result_data,
                                    nodes_map,
                                    relationships_data,
                                    added_relationship_keys,
                                )
                                # --------------------------------------------------------------------------

                            # 转换节点映射为列表
                            nodes_data = list(nodes_map.values())

                            # 检查是否只有节点没有关系 - 如果是则执行补充查询
                            if len(nodes_data) > 0 and len(relationships_data) == 0:
                                logger.info(
                                    f"结果数据只包含节点({len(nodes_data)}个)但没有关系，尝试执行补充查询"
                                )
                                # ... (获取 node_ids 的逻辑) ...
                                if node_ids:
                                    try:
                                        # ... (构建 rel_query 的逻辑) ...
                                        logger.info(
                                            f"执行关系补充查询，涉及{len(node_ids)}个节点"
                                        )
                                        rel_result = neo4j_session.run(rel_query)

                                        for record in rel_result:
                                            # --------------------- 修改：传递 added_relationship_keys ---------------------
                                            self._extract_graph_elements_from_result(
                                                record,
                                                nodes_map,
                                                relationships_data,
                                                added_relationship_keys,
                                            )
                                            # --------------------------------------------------------------------------
                                            # (移除原有的添加和统计逻辑)

                                        # 更新 nodes_data
                                        nodes_data = list(nodes_map.values())

                                    except Exception as e:
                                        logger.error(
                                            f"执行结果数据补充查询失败: {str(e)}"
                                        )

                            # ... (计算统计信息的逻辑) ...

                # 如果没有指定Cypher查询但有结果数据
                elif result_data:
                    # nodes_map = {} # nodes_map 移到 with 外部初始化

                    if isinstance(result_data, list):
                        for item in result_data:
                            # --------------------- 修改：传递 added_relationship_keys ---------------------
                            self._extract_graph_elements_from_result(
                                item,
                                nodes_map,
                                relationships_data,
                                added_relationship_keys,
                            )
                            # --------------------------------------------------------------------------
                    elif isinstance(result_data, dict):
                        # --------------------- 修改：传递 added_relationship_keys ---------------------
                        self._extract_graph_elements_from_result(
                            result_data,
                            nodes_map,
                            relationships_data,
                            added_relationship_keys,
                        )
                        # --------------------------------------------------------------------------

                    # 转换节点映射为列表
                    nodes_data = list(nodes_map.values())

                    # 检查是否只有节点没有关系 - 如果是则执行补充查询
                    if len(nodes_data) > 0 and len(relationships_data) == 0:
                        logger.info(
                            f"结果数据只包含节点({len(nodes_data)}个)但没有关系，尝试执行补充查询"
                        )

                        # 获取节点ID用于查询
                        node_ids = []
                        for node in nodes_data:
                            # 尝试从不同属性中获取唯一标识符
                            if "id" in node:
                                node_ids.append(node["id"])
                            elif "unique_id" in node.get("attributes", {}):
                                node_ids.append(node["attributes"]["unique_id"])

                        # 如果找到了节点ID，执行补充查询
                        if node_ids:
                            try:
                                # 当节点数量过多时，限制补充查询的节点数量
                                max_nodes_for_query = 50
                                if len(node_ids) > max_nodes_for_query:
                                    logger.warning(
                                        f"节点数量({len(node_ids)})超过补充查询上限({max_nodes_for_query})，将只使用部分节点"
                                    )
                                    node_ids = node_ids[:max_nodes_for_query]

                                # 为节点ID添加引号并拼接
                                node_ids_str = ", ".join(
                                    [f"'{node_id}'" for node_id in node_ids]
                                )

                                # 构建查询
                                if len(node_ids) <= 20:
                                    rel_query = f"""
                                        MATCH (source)-[r]->(target)
                                        WHERE source.unique_id IN [{node_ids_str}] AND target.unique_id IN [{node_ids_str}]
                                        RETURN source, r, target, type(r) as rel_type
                                        LIMIT 1000
                                    """
                                else:
                                    rel_query = f"""
                                        MATCH (source)-[r]->(target)
                                        WHERE source.unique_id IN [{node_ids_str}] AND target.unique_id IN [{node_ids_str}]
                                        RETURN source, r, target, type(r) as rel_type
                                        LIMIT 5000
                                    """

                                logger.info(
                                    f"执行关系补充查询，涉及{len(node_ids)}个节点"
                                )
                                rel_result = neo4j_session.run(rel_query)

                                for record in rel_result:
                                    # --------------------- 修改：传递 added_relationship_keys ---------------------
                                    self._extract_graph_elements_from_result(
                                        record,
                                        nodes_map,
                                        relationships_data,
                                        added_relationship_keys,
                                    )
                                    # --------------------------------------------------------------------------
                                    # (移除原有的添加和统计逻辑)

                                # 更新 nodes_data
                                nodes_data = list(nodes_map.values())

                            except Exception as e:
                                logger.error(f"执行结果数据补充查询失败: {str(e)}")

                    # 计算统计信息
                    for node in nodes_data:
                        node_type = node.get("type", "Unknown")
                        node_types_stats[node_type] = (
                            node_types_stats.get(node_type, 0) + 1
                        )

                    for rel in relationships_data:
                        rel_type = rel.get("type", "Unknown")
                        relationship_types_stats[rel_type] = (
                            relationship_types_stats.get(rel_type, 0) + 1
                        )

                # 如果仍然没有数据，尝试获取基本的图谱节点和关系
                if not nodes_data:
                    logger.info("尝试获取基本图谱数据")

                    # 查询节点
                    node_result = neo4j_session.run(
                        """
                        MATCH (n {graph_id: $graph_id})
                        RETURN n, labels(n) as types
                        LIMIT 1000 
                    """,
                        graph_id=session_id,
                    )

                    for record in node_result:
                        node = record["n"]
                        types = record["types"]

                        # 转换Neo4j节点为字典
                        node_props = dict(node)
                        node_id = str(node.id)
                        node_type = types[0] if types else "Unknown"

                        # 清理内部属性
                        node_props.pop("graph_id", None)

                        node_dict = {
                            "id": node_id,
                            "type": node_type,
                            "attributes": node_props,
                        }

                        nodes_data.append(node_dict)

                        # 更新统计
                        node_types_stats[node_type] = (
                            node_types_stats.get(node_type, 0) + 1
                        )

                    # 查询关系
                    rel_result = neo4j_session.run(
                        """
                        MATCH (source {graph_id: $graph_id})-[r]->(target {graph_id: $graph_id})
                        RETURN source, r, target, type(r) as rel_type
                        LIMIT 2000 
                    """,
                        graph_id=session_id,
                    )

                    for record in rel_result:
                        source = record["source"]
                        target = record["target"]
                        rel = record["r"]
                        rel_type = record["rel_type"]

                        # 获取关系属性
                        rel_props = dict(rel)
                        rel_props.pop("graph_id", None)

                        # 创建关系字典
                        rel_dict = {
                            "source": str(source.id),
                            "target": str(target.id),
                            "type": rel_type,
                            "properties": rel_props,
                        }

                        relationships_data.append(rel_dict)

                        # 更新统计
                        relationship_types_stats[rel_type] = (
                            relationship_types_stats.get(rel_type, 0) + 1
                        )

                # 检查是否需要进行第三种补充：有节点有关系但节点之间没有连接
                if len(nodes_data) > 1 and len(relationships_data) > 0:
                    # 构建节点的连接图，检查是否存在未连接的子图
                    node_connections = {}
                    for node in nodes_data:
                        node_connections[node["id"]] = set()

                    # 记录关系
                    for rel in relationships_data:
                        source_id = rel["source"]
                        target_id = rel["target"]
                        if (
                            source_id in node_connections
                            and target_id in node_connections
                        ):
                            node_connections[source_id].add(target_id)
                            node_connections[target_id].add(source_id)  # 无向连接

                    # 查找连通分量
                    components = []
                    visited = set()

                    def dfs(node_id, component):
                        visited.add(node_id)
                        component.append(node_id)
                        for neighbor in node_connections[node_id]:
                            if neighbor not in visited:
                                dfs(neighbor, component)

                    for node_id in node_connections:
                        if node_id not in visited:
                            component = []
                            dfs(node_id, component)
                            if component:
                                components.append(component)

                    # 如果有多个连通分量，尝试在它们之间添加连接
                    if (
                        len(components) > 1 and len(components) <= 5
                    ):  # 限制组件数量，避免过度查询
                        logger.info(
                            f"检测到{len(components)}个未连接的子图，尝试添加它们之间的关系"
                        )

                        try:
                            # 为每个组件选择一个代表节点
                            component_reps = [comp[0] for comp in components]

                            # 查询这些代表节点之间的最短路径
                            for i in range(len(component_reps)):
                                for j in range(i + 1, len(component_reps)):
                                    source_id = component_reps[i]
                                    target_id = component_reps[j]

                                    # 查询最短路径
                                    path_query = f"""
                                        MATCH (source:装备名称), (target:装备名称), 
                                            path = shortestPath((source)-[*..5]-(target))
                                        WHERE source.unique_id = "{source_id}" AND target.unique_id = "{target_id}"
                                        RETURN path
                                        LIMIT 1
                                    """

                                    try:
                                        path_result = neo4j_session.run(path_query)
                                        for record in path_result:
                                            if "path" in record:
                                                path = record["path"]
                                                # 提取路径中的所有节点和关系
                                                self._extract_graph_elements_from_result(
                                                    path,
                                                    nodes_map,
                                                    relationships_data,
                                                    added_relationship_keys,
                                                )
                                    except Exception as e:
                                        logger.error(
                                            f"执行路径查询失败 ({source_id}->{target_id}): {str(e)}"
                                        )
                        except Exception as e:
                            logger.error(f"尝试连接子图失败: {str(e)}")

            # 最终计算统计信息
            # 重新计算，因为去重可能改变了数量
            node_types_stats = {}
            relationship_types_stats = {}
            for node in nodes_data:
                node_type = node.get("type", "Unknown")
                node_types_stats[node_type] = node_types_stats.get(node_type, 0) + 1
            for rel in relationships_data:
                rel_type = rel.get("type", "Unknown")
                relationship_types_stats[rel_type] = (
                    relationship_types_stats.get(rel_type, 0) + 1
                )

            # 更新返回数据
            graph_data["nodes"] = nodes_data
            graph_data["relationships"] = relationships_data
            graph_data["statistics"]["node_types"] = node_types_stats
            graph_data["statistics"]["relationship_types"] = relationship_types_stats
            graph_data["statistics"]["node_count"] = len(nodes_data)
            graph_data["statistics"]["relationship_count"] = len(relationships_data)

            # 添加警告信息
            if len(nodes_data) >= 1000 or len(relationships_data) >= 2000:
                graph_data["warning"] = (
                    "注意：返回的数据量可能因达到上限而被截断。请考虑在可视化或查询中添加更具体的过滤条件。"
                )
            logger.info(f"{graph_data}")
            logger.info(
                f"构建可视化子图完成 (去重后): {graph_data['statistics']['node_count']} 个节点, {graph_data['statistics']['relationship_count']} 条关系"
            )
            return graph_data

        except Exception as e:
            logger.error(f"构建可视化子图失败: {str(e)}")
            return {
                "nodes": [],
                "relationships": [],
                "statistics": {
                    "node_types": {},
                    "relationship_types": {},
                    "node_count": 0,
                    "relationship_count": 0,
                },
            }

    def _extract_graph_elements_from_result(
        self, value, nodes_map, relationships, added_relationship_keys
    ):
        """从结果中提取图元素

        Args:
            value: 要处理的值
            nodes_map: 节点映射，用于存储唯一节点
            relationships: 关系列表
            added_relationship_keys: 用于关系去重的集合
        """
        if value is None:
            return

        # 处理Neo4j节点对象 (neo4j.graph.Node)
        # isinstance(value, neo4j.graph.Node) # 更可靠的检查方式，但需要导入
        if (
            hasattr(value, "id")
            and hasattr(value, "labels")
            and hasattr(value, "items")
        ):
            node_internal_id = str(value.id)
            node_labels = list(value.labels)
            node_type = node_labels[0] if node_labels else "Unknown"

            # 获取节点的所有属性
            node_props = dict(value.items())

            # 确定节点ID（优先使用unique_id属性，其次使用内部ID）
            node_id = node_props.get("unique_id", node_internal_id)

            # 只有在节点尚未添加时才添加
            if node_id not in nodes_map:
                # 从属性中移除内部可能存在的 graph_id
                node_props.pop("graph_id", None)
                nodes_map[node_id] = {
                    "id": node_id,
                    "type": node_type,
                    "attributes": node_props,
                    "_internal_id": node_internal_id,  # 保留内部ID用于关系连接
                }

        # 处理Neo4j关系对象 (neo4j.graph.Relationship)
        # isinstance(value, neo4j.graph.Relationship)
        elif (
            hasattr(value, "id")
            and hasattr(value, "type")
            and hasattr(value, "start_node")
            and hasattr(value, "end_node")
            and hasattr(value, "items")
        ):
            rel_type = value.type
            start_node_obj = value.start_node
            end_node_obj = value.end_node

            # 递归处理关系两端的节点，确保它们在 nodes_map 中
            self._extract_graph_elements_from_result(
                start_node_obj, nodes_map, relationships, added_relationship_keys
            )
            self._extract_graph_elements_from_result(
                end_node_obj, nodes_map, relationships, added_relationship_keys
            )

            # 获取节点ID（此时应已存在于 nodes_map 中，优先用unique_id）
            start_node_props = dict(start_node_obj.items())
            end_node_props = dict(end_node_obj.items())
            start_node_id = start_node_props.get("unique_id", str(start_node_obj.id))
            end_node_id = end_node_props.get("unique_id", str(end_node_obj.id))

            # 获取关系属性
            rel_props = dict(value.items())
            # 从属性中移除内部可能存在的 graph_id
            rel_props.pop("graph_id", None)

            # 创建关系字典，确保 source 和 target 使用 nodes_map 中的 id
            rel_dict = {
                "source": start_node_id,
                "target": end_node_id,
                "type": rel_type,
                "properties": rel_props,
                "_internal_id": str(value.id),  # 保留关系内部ID用于去重
            }

            # 添加关系（简单去重）
            rel_key = f"{start_node_id}-{rel_type}-{end_node_id}"  # 简单标识符
            is_duplicate = False
            for existing_rel in relationships:
                # 更严格的去重：检查内部ID或source/target/type/props完全一致
                if existing_rel.get("_internal_id") == rel_dict["_internal_id"]:
                    is_duplicate = True
                    break
                # if existing_rel['source'] == start_node_id and \
                #    existing_rel['target'] == end_node_id and \
                #    existing_rel['type'] == rel_type and \
                #    existing_rel['properties'] == rel_props:
                #      is_duplicate = True
                #      break
            if not is_duplicate:
                relationships.append(rel_dict)

        # 处理路径对象 (neo4j.graph.Path)
        # isinstance(value, neo4j.graph.Path)
        elif (
            hasattr(value, "start_node")
            and hasattr(value, "end_node")
            and hasattr(value, "relationships")
        ):
            # 路径对象，处理其包含的所有节点和关系
            # 路径中的节点和关系对象会被上面对应的逻辑处理
            for node in value.nodes:
                self._extract_graph_elements_from_result(
                    node, nodes_map, relationships, added_relationship_keys
                )
            for rel in value.relationships:
                self._extract_graph_elements_from_result(
                    rel, nodes_map, relationships, added_relationship_keys
                )

        # 处理字典和嵌套结构 (通常是查询结果中的 Map 或 List<Map>)
        elif isinstance(value, dict):
            # 检查字典是否代表一个节点或关系（有时驱动会返回字典而非对象）
            # 启发式检查：如果字典包含像 'id', 'labels', 'properties' 或 'start', 'end', 'type' 这样的键
            if (
                "labels" in value
                and "properties" in value
                and ("id" in value or "_id" in value)
            ):  # 可能是节点字典
                node_internal_id = str(value.get("id", value.get("_id")))
                node_labels = value.get("labels", [])
                node_type = node_labels[0] if node_labels else "Unknown"
                node_props = value.get("properties", {})
                node_id = node_props.get("unique_id", node_internal_id)
                if node_id not in nodes_map:
                    node_props.pop("graph_id", None)
                    nodes_map[node_id] = {
                        "id": node_id,
                        "type": node_type,
                        "attributes": node_props,
                        "_internal_id": node_internal_id,
                    }
            elif (
                "type" in value
                and "properties" in value
                and ("start" in value or "startNodeElementId" in value)
                and ("end" in value or "endNodeElementId" in value)
            ):  # 可能是关系字典
                # 需要递归查找或确保 start/end 节点已存在
                # 这种处理比较复杂，最好是确保查询返回实际对象
                pass  # 暂时跳过复杂字典关系处理
            else:
                # 否则，递归处理字典中的每个值
                for sub_value in value.values():
                    self._extract_graph_elements_from_result(
                        sub_value, nodes_map, relationships, added_relationship_keys
                    )

        elif isinstance(value, list):
            # 递归处理列表中的每个元素
            for item in value:
                self._extract_graph_elements_from_result(
                    item, nodes_map, relationships, added_relationship_keys
                )

        # 处理查询结果记录 (neo4j.Result.Record)
        elif hasattr(value, "keys") and callable(value.get):  # 检查是否像 Record 对象
            # 处理查询结果记录中的每个字段值
            for key in value.keys():
                self._extract_graph_elements_from_result(
                    value.get(key), nodes_map, relationships, added_relationship_keys
                )

        # 其他标量类型 (int, float, str, bool) 则忽略

    def _generate_response_content(
        self, question, sql, result, reasoning=None, explanation=None
    ):
        """
        生成格式化的回复内容 - 此函数现在不再直接生成前端HTML，
        其主要作用是确定最终的解释文本（explanation）。
        在 ask_question 中，explanation 会被直接用作 content。
        这个函数可以保留用于未来可能的扩展，或者直接在 ask_question 中处理 explanation 逻辑。
        当前实现下，此函数不再被 ask_question 直接调用来构建复杂的 content 字符串。
        """
        if explanation:
            return explanation
        elif result is not None and isinstance(result, dict) and result.get("data"):
            count = len(result["data"])
            return f"查询完成，找到 {count} 条记录。"
        elif sql:  # 如果有SQL但没数据或解释
            return "SQL 语句已成功执行。"
        else:  # 默认的回退文本
            return "请求已处理。"

    def save_query_result(
        self, title, question, sql, result=None, visualization=None, description=None
    ):
        """保存查询结果"""
        try:
            # 确保转换NumPy类型
            if result is not None:
                result = convert_numpy_types(result)
            if visualization is not None:
                visualization = convert_numpy_types(visualization)

            return save_kgqa_query(
                title=title,
                question=question,
                sql=sql,
                result=result,
                visualization=visualization,
                description=description,
            )
        except Exception as e:
            logger.error(f"保存查询结果失败: {str(e)}")
            raise

    def get_all_saved_queries(self):
        """获取所有保存的查询"""
        return get_saved_kgqa_queries()

    def get_query(self, query_id):
        """获取保存的查询"""
        return get_saved_kgqa_query(query_id)

    def delete_query(self, query_id):
        """删除保存的查询"""
        return delete_saved_kgqa_query(query_id)

    def toggle_table_lock(self, session_id, lock_status=True):
        """锁定或解锁会话的数据表

        Args:
            session_id: 会话ID
            lock_status: True表示锁定，False表示解锁

        Returns:
            更新后的会话信息字典，如果会话不存在则返回None
        """
        try:
            # 获取会话信息
            session = get_kgqa_chat_session(session_id)
            if not session:
                return None

            # 检查锁定条件 - 只有当有表名时才能锁定
            if lock_status and not session.get("table_name"):
                logger.warning(f"会话 {session_id} 尝试锁定表，但没有设置表名")
                return update_kgqa_chat_session(session_id, is_table_locked=False)

            # 更新锁定状态
            return update_kgqa_chat_session(session_id, is_table_locked=lock_status)
        except Exception as e:
            logger.error(f"切换表锁定状态失败: {str(e)}")
            return None

    def add_cypher_question(self, question, query, result=None, answer=None):
        """添加一个新的Cypher问答对到图问答链

        Args:
            question: 用户问题
            query: 对应的Cypher查询
            result: 可选的查询结果
            answer: 可选的生成答案

        Returns:
            成功返回True，失败返回False
        """
        try:
            if self.graph_qa_chain is None:
                self.init_graph_qa_chain()

            self.graph_qa_chain.add_cypher_question(
                question=question, query=query, result=result, answer=answer
            )

            return True
        except Exception as e:
            logger.error(f"添加Cypher问答对失败: {str(e)}")
            return False

    def add_documentation(self, documentation):
        """添加文档到图问答链的向量存储

        Args:
            documentation: 文档列表，每个文档包含"text"和"metadata"

        Returns:
            成功返回True，失败返回False
        """
        try:
            if self.graph_qa_chain is None:
                self.init_graph_qa_chain()

            self.graph_qa_chain.add_documentation(documentation)
            return True
        except Exception as e:
            logger.error(f"添加文档失败: {str(e)}")
            return False

    def get_training_data(self):
        """获取训练数据"""
        try:
            if self.graph_qa_chain is None:
                self.init_graph_qa_chain()
            return self.graph_qa_chain.get_training_data()
        except Exception as e:
            logger.error(f"获取训练数据失败: {str(e)}")
            return pd.DataFrame()

    def delete_training_data(self, id: str) -> None:
        """删除指定的训练数据"""
        try:
            if self.graph_qa_chain is None:
                self.init_graph_qa_chain()
            return self.graph_qa_chain.delete_training_data(id)
        except Exception as e:
            logger.error(f"删除训练数据失败: {str(e)}")
            return False

    def clear_examples(self):
        """清除所有问答示例

        Returns:
            成功返回True，失败返回False
        """
        try:
            if self.graph_qa_chain is None:
                return False

            self.graph_qa_chain.clear_examples()
            return True
        except Exception as e:
            logger.error(f"清除问答示例失败: {str(e)}")
            return False

    def generate_cypher_and_result(
        self, question: str, graph_identifier: str, auto_train: bool = True
    ):
        """
        直接生成 Cypher、结果和解释，不依赖于特定的聊天会话 ID。

        Args:
            question (str): 用户的问题。
            graph_identifier (str): 目标知识图谱标识符 (例如 'graph_id:graph_name')。
            auto_train (bool): 是否启用自动训练。 Defaults to True.

        Returns:
            dict: 包含 status, cypher, result, explanation, (可能还有 visualization, thinking) 的字典。
                  如果失败，返回包含 status 和 message 的字典。
        """
        try:
            # 1. 确保 GraphQAChain 已初始化
            if self.graph_qa_chain is None:
                logger.info(
                    "GraphQAChain not initialized for agent tool, initializing now..."
                )
                # Use current config to initialize
                current_config = self.vanna_manager.get_config()
                self.init_graph_qa_chain(current_config)
                if self.graph_qa_chain is None:  # Check again after init attempt
                    raise ValueError("Failed to initialize GraphQAChain.")

            # 2. 准备问题上下文
            enhanced_question = question
            graph_id = None
            if graph_identifier:
                try:
                    # 解析标识符以获取 graph_id (如果需要)
                    parts = graph_identifier.split(":", 1)
                    if len(parts) == 2 and parts[0].isdigit():
                        graph_id = int(parts[0])
                        target_graph_id_prop = graph_id
                        self.graph_qa_chain.graph_schema = (
                            self.graph_qa_chain._get_graph_schema(target_graph_id_prop)
                        )
                        logger.info(
                            f"图谱Schema更新: {self.graph_qa_chain.graph_schema}"
                        )
                        graph_name = parts[1]
                        context = f"Querying knowledge graph '{graph_name}' (ID: {graph_id}). All relevant nodes and relationships have the property `graph_id: {graph_id}`. Your Cypher query should include appropriate `WHERE` conditions to filter this graph."
                        enhanced_question = f"{context}\nQuestion: {question}"
                    else:
                        # 如果格式不对，仅使用名称
                        graph_name = graph_identifier
                        context = f"Querying knowledge graph '{graph_name}'."
                        enhanced_question = f"{context}\nQuestion: {question}"
                except Exception as e:
                    logger.warning(
                        f"无法解析图谱标识符 '{graph_identifier}': {e}. 将按原样使用。"
                    )
                    context = f"Querying knowledge graph '{graph_identifier}'."
                    enhanced_question = f"{context}\nQuestion: {question}"

            # 3. 设置 Autotrain
            self.graph_qa_chain.set_autotrain(auto_train)
            logger.info(
                f"Calling GraphQAChain invoke for Agent tool: Question='{enhanced_question[:100]}...', AutoTrain={auto_train}"
            )

            # 4. 调用 GraphQAChain 核心方法
            invoke_args = {"query": enhanced_question}
            if graph_id is not None:
                invoke_args["target_graph_id"] = graph_id
            response = self.graph_qa_chain.invoke(invoke_args)

            # 5. 解析返回结果
            cypher = None
            result_data_raw = None
            result_data_processed = None
            explanation = response.get(
                "result", ""
            )  # Use LLM result as base explanation
            thinking = None
            logger.info(f"GraphQAChain response: {response}")
            if "intermediate_steps" in response:
                for step in response.get("intermediate_steps", []):
                    if "query" in step:
                        cypher = step["query"]
                    if "context" in step:
                        # context 这里是原始查询结果，可能不是 JSON 序列化的
                        result_data_raw = step["context"]
                        if (
                            result_data_raw
                            and result_data_raw
                            != "I couldn't find any relevant information in the database"
                        ):
                            try:
                                result_data_processed = convert_numpy_types(
                                    result_data_raw
                                )
                            except Exception as conv_err:
                                logger.error(
                                    f"为 Agent 工具处理图数据时出错: {conv_err}"
                                )
                                result_data_processed = {
                                    "error": "Error processing graph data after conversion",
                                    "details": str(conv_err),
                                }

            # 6. 提取思考过程
            if explanation:
                think_match = re.search(r"<think>(.*?)</think>", explanation, re.DOTALL)
                if think_match:
                    thinking = think_match.group(1).strip()
                    clean_explanation = re.sub(
                        r"<think>.*?</think>", "", explanation, flags=re.DOTALL
                    ).strip()
                    if clean_explanation:
                        explanation = clean_explanation

            # 7. 构建可视化数据 (可选，但对 Agent 可能意义不大，除非 Agent 要描述图)
            # visualization = None
            # if cypher and result_data_processed and graph_id is not None: # Need graph_id for visualization context
            #     try:
            #         # Call the visualization builder using the processed data
            #         # Note: _build_visualization_subgraph might need graph_id as an argument now
            #         graph_data = self._build_visualization_subgraph(graph_id, cypher, result_data_processed) # Pass processed data
            #         graph_data_converted = convert_numpy_types(graph_data) # Ensure final conversion
            #         if not self.graph_data_is_empty(graph_data_converted):
            #              visualization = graph_data_converted
            #     except Exception as viz_err:
            #         logger.warning(f"Agent tool: Failed to build KG visualization: {viz_err}")

            # 8. 返回结果
            return {
                "status": "success",
                "cypher": cypher,
                "result": result_data_processed,  # 返回处理过的、可序列化的结果
                # 'visualization': visualization,
                "explanation": explanation,
                "thinking": thinking,
            }

        except Exception as e:
            logger.error(f"Agent 工具执行 KG 查询失败: {str(e)}", exc_info=True)
            return {"status": "error", "message": f"执行 KG 查询时出错: {str(e)}"}

    def get_kg_schema_string(
        self, target_graph_id_prop=None, include_types=[], exclude_types=[]
    ) -> str:
        """
        获取当前初始化的知识图谱的 schema 字符串。

        Returns:
            str: 图谱 schema 字符串，或者未初始化或出错时的提示信息。
        """
        try:
            # **关键修复**: 主动刷新Neo4j schema缓存
            # 解决schema获取为空的时序问题
            if hasattr(self, "neo4j_graph") and self.neo4j_graph:
                try:
                    logger.info("主动刷新Neo4j schema缓存...")
                    # self.neo4j_graph.refresh_schema()
                    logger.info("Neo4j schema缓存刷新完成")
                except Exception as refresh_error:
                    logger.warning(f"主动刷新Neo4j schema缓存失败: {refresh_error}")

            # 1. 确保 GraphQAChain 已初始化 (它会持有 schema)
            if self.graph_qa_chain is None:
                logger.info(
                    "GraphQAChain not initialized for schema request, initializing now..."
                )
                current_config = self.vanna_manager.get_config()
                self.init_graph_qa_chain(
                    current_config, target_graph_id_prop, include_types, exclude_types
                )
                if self.graph_qa_chain is None:
                    return "错误: 无法初始化知识图谱组件以获取 schema。"

            # 2. 访问 schema 属性
            # .graph_schema 属性通常包含一个描述性的字符串
            if (
                hasattr(self.graph_qa_chain, "graph_schema")
                and self.graph_qa_chain.graph_schema
            ):
                if target_graph_id_prop:
                    schema_str = self.graph_qa_chain._get_graph_schema(
                        target_graph_id_prop, include_types, exclude_types
                    )
                else:
                    schema_str = self.graph_qa_chain.graph_schema
                logger.info(f"获取到知识图谱 Schema:\n{schema_str}")
                # 返回 schema，可能添加一些上下文
                return f"当前知识图谱的 Schema 结构如下:\n{schema_str}"
            else:
                logger.warning(
                    "graph_qa_chain 已初始化，但 graph_schema 属性为空或不存在。"
                )
                # 尝试重新获取 schema?
                try:
                    # self.neo4j_graph.refresh_schema()
                    schema_str_refreshed = self.neo4j_graph.schema
                    if schema_str_refreshed:
                        logger.info("通过 refresh_schema 获取到 Schema。")
                        # 更新 chain 中的 schema (如果可能)
                        if hasattr(self.graph_qa_chain, "graph"):
                            self.graph_qa_chain.graph.schema = schema_str_refreshed
                        return f"当前知识图谱的 Schema 结构如下 (刷新后):\n{schema_str_refreshed}"
                    else:
                        return "错误: 无法获取知识图谱的 schema 信息。"
                except Exception as refresh_err:
                    logger.error(f"尝试刷新 KG schema 失败: {refresh_err}")
                    return "错误: 无法获取知识图谱的 schema 信息。"

        except Exception as e:
            logger.error(f"获取知识图谱 schema 时出错: {str(e)}", exc_info=True)
            return f"错误: 获取知识图谱 schema 时发生错误: {str(e)}"

    def create_graph_id_indexes(self):
        """
        手动创建graph_id索引以提高查询性能

        Returns:
            dict: 包含创建结果的字典
        """
        try:
            # 确保GraphQAChain已初始化
            if self.graph_qa_chain is None:
                self.init_graph_qa_chain()

            if self.graph_qa_chain is None:
                return {
                    "status": "error",
                    "message": self._get_localized_message(
                        "GraphQAChain未能初始化", "GraphQAChain failed to initialize"
                    ),
                }

            # 调用创建索引方法
            created_indexes = self.graph_qa_chain.create_graph_id_indexes()

            return {
                "status": "success",
                "message": f"索引创建完成",
                "created_indexes": created_indexes,
                "total_created": len(created_indexes),
            }

        except Exception as e:
            logger.error(f"手动创建graph_id索引失败: {str(e)}")
            return {"status": "error", "message": f"创建索引失败: {str(e)}"}

    def get_index_status(self):
        """
        获取当前数据库中graph_id相关索引的状态

        Returns:
            dict: 包含索引状态信息的字典
        """
        try:
            if not hasattr(self, "neo4j_driver") or not self.neo4j_driver:
                return {
                    "status": "error",
                    "message": self._get_localized_message(
                        "Neo4j驱动未初始化", "Neo4j driver not initialized"
                    ),
                }

            with self.neo4j_driver.session() as session:
                # 查询所有包含graph_id的索引
                index_query = """
                SHOW INDEXES 
                WHERE properties = ['graph_id']
                RETURN name, labelsOrTypes, properties, state, type
                """

                result = session.run(index_query)
                indexes = []

                for record in result:
                    indexes.append(
                        {
                            "name": record.get("name"),
                            "labels": record.get("labelsOrTypes", []),
                            "properties": record.get("properties", []),
                            "state": record.get("state"),
                            "type": record.get("type"),
                        }
                    )

                # 统计信息
                total_indexes = len(indexes)
                online_indexes = len(
                    [idx for idx in indexes if idx.get("state") == "ONLINE"]
                )

                return {
                    "status": "success",
                    "total_indexes": total_indexes,
                    "online_indexes": online_indexes,
                    "indexes": indexes,
                }

        except Exception as e:
            logger.error(f"获取索引状态失败: {str(e)}")
            return {"status": "error", "message": f"获取索引状态失败: {str(e)}"}

    def drop_graph_id_indexes(self, confirm=False):
        """
        删除graph_id相关的索引（慎用）

        Args:
            confirm: 是否确认删除，防止误操作

        Returns:
            dict: 包含删除结果的字典
        """
        if not confirm:
            return {"status": "error", "message": "请设置confirm=True以确认删除操作"}

        try:
            if not hasattr(self, "neo4j_driver") or not self.neo4j_driver:
                return {
                    "status": "error",
                    "message": self._get_localized_message(
                        "Neo4j驱动未初始化", "Neo4j driver not initialized"
                    ),
                }

            with self.neo4j_driver.session() as session:
                # 首先获取所有graph_id索引
                index_query = """
                SHOW INDEXES 
                WHERE properties = ['graph_id']
                RETURN name
                """

                result = session.run(index_query)
                index_names = [record["name"] for record in result]

                # 删除索引
                dropped_indexes = []
                for index_name in index_names:
                    try:
                        drop_query = f"DROP INDEX {index_name}"
                        session.run(drop_query)
                        dropped_indexes.append(index_name)
                        logger.info(f"成功删除索引: {index_name}")
                    except Exception as e:
                        logger.error(f"删除索引 {index_name} 失败: {str(e)}")

                return {
                    "status": "success",
                    "message": f"删除了 {len(dropped_indexes)} 个索引",
                    "dropped_indexes": dropped_indexes,
                }

        except Exception as e:
            logger.error(f"删除graph_id索引失败: {str(e)}")
            return {"status": "error", "message": f"删除索引失败: {str(e)}"}


# 创建全局问答管理器实例
kgqa_manager = KGQAManager()


# 配置更新回调函数
def _on_kgqa_config_update(new_config_dict):
    """知识图谱QA管理器配置更新回调"""
    try:
        # 重新初始化图数据库连接和LLM
        logger.info("知识图谱QA管理器配置已更新，重新初始化连接...")
        kgqa_manager.init_graph_qa_chain(new_config_dict)
        logger.info("知识图谱QA管理器配置更新完成")
    except Exception as e:
        logger.error(f"知识图谱QA管理器配置更新失败: {str(e)}")


# 注册配置更新回调
from backend.services.vanna_service import vanna_manager

vanna_manager.register_config_callback(_on_kgqa_config_update)
