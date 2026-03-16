import logging
import json
import asyncio  # Added for sleep
from backend.utils.db_utils import (
    create_agent_task_session,
    update_agent_task_session,
    get_agent_task_session,
    get_agent_task_sessions,
    delete_agent_task_session,
    get_agent_task_messages,
    add_agent_task_message,
)
from backend.manager.qa_manager import qa_manager
from backend.manager.kgqa_manager import kgqa_manager
from backend.services.vanna_service import vanna_manager  # 用于获取配置
from backend.utils.token_tracking import global_token_tracker, create_token_callback
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools import Tool
from datetime import datetime
from langchain_core.exceptions import OutputParserException
from backend.utils.openai_compat import normalize_openai_base_url

TOOL_TRACKING_AVAILABLE = True

logger = logging.getLogger(__name__)


class AgentTaskManager:
    """Agent 任务管理器"""

    def __init__(self):
        self.vanna_manager = vanna_manager
        self.llm = None
        self._initialize_llm()
        # Add a small delay for stop request checks to avoid hammering DB
        self.stop_check_interval = 0.5  # seconds

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

    def _initialize_llm(self):
        """根据配置初始化 LLM"""
        try:
            # 创建token跟踪callback
            token_callback = None
            try:
                token_callback = create_token_callback("agent")
                logger.info("Agent token跟踪callback已创建")
            except Exception:
                logger.info("无法导入Agent token跟踪callback，将跳过token跟踪")

            callbacks = [token_callback] if token_callback else []

            # Try to reuse the LLM from KGQA manager first as it might be more capable
            # Fallback to Vanna/config based initialization
            config = self.vanna_manager.get_config()
            model_config = config.get("model", {})
            model_type = model_config.get("type", "ollama")
            temperature = model_config.get("temperature", 0.7)
            logger.info(
                f"AgentTaskManager: Initializing LLM based on main config (type: {model_type})"
            )

            if model_type == "ollama":
                from langchain_ollama import ChatOllama
                import httpx

                # 设置HTTP客户端参数，提升稳定性
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
                    logger.info("AgentTaskManager: HTTP/2 client created successfully")
                except ImportError as e:
                    # 如果没有h2包，回退到HTTP/1.1
                    logger.warning(
                        f"AgentTaskManager: HTTP/2 not available ({str(e)}), falling back to HTTP/1.1"
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

                ollama_model = model_config.get("ollama_model", "gemma3:27b-it-q8_0")
                self.llm = ChatOllama(
                    base_url=model_config.get("ollama_url", "http://localhost:11434"),
                    model=ollama_model,
                    temperature=min(
                        temperature, 0.3
                    ),  # 降低temperature以提高格式遵循性
                    extract_reasoning=True,
                    num_ctx=model_config.get("num_ctx", 25600),
                    client=http_client,  # 使用自定义HTTP客户端
                    request_timeout=60.0,  # 总体请求超时
                    stop_sequences=[
                        "<stop>",
                        "Human:",
                        "Assistant:",
                        "\n\nQuestion:",
                        "\n\nThought: I need",  # 防止重复思考循环
                        "Final Answer: I cannot",  # 防止过早终止
                    ],  # 添加更多停止序列来控制格式
                    num_retries=2,  # 添加自动重试
                    callbacks=callbacks,  # 添加token跟踪callback
                    format="",  # 确保不使用特殊格式
                    top_p=0.9,  # 添加top_p控制
                    repeat_penalty=1.1,  # 减少重复内容
                )
                logger.info(
                    f"AgentTaskManager: Initialized Ollama LLM: {ollama_model} with enhanced format control"
                )
            elif model_type == "openai":
                from langchain_openai import ChatOpenAI

                openai_model = model_config.get("model_name", "gpt-3.5-turbo")
                frequency_penalty = 0.1
                presence_penalty = 0.1
                # 检查是否是DeepSeek模型，应用特殊配置
                is_deepseek = "deepseek" in openai_model.lower()
                is_qwen = "qwen" in openai_model.lower()
                final_temperature = (
                    min(temperature, 0.1)
                    if is_deepseek or is_qwen
                    else min(temperature, 0.3)
                )
                if is_deepseek or is_qwen:
                    # DeepSeek模型使用更严格的top_p和惩罚参数
                    frequency_penalty = 0.2
                    presence_penalty = 0.2
                is_gemini = "gemini" in openai_model.lower()
                if is_gemini:
                    # Gemini模型需要特殊处理 - 移除不支持的参数
                    frequency_penalty = None
                    presence_penalty = None
                    logger.info(
                        f"AgentTaskManager: Detected Gemini model {openai_model}, using Gemini-optimized configuration"
                    )

                    self.llm = ChatOpenAI(
                        model=openai_model,
                        temperature=final_temperature,
                        api_key=model_config.get("api_key"),
                        base_url=normalize_openai_base_url(
                            model_config.get("api_base", "https://api.openai.com/v1")
                        ),
                        request_timeout=60.0,  # 请求超时
                        max_retries=2,  # 自动重试
                        callbacks=callbacks,  # 添加token跟踪callback
                        # 移除Gemini不支持的参数
                        # max_tokens=model_config.get('num_ctx', 25600),  # Gemini可能不支持
                        # top_p=0.9,  # Gemini可能不支持
                        # frequency_penalty=None,  # Gemini不支持
                        # presence_penalty=None   # Gemini不支持
                    )
                else:
                    self.llm = ChatOpenAI(
                        model=openai_model,
                        temperature=final_temperature,  # DeepSeek使用更低的温度
                        api_key=model_config.get("api_key"),
                        base_url=normalize_openai_base_url(
                            model_config.get("api_base", "https://api.openai.com/v1")
                        ),
                        max_tokens=model_config.get("num_ctx", 25600),
                        request_timeout=60.0,  # 请求超时
                        max_retries=2,  # 自动重试
                        callbacks=callbacks,  # 添加token跟踪callback
                        top_p=0.7
                        if is_deepseek or is_qwen
                        else 0.9,  # DeepSeek使用更严格的top_p
                        frequency_penalty=frequency_penalty,  # DeepSeek增加频率惩罚
                        presence_penalty=presence_penalty,  # DeepSeek增加存在惩罚
                    )
                logger.info(
                    f"AgentTaskManager: Initialized OpenAI-compatible LLM: {openai_model} (DeepSeek optimized: {is_deepseek})"
                )
            else:
                logger.error(
                    f"AgentTaskManager: Unsupported LLM type in config: {model_type}"
                )
                raise ValueError(f"Unsupported LLM type: {model_type}")
        except Exception as e:
            logger.error(
                f"AgentTaskManager: Failed to initialize LLM: {e}", exc_info=True
            )
            self.llm = None  # Ensure LLM is None on failure

    def create_task_session(
        self,
        name=None,
        user_goal=None,
        sql_database_name=None,
        sql_table_name=None,
        kg_graph_name=None,
        model_name=None,
        temperature=0.7,
        max_iterations=20,
    ):
        """
        创建新的Agent任务会话, 直接关联数据库表和知识图谱.

        Args:
            name (str, optional): 任务名称. Defaults to None.
            user_goal (str, optional): 用户的任务目标. Defaults to None.
            sql_database_name (str, optional): 关联的SQL数据库名称. Defaults to None.
            sql_table_name (str, optional): 关联的SQL数据表名称. Defaults to None.
            kg_graph_name (str, optional): 关联的知识图谱标识符 (例如 'graph_id:graph_name'). Defaults to None.
            model_name (str, optional): 指定使用的模型名称. Defaults to None.
            temperature (float, optional): 模型温度. Defaults to 0.7.
            max_iterations (int, optional): Agent最大迭代次数. Defaults to 20.

        Returns:
            int: 创建的任务会话ID.
        """
        if not name:
            name = f"新任务 {datetime.now().strftime('%Y-%m-%d %H:%M')}"

        # --- Determine Database Name --- START
        final_sql_database_name = sql_database_name
        if sql_table_name and not sql_database_name:
            try:
                config = self.vanna_manager.get_config()
                db_config = config.get("database", {})
                configured_db_name = db_config.get("database_name")
                if configured_db_name:
                    final_sql_database_name = configured_db_name
                    logger.info(
                        f"Agent Task: Using configured database '{configured_db_name}' for table '{sql_table_name}'"
                    )
                else:
                    logger.warning(
                        f"Agent Task: Table '{sql_table_name}' provided without DB name, and no database_name found in Vanna config."
                    )
            except Exception as e:
                logger.warning(
                    f"Agent Task: Error getting configured DB name: {e}. DB name will remain unset."
                )
        # --- Determine Database Name --- END

        # 获取模型名称和温度，优先使用传入参数，否则使用全局配置
        final_model_name = model_name
        final_temperature = temperature
        if not final_model_name:
            try:
                # Try getting from config if not provided
                config = self.vanna_manager.get_config()
                model_config = config.get("model", {})
                model_type = model_config.get("type", "ollama")
                if model_type == "ollama":
                    final_model_name = model_config.get("ollama_model")
                elif model_type == "openai":
                    final_model_name = model_config.get("model_name")
                # Set default temperature from config if not provided
                if final_temperature is None:
                    final_temperature = model_config.get("temperature", 0.7)
            except Exception as e:
                logger.warning(
                    f"Could not determine model name or temp from config: {e}. Using defaults."
                )
                final_model_name = (
                    final_model_name or "default_model"
                )  # Provide a fallback
                final_temperature = final_temperature or 0.7

        # Ensure max_iterations is a reasonable integer
        try:
            final_max_iterations = (
                int(max_iterations) if max_iterations is not None else 20
            )
            if final_max_iterations <= 0:
                final_max_iterations = 20  # Default if invalid
        except (ValueError, TypeError):
            final_max_iterations = 20

        task_session_id = create_agent_task_session(
            name=name,
            user_goal=user_goal,
            sql_database_name=final_sql_database_name,
            sql_table_name=sql_table_name,
            kg_graph_name=kg_graph_name,
            model_name=final_model_name,
            temperature=final_temperature,
            sql_session_id_ref=None,
            kg_session_id_ref=None,
            max_iterations=final_max_iterations,  # Store max_iterations
            stop_requested=False,  # Initialize stop_requested flag
        )
        logger.info(
            f"Created agent task {task_session_id} for goal: '{user_goal}', SQL Table: '{sql_database_name}.{sql_table_name}', KG: '{kg_graph_name}', Max Iterations: {final_max_iterations}"
        )
        return task_session_id

    def get_task_session(self, session_id):
        """获取Agent任务会话信息"""
        return get_agent_task_session(session_id)

    def get_all_task_sessions(self):
        """获取所有Agent任务会话"""
        return get_agent_task_sessions()

    def update_task_session(self, session_id, **kwargs):
        """更新Agent任务会话信息"""
        # Can be used to update status ('running', 'completed', 'failed')
        # Prevent changing the core table/graph associations after creation? Add logic if needed.
        return update_agent_task_session(session_id, **kwargs)

    def delete_task_session(self, session_id):
        """删除Agent任务会话"""
        return delete_agent_task_session(session_id)

    def get_task_messages(self, session_id):
        """获取任务会话的所有消息/步骤"""
        return get_agent_task_messages(session_id)

    def _add_task_message(
        self,
        session_id,
        role,
        content,
        message_type,
        tool_name=None,
        tool_input=None,
        tool_output=None,
    ):
        """内部辅助方法，用于添加消息并处理可能的错误"""
        try:
            add_agent_task_message(
                session_id,
                role,
                content,
                message_type,
                tool_name,
                tool_input,
                tool_output,
            )
        except Exception as e:
            logger.error(
                f"Failed to add message to task {session_id}: {e}", exc_info=True
            )
            # Decide if we should raise this or just log it

    def request_task_stop(self, session_id):
        """请求停止一个正在运行的任务"""
        logger.info(f"Received stop request for task {session_id}")
        session = self.get_task_session(session_id)
        if not session:
            logger.warning(f"Stop request failed: Task {session_id} not found.")
            return False, f"Task {session_id} not found."

        current_status = session.get("status")
        if current_status != "running":
            logger.warning(
                f"Stop request ignored: Task {session_id} is not running (status: {current_status})."
            )
            # Return success=True because the task is already not running, the goal is achieved.
            # Or return False if you want to signal the request wasn't needed. Let's use False.
            return (
                False,
                f"Task {session_id} is not currently running (status: {current_status}).",
            )

        # Check if already requested
        if session.get("stop_requested", False):
            logger.info(f"Stop request for task {session_id} already processed.")
            return True, f"Stop request for task {session_id} was already sent."

        try:
            # Update the flag in the database
            updated = self.update_task_session(session_id, stop_requested=True)
            if updated:
                logger.info(
                    f"Successfully set stop_requested flag for task {session_id}."
                )
                return (
                    True,
                    f"Stop request sent for task {session_id}. The task will stop shortly.",
                )
            else:
                # This case might happen if update_agent_task_session fails for some reason
                logger.error(
                    f"Failed to set stop_requested flag for task {session_id} during update."
                )
                return (
                    False,
                    f"Failed to update stop request flag for task {session_id}.",
                )
        except Exception as e:
            logger.error(
                f"Error setting stop_requested flag for task {session_id}: {e}",
                exc_info=True,
            )
            return (
                False,
                f"An error occurred while processing the stop request for task {session_id}.",
            )

    async def run_task(self, task_session_id):
        """运行Agent任务，支持停止请求和最大迭代次数"""
        # 初始化状态变量
        final_answer_yielded = False
        stop_requested_flag = False
        current_iteration = 0
        last_status_check_time = datetime.now()

        try:
            # 检查LLM是否初始化
            self._initialize_llm()
            if self.llm is None:
                errmsg = "Agent LLM not initialized. Cannot run task."
                logger.error(f"{errmsg} Task ID: {task_session_id}.")
                self._add_task_message(
                    task_session_id, "system", errmsg, message_type="error"
                )
                update_agent_task_session(task_session_id, status="failed")
                yield {"type": "error", "content": errmsg}
                return

            # 获取会话信息
            session = self.get_task_session(task_session_id)
            if not session:
                errmsg = f"Task session {task_session_id} not found."
                logger.error(errmsg)
                yield {"type": "error", "content": errmsg}
                return

            # 获取任务配置
            user_goal = session.get("user_goal")
            sql_db = session.get("sql_database_name")
            sql_table = session.get("sql_table_name")
            kg_graph = session.get("kg_graph_name")
            task_max_iterations = session.get("max_iterations", 20)

            logger.info(
                f"Starting agent task {task_session_id} for goal: '{user_goal}' with context SQL='{sql_db}.{sql_table}', KG='{kg_graph}', Max Iterations: {task_max_iterations}"
            )

            # 更新任务状态为运行中
            update_agent_task_session(
                task_session_id,
                status="running",
                stop_requested=False,
                current_iteration=0,
                last_status_update=datetime.now().isoformat(),
            )
            yield {"type": "status", "content": "running"}

            # 创建工具
            tools = []
            # SQL工具创建
            if sql_db and sql_table:
                # Define the function within the scope where sql_db and sql_table are available
                def sql_query_func(query: str) -> str:
                    # Make sure qa_manager is available
                    if not qa_manager:
                        logger.error(
                            f"Agent Task {task_session_id}: qa_manager is not available."
                        )
                        error_msg = self._get_localized_message(
                            "错误: SQL查询工具配置不正确（问答管理器不可用）。",
                            "Error: SQL Query Tool is not configured correctly (QA Manager unavailable).",
                        )
                        # 记录工具使用失败
                        if TOOL_TRACKING_AVAILABLE:
                            global_token_tracker.add_tool_usage(
                                "sql_database_query", success=False
                            )
                        # 更新任务状态以反映工具错误
                        try:
                            update_agent_task_session(
                                task_session_id,
                                error_message=error_msg,
                                last_status_update=datetime.now().isoformat(),
                            )
                        except Exception as e:
                            logger.error(
                                f"Agent Task {task_session_id}: Failed to update status on tool error: {e}"
                            )
                        return error_msg
                    try:
                        logger.info(
                            f"Agent Task {task_session_id}: Calling qa_manager.generate_sql_and_result for table '{sql_db}.{sql_table}' with question: '{query}'"
                        )
                        # *** MODIFICATION: Call the new direct method ***
                        result = qa_manager.generate_sql_and_result(
                            question=query, database_name=sql_db, table_name=sql_table
                        )

                        # Process the result (same logic as before)
                        if result.get("status") == "success":
                            # 记录工具使用成功
                            if TOOL_TRACKING_AVAILABLE:
                                global_token_tracker.add_tool_usage(
                                    "sql_database_query", success=True
                                )

                            content = result.get(
                                "explanation",
                                self._get_localized_message(
                                    "查询已执行。", "Query executed."
                                ),
                            )
                            sql = result.get(
                                "sql",
                                self._get_localized_message(
                                    "未生成SQL。", "No SQL generated."
                                ),
                            )
                            res_data = result.get("result")
                            res_summary = self._get_localized_message(
                                "未返回数据。", "No data returned."
                            )
                            if (
                                res_data
                                and isinstance(res_data, dict)
                                and res_data.get("data")
                            ):
                                res_summary = self._get_localized_message(
                                    f"返回了 {len(res_data['data'])} 行数据。",
                                    f"Returned {len(res_data['data'])} rows.",
                                )
                            elif isinstance(res_data, list):
                                res_summary = self._get_localized_message(
                                    f"返回了 {len(res_data)} 个项目。",
                                    f"Returned {len(res_data)} items.",
                                )
                            elif sql and not res_data:
                                res_summary = self._get_localized_message(
                                    "查询执行成功但未返回数据。",
                                    "Query executed successfully but returned no data.",
                                )

                            output = f"Executed SQL:\n```sql\n{sql}\n```\nResult: {res_summary}\nExplanation: {content}"
                            logger.info(
                                f"Agent Task {task_session_id}: SQLQueryTool result: {output[:300]}..."
                            )
                            # Add result to agent messages for context? Maybe just summary.
                            self._add_task_message(
                                task_session_id,
                                "tool",
                                output,
                                "observation",
                                tool_name="sql_database_query",
                            )
                            return output  # Return the formatted string to the agent scratchpad
                        else:
                            # 记录工具使用失败
                            if TOOL_TRACKING_AVAILABLE:
                                global_token_tracker.add_tool_usage(
                                    "sql_database_query", success=False
                                )

                            errmsg = self._get_localized_message(
                                f"执行SQL查询时出错: {result.get('message', '未知错误')}",
                                f"Error executing SQL query: {result.get('message', 'Unknown error')}",
                            )
                            logger.error(
                                f"Agent Task {task_session_id}: SQLQueryTool error: {errmsg}"
                            )
                            # Add error as observation
                            self._add_task_message(
                                task_session_id,
                                "tool",
                                errmsg,
                                "observation",
                                tool_name="sql_database_query",
                            )
                            # 更新任务状态以反映SQL错误（但不终止任务）
                            try:
                                update_agent_task_session(
                                    task_session_id,
                                    last_tool_error=errmsg,
                                    last_status_update=datetime.now().isoformat(),
                                )
                            except Exception as e:
                                logger.error(
                                    f"Agent Task {task_session_id}: Failed to update status on SQL error: {e}"
                                )
                            return errmsg
                    except Exception as e:
                        # 记录工具使用失败（异常情况）
                        if TOOL_TRACKING_AVAILABLE:
                            global_token_tracker.add_tool_usage(
                                "sql_database_query", success=False
                            )

                        logger.error(
                            f"Error in SQL Query Tool execution for task {task_session_id}: {e}",
                            exc_info=True,
                        )
                        errmsg = self._get_localized_message(
                            f"执行SQL查询时出错: {str(e)}",
                            f"Error executing SQL query: {str(e)}",
                        )
                        self._add_task_message(
                            task_session_id,
                            "tool",
                            errmsg,
                            "observation",
                            tool_name="sql_database_query",
                        )
                        # 更新任务状态以反映工具执行异常
                        try:
                            update_agent_task_session(
                                task_session_id,
                                last_tool_error=errmsg,
                                last_status_update=datetime.now().isoformat(),
                            )
                        except Exception as update_e:
                            logger.error(
                                f"Agent Task {task_session_id}: Failed to update status on tool exception: {update_e}"
                            )
                        return errmsg

                tools.append(
                    Tool(
                        name="sql_database_query",
                        func=sql_query_func,  # Use the updated function
                        description=f"Query the SQL database table '{sql_table}' in database '{sql_db}'. Use natural language questions to explore data structure, discover available values, or query specific information.",
                    )
                )
                logger.info(
                    f"AgentTaskManager: Added SQLQueryTool for task {task_session_id} targeting table '{sql_db}.{sql_table}'"
                )

            # KG工具创建
            if kg_graph:
                # Define the function within the scope where kg_graph is available
                def kg_query_func(query: str) -> str:
                    # Make sure kgqa_manager is available
                    if not kgqa_manager:
                        logger.error(
                            f"Agent Task {task_session_id}: kgqa_manager is not available."
                        )
                        error_msg = self._get_localized_message(
                            "错误: 知识图谱工具配置不正确（KGQA管理器不可用）。",
                            "Error: Knowledge Graph Tool is not configured correctly (KGQA Manager unavailable).",
                        )
                        # 记录工具使用失败
                        if TOOL_TRACKING_AVAILABLE:
                            global_token_tracker.add_tool_usage(
                                "knowledge_graph_query", success=False
                            )
                        # 更新任务状态以反映工具错误
                        try:
                            update_agent_task_session(
                                task_session_id,
                                error_message=error_msg,
                                last_status_update=datetime.now().isoformat(),
                            )
                        except Exception as e:
                            logger.error(
                                f"Agent Task {task_session_id}: Failed to update status on tool error: {e}"
                            )
                        return error_msg
                    try:
                        logger.info(
                            f"Agent Task {task_session_id}: Calling kgqa_manager.generate_cypher_and_result for graph '{kg_graph}' with question: '{query}'"
                        )
                        # *** MODIFICATION: Call the new direct method ***
                        kg_result = kgqa_manager.generate_cypher_and_result(
                            question=query,
                            graph_identifier=kg_graph,
                            auto_train=True,  # Or get from config/task settings if needed
                        )

                        # Process the result (same logic as before)
                        if kg_result.get("status") == "success":
                            # 记录工具使用成功
                            if TOOL_TRACKING_AVAILABLE:
                                global_token_tracker.add_tool_usage(
                                    "knowledge_graph_query", success=True
                                )

                            content = kg_result.get(
                                "explanation",
                                self._get_localized_message(
                                    "知识图谱查询已执行。", "KG Query executed."
                                ),
                            )
                            cypher = kg_result.get(
                                "cypher",
                                self._get_localized_message(
                                    "未生成Cypher查询。", "No Cypher generated."
                                ),
                            )
                            res_data = kg_result.get(
                                "result"
                            )  # This is already processed/serializable
                            res_summary = self._get_localized_message(
                                "未返回具体数据点。",
                                "No specific data points returned.",
                            )
                            if isinstance(res_data, list) and len(res_data) > 0:
                                res_summary = self._get_localized_message(
                                    f"返回了 {len(res_data)} 个结果项/节点/路径。",
                                    f"Returned {len(res_data)} result items/nodes/paths.",
                                )
                            elif isinstance(res_data, dict) and res_data:
                                # Check for specific error structure from conversion
                                if "error" in res_data:
                                    res_summary = self._get_localized_message(
                                        f"处理图数据时出错: {res_data.get('details', '未知')}",
                                        f"Error processing graph data: {res_data.get('details', 'Unknown')}",
                                    )
                                    content = self._get_localized_message(
                                        f"查询可能已运行，但结果无法处理。{content}",
                                        f"Query may have run, but results could not be processed. {content}",
                                    )
                                else:
                                    res_summary = self._get_localized_message(
                                        "返回了一个结果对象/字典。",
                                        "Returned a result object/dictionary.",
                                    )
                            elif cypher and not res_data:
                                res_summary = self._get_localized_message(
                                    "查询执行成功但未返回具体数据。",
                                    "Query executed successfully but returned no specific data.",
                                )

                            output = f"Executed Cypher:\n```cypher\n{cypher}\n```\nResult: {res_summary}\nExplanation: {content}"
                            logger.info(
                                f"Agent Task {task_session_id}: KGQueryTool result: {output[:300]}..."
                            )
                            self._add_task_message(
                                task_session_id,
                                "tool",
                                output,
                                "observation",
                                tool_name="knowledge_graph_query",
                            )
                            return output  # Return formatted string to agent
                        else:
                            # 记录工具使用失败
                            if TOOL_TRACKING_AVAILABLE:
                                global_token_tracker.add_tool_usage(
                                    "knowledge_graph_query", success=False
                                )

                            errmsg = self._get_localized_message(
                                f"执行知识图谱查询时出错: {kg_result.get('message', '未知错误')}",
                                f"Error executing KG query: {kg_result.get('message', 'Unknown error')}",
                            )
                            logger.error(
                                f"Agent Task {task_session_id}: KGQueryTool error: {errmsg}"
                            )
                            self._add_task_message(
                                task_session_id,
                                "tool",
                                errmsg,
                                "observation",
                                tool_name="knowledge_graph_query",
                            )
                            # 更新任务状态以反映KG错误（但不终止任务）
                            try:
                                update_agent_task_session(
                                    task_session_id,
                                    last_tool_error=errmsg,
                                    last_status_update=datetime.now().isoformat(),
                                )
                            except Exception as e:
                                logger.error(
                                    f"Agent Task {task_session_id}: Failed to update status on KG error: {e}"
                                )
                            return errmsg
                    except Exception as e:
                        # 记录工具使用失败（异常情况）
                        if TOOL_TRACKING_AVAILABLE:
                            global_token_tracker.add_tool_usage(
                                "knowledge_graph_query", success=False
                            )

                        logger.error(
                            f"Error in KG Query Tool execution for task {task_session_id}: {e}",
                            exc_info=True,
                        )
                        errmsg = self._get_localized_message(
                            f"执行知识图谱查询时出错: {str(e)}",
                            f"Error executing KG query: {str(e)}",
                        )
                        self._add_task_message(
                            task_session_id,
                            "tool",
                            errmsg,
                            "observation",
                            tool_name="knowledge_graph_query",
                        )
                        # 更新任务状态以反映工具执行异常
                        try:
                            update_agent_task_session(
                                task_session_id,
                                last_tool_error=errmsg,
                                last_status_update=datetime.now().isoformat(),
                            )
                        except Exception as update_e:
                            logger.error(
                                f"Agent Task {task_session_id}: Failed to update status on tool exception: {update_e}"
                            )
                        return errmsg

                tools.append(
                    Tool(
                        name="knowledge_graph_query",
                        func=kg_query_func,  # Use the updated function
                        description=f"Query the knowledge graph '{kg_graph}'. Use natural language questions to explore graph structure, discover available entities and relationships, or query specific information.",
                    )
                )
                logger.info(
                    f"AgentTaskManager: Added KGQueryTool for task {task_session_id} targeting graph '{kg_graph}'"
                )

            # 检查工具和模式
            if not tools:
                warn_msg = self._get_localized_message(
                    "警告: 未为此任务配置特定的SQL表或知识图谱。代理将缺乏数据查询和模式查看功能。",
                    "Warning: No specific SQL table or Knowledge Graph was configured for this task. The agent will lack data query and schema viewing capabilities.",
                )  # Updated warning
                logger.warning(f"Agent task {task_session_id}: {warn_msg}")
                self._add_task_message(
                    task_session_id, "system", warn_msg, message_type="warning"
                )
                yield {"type": "warning", "content": warn_msg}
                # If you want the agent to fail here, you could raise an error or return
                # return

            if kg_graph and ":" in kg_graph:
                parts = kg_graph.split(":", 1)
                if len(parts) == 2 and parts[0].isdigit():
                    graph_id = int(parts[0])
            else:
                graph_id = None

            # 直接得到sql和kg的schema
            if sql_table:
                sql_schema_info = qa_manager.get_sql_schema_for_query(sql_table)
            else:
                sql_schema_info = None
            if kg_graph:
                graph_schema_str = kgqa_manager.get_kg_schema_string(graph_id)
            else:
                graph_schema_str = None
            logger.info(
                f"Agent Task {task_session_id}: SQL_TABLE: {sql_table}, KG_GRAPH: {kg_graph}, SQL Schema: {sql_schema_info}, KG Schema: {graph_schema_str}"
            )
            # 创建提示模板和Agent
            prompt_template = """You are a systematic data analysis expert. 

LANGUAGE ADAPTATION:
- If question is in Chinese, use Chinese throughout
- If question is in English, use English throughout
- Maintain consistency in your chosen language

SYSTEMATIC EXPLORATION APPROACH:
Phase 1 - ALWAYS START with data discovery:
- First Action MUST be to discover what data types/columns/entities exist
- Second Action MUST be to sample actual values in relevant fields
- Only after discovery, search for specific entities

Phase 2 - Evidence-based querying:
- Use only values confirmed through discovery
- Design queries based on actual data patterns found

Phase 3 - Comprehensive analysis:
- Provide context about data limitations discovered
- Suggest alternatives if target entities don't exist

ITERATION BUDGET AWARENESS:
- Maximum allowed action iterations: {task_max_iterations}
- Current iteration at start: {current_iteration}
- Remaining iterations at start: {remaining_iterations}
- Plan your tool usage to finish early and reserve at least 2 iterations for synthesis and final answer
- If information is still incomplete near the budget limit, stop exploring and provide the best possible final answer based on collected evidence

EXPLORATION EXAMPLES:

For Actor/Movie queries:
✅ Correct First Actions:
Thought: I need to discover what data is available before searching for specific actors.
Action: sql_database_query
Action Input: What columns exist in this table and what types of data are available?

❌ Wrong First Actions:
Action: sql_database_query  
Action Input: Find movies starring [specific actor]

DATA CONTEXT:
- SQL Database: '{sql_db}' 
- SQL Table: '{sql_table}'
- Knowledge Graph: '{kg_graph}'

SQL Schema: {sql_schema_info}
KG Schema: {graph_schema_str}

Remember: 
1. FIRST action must ALWAYS be data discovery
2. NEVER assume specific entities exist
3. Follow the exact Thought->Action->Action Input format
4. Use natural language for Action Input, not SQL/Cypher
5. CRITICAL: When direct data is unavailable, use your domain knowledge to identify the entity type and explore related data patterns - for example, if asked about "Tony Lema" and no player data exists, recognize he's a golfer and examine tournament/golf-related data for indirect verification.
6. CRITICAL: When you receive "Table Context: The data of the table is about 'X'", understand that the ENTIRE table contains data belonging to entity X. Do NOT search for X's name within the table - instead, treat all table rows as X's data records. For example, if table context says "about 'tony lema'" and you see tournament performance data, interpret it as Tony Lema's personal tournament performance records.

Follow the EXACT format below.

CRITICAL FORMATTING REQUIREMENTS:
1. ALWAYS use this exact sequence: Thought: -> Action: -> Action Input: -> Observation:
2. NEVER skip the "Action:" line after "Thought:"
3. Use ONLY the exact tool names provided
4. Each cycle must be complete before the next

AVAILABLE TOOLS:
{tools}

EXACT TOOL NAMES: {tool_names}

RESPONSE FORMAT (MANDATORY):
Question: [the input question]
Thought: [your reasoning about what to do next]
Action: [EXACTLY one tool name from: {tool_names}]
Action Input: [natural language question for the tool]
Observation: [result from the tool]
Thought: [analysis of the observation and next steps]
Action: [next tool name if needed]
Action Input: [input for next tool]
Observation: [result from tool]
... (continue until you have enough information)
Thought: [final analysis with complete understanding]
Final Answer: [comprehensive answer to the original question]

Begin!


Question: {input}
Thought: {agent_scratchpad}"""

            prompt = ChatPromptTemplate.from_template(prompt_template)

            # Inject specific context into the prompt
            prompt = prompt.partial(
                sql_db=sql_db if sql_db else "N/A",
                sql_table=sql_table if sql_table else "N/A",
                kg_graph=kg_graph
                if kg_graph
                else "N/A",  # kg_graph now holds the identifier string,
                sql_schema_info=sql_schema_info if sql_schema_info else "N/A",
                graph_schema_str=graph_schema_str if graph_schema_str else "N/A",
                task_max_iterations=task_max_iterations,
                current_iteration=current_iteration,
                remaining_iterations=max(task_max_iterations - current_iteration, 0),
            )

            # 创建Agent
            try:
                # Ensure LLM is available before creating the agent
                if self.llm is None:
                    raise ValueError("LLM is not initialized")

                # 创建自定义错误处理函数
                def handle_parsing_error(error):
                    """自定义错误处理函数，修复常见的格式问题"""
                    error_msg = str(error)
                    logger.warning(
                        f"Task {task_session_id}: Parsing error detected: {error_msg}"
                    )

                    # 检查是否是缺少Action的错误
                    if "Missing 'Action:' after 'Thought:'" in error_msg:
                        return "Invalid format detected. Please follow the exact format:\nThought: [your reasoning]\nAction: [tool name]\nAction Input: [natural language question]"
                    elif "Could not parse LLM output" in error_msg:
                        return "Format error. Remember to use:\nThought: [reasoning]\nAction: [exact tool name]\nAction Input: [question for tool]"
                    else:
                        return f"Parsing error occurred. Please ensure you follow the exact format:\nThought: [analysis]\nAction: [tool name from available tools]\nAction Input: [natural language question]\n\nError details: {error_msg}"

                agent = create_react_agent(self.llm, tools, prompt)
                logger.info(f"Agent created successfully for task {task_session_id}")
            except Exception as e:
                errmsg = f"Failed to create agent: {e}"
                logger.error(f"{errmsg} Task ID: {task_session_id}", exc_info=True)
                self._add_task_message(
                    task_session_id, "system", errmsg, message_type="error"
                )
                update_agent_task_session(task_session_id, status="failed")
                yield {"type": "error", "content": errmsg}
                return

            # 创建Agent执行器，使用改进的错误处理
            agent_executor = AgentExecutor(
                agent=agent,
                tools=tools,
                verbose=True,
                handle_parsing_errors=handle_parsing_error,  # 使用自定义错误处理函数
                max_iterations=task_max_iterations,
                return_intermediate_steps=True,
                max_execution_time=300.0,  # 5分钟超时限制
                early_stopping_method="force",
            )
            logger.info(
                f"AgentExecutor created for task {task_session_id} with max_iterations={task_max_iterations}"
            )

            # 执行Agent
            input_data = {"input": user_goal}

            # 开始流式执行
            try:
                async for chunk in agent_executor.astream_log(
                    input_data, include_names=["run_agent"]
                ):
                    # 定期检查停止请求
                    current_time = datetime.now()
                    time_since_last_check = (
                        current_time - last_status_check_time
                    ).total_seconds()

                    if time_since_last_check >= self.stop_check_interval:
                        last_status_check_time = current_time

                        # 检查停止请求
                        try:
                            current_session_state = self.get_task_session(
                                task_session_id
                            )
                            if current_session_state and current_session_state.get(
                                "stop_requested", False
                            ):
                                stop_requested_flag = True
                                logger.warning(
                                    f"Task {task_session_id}: Stop requested detected during execution."
                                )
                                self._add_task_message(
                                    task_session_id,
                                    "system",
                                    "Task stop requested by user. Stopping execution.",
                                    message_type="warning",
                                )
                                break  # 退出循环

                            # 更新迭代计数
                            if current_iteration != current_session_state.get(
                                "current_iteration", 0
                            ):
                                update_agent_task_session(
                                    task_session_id,
                                    current_iteration=current_iteration,
                                    last_status_update=current_time.isoformat(),
                                )
                        except Exception as check_err:
                            logger.error(
                                f"Task {task_session_id}: Error during status check: {check_err}",
                                exc_info=True,
                            )

                    # 处理流数据块
                    for op in chunk.ops:
                        op_path = op.get("path")
                        op_value = op.get("value")

                        # 处理思考和动作
                        if op_path == "/streamed_output/-" and isinstance(
                            op_value, dict
                        ):
                            # --- Agent Action / Thought ---
                            if (
                                "actions" in op_value
                                and isinstance(op_value["actions"], list)
                                and op_value["actions"]
                            ):
                                action_data = op_value["actions"][0]
                                if (
                                    hasattr(action_data, "tool")
                                    and hasattr(action_data, "tool_input")
                                    and hasattr(action_data, "log")
                                ):
                                    # 增加迭代计数
                                    current_iteration += 1

                                    tool_name = action_data.tool
                                    tool_input = action_data.tool_input
                                    log_content = action_data.log.strip()

                                    # 提取思考过程
                                    thought = None
                                    if log_content.startswith("Thought:"):
                                        thought = (
                                            log_content.split("Action:")[0]
                                            .replace("Thought:", "")
                                            .strip()
                                        )
                                    elif "\\nThought:" in log_content:
                                        parts = log_content.split("\\nThought:")
                                        if len(parts) > 1:
                                            thought = (
                                                parts[1].split("\\nAction:")[0].strip()
                                            )
                                    elif log_content:
                                        action_marker = f"Action: {tool_name}"
                                        if action_marker in log_content:
                                            potential_thought = log_content.split(
                                                action_marker
                                            )[0].strip()
                                            if (
                                                potential_thought
                                                and not potential_thought.startswith(
                                                    "Action:"
                                                )
                                            ):
                                                thought = potential_thought

                                    # 记录并推送思考
                                    if thought:
                                        logger.info(
                                            f"Task {task_session_id} Thought: {thought}"
                                        )
                                        self._add_task_message(
                                            task_session_id, "agent", thought, "thought"
                                        )
                                        yield {"type": "thought", "content": thought}

                                    # 记录并推送动作
                                    logger.info(
                                        f"Task {task_session_id} Action: Tool={tool_name}, Input='{tool_input}'"
                                    )
                                    self._add_task_message(
                                        task_session_id,
                                        "agent",
                                        f"Using tool '{tool_name}'.",
                                        "action",
                                        tool_name=tool_name,
                                        tool_input=tool_input,
                                    )
                                    yield {
                                        "type": "action",
                                        "tool": tool_name,
                                        "tool_input": tool_input,
                                    }

                                    # 每5次迭代更新一次状态
                                    if current_iteration % 5 == 0:
                                        try:
                                            update_agent_task_session(
                                                task_session_id,
                                                current_iteration=current_iteration,
                                                last_status_update=datetime.now().isoformat(),
                                            )
                                        except Exception as update_err:
                                            logger.error(
                                                f"Task {task_session_id}: Error updating iteration count: {update_err}"
                                            )

                                    continue

                            # --- Tool Observation ---
                            if (
                                "steps" in op_value
                                and isinstance(op_value["steps"], list)
                                and op_value["steps"]
                            ):
                                step_data = op_value["steps"][0]
                                if hasattr(step_data, "action") and hasattr(
                                    step_data, "observation"
                                ):
                                    action_taken = step_data.action
                                    observation = str(step_data.observation)
                                    tool_name = action_taken.tool

                                    logger.info(
                                        f"Task {task_session_id} Observation ({tool_name}): {observation[:200]}..."
                                    )

                                    # 检查观察结果是否包含错误信号
                                    if (
                                        "_Exception" in observation
                                        or "Invalid or incomplete response"
                                        in observation
                                    ):
                                        logger.warning(
                                            f"Task {task_session_id}: Exception detected in observation: {observation[:100]}..."
                                        )
                                        # 记录错误并更新任务状态
                                        try:
                                            error_msg = (
                                                f"工具执行异常: {observation[:200]}..."
                                            )
                                            update_agent_task_session(
                                                task_session_id,
                                                status="failed",
                                                last_tool_error=observation,
                                                error_message=error_msg,
                                                last_status_update=datetime.now().isoformat(),
                                            )
                                            # 推送错误
                                            yield {
                                                "type": "observation",
                                                "tool": tool_name,
                                                "content": observation,
                                            }
                                            yield {
                                                "type": "error",
                                                "content": error_msg,
                                            }
                                            yield {
                                                "type": "status",
                                                "content": "failed",
                                            }
                                            return  # 终止执行
                                        except Exception as update_err:
                                            logger.error(
                                                f"Task {task_session_id}: Failed to update status on observation exception: {update_err}"
                                            )

                                    # 检查是否是格式错误相关的观察
                                    elif any(
                                        error_indicator in observation
                                        for error_indicator in [
                                            "Invalid format detected",
                                            "Format error",
                                            "Parsing error occurred",
                                            "Missing 'Action:' after 'Thought:'",
                                            "Could not parse LLM output",
                                        ]
                                    ):
                                        logger.warning(
                                            f"Task {task_session_id}: Format error in observation, but continuing: {observation[:100]}..."
                                        )
                                        # 推送格式错误观察，但不终止执行
                                        yield {
                                            "type": "observation",
                                            "tool": tool_name,
                                            "content": observation,
                                        }
                                        yield {
                                            "type": "warning",
                                            "content": f"格式解析问题，Agent正在尝试恢复: {observation[:100]}...",
                                        }
                                        continue

                                    # 推送正常观察结果
                                    else:
                                        yield {
                                            "type": "observation",
                                            "tool": tool_name,
                                            "content": observation,
                                        }
                                        continue

                            # --- Final Output ---
                            if "output" in op_value and not final_answer_yielded:
                                final_answer = op_value.get("output")
                                if final_answer:
                                    # 提取最终思考
                                    final_thought = None
                                    if (
                                        "messages" in op_value
                                        and isinstance(op_value["messages"], list)
                                        and op_value["messages"]
                                    ):
                                        last_message = op_value["messages"][-1]
                                        if hasattr(last_message, "content"):
                                            last_message_content = last_message.content
                                            if (
                                                "Thought:" in last_message_content
                                                and "Final Answer:"
                                                in last_message_content
                                            ):
                                                thought_part = (
                                                    last_message_content.split(
                                                        "Final Answer:", 1
                                                    )[0]
                                                )
                                                if "Thought:" in thought_part:
                                                    final_thought = thought_part.split(
                                                        "Thought:", 1
                                                    )[1].strip()

                                    # 推送最终思考
                                    if final_thought:
                                        logger.info(
                                            f"Task {task_session_id} Final Thought: {final_thought}"
                                        )
                                        self._add_task_message(
                                            task_session_id,
                                            "agent",
                                            final_thought,
                                            "thought",
                                        )
                                        yield {
                                            "type": "thought",
                                            "content": final_thought,
                                        }

                                    # 推送最终答案
                                    logger.info(
                                        f"Task {task_session_id} Final Answer: {final_answer}"
                                    )
                                    self._add_task_message(
                                        task_session_id,
                                        "agent",
                                        final_answer,
                                        "final_answer",
                                    )
                                    yield {
                                        "type": "final_answer",
                                        "content": final_answer,
                                    }
                                    final_answer_yielded = True

                                    # 更新任务状态为完成
                                    try:
                                        update_agent_task_session(
                                            task_session_id,
                                            status="completed",
                                            current_iteration=current_iteration,
                                            last_status_update=datetime.now().isoformat(),
                                        )
                                    except Exception as update_err:
                                        logger.error(
                                            f"Task {task_session_id}: Error updating completion status: {update_err}"
                                        )

                            # 处理后续的最终输出块
                            elif "output" in op_value and final_answer_yielded:
                                chunk_content = op_value.get("output")
                                if chunk_content:
                                    yield {
                                        "type": "final_answer",
                                        "content": chunk_content,
                                    }

                # 检查最大迭代次数
                if (
                    current_iteration >= task_max_iterations
                    and not final_answer_yielded
                ):
                    max_iter_msg = f"Task reached maximum iterations limit ({task_max_iterations}) without completing."
                    logger.warning(f"Task {task_session_id}: {max_iter_msg}")
                    self._add_task_message(
                        task_session_id, "system", max_iter_msg, message_type="warning"
                    )

                    # 更新状态为失败
                    try:
                        update_agent_task_session(
                            task_session_id,
                            status="failed",
                            current_iteration=current_iteration,
                            last_status_update=datetime.now().isoformat(),
                            error_message=max_iter_msg,
                        )
                    except Exception as update_err:
                        logger.error(
                            f"Task {task_session_id}: Error updating max iterations status: {update_err}"
                        )

                    # 推送错误消息
                    yield {
                        "type": "error",
                        "content": f"任务达到最大迭代次数 ({task_max_iterations}) 限制，未能完成。",
                    }
                    yield {"type": "status", "content": "failed"}
                    return

                # 处理停止请求
                if stop_requested_flag:
                    stop_msg = "Task was stopped by user request."
                    logger.info(f"Task {task_session_id}: {stop_msg}")

                    # 更新状态为停止
                    try:
                        update_agent_task_session(
                            task_session_id,
                            status="stopped",
                            current_iteration=current_iteration,
                            last_status_update=datetime.now().isoformat(),
                            error_message=stop_msg,
                        )
                    except Exception as update_err:
                        logger.error(
                            f"Task {task_session_id}: Error updating stopped status: {update_err}"
                        )

                    # 推送停止消息
                    yield {"type": "error", "content": "任务已被用户请求停止。"}
                    yield {"type": "status", "content": "stopped"}
                    return

            except OutputParserException as ope:
                # 处理输出解析异常
                errmsg = f"Agent output parsing error: {ope}. The LLM might not be following the expected format."
                logger.error(f"Task {task_session_id}: {errmsg}", exc_info=True)
                partial_output = getattr(ope, "llm_output", "")
                self._add_task_message(
                    task_session_id,
                    "system",
                    f"{errmsg}\\nLLM Output:\\n{partial_output}",
                    message_type="error",
                    tool_name="system_parser_error",
                )

                # 更新状态为失败
                try:
                    update_agent_task_session(
                        task_session_id,
                        status="failed",
                        current_iteration=current_iteration,
                        last_status_update=datetime.now().isoformat(),
                        error_message=errmsg,
                    )
                except Exception as update_err:
                    logger.error(
                        f"Task {task_session_id}: Error updating error status: {update_err}"
                    )

                # 推送错误消息
                yield {"type": "error", "content": errmsg}
                yield {"type": "status", "content": "failed"}
                return

            except Exception as e:
                # 处理其他运行时异常
                exception_type = type(e).__name__
                errmsg = f"Agent execution failed due to {exception_type}: {str(e)}"
                logger.error(f"Task {task_session_id}: {errmsg}", exc_info=True)

                tool_name_for_msg = "system_runtime_error"
                if isinstance(e, KeyboardInterrupt):
                    errmsg = "Agent execution interrupted by user (Ctrl+C)."
                    logger.warning(f"Task {task_session_id}: {errmsg}")
                    tool_name_for_msg = "interrupt"

                self._add_task_message(
                    task_session_id,
                    "system",
                    errmsg,
                    message_type="error",
                    tool_name=tool_name_for_msg,
                )

                # 更新状态为失败
                try:
                    update_agent_task_session(
                        task_session_id,
                        status="failed",
                        current_iteration=current_iteration,
                        last_status_update=datetime.now().isoformat(),
                        error_message=errmsg,
                    )
                except Exception as update_err:
                    logger.error(
                        f"Task {task_session_id}: Error updating error status: {update_err}"
                    )

                # 推送错误消息
                yield {"type": "error", "content": errmsg}
                yield {"type": "status", "content": "failed"}
                return

            # 任务结束但没有最终答案
            if not final_answer_yielded and not stop_requested_flag:
                errmsg = "Task finished without producing a final answer."
                logger.warning(f"Task {task_session_id}: {errmsg}")

                # 更新状态为失败
                try:
                    update_agent_task_session(
                        task_session_id,
                        status="failed",
                        current_iteration=current_iteration,
                        last_status_update=datetime.now().isoformat(),
                        error_message=errmsg,
                    )
                except Exception as update_err:
                    logger.error(
                        f"Task {task_session_id}: Error updating final status: {update_err}"
                    )

                # 推送错误消息
                yield {"type": "error", "content": "任务结束但没有产生最终答案。"}
                yield {"type": "status", "content": "failed"}

        except Exception as outer_e:
            # 处理最外层异常
            logger.error(
                f"Task {task_session_id}: Critical error in run_task: {outer_e}",
                exc_info=True,
            )

            # 更新状态为失败
            try:
                update_agent_task_session(
                    task_session_id,
                    status="failed",
                    error_message=f"Critical error: {str(outer_e)}",
                )
            except Exception as update_err:
                logger.error(
                    f"Task {task_session_id}: Failed to update status after critical error: {update_err}"
                )

            # 推送错误消息
            yield {"type": "error", "content": f"严重错误: {str(outer_e)}"}
            yield {"type": "status", "content": "failed"}


# 全局实例
agent_task_manager = AgentTaskManager()


# 配置更新回调函数
def _on_agent_config_update(new_config_dict):
    """Agent任务管理器配置更新回调"""
    try:
        # AgentTaskManager可能需要重新初始化LLM
        logger.info("Agent任务管理器配置已更新")
        # 如果需要，可以重新初始化LLM
        # agent_task_manager._initialize_llm()
    except Exception as e:
        logger.error(f"Agent任务管理器配置更新失败: {str(e)}")


# 注册配置更新回调
from backend.services.vanna_service import vanna_manager

vanna_manager.register_config_callback(_on_agent_config_update)
