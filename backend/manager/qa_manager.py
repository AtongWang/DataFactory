import json
import logging
import pandas as pd
import numpy as np
import re
from decimal import Decimal
from backend.services.vanna_service import vanna_manager
from backend.utils.db_utils import (
    create_chat_session,
    update_chat_session,
    get_chat_session,
    get_chat_sessions,
    delete_chat_session,
    get_chat_messages,
    add_chat_message,
    delete_chat_message,
    save_query,
    get_saved_queries,
    get_saved_query,
    delete_saved_query,
)
import datetime

logger = logging.getLogger(__name__)


# 添加辅助函数，用于处理NumPy和日期时间类型的JSON序列化
def convert_numpy_types(obj):
    """递归转换字典或列表中的NumPy和日期时间类型为Python原生或兼容JSON的类型"""
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        # 检查是否为 NaN 或无穷大，如果是，则返回 None 或字符串表示
        if np.isnan(obj):
            return None
        if np.isinf(obj):
            return str(obj)  # 或 None，取决于需求
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return convert_numpy_types(obj.tolist())
    elif isinstance(obj, (datetime.datetime, pd.Timestamp)):
        # 检查是否为 NaT (Not a Time)
        if pd.isna(obj):
            return None
        try:
            return obj.isoformat()  # 转换为 ISO 8601 字符串
        except ValueError:  # 处理可能的 NaT 情况或其他无法转换的时间戳
            return None
    elif isinstance(obj, Decimal):
        # 处理Decimal类型 - 转换为字符串或浮点数
        return float(obj)
    elif pd.isna(obj):  # 捕获其他类型的 NaN，如 pandas Float64 类型中的 <NA>
        return None
    elif hasattr(obj, "item"):  # 捕获其他 numpy 标量类型
        # 需要进一步检查 item() 的结果是否是 NaN 或 Inf
        item_value = obj.item()
        if isinstance(item_value, float):
            if np.isnan(item_value):
                return None
            if np.isinf(item_value):
                return str(item_value)
        return item_value
    else:
        return obj


class QAManager:
    """数据库问答管理器"""

    def __init__(self):
        self.vanna_manager = vanna_manager

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

    def _normalize_visualization_text(self, content, visualization):
        if not content or not isinstance(content, str):
            return content

        def strip_plot_code_block(match):
            block = match.group(0)
            lower_block = block.lower()
            if any(
                marker in lower_block
                for marker in (
                    "matplotlib",
                    "plt.",
                    "import matplotlib",
                    "plotly",
                    "px.",
                    "go.",
                    "fig =",
                )
            ):
                return ""
            return block

        normalized = re.sub(r"```[\s\S]*?```", strip_plot_code_block, content)
        normalized = re.sub(
            r"^\s*(import\s+matplotlib|from\s+matplotlib\s+import|plt\.)[\s\S]*$",
            "",
            normalized,
            flags=re.IGNORECASE | re.MULTILINE,
        )
        normalized = re.sub(
            r"^\s*.*python\s*/\s*matplotlib.*$",
            "",
            normalized,
            flags=re.IGNORECASE | re.MULTILINE,
        )
        normalized = re.sub(r"\n{3,}", "\n\n", normalized).strip()
        if visualization:
            tab_hint = self._get_localized_message(
                "图表已按系统内置样式生成，请查看下方“图表”标签页。",
                'The chart has been generated using built-in styles. Please check the "Chart" tab below.',
            )
            if "图表" not in normalized and "chart" not in normalized.lower():
                normalized = f"{normalized}\n\n{tab_hint}" if normalized else tab_hint

        return normalized

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
                name = f"新对话 {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}"

            session_id = create_chat_session(
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
        return get_chat_session(session_id)

    def get_all_sessions(self):
        """获取所有会话"""
        return get_chat_sessions()

    def update_session(self, session_id, **kwargs):
        """更新会话信息"""
        return update_chat_session(session_id, **kwargs)

    def delete_session(self, session_id):
        """删除会话"""
        return delete_chat_session(session_id)

    def get_messages(self, session_id):
        """获取会话的所有消息"""
        return get_chat_messages(session_id)

    def add_user_message(self, session_id, content):
        """添加用户消息"""
        return add_chat_message(session_id, "user", content)

    def add_assistant_message(
        self,
        session_id,
        content,
        sql=None,
        result=None,
        visualization=None,
        reasoning=None,
        thinking=None,
    ):
        """添加助手消息"""
        return add_chat_message(
            session_id=session_id,
            role="assistant",
            content=content,
            sql=sql,
            result=result,
            visualization=visualization,
            reasoning=reasoning,
            thinking=thinking,
        )

    def delete_message(self, message_id):
        """删除消息"""
        return delete_chat_message(message_id)

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
                messages = get_chat_messages(session_id)
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
                update_chat_session(session_id, name=session_name)
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
            session = get_chat_session(session_id)
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
                    update_chat_session(session_id, **update_data)
                    # 重新获取更新后的会话信息
                    session = get_chat_session(session_id)

            # 添加用户消息
            message_id = self.add_user_message(session_id, question)

            # # 是否是第一条消息，如果是则尝试生成会话名称
            messages = get_chat_messages(session_id)
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

            # 准备上下文，提取最近几条消息作为上下文
            # recent_messages = messages[-10:] if len(messages) > 10 else messages

            # 设置温度
            temperature = session.get("temperature", 0.7)

            # 获取当前系统配置
            current_config = self.vanna_manager.get_config()

            # 检查会话中是否有设置模型名称
            model_name = session.get("model_name")
            if model_name:
                # 当前模型类型
                model_type = current_config.get("model", {}).get("type", "vanna")

                # 根据模型类型更新配置
                if model_type == "vanna":
                    current_config["model"]["model_name"] = model_name
                elif model_type == "ollama":
                    current_config["model"]["ollama_model"] = model_name
                elif model_type == "openai":
                    current_config["model"]["model_name"] = model_name

                # 更新温度
                if "temperature" in current_config.get("model", {}):
                    current_config["model"]["temperature"] = temperature

                # 更新vanna_manager配置
                self.vanna_manager.update_config(current_config)

            # 调用Vanna生成SQL和结果
            if self.vanna_manager.vn is None:
                raise ValueError("Vanna未初始化")

            # 如果指定了表名，为问题添加上下文
            context = ""
            table_to_query = session.get("table_name")
            if table_to_query:
                context = f"For the table {table_to_query} "

            self.vanna_manager.vn.question_table_name = table_to_query
            # # 构建历史消息上下文
            # history_context = ""
            # if recent_messages:
            #     history_context = "以下是之前的对话历史：\n"
            #     if recent_messages[0]['role'] != 'user':
            #         recent_messages = recent_messages[1:]
            #     if recent_messages[-1]['role'] != 'assistant':
            #         recent_messages = recent_messages[:-1]
            #     for msg in recent_messages:
            #         if msg['role'] == 'user' or msg['role'] == 'assistant':
            #             history_context += f"{msg['role']}: {msg['content']}\n"
            #     history_context += "\n现在请回答我的问题：\n"

            enhanced_question = f"{context}{question}"

            # 获取会话的 auto_train 设置，默认为 True
            auto_train_setting = session.get("auto_train", True)
            logger.info(f"会话 {session_id} 的 auto_train 设置为: {auto_train_setting}")

            visualize_need = self.vanna_manager.vn.check_user_visualize_goal(question)
            # visualize_need = False # 默认不需要可视化，除非有明确的需求
            # 查询Vanna获取结果
            try:
                # 调用Vanna的ask方法，符合其API规范
                response = self.vanna_manager.vn.ask(
                    question=enhanced_question,
                    print_results=False,  # 注意：print_results通常用于调试，Web应用中可能不需要
                    auto_train=auto_train_setting,  # 使用从会话获取的设置
                    visualize=visualize_need,
                    allow_llm_to_see_data=True,
                )

                # Vanna的ask方法返回元组(sql, dataframe, figure)或None
                if response is None:
                    # 如果返回None，表示整个处理失败
                    error_msg = self._get_localized_message(
                        "无法处理您的问题，请尝试重新表述。",
                        "Unable to process your question, please try rephrasing it.",
                    )
                    self.add_assistant_message(session_id=session_id, content=error_msg)
                    return {"status": "error", "message": error_msg}

                # 解析返回的元组
                sql, result_df, plotly_fig = response

                # SQL可能为None
                if sql is None:
                    error_msg = self._get_localized_message(
                        "无法为您的问题生成有效的SQL查询。",
                        "Unable to generate a valid SQL query for your question.",
                    )
                    self.add_assistant_message(session_id=session_id, content=error_msg)
                    return {"status": "error", "message": error_msg}

                # 处理结果
                try:
                    # 处理结果数据 - result_df可能为None
                    if result_df is not None:
                        # 将DataFrame转换为字典格式，确保NumPy类型被转换为Python原生类型
                        # 先转换列名
                        columns = [str(col) for col in result_df.columns.tolist()]
                        # 转换数据，确保所有类型都被处理
                        data = []
                        for row in result_df.values:
                            processed_row = []
                            for item in row:
                                # 使用增强后的转换函数处理每个单元格的值
                                processed_item = convert_numpy_types(item)
                                processed_row.append(processed_item)
                            data.append(processed_row)

                        result_data = {"columns": columns, "data": data}
                    else:
                        result_data = None

                    # 处理可视化数据 - plotly_fig可能为None
                    visualization = None
                    if plotly_fig is not None:
                        try:
                            # 使用辅助函数处理图表中的NumPy类型
                            fig_dict = plotly_fig.to_dict()
                            visualization = convert_numpy_types(fig_dict)
                        except Exception as viz_error:
                            logger.warning(f"无法将可视化转换为字典: {str(viz_error)}")

                    # 由于API不返回推理过程，这里将reasoning设为None
                    reasoning = None

                    # 生成数据分析解释文本
                    explanation = None
                    try:
                        logger.info("开始生成解释...")
                        # 传递原始问题，Vanna 内部会处理
                        explanation = self.vanna_manager.vn.generate_summary(
                            question, result_df
                        )
                        # 安全地记录日志，处理非字符串类型的explanation
                        if explanation:
                            if isinstance(explanation, str):
                                logger.info(f"解释生成完成: {explanation[:100]}...")
                            else:
                                logger.info(
                                    f"解释生成完成: 类型={type(explanation)}, 内容={str(explanation)[:100]}..."
                                )
                        else:
                            logger.info("解释生成完成: 返回值为空")
                        if not explanation:  # Fallback if summary fails
                            explanation = self._get_localized_message(
                                f"查询成功执行，返回 {len(data)} 行数据。",
                                f"Query executed successfully, returned {len(data)} rows of data.",
                            )
                    except Exception as explain_error:
                        logger.warning(f"为 Agent 工具生成解释失败: {explain_error}")
                        explanation = self._get_localized_message(
                            f"查询成功执行，返回 {len(data)} 行数据 (无法生成摘要)。",
                            f"Query executed successfully, returned {len(data)} rows of data (unable to generate summary).",
                        )

                    explanation = self._normalize_visualization_text(
                        explanation, visualization
                    )

                    # 生成回复内容 (现在 content 就是 explanation)
                    content = (
                        explanation  # 直接使用生成的或默认的 explanation 作为 content
                    )
                    logger.info(f"生成回复内容: {content}")
                    # 尝试从content中提取思考过程
                    thinking = None
                    if content:
                        if isinstance(content, str):
                            # 检查是否包含<think>标签
                            think_match = re.search(
                                r"<think>(.*?)</think>", content, re.DOTALL
                            )
                            if think_match:
                                thinking = think_match.group(1).strip()
                                # 从content中移除思考过程
                                clean_content = re.sub(
                                    r"<think>.*?</think>", "", content, flags=re.DOTALL
                                ).strip()
                                if clean_content:
                                    content = clean_content
                                logger.info(f"生成思考过程: {thinking}")
                        elif isinstance(content, dict):
                            thinking = content.get("reasoning")
                            content = content.get("content")
                        else:
                            logger.warning(
                                f"生成回复内容时遇到未知类型: {type(content)}"
                            )
                    # 添加助手消息
                    # 注意: content 现在只包含解释文本
                    self.add_assistant_message(
                        session_id=session_id,
                        content=content,
                        sql=sql,
                        result=result_data,
                        visualization=visualization,
                        reasoning=reasoning,
                        thinking=thinking,
                    )

                    # ask_question 返回的字典，确保 explanation 字段存在
                    return {
                        "status": "success",
                        "sql": sql,
                        "result": result_data,
                        "visualization": visualization,
                        "content": content,  # content 字段现在是解释文本
                        "reasoning": reasoning,
                        "explanation": content,  # 明确添加 explanation 字段，与 content 内容一致
                        "thinking": thinking,  # 添加思考过程字段
                    }
                except Exception as e:
                    logger.error(f"处理查询结果失败: {str(e)}")
                    error_msg = self._get_localized_message(
                        f"生成SQL但处理结果失败: {str(e)}",
                        f"SQL generated but failed to process results: {str(e)}",
                    )
                    self.add_assistant_message(
                        session_id=session_id, content=error_msg, sql=sql
                    )
                    return {"status": "error", "message": error_msg, "sql": sql}
            except Exception as e:
                logger.error(f"Vanna处理问题失败: {str(e)}")
                error_msg = self._get_localized_message(
                    f"处理问题时出错: {str(e)}", f"Error processing question: {str(e)}"
                )
                self.add_assistant_message(session_id=session_id, content=error_msg)
                return {"status": "error", "message": error_msg}
            finally:
                # 清理状态，避免污染后续调用
                if hasattr(self.vanna_manager.vn, "question_table_name"):
                    self.vanna_manager.vn.question_table_name = None
        except Exception as e:
            logger.error(f"问答处理失败: {str(e)}")
            error_msg = self._get_localized_message(
                f"问答处理失败: {str(e)}", f"QA processing failed: {str(e)}"
            )
            return {"status": "error", "message": error_msg}

    def ask_question_stream(
        self, session_id, question, database_name=None, table_name=None
    ):
        try:
            session = get_chat_session(session_id)
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
                    update_chat_session(session_id, **update_data)
                    session = get_chat_session(session_id)

            self.add_user_message(session_id, question)

            messages = get_chat_messages(session_id)
            if len(messages) == 1:
                try:
                    self.generate_session_name(session_id, question)
                except Exception:
                    pass

            temperature = session.get("temperature", 0.7)
            current_config = self.vanna_manager.get_config()
            model_name = session.get("model_name")
            if model_name:
                model_type = current_config.get("model", {}).get("type", "vanna")
                if model_type == "vanna":
                    current_config["model"]["model_name"] = model_name
                elif model_type == "ollama":
                    current_config["model"]["ollama_model"] = model_name
                elif model_type == "openai":
                    current_config["model"]["model_name"] = model_name
                if "temperature" in current_config.get("model", {}):
                    current_config["model"]["temperature"] = temperature
                self.vanna_manager.update_config(current_config)

            if self.vanna_manager.vn is None:
                yield {"type": "error", "message": "Vanna未初始化"}
                yield {"type": "end"}
                return

            context = ""
            table_to_query = session.get("table_name")
            if table_to_query:
                context = f"For the table {table_to_query} "

            self.vanna_manager.vn.question_table_name = table_to_query
            enhanced_question = f"{context}{question}"
            auto_train_setting = session.get("auto_train", True)
            visualize_need = self.vanna_manager.vn.check_user_visualize_goal(question)

            yield {"type": "start"}
            yield {
                "type": "status",
                "stage": "prepare",
                "progress": 10,
                "message": self._get_localized_message(
                    "正在准备查询上下文", "Preparing query context"
                ),
            }

            yield {
                "type": "status",
                "stage": "sql",
                "progress": 30,
                "message": self._get_localized_message(
                    "正在生成SQL并执行查询", "Generating SQL and running query"
                ),
            }

            response = self.vanna_manager.vn.ask(
                question=enhanced_question,
                print_results=False,
                auto_train=auto_train_setting,
                visualize=visualize_need,
                allow_llm_to_see_data=True,
            )

            if response is None:
                error_msg = self._get_localized_message(
                    "无法处理您的问题，请尝试重新表述。",
                    "Unable to process your question, please try rephrasing it.",
                )
                self.add_assistant_message(session_id=session_id, content=error_msg)
                yield {"type": "error", "message": error_msg}
                yield {"type": "end"}
                return

            sql, result_df, plotly_fig = response
            yield {
                "type": "status",
                "stage": "process",
                "progress": 55,
                "message": self._get_localized_message(
                    "正在处理结果和图表", "Processing result and chart"
                ),
            }
            if sql is None:
                error_msg = self._get_localized_message(
                    "无法为您的问题生成有效的SQL查询。",
                    "Unable to generate a valid SQL query for your question.",
                )
                self.add_assistant_message(session_id=session_id, content=error_msg)
                yield {"type": "error", "message": error_msg}
                yield {"type": "end"}
                return

            result_data = None
            if result_df is not None:
                columns = [str(col) for col in result_df.columns.tolist()]
                data = []
                for row in result_df.values:
                    processed_row = []
                    for item in row:
                        processed_row.append(convert_numpy_types(item))
                    data.append(processed_row)
                result_data = {"columns": columns, "data": data}

            visualization = None
            if plotly_fig is not None:
                try:
                    fig_dict = plotly_fig.to_dict()
                    visualization = convert_numpy_types(fig_dict)
                except Exception:
                    visualization = None

            explanation = ""
            if result_df is not None:
                try:
                    yield {
                        "type": "status",
                        "stage": "summary",
                        "progress": 70,
                        "message": self._get_localized_message(
                            "LLM 正在生成分析", "LLM is generating analysis"
                        ),
                    }
                    stream_used = False
                    if hasattr(self.vanna_manager.vn, "generate_summary_stream"):
                        stream_used = True
                        for delta in self.vanna_manager.vn.generate_summary_stream(
                            question, result_df
                        ):
                            if isinstance(delta, dict):
                                delta = delta.get("content", "")
                            if delta:
                                explanation += str(delta)
                                yield {"type": "chunk", "delta": str(delta)}

                    if not stream_used:
                        summary = self.vanna_manager.vn.generate_summary(
                            question, result_df
                        )
                        if isinstance(summary, dict):
                            explanation = summary.get("content") or ""
                        else:
                            explanation = summary or ""
                        if explanation:
                            yield {"type": "chunk", "delta": explanation}

                    if not explanation:
                        row_count = (
                            len(result_data.get("data", [])) if result_data else 0
                        )
                        explanation = self._get_localized_message(
                            f"查询成功执行，返回 {row_count} 行数据。",
                            f"Query executed successfully, returned {row_count} rows of data.",
                        )
                        yield {"type": "chunk", "delta": explanation}
                except Exception as explain_error:
                    row_count = len(result_data.get("data", [])) if result_data else 0
                    explanation = self._get_localized_message(
                        f"查询成功执行，返回 {row_count} 行数据 (无法生成摘要)。",
                        f"Query executed successfully, returned {row_count} rows of data (unable to generate summary).",
                    )
                    logger.warning(f"流式生成解释失败: {explain_error}")
                    yield {"type": "chunk", "delta": explanation}
            else:
                explanation = self._get_localized_message(
                    "SQL 语句已成功执行，但未返回数据。",
                    "SQL statement executed successfully but returned no data.",
                )
                yield {"type": "chunk", "delta": explanation}

            explanation = self._normalize_visualization_text(explanation, visualization)
            thinking = None
            if isinstance(explanation, str):
                think_match = re.search(r"<think>(.*?)</think>", explanation, re.DOTALL)
                if think_match:
                    thinking = think_match.group(1).strip()
                    explanation = re.sub(
                        r"<think>.*?</think>", "", explanation, flags=re.DOTALL
                    ).strip()

            content = explanation
            yield {
                "type": "status",
                "stage": "persist",
                "progress": 90,
                "message": self._get_localized_message(
                    "正在保存消息", "Saving message"
                ),
            }
            self.add_assistant_message(
                session_id=session_id,
                content=content,
                sql=sql,
                result=result_data,
                visualization=visualization,
                reasoning=None,
                thinking=thinking,
            )

            yield {
                "type": "meta",
                "payload": {
                    "sql": sql,
                    "result": result_data,
                    "visualization": visualization,
                    "thinking": thinking,
                    "explanation": content,
                },
            }
            yield {
                "type": "status",
                "stage": "done",
                "progress": 100,
                "message": self._get_localized_message("已完成", "Completed"),
            }
            yield {"type": "end"}
        except Exception as e:
            logger.error(f"流式问答处理失败: {str(e)}")
            yield {"type": "error", "message": str(e)}
            yield {"type": "end"}
        finally:
            if self.vanna_manager.vn is not None and hasattr(
                self.vanna_manager.vn, "question_table_name"
            ):
                self.vanna_manager.vn.question_table_name = None

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
            return self._get_localized_message(
                f"查询完成，找到 {count} 条记录。",
                f"Query completed, found {count} records.",
            )
        elif sql:  # 如果有SQL但没数据或解释
            return self._get_localized_message(
                "SQL 语句已成功执行。", "SQL statement executed successfully."
            )
        else:  # 默认的回退文本
            return self._get_localized_message("请求已处理。", "Request processed.")

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

            return save_query(
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
        return get_saved_queries()

    def get_query(self, query_id):
        """获取保存的查询"""
        return get_saved_query(query_id)

    def delete_query(self, query_id):
        """删除保存的查询"""
        return delete_saved_query(query_id)

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
            session = get_chat_session(session_id)
            if not session:
                return None

            # 检查锁定条件 - 只有当有表名时才能锁定
            if lock_status and not session.get("table_name"):
                logger.warning(f"会话 {session_id} 尝试锁定表，但没有设置表名")
                return update_chat_session(session_id, is_table_locked=False)

            # 更新锁定状态
            return update_chat_session(session_id, is_table_locked=lock_status)
        except Exception as e:
            logger.error(f"切换表锁定状态失败: {str(e)}")
            return None

    def generate_sql_and_result(
        self,
        question: str,
        database_name: str,
        table_name: str,
        model_config_override: dict = None,
    ):
        """
        直接生成 SQL、结果和解释，不依赖于特定的聊天会话 ID。

        Args:
            question (str): 用户的问题。
            database_name (str): 目标数据库名称。
            table_name (str): 目标数据表名称。
            model_config_override (dict, optional): 覆盖当前模型配置（例如温度、模型名称）。 Defaults to None.

        Returns:
            dict: 包含 status, sql, result, explanation, (可能还有 visualization, thinking) 的字典。
                  如果失败，返回包含 status 和 message 的字典。
        """
        try:
            logger.info(
                f"开始generate_sql_and_result: question='{question}', table='{table_name}'"
            )

            # 0. 检查 Vanna 是否初始化
            if self.vanna_manager.vn is None:
                logger.error("Vanna未初始化")
                raise ValueError("Vanna未初始化")

            logger.info("Vanna实例检查通过")

            # 1. 准备 Vanna 配置 (可选的模型覆盖)
            # 注意：这里不处理会话级别的模型覆盖，如果需要，需要传递 model_config_override
            # TODO: Consider how to apply temporary model overrides if needed without affecting global state.
            # For now, we assume it uses the currently configured Vanna model.

            # 2. 准备问题上下文
            context = ""
            if table_name:
                context = f"For the table {table_name} "
                # 设置 Vanna 实例的目标表 (如果 Vanna 实现支持这种临时设置)
                # 这依赖于 Vanna 内部实现，可能需要调整 Vanna 或用其他方式传递上下文
                if hasattr(self.vanna_manager.vn, "question_table_name"):
                    self.vanna_manager.vn.question_table_name = table_name
                    logger.info(f"设置question_table_name为: {table_name}")
                else:
                    # 如果 Vanna 不支持，将上下文添加到问题前缀
                    context = f"For the table '{table_name}'： "
                    logger.info(f"使用上下文前缀: {context}")

            enhanced_question = f"{context}{question}"
            logger.info(f"增强后的问题: {enhanced_question}")

            # 3. 调用 Vanna 核心方法
            # 注意：auto_train 和 visualize 设置需要决定默认值或从参数传入
            auto_train_setting = True  # 或者从配置读取默认值
            # visualize_need = self.vanna_manager.vn.check_user_visualize_goal(question)
            visualize_need = False  # 默认不需要可视化，除非有明确的需求
            logger.info(
                f"Calling Vanna ask for Agent tool: Question='{enhanced_question}', Table='{table_name}', AutoTrain={auto_train_setting}, Visualize={visualize_need}"
            )

            logger.info("准备调用Vanna.ask方法...")
            response = self.vanna_manager.vn.ask(
                question=enhanced_question,
                print_results=False,  # Agent 不需要打印
                auto_train=auto_train_setting,
                visualize=visualize_need,
                allow_llm_to_see_data=True,
            )
            logger.info(f"Vanna.ask调用完成，返回结果类型: {type(response)}")
            logger.info(f"Vanna.ask返回结果: {response}")
            # 4. 处理 Vanna 返回结果 (与 ask_question 类似)
            if response is None:
                error_msg = self._get_localized_message(
                    "无法处理您的问题，请尝试重新表述。",
                    "Unable to process your question, please try rephrasing it.",
                )
                logger.warning("Vanna ask returned None for agent tool.")
                return {"status": "error", "message": error_msg}

            logger.info("开始解析Vanna返回结果...")
            sql, result_df, plotly_fig = response
            logger.info(
                f"解析结果: sql={sql is not None}, result_df={result_df is not None}, plotly_fig={plotly_fig is not None}"
            )

            if sql is None:
                error_msg = self._get_localized_message(
                    "无法为您的问题生成有效的SQL查询。",
                    "Unable to generate a valid SQL query for your question.",
                )
                logger.warning("Vanna ask returned None for SQL for agent tool.")
                return {"status": "error", "message": error_msg}

            # 5. 处理结果和可视化 (使用 convert_numpy_types)
            result_data = None
            visualization = None
            explanation = "查询已执行。"  # Default explanation

            logger.info("开始处理结果数据...")
            if result_df is not None:
                # try:
                columns = [str(col) for col in result_df.columns.tolist()]
                data = []
                for row in result_df.values:
                    processed_row = [convert_numpy_types(item) for item in row]
                    data.append(processed_row)
                result_data = {"columns": columns, "data": data}
                logger.info(
                    f"结果数据处理完成，行数: {len(data)}, 列数: {len(columns)}"
                )

                # 生成解释
                try:
                    logger.info("开始生成解释...")
                    # 传递原始问题，Vanna 内部会处理
                    explanation = self.vanna_manager.vn.generate_summary(
                        question, result_df
                    )
                    # 安全地记录日志，处理非字符串类型的explanation
                    if explanation:
                        if isinstance(explanation, str):
                            logger.info(f"解释生成完成: {explanation[:100]}...")
                        else:
                            logger.info(
                                f"解释生成完成: 类型={type(explanation)}, 内容={str(explanation)[:100]}..."
                            )
                    else:
                        logger.info("解释生成完成: 返回值为空")
                    if not explanation:  # Fallback if summary fails
                        explanation = self._get_localized_message(
                            f"查询成功执行，返回 {len(data)} 行数据。",
                            f"Query executed successfully, returned {len(data)} rows of data.",
                        )
                except Exception as explain_error:
                    logger.warning(f"为 Agent 工具生成解释失败: {explain_error}")
                    explanation = self._get_localized_message(
                        f"查询成功执行，返回 {len(data)} 行数据 (无法生成摘要)。",
                        f"Query executed successfully, returned {len(data)} rows of data (unable to generate summary).",
                    )

            elif sql:  # SQL executed but no DataFrame (e.g., DDL, or empty result)
                explanation = self._get_localized_message(
                    "SQL 语句已成功执行，但未返回数据。",
                    "SQL statement executed successfully but returned no data.",
                )
                logger.info("SQL执行成功但无数据返回")

            logger.info("开始处理可视化...")
            if plotly_fig is not None:
                try:
                    fig_dict = plotly_fig.to_dict()
                    visualization = convert_numpy_types(fig_dict)
                    logger.info("可视化处理完成")
                except Exception as viz_error:
                    logger.warning(f"为 Agent 工具转换可视化失败: {viz_error}")

            explanation = self._normalize_visualization_text(explanation, visualization)

            # 6. 提取思考过程 (如果需要且 Vanna 支持)
            thinking = None
            logger.info("开始提取思考过程...")
            # 假设 explanation 可能包含 <think> 标签
            if explanation:
                if isinstance(explanation, str):
                    think_match = re.search(
                        r"<think>(.*?)</think>", explanation, re.DOTALL
                    )
                    if think_match:
                        thinking = think_match.group(1).strip()
                        clean_explanation = re.sub(
                            r"<think>.*?</think>", "", explanation, flags=re.DOTALL
                        ).strip()
                        if clean_explanation:
                            explanation = clean_explanation  # Use cleaned explanation
                        logger.info(f"提取到思考过程: {thinking and thinking[:50]}...")
                elif isinstance(explanation, dict):
                    thinking = explanation.get("reasoning")
                    explanation = explanation.get("content")
                else:
                    logger.warning(
                        f"为 Agent 工具生成解释时遇到未知类型: {type(explanation)}"
                    )

            # 7. 返回结果
            logger.info("准备返回最终结果...")
            return {
                "status": "success",
                "sql": sql,
                "result": result_data,
                "visualization": visualization,
                "explanation": explanation,  # 使用处理后的解释
                "thinking": thinking,
            }

        except Exception as e:
            logger.error(f"Agent 工具执行 SQL 查询失败: {str(e)}", exc_info=True)
            error_msg = self._get_localized_message(
                f"执行 SQL 查询时出错: {str(e)}", f"Error executing SQL query: {str(e)}"
            )
            return {"status": "error", "message": error_msg}
        finally:
            # 清理状态，避免污染后续调用
            if hasattr(self.vanna_manager.vn, "question_table_name"):
                self.vanna_manager.vn.question_table_name = None
                logger.info("清理question_table_name状态")

    def get_sql_schema_for_query(self, question_or_table_name: str) -> str:
        """
        获取与问题或表名相关的 SQL DDL 模式信息。

        Args:
            question_or_table_name (str): 用户的问题或具体的表名。

        Returns:
            str: 相关的 DDL 字符串，或者未找到或出错时的提示信息。
        """
        try:
            if self.vanna_manager.vn is None:
                return self._get_localized_message(
                    "错误: Vanna 服务未初始化，无法获取 schema。",
                    "Error: Vanna service is not initialized, unable to get schema.",
                )

            logger.info(f"调用 Vanna get_related_ddl，输入: '{question_or_table_name}'")

            # 如果输入可能是表名，则直接传递表名参数

            ddl = self.vanna_manager.vn.get_related_ddl(
                question=question_or_table_name, table_name=question_or_table_name
            )

            if ddl:
                logger.info(f"找到相关 DDL 数量: {len(ddl)}")
                # 将所有找到的DDL合并为一个字符串返回
                if isinstance(ddl, list):
                    ddl_str = "\n\n".join(ddl)
                else:
                    ddl_str = str(ddl)

                return self._get_localized_message(
                    f"以下是与 '{question_or_table_name}' 相关的数据库表结构 (DDL):\n```sql\n{ddl_str}\n```",
                    f"The following is the database table structure (DDL) related to '{question_or_table_name}':\n```sql\n{ddl_str}\n```",
                )
            else:
                logger.warning(
                    f"未能找到与 '{question_or_table_name}' 相关的 DDL 信息。"
                )
                return self._get_localized_message(
                    f"未能找到与 '{question_or_table_name}' 直接相关的表结构信息。可能需要更具体的问题或检查训练数据。",
                    f"Unable to find table structure information directly related to '{question_or_table_name}'. You may need a more specific question or check the training data.",
                )

        except AttributeError:
            logger.error("当前 Vanna 实例可能没有 get_related_ddl 方法。")
            return self._get_localized_message(
                "错误: 当前配置不支持获取相关的 DDL。",
                "Error: Current configuration does not support getting related DDL.",
            )
        except Exception as e:
            logger.error(f"获取相关 DDL 时出错: {str(e)}", exc_info=True)
            return self._get_localized_message(
                f"错误: 获取数据库 schema 时发生错误: {str(e)}",
                f"Error: An error occurred while getting database schema: {str(e)}",
            )


# 创建全局问答管理器实例
qa_manager = QAManager()


# 配置更新回调函数
def _on_qa_config_update(new_config_dict):
    """QA管理器配置更新回调"""
    try:
        # QAManager通过vanna_manager获取配置，无需重新初始化
        # 但可以在这里处理特定的配置更新逻辑
        logger.info("QA管理器配置已更新")
    except Exception as e:
        logger.error(f"QA管理器配置更新失败: {str(e)}")


# 注册配置更新回调
from backend.services.vanna_service import vanna_manager

vanna_manager.register_config_callback(_on_qa_config_update)
