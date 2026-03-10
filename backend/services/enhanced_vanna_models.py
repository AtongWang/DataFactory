from vanna.ollama import Ollama
from vanna.openai import OpenAI_Chat
from backend.services.vanna_new_class import EnhancedVannaBase
import json
from pydantic import BaseModel
import re
import logging
from backend.utils.token_tracking import (
    global_token_tracker,
    estimate_tokens_from_messages,
)

logger = logging.getLogger(__name__)


def clean_json_response(response_content):
    """
    清理模型响应中的markdown格式，提取纯JSON内容
    """
    if not response_content:
        return response_content

    # 移除可能的markdown代码块标记
    # 处理 ```json...``` 格式
    cleaned = re.sub(
        r"^```json\s*\n?", "", response_content.strip(), flags=re.MULTILINE
    )
    cleaned = re.sub(r"\n?```\s*$", "", cleaned, flags=re.MULTILINE)

    # 处理 ```...``` 格式
    cleaned = re.sub(r"^```\s*\n?", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"\n?```\s*$", "", cleaned, flags=re.MULTILINE)

    # 移除前后空白字符
    cleaned = cleaned.strip()

    return cleaned


def is_provider_supported(model_name, api_base):
    """
    检查模型是否支持 provider 参数
    """
    if not model_name or not api_base:
        return False

    # 只有OpenRouter支持provider参数
    if "openrouter" not in api_base.lower():
        return False

    # 某些模型已知不支持provider参数
    unsupported_models = [
        "gemini",  # Gemini系列模型
        "claude-3-haiku",  # 某些Claude模型版本
    ]

    model_lower = model_name.lower()
    for unsupported in unsupported_models:
        if unsupported in model_lower:
            logger.warning(f"模型 {model_name} 已知不支持provider参数")
            return False

    return True


def safe_create_completion(client, completion_params, model_name="unknown"):
    """
    安全的创建completion请求，包含provider参数错误的自动回退机制
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
                f"模型 {model_name} 不支持provider参数，尝试移除provider参数后重试"
            )

            # 创建不包含provider参数的副本
            fallback_params = completion_params.copy()
            if "provider" in fallback_params:
                del fallback_params["provider"]

            try:
                # 重试不带provider参数的请求
                response = client.chat.completions.create(**fallback_params)
                logger.info(f"模型 {model_name} 成功使用fallback请求（无provider参数）")
                return response
            except Exception as fallback_e:
                logger.error(f"模型 {model_name} fallback请求也失败: {str(fallback_e)}")
                raise fallback_e
        else:
            # 其他类型的错误，直接抛出
            raise e


# 定义 Pydantic 模型
class VisualizeGoalResponse(BaseModel):
    requires_visualization: bool


class EnhancedOllama(EnhancedVannaBase, Ollama):
    """
    增强版Ollama类，继承EnhancedVannaBase中的所有增强功能
    同时保留原始Ollama类的所有功能
    """

    def __init__(self, config=None):
        Ollama.__init__(self, config=config)
        # EnhancedVannaBase不需要单独初始化，因为Ollama已经初始化了VannaBase
        self.system_prompt = config.get("system_prompt", "")

    def __str__(self):
        return f"EnhancedOllama({self.model})"

    def submit_prompt(self, prompt, **kwargs) -> str:
        logger.info(
            f"Ollama parameters:\n"
            f"model={self.model},\n"
            f"options={self.ollama_options},\n"
            f"keep_alive={self.keep_alive}"
        )

        # 计算输入token数量
        prompt_tokens = estimate_tokens_from_messages(prompt)

        # system prompt 放在prompt的最前面
        for message in prompt:
            if message["role"] == "system":
                message["content"] += "\n" + self.system_prompt
        logger.info(f"Prompt Content:\n{json.dumps(prompt, ensure_ascii=False)}")
        response_dict = self.ollama_client.chat(
            model=self.model,
            messages=prompt,
            stream=False,
            options=self.ollama_options,
            keep_alive=self.keep_alive,
        )

        logger.info(f"Ollama Response:\n{str(response_dict)}")

        # 计算输出token数量
        response_content = response_dict["message"]["content"]
        completion_tokens = len(response_content) // 4  # 粗略估算

        # 记录token使用情况
        global_token_tracker.add_vanna_tokens(
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
        )

        return response_content

    def submit_prompt_stream(self, prompt, **kwargs):
        for message in prompt:
            if message["role"] == "system":
                message["content"] += "\n" + self.system_prompt

        stream_iter = self.ollama_client.chat(
            model=self.model,
            messages=prompt,
            stream=True,
            options=self.ollama_options,
            keep_alive=self.keep_alive,
        )

        for part in stream_iter:
            delta = (part.get("message") or {}).get("content")
            if delta:
                yield delta

    def check_user_visualize_goal(self, question):
        # 检查用户是否明确要求绘图
        logger.info(f"Checking user visualize goal for question: {question}")

        # 更新 system prompt 以引导模型专注于判断，并要求其遵循 Pydantic schema
        prompt = [
            # 移除旧的 system prompt，因为 format 参数会处理结构化输出
            # {"role": "system", "content": f"请判断用户的问题是否明确要求绘图。请返回JSON格式，必须包含requires_visualization字段，值为true或false。例如：{{\"requires_visualization\": true}}"},
            {
                "role": "system",
                "content": f"基于用户的问题，判断是否明确要求进行数据可视化或绘图。",
            },  # 更简洁的引导
            {"role": "user", "content": question},
        ]

        try:
            # 使用 Pydantic schema 进行结构化输出
            schema = VisualizeGoalResponse.model_json_schema()
            logger.info(f"Using Pydantic schema for structured output: {schema}")

            response = self.ollama_client.chat(
                model=self.model,
                messages=prompt,
                # 指定 Pydantic schema 作为输出格式
                format=schema,
                stream=False,  # 确保获取完整响应
                options=self.ollama_options,
                keep_alive=self.keep_alive,
            )

            logger.info(f"Ollama Structured Response:\n{str(response)}")

            # 检查响应格式
            if "message" not in response or "content" not in response["message"]:
                logger.info(f"Invalid Ollama response format: {response}")
                return False

            response_content = response["message"]["content"]

            # 清理JSON响应
            cleaned_content = clean_json_response(response_content)
            logger.info(f"Cleaned JSON content: {cleaned_content}")

            # 使用 Pydantic 直接验证 JSON 响应
            result = VisualizeGoalResponse.model_validate_json(cleaned_content)
            return result.requires_visualization

        except json.JSONDecodeError as json_err:
            logger.info(
                f"Failed to decode Ollama JSON response: {json_err}\nOriginal content: {response_content}\nCleaned content: {cleaned_content if 'cleaned_content' in locals() else 'N/A'}"
            )
            return False
        except Exception as e:  # 捕获 Pydantic 验证错误和其他潜在异常
            logger.info(
                f"Error in check_user_visualize_goal: {e}\nResponse content: {response_content if 'response_content' in locals() else 'N/A'}"
            )
            return False


class EnhancedOpenAI_Chat(EnhancedVannaBase, OpenAI_Chat):
    """
    增强版OpenAI_Chat类，继承EnhancedVannaBase中的所有增强功能
    同时保留原始OpenAI_Chat类的所有功能
    """

    def __init__(self, client=None, config=None):
        OpenAI_Chat.__init__(self, client=client, config=config)
        # EnhancedVannaBase不需要单独初始化，因为OpenAI_Chat已经初始化了VannaBase
        self.system_prompt = config.get("system_prompt", "")

    def __str__(self):
        return f"EnhancedOpenAI_Chat({self.model})"

    def submit_prompt(self, prompt, **kwargs) -> str:
        if prompt is None:
            raise Exception("Prompt is None")

        if len(prompt) == 0:
            raise Exception("Prompt is empty")

        # 计算输入token数量
        prompt_tokens = estimate_tokens_from_messages(prompt)

        # system prompt 放在prompt的最前面
        for message in prompt:
            if message["role"] == "system":
                message["content"] += "\n" + self.system_prompt

        # Count the number of tokens in the message log
        # Use 4 as an approximation for the number of characters per token
        num_tokens = 0
        for message in prompt:
            num_tokens += len(message["content"]) / 4

        # 检测是否为Gemini模型
        current_model = (
            kwargs.get("model")
            or (self.config.get("model") if self.config else None)
            or (self.config.get("engine") if self.config else None)
        )
        is_gemini = "gemini" in current_model.lower() if current_model else False
        is_qwen = "qwen" in current_model.lower() if current_model else False

        # 为不同模型准备特殊的请求参数
        def create_completion_params(model_name):
            base_params = {
                "model": model_name,
                "messages": prompt,
                "stop": None,
                "temperature": self.temperature,
            }
            is_openrouter = (
                "openrouter" in self.client.base_url.host
                if self.client and hasattr(self.client, "base_url")
                else False
            )

            # 检查是否启用provider routing功能
            use_provider_routing = self.config.get("use_provider_routing", True)

            if is_openrouter and use_provider_routing:
                # 先检查模型是否支持provider参数
                api_base = (
                    self.client.base_url.host
                    if self.client and hasattr(self.client, "base_url")
                    else ""
                )
                if is_provider_supported(model_name, api_base):
                    # 添加 Provider Routing 配置
                    if self.config.get("provider"):
                        provider_config = self.config.get("provider", {})
                        # 过滤掉空值的provider配置
                        filtered_provider = {}

                        if provider_config.get("order"):
                            filtered_provider["order"] = provider_config["order"]

                        if "allow_fallbacks" in provider_config:
                            filtered_provider["allow_fallbacks"] = provider_config[
                                "allow_fallbacks"
                            ]

                        if "require_parameters" in provider_config:
                            filtered_provider["require_parameters"] = provider_config[
                                "require_parameters"
                            ]

                        if (
                            provider_config.get("data_collection")
                            and provider_config["data_collection"] != "allow"
                        ):
                            filtered_provider["data_collection"] = provider_config[
                                "data_collection"
                            ]

                        if provider_config.get("only"):
                            filtered_provider["only"] = provider_config["only"]

                        if provider_config.get("ignore"):
                            filtered_provider["ignore"] = provider_config["ignore"]

                        if provider_config.get("quantizations"):
                            filtered_provider["quantizations"] = provider_config[
                                "quantizations"
                            ]

                        if provider_config.get("sort"):
                            filtered_provider["sort"] = provider_config["sort"]

                        # 处理max_price配置
                        max_price = provider_config.get("max_price", {})
                        if any(max_price.values()):  # 如果有任何价格限制
                            filtered_max_price = {}
                            for key, value in max_price.items():
                                if value is not None:
                                    filtered_max_price[key] = value
                            if filtered_max_price:
                                filtered_provider["max_price"] = filtered_max_price

                        # 只有在有实际配置时才添加provider参数
                        if filtered_provider:
                            base_params["provider"] = filtered_provider
                            logger.info(
                                f"EnhancedOpenAI_Chat: 使用Provider配置: {filtered_provider}"
                            )
                else:
                    logger.info(
                        f"EnhancedOpenAI_Chat: 模型 {model_name} 不支持provider参数，跳过provider配置"
                    )

            # 如果是Gemini模型，移除不支持的参数
            if is_gemini:
                logger.info(
                    f"EnhancedOpenAI_Chat: 检测到Gemini模型 {model_name}，移除不支持的参数"
                )
                # Gemini可能不支持某些参数，只保留基本参数
                # 移除可能导致400错误的参数
                base_params = {
                    "model": model_name,
                    "messages": prompt,
                    "temperature": min(
                        self.temperature, 1.0
                    ),  # 确保temperature在合理范围内
                    # 移除 stop, max_tokens, top_p, frequency_penalty, presence_penalty 等
                }
            # 如果是Qwen模型，处理thinking参数
            elif is_qwen:
                logger.info(
                    f"EnhancedOpenAI_Chat: 检测到Qwen模型 {model_name}，配置thinking参数"
                )
                # Qwen模型的特殊处理：对于非流式调用，enable_thinking必须为false
                base_params = {
                    "model": model_name,
                    "messages": prompt,
                    "temperature": self.temperature,
                    # 对于非流式调用，必须设置enable_thinking为false
                    "extra_body": {"enable_thinking": False},
                    # 移除可能导致冲突的参数如stop
                }
                # 移除stop参数，因为它可能与qwen的thinking机制冲突
                # 注意：这里不包含stop参数

            return base_params

        if kwargs.get("model", None) is not None:
            model = kwargs.get("model", None)
            logger.info(f"Using model {model} for {num_tokens} tokens (approx)")
            completion_params = create_completion_params(model)
            response = safe_create_completion(self.client, completion_params, model)
        elif kwargs.get("engine", None) is not None:
            engine = kwargs.get("engine", None)
            logger.info(f"Using model {engine} for {num_tokens} tokens (approx)")
            completion_params = create_completion_params(engine)
            # 对于engine参数，需要特殊处理
            if is_gemini:
                # Gemini可能不支持engine参数，改用model
                pass  # completion_params已经包含model参数
            else:
                # 将model参数改为engine参数
                completion_params["engine"] = completion_params.pop("model")
            response = safe_create_completion(self.client, completion_params, engine)
        elif self.config is not None and "engine" in self.config:
            logger.info(
                f"Using engine {self.config['engine']} for {num_tokens} tokens (approx)"
            )
            completion_params = create_completion_params(self.config["engine"])
            if is_gemini:
                # Gemini使用model参数
                pass  # completion_params已经包含model参数
            else:
                # 将model参数改为engine参数
                completion_params["engine"] = completion_params.pop("model")
            response = safe_create_completion(
                self.client, completion_params, self.config["engine"]
            )
        elif self.config is not None and "model" in self.config:
            logger.info(
                f"Using model {self.config['model']} for {num_tokens} tokens (approx)"
            )
            completion_params = create_completion_params(self.config["model"])
            response = safe_create_completion(
                self.client, completion_params, self.config["model"]
            )
        else:
            if num_tokens > 3500:
                model = "gpt-3.5-turbo-16k"
            else:
                model = "gpt-3.5-turbo"

            logger.info(f"Using model {model} for {num_tokens} tokens (approx)")
            completion_params = create_completion_params(model)
            response = safe_create_completion(self.client, completion_params, model)

        # 提取实际的token使用情况（如果OpenAI响应中包含）
        actual_prompt_tokens = (
            getattr(response, "usage", {}).prompt_tokens
            if hasattr(response, "usage")
            else prompt_tokens
        )
        actual_completion_tokens = (
            getattr(response, "usage", {}).completion_tokens
            if hasattr(response, "usage")
            else 0
        )

        # 记录token使用情况
        global_token_tracker.add_vanna_tokens(
            prompt_tokens=actual_prompt_tokens,
            completion_tokens=actual_completion_tokens,
        )

        if kwargs.get("need_reasoning", False) == True:
            result_content = (
                response.choices[0].message.content
                if hasattr(response.choices[0].message, "content")
                else "未返回内容"
            )
            result_reasoning = (
                response.choices[0].message.reasoning_content
                if hasattr(response.choices[0].message, "reasoning_content")
                else None
            )
            return {"content": result_content, "reasoning": result_reasoning}
        else:
            pass
        # Find the first response from the chatbot that has text in it (some responses may not have text)
        for choice in response.choices:
            if "text" in choice:
                return choice.text

        # If no response with text is found, return the first response's content (which may be empty)
        return response.choices[0].message.content

    def submit_prompt_stream(self, prompt, **kwargs):
        if prompt is None or len(prompt) == 0:
            return

        for message in prompt:
            if message["role"] == "system":
                message["content"] += "\n" + self.system_prompt

        model_to_use = kwargs.get("model") or kwargs.get("engine")
        if not model_to_use and self.config is not None:
            model_to_use = self.config.get("model") or self.config.get("engine")
        if not model_to_use:
            model_to_use = "gpt-3.5-turbo"

        is_qwen = "qwen" in str(model_to_use).lower()

        completion_params = {
            "model": model_to_use,
            "messages": prompt,
            "temperature": self.temperature,
            "stream": True,
        }

        if is_qwen:
            completion_params["extra_body"] = {"enable_thinking": False}

        stream_resp = safe_create_completion(
            self.client, completion_params, model_to_use
        )

        for chunk in stream_resp:
            try:
                delta = chunk.choices[0].delta.content
            except Exception:
                delta = None
            if delta:
                yield delta

    def check_user_visualize_goal(self, question):
        # 检查用户是否明确要求绘图
        logger.info(f"Checking user visualize goal for question: {question}")

        # 使用 Pydantic 模型获取 JSON schema
        schema = VisualizeGoalResponse.model_json_schema()

        prompt = [
            {
                "role": "system",
                "content": f"请你判断用户的问题 '{question}' 是否明确要求绘图。请务必以 JSON 格式返回结果，格式必须遵循以下 JSON Schema：{json.dumps(schema)}。只返回 JSON 对象本身，不要包含任何其他说明文字。",
            },
            {"role": "user", "content": question},  # 再次提供问题可能有助于某些模型
        ]

        try:
            # 确定要使用的模型
            # 优先使用 config 中定义的 model 或 engine
            model_to_use = self.config.get("model") or self.config.get("engine")
            if not model_to_use:
                # 根据 token 数选择模型 (这里可能不需要，因为这个判断任务通常不长，可以选择一个默认支持JSON模式的模型)
                # 选择一个已知支持 JSON 模式的模型，例如 gpt-4o 或较新的 gpt-3.5-turbo
                model_to_use = "gpt-3.5-turbo"  # 或者根据可用性选择 gpt-4o 等
                logger.info(
                    f"No model/engine specified in config, defaulting to {model_to_use} for JSON mode."
                )

            logger.info(f"Using model {model_to_use} for visualize goal check.")

            # 检查是否使用的是特殊模型
            is_deepseek_model = "deepseek" in str(model_to_use).lower()
            is_gemini_model = "gemini" in str(model_to_use).lower()
            is_qwen_model = "qwen" in str(model_to_use).lower()

            # 对于特殊模型（deepseek、gemini、qwen），不使用JSON响应格式
            if is_deepseek_model or is_gemini_model or is_qwen_model:
                if is_deepseek_model:
                    model_type = "DeepSeek"
                elif is_gemini_model:
                    model_type = "Gemini"
                else:
                    model_type = "Qwen"
                logger.info(f"检测到{model_type}模型，使用文本响应格式")

                # 为特殊模型准备参数
                completion_params = {
                    "model": model_to_use,
                    "messages": prompt,
                    "temperature": 0,
                }

                # Gemini可能需要移除某些参数
                if is_gemini_model:
                    # 确保temperature在合理范围内
                    completion_params["temperature"] = min(
                        completion_params["temperature"], 1.0
                    )
                # Qwen模型需要处理thinking参数
                elif is_qwen_model:
                    # 对于非流式调用，enable_thinking必须为false
                    completion_params["extra_body"] = {"enable_thinking": False}

                response = safe_create_completion(
                    self.client, completion_params, model_to_use
                )

                # 手动解析文本响应
                response_content = response.choices[0].message.content
                logger.info(f"{model_type} Text Response:\n{response_content}")

                # 清理响应内容
                cleaned_content = clean_json_response(response_content)
                logger.info(f"{model_type} Cleaned content: {cleaned_content}")

                # 尝试解析为JSON，如果失败则使用关键词匹配
                try:
                    result = VisualizeGoalResponse.model_validate_json(cleaned_content)
                    return result.requires_visualization
                except:
                    # 简单解析文本响应，寻找yes/no/true/false等关键词
                    requires_visualization = any(
                        keyword in response_content.lower()
                        for keyword in ["yes", "true", "需要", "要求", "绘图", "可视化"]
                    )
                    return requires_visualization
            else:
                # 对于其他模型，使用JSON响应格式
                completion_params = {
                    "model": model_to_use,
                    "messages": prompt,
                    "temperature": 0,  # 对于判断任务，温度设为0
                    "response_format": {"type": "json_object"},  # 请求 JSON 输出
                }
                response = safe_create_completion(
                    self.client, completion_params, model_to_use
                )

                response_content = response.choices[0].message.content
                logger.info(f"OpenAI JSON Response:\n{response_content}")

                # 清理JSON响应
                cleaned_content = clean_json_response(response_content)
                logger.info(f"OpenAI Cleaned JSON content: {cleaned_content}")

                # 解析 JSON 响应
                result = VisualizeGoalResponse.model_validate_json(cleaned_content)
                return result.requires_visualization

        except Exception as e:
            logger.info(f"Error checking visualize goal with OpenAI: {e}")
            # 如果 API 调用或解析失败，返回默认值 False
            return False
