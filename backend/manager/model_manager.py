from typing import List, Dict, Optional, Union
from pydantic import BaseModel, Field
import json
import ollama
import openai
from abc import ABC, abstractmethod
import logging
from backend.utils.openai_compat import normalize_openai_base_url

logger = logging.getLogger(__name__)


def safe_create_completion_for_model_manager(
    model_name, messages, temperature=0.1, response_format=None
):
    """
    安全的创建completion请求，包含provider参数错误的自动回退机制
    专门用于模型管理器中的OpenAI调用
    """
    # 构建完整的请求参数
    completion_params = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
    }

    # 只有在指定了response_format时才添加
    if response_format:
        completion_params["response_format"] = response_format

    try:
        # 首先尝试原始请求
        response = openai.chat.completions.create(**completion_params)
        return response
    except Exception as e:
        error_msg = str(e).lower()

        # 检查是否是provider参数相关的错误
        if "provider" in error_msg and (
            "unexpected" in error_msg or "invalid" in error_msg
        ):
            logger.warning(
                f"模型管理器: 模型 {model_name} 不支持provider参数，尝试移除provider参数后重试"
            )

            # 创建不包含provider参数的副本
            fallback_params = completion_params.copy()
            if "provider" in fallback_params:
                del fallback_params["provider"]

            try:
                # 重试不带provider参数的请求
                response = openai.chat.completions.create(**fallback_params)
                logger.info(
                    f"模型管理器: 模型 {model_name} 成功使用fallback请求（无provider参数）"
                )
                return response
            except Exception as fallback_e:
                logger.error(
                    f"模型管理器: 模型 {model_name} fallback请求也失败: {str(fallback_e)}"
                )
                raise fallback_e
        else:
            # 其他类型的错误，直接抛出
            raise e


class ColumnInfo(BaseModel):
    """列信息模型"""

    original_name: str
    translated_name: str
    recommended_type: str
    type_description: str
    sample_values: List[str]


class ColumnAnalysis(BaseModel):
    """列分析结果模型"""

    columns: List[ColumnInfo]
    total_rows: int
    total_columns: int


class BaseModelProvider(ABC):
    """模型提供商基类"""

    @abstractmethod
    def chat(
        self, messages: List[Dict[str, str]], format: Optional[Dict] = None
    ) -> str:
        pass

    @abstractmethod
    def list_models(self) -> List[str]:
        pass


class OllamaProvider(BaseModelProvider):
    """Ollama提供商"""

    def __init__(self, config: Dict):
        self.config = config
        self.ollama_url = config.get("ollama_url", "http://localhost:11434")
        self.model = config.get("ollama_model", "llama2")
        self.client = ollama.Client(host=self.ollama_url)

    def chat(
        self, messages: List[Dict[str, str]], format: Optional[Dict] = None
    ) -> str:
        try:
            response = self.client.chat(
                model=self.model, messages=messages, format=format
            )
            return response.message.content or ""
        except Exception as e:
            raise ValueError(f"Ollama调用失败: {str(e)}")

    def list_models(self) -> List[str]:
        try:
            response = self.client.list()
            return [model.get("model", "") for model in response.get("models", [])]
        except Exception:
            return []


class OpenAIProvider(BaseModelProvider):
    """OpenAI提供商"""

    def __init__(self, config: Dict):
        self.config = config
        openai.api_key = config.get("api_key")
        openai.base_url = normalize_openai_base_url(config.get("api_base"))
        self.model = config.get("model_name", "gpt-3.5-turbo")
        self.temperature = config.get("temperature", 0.1)

        # 尝试验证模型名称
        self._verify_model()

    def _verify_model(self):
        """验证配置的模型名称是否有效"""
        logger = logging.getLogger(__name__)
        try:
            # 获取可用模型列表
            available_models = self.list_models()
            if available_models and self.model not in available_models:
                logger.warning(
                    f"警告: 配置的模型 '{self.model}' 不在可用模型列表中。可用模型: {', '.join(available_models[:5])}{'...' if len(available_models) > 5 else ''}"
                )
                # 不抛出异常，只记录警告，因为模型可能仍然可用（可能是新模型或私有模型）
        except Exception as e:
            # 如果无法获取模型列表，只记录警告
            logger.warning(f"无法验证模型 '{self.model}' 是否有效: {str(e)}")

    def chat(
        self, messages: List[Dict[str, str]], format: Optional[Dict] = None
    ) -> str:
        try:
            logger = logging.getLogger(__name__)
            logger.info(
                f"调用OpenAI，模型: {self.model}, API Base: {openai.base_url if hasattr(openai, 'base_url') else '默认'}"
            )

            response = safe_create_completion_for_model_manager(
                model_name=self.model,
                messages=messages,
                temperature=self.temperature,
                response_format={"type": "json_object"} if format else None,
            )
            return response.choices[0].message.content or ""
        except openai.NotFoundError as e:
            # 模型不存在或API端点错误
            error_msg = f"OpenAI模型或API端点错误: {str(e)}。请检查模型名称'{self.model}'是否存在以及API基础URL是否正确。"
            logging.error(error_msg)
            raise ValueError(error_msg)
        except openai.AuthenticationError as e:
            # API密钥无效
            error_msg = f"OpenAI API认证失败: {str(e)}。请检查API密钥是否有效。"
            logging.error(error_msg)
            raise ValueError(error_msg)
        except Exception as e:
            error_msg = f"OpenAI调用失败: {str(e)}"
            logging.error(error_msg)
            raise ValueError(error_msg)

    def list_models(self) -> List[str]:
        logger = logging.getLogger(__name__)
        try:
            logger.info(
                f"正在获取OpenAI可用模型列表, API Base: {openai.base_url if hasattr(openai, 'base_url') else '默认'}"
            )
            response = openai.models.list()
            models = [model.id for model in response.data]
            logger.info(f"成功获取到 {len(models)} 个OpenAI模型")
            return models
        except openai.NotFoundError as e:
            logger.error(f"获取模型列表失败(404): {str(e)}")
            return []
        except openai.AuthenticationError as e:
            logger.error(f"获取模型列表认证失败: {str(e)}")
            return []
        except Exception as e:
            logger.error(f"获取模型列表时出错: {str(e)}")
            return []


class ModelManager:
    def __init__(self, config: Dict):
        """初始化模型管理器"""
        self.config = config
        self.provider = self._get_provider()

    def _get_provider(self) -> BaseModelProvider:
        """获取模型提供商实例"""
        provider = self.config.get("type", "ollama").lower()
        providers = {
            "ollama": OllamaProvider,
            "openai": OpenAIProvider,
            "vanna": OpenAIProvider,  # Vanna使用OpenAI作为后端
        }
        provider_class = providers.get(provider)
        if not provider_class:
            raise ValueError(f"不支持的模型提供商: {provider}")
        return provider_class(self.config)

    def analyze_columns(
        self, columns: List[str], sample_data: Dict[str, List[str]], total_rows: int
    ) -> ColumnAnalysis:
        """分析列名并返回结构化结果"""
        # 获取当前语言设置
        from backend.services.vanna_service import vanna_manager

        config = vanna_manager.get_config()
        language = config.get("language", {}).get("language", "zh-CN")

        # 根据语言设置构建不同的提示词
        if language == "zh-CN":
            prompt = f"""请分析以下数据列并提供结构化信息：
1. 将中文列名翻译成英文
2. 根据数据内容推荐最合适的MySQL数据类型
3. 提供数据类型的详细说明（使用中文）

数据列信息：
{json.dumps(columns, ensure_ascii=False)}

数据样本：
{json.dumps(sample_data, ensure_ascii=False)}

请以JSON格式返回结果，必须使用"columns"作为顶层键，包含以下字段：
- original_name: 原始列名
- translated_name: 英文翻译
- recommended_type: 推荐的MySQL数据类型
- type_description: 数据类型说明（中文）
- sample_values: 样本值列表

格式示例:
{{"columns": [
  {{
    "original_name": "列名",
    "translated_name": "column_name",
    "recommended_type": "类型",
    "type_description": "描述",
    "sample_values": ["样本1", "样本2"]
  }}
]}}

请确保返回的是有效的JSON格式，且顶层键必须是"columns"。"""
        else:
            # 英文提示词
            prompt = f"""Please analyze the following data columns and provide structured information:
1. Translate Chinese column names to English
2. Recommend the most suitable MySQL data type based on data content
3. Provide detailed description of the data type (in English)

Column information:
{json.dumps(columns, ensure_ascii=False)}

Data samples:
{json.dumps(sample_data, ensure_ascii=False)}

Please return results in JSON format, using "columns" as the top-level key, containing the following fields:
- original_name: Original column name
- translated_name: English translation
- recommended_type: Recommended MySQL data type
- type_description: Data type description (in English)
- sample_values: Sample values list

Format example:
{{"columns": [
  {{
    "original_name": "Column Name",
    "translated_name": "column_name",
    "recommended_type": "Type",
    "type_description": "Description",
    "sample_values": ["Sample1", "Sample2"]
  }}
]}}

Please ensure the returned result is valid JSON format with "columns" as the top-level key."""

        # 使用模型提供商进行结构化输出（移除format参数以兼容OLLAMA）
        response = self.provider.chat(messages=[{"role": "user", "content": prompt}])

        # 解析响应并创建ColumnAnalysis对象
        try:
            logger.info(f"解析模型响应llm: {response}")

            # 使用健壮的JSON解析逻辑（借鉴DDL生成过程）
            json_string = None
            result = None

            try:
                # Find the first opening curly brace
                start_index = response.find("{")
                if start_index == -1:
                    raise ValueError(
                        "Could not find the start of a JSON object ('{') in the response."
                    )

                brace_level = 0
                end_index = -1
                # Iterate through the string to find the matching closing brace
                for i, char in enumerate(response[start_index:]):
                    if char == "{":
                        brace_level += 1
                    elif char == "}":
                        brace_level -= 1
                        if brace_level == 0:
                            end_index = start_index + i
                            break  # Found the matching brace

                if end_index == -1:
                    # Fallback: Try cleaning markdown if matching brace wasn't found cleanly
                    import re

                    cleaned_response = re.sub(
                        r"^```json\s*|\s*```$", "", response.strip(), flags=re.MULTILINE
                    )
                    if cleaned_response.startswith("{") and cleaned_response.endswith(
                        "}"
                    ):
                        json_string = cleaned_response
                        logger.warning(
                            "JSON object boundaries possibly unclear, using regex cleaned response"
                        )
                    else:
                        raise ValueError(
                            "Could not find the matching end brace ('}') for the JSON object."
                        )
                else:
                    # Extract the substring containing the JSON object
                    json_string = response[start_index : end_index + 1]

                # Attempt to parse the extracted string
                result = json.loads(json_string)
                logger.info(f"解析模型响应json: {result}")

            except json.JSONDecodeError as jde:
                # Re-raise with more context
                error_message = f"Failed to parse JSON: {jde}. "
                if json_string is not None:
                    error_message += f"Attempted to parse: {json_string[:200]}..."  # Log beginning of attempted parse
                else:
                    error_message += f"Could not extract valid JSON boundaries. Original response: {response[:200]}..."
                # Log the full problematic string if available for easier debugging
                logger.error(f"Full string attempted for JSON parsing: {json_string}")
                raise json.JSONDecodeError(error_message, jde.doc, jde.pos)

            if not isinstance(result, dict):
                raise ValueError(
                    f"LLM response parsed, but it's not a dictionary. Type: {type(result)}"
                )

            column_infos = [ColumnInfo(**col) for col in result.get("columns", [])]
            logger.info(f"解析模型响应column_infos: {column_infos}")
            return ColumnAnalysis(
                columns=column_infos, total_rows=total_rows, total_columns=len(columns)
            )
        except Exception as e:
            logger.error(f"解析模型响应失败: {str(e)}")
            # Log the original response for debugging
            logger.error(f"原始响应内容: {response}")
            raise ValueError(f"解析模型响应失败: {str(e)}")

    def translate_text(self, text: str) -> str:
        """翻译文本"""
        prompt = f"请将以下中文翻译成英文，只返回英文翻译结果，不要其他解释：{text}"
        response = self.provider.chat(messages=[{"role": "user", "content": prompt}])
        return response.strip()

    def get_available_models(self) -> List[str]:
        """获取可用的模型列表"""
        return self.provider.list_models()
