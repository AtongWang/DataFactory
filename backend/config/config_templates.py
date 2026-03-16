import os


class DatabaseConfig:
    """数据库配置基类"""

    def __init__(self):
        self.type = "sqlite"  # 默认类型
        self.database_name = ""  # 数据库名称


class SQLiteConfig(DatabaseConfig):
    """SQLite数据库配置"""

    def __init__(self):
        super().__init__()
        self.type = "sqlite"
        self.url = "db_file/wisdomindata.db"


class MySQLConfig(DatabaseConfig):
    """MySQL数据库配置"""

    def __init__(self):
        super().__init__()
        self.type = "mysql"
        self.host = ""
        self.port = "3306"
        self.username = ""
        self.password = ""


class PostgreSQLConfig(DatabaseConfig):
    """PostgreSQL数据库配置"""

    def __init__(self):
        super().__init__()
        self.type = "postgres"
        self.host = ""
        self.port = "5432"
        self.username = ""
        self.password = ""


class ChromaDBConfig(DatabaseConfig):
    """ChromaDB数据库配置"""

    def __init__(self):
        super().__init__()
        self.type = "chroma"
        self.path = "./chroma_db"  # 默认路径
        self.embedding_provider = "ollama"
        self.embedding_function = "llama2"  # 默认使用llama2作为嵌入模型
        self.embedding_ollama_url = "http://localhost:11434"  # 嵌入模型专用Ollama URL
        self.embedding_api_key = ""
        self.embedding_api_base = "https://api.openai.com/v1"


class ModelConfig:
    """模型配置基类"""

    def __init__(self):
        self.type = "vanna"  # 默认类型


class VannaConfig(ModelConfig):
    """Vanna.AI模型配置"""

    def __init__(self):
        super().__init__()
        self.type = "vanna"
        self.api_key = ""
        self.model_name = ""
        self.temperature = 0.7


class OllamaConfig(ModelConfig):
    """Ollama模型配置"""

    def __init__(self):
        super().__init__()
        self.type = "ollama"
        self.ollama_url = "http://localhost:11434"
        self.ollama_model = "llama2"
        self.temperature = 0.7
        self.system_prompt = "请你根据用户的语言类型，使用同样的语言类型回答用户的问题。同时在图表代码生成时，图表名称、坐标轴名称等图表文字要素使用相同的语言类型。"
        self.num_ctx = 128000


class OpenAIConfig(ModelConfig):
    """OpenAI模型配置"""

    def __init__(self):
        super().__init__()
        self.type = "openai"
        self.api_key = ""
        self.model_name = "gpt-3.5-turbo"
        self.api_base = "https://api.openai.com/v1"
        self.temperature = 0.7
        self.system_prompt = "请你根据用户的语言类型，使用同样的语言类型回答用户的问题。同时在图表代码生成时，图表名称、坐标轴名称等图表文字要素使用相同的语言类型。"
        self.num_ctx = 128000
        # OpenRouter Provider Routing 配置
        self.provider = {
            "order": [],  # 提供商优先级列表 ["anthropic", "openai"]
            "allow_fallbacks": True,  # 是否允许备用提供商
            "require_parameters": False,  # 仅使用支持所有参数的提供商
            "data_collection": "allow",  # "allow" | "deny" 数据收集策略
            "only": [],  # 仅允许的提供商列表
            "ignore": [],  # 忽略的提供商列表
            "quantizations": [],  # 量化级别过滤 ["int4", "int8"]
            "sort": None,  # 排序策略 "price" | "throughput" | "latency"
            "max_price": {  # 最大价格限制
                "prompt": None,  # 每百万token提示价格上限
                "completion": None,  # 每百万token完成价格上限
                "image": None,  # 每张图像价格上限
                "request": None,  # 每次请求价格上限
            },
        }
        # Provider Routing 功能开关
        self.use_provider_routing = True  # 默认启用，可设为False来禁用


class NamingModelConfig:
    """会话自动命名模型配置"""

    def __init__(self):
        self.enabled = False
        self.prompt_template = "根据以下对话内容，为这个对话生成一个简短的标题（不超过20个字符）：{conversation}"
        self.use_system_model = True  # 是否使用系统模型
        self.model_type = "ollama"  # 默认使用Ollama
        self.ollama_url = "http://localhost:11434"
        self.ollama_model = "llama2"
        self.openai_api_key = ""
        self.openai_model = "gpt-3.5-turbo"
        self.openai_api_base = "https://api.openai.com/v1"
        self.vanna_api_key = ""
        self.vanna_model_name = ""


class Neo4jConfig:
    """Neo4j图数据库配置"""

    def __init__(self):
        self.uri = "bolt://localhost:7687"
        self.user = "neo4j"
        self.password = "12345678"


class LanguageConfig:
    """语言配置"""

    def __init__(self):
        self.language = "zh-CN"  # 默认中文


class AppConfig:
    """应用配置类"""

    def __init__(self):
        self.database = SQLiteConfig()  # 默认使用SQLite
        self.model = VannaConfig()  # 默认使用Vanna
        self.store_database = ChromaDBConfig()
        self.naming_model = NamingModelConfig()  # 添加命名模型配置
        self.neo4j = Neo4jConfig()  # 添加Neo4j配置
        self.language = LanguageConfig()  # 添加语言配置

    @classmethod
    def from_dict(cls, data):
        """从字典创建配置对象"""
        config = cls()

        # 设置向量数据库配置
        config.store_database = ChromaDBConfig()
        config.store_database.path = data.get("store_database", {}).get("path", "")
        config.store_database.embedding_provider = data.get("store_database", {}).get(
            "embedding_provider", "ollama"
        )
        config.store_database.embedding_function = data.get("store_database", {}).get(
            "embedding_function", "llama2"
        )
        config.store_database.embedding_ollama_url = data.get("store_database", {}).get(
            "embedding_ollama_url", "http://localhost:11434"
        )
        config.store_database.embedding_api_key = data.get("store_database", {}).get(
            "embedding_api_key", ""
        )
        config.store_database.embedding_api_base = data.get("store_database", {}).get(
            "embedding_api_base", "https://api.openai.com/v1"
        )

        # 设置数据库配置
        db_type = data.get("database", {}).get("type", "sqlite")
        if db_type == "sqlite":
            config.database = SQLiteConfig()
            config.database.url = data.get("database", {}).get("url", "")
        elif db_type == "mysql":
            config.database = MySQLConfig()
            db_data = data.get("database", {})
            config.database.host = db_data.get("host", "")
            config.database.port = db_data.get("port", "3306")
            config.database.username = db_data.get("username", "")
            config.database.password = db_data.get("password", "")
            config.database.database_name = db_data.get("database_name", "")
        elif db_type == "postgres":
            config.database = PostgreSQLConfig()
            db_data = data.get("database", {})
            config.database.host = db_data.get("host", "")
            config.database.port = db_data.get("port", "5432")
            config.database.username = db_data.get("username", "")
            config.database.password = db_data.get("password", "")
            config.database.database_name = db_data.get("database_name", "")

        # 设置模型配置
        model_type = data.get("model", {}).get("type", "vanna")
        if model_type == "vanna":
            config.model = VannaConfig()
            model_data = data.get("model", {})
            config.model.api_key = model_data.get("api_key", "")
            config.model.model_name = model_data.get("model_name", "")
            config.model.temperature = model_data.get("temperature", 0.7)
        elif model_type == "ollama":
            config.model = OllamaConfig()
            model_data = data.get("model", {})
            config.model.ollama_url = model_data.get(
                "ollama_url", config.model.ollama_url
            )
            config.model.ollama_model = model_data.get(
                "ollama_model", config.model.ollama_model
            )
            config.model.temperature = model_data.get(
                "temperature", config.model.temperature
            )
            config.model.system_prompt = model_data.get(
                "system_prompt", config.model.system_prompt
            )
            config.model.num_ctx = model_data.get("num_ctx", config.model.num_ctx)
        elif model_type == "openai":
            config.model = OpenAIConfig()
            model_data = data.get("model", {})
            config.model.api_key = model_data.get("api_key", config.model.api_key)
            config.model.model_name = model_data.get(
                "model_name", config.model.model_name
            )
            config.model.api_base = model_data.get("api_base", config.model.api_base)
            config.model.temperature = model_data.get(
                "temperature", config.model.temperature
            )
            config.model.system_prompt = model_data.get(
                "system_prompt", config.model.system_prompt
            )
            config.model.num_ctx = model_data.get("num_ctx", config.model.num_ctx)

            # 解析 provider 配置
            provider_data = model_data.get("provider", {})
            if provider_data:
                config.model.provider.update(
                    {
                        "order": provider_data.get("order", []),
                        "allow_fallbacks": provider_data.get("allow_fallbacks", True),
                        "require_parameters": provider_data.get(
                            "require_parameters", False
                        ),
                        "data_collection": provider_data.get(
                            "data_collection", "allow"
                        ),
                        "only": provider_data.get("only", []),
                        "ignore": provider_data.get("ignore", []),
                        "quantizations": provider_data.get("quantizations", []),
                        "sort": provider_data.get("sort", None),
                        "max_price": provider_data.get(
                            "max_price",
                            {
                                "prompt": None,
                                "completion": None,
                                "image": None,
                                "request": None,
                            },
                        ),
                    }
                )

            # 解析 use_provider_routing 配置
            config.model.use_provider_routing = model_data.get(
                "use_provider_routing", True
            )
        # 设置命名模型配置
        naming_data = data.get("naming_model", {})
        config.naming_model = NamingModelConfig()
        config.naming_model.enabled = naming_data.get("enabled", False)
        config.naming_model.prompt_template = naming_data.get(
            "prompt_template", config.naming_model.prompt_template
        )
        config.naming_model.use_system_model = naming_data.get("use_system_model", True)
        config.naming_model.model_type = naming_data.get("model_type", "ollama")

        # 根据命名模型类型设置相应字段
        config.naming_model.ollama_url = naming_data.get(
            "ollama_url", "http://localhost:11434"
        )
        config.naming_model.ollama_model = naming_data.get("ollama_model", "llama2")
        config.naming_model.openai_api_key = naming_data.get("openai_api_key", "")
        config.naming_model.openai_model = naming_data.get(
            "openai_model", "gpt-3.5-turbo"
        )
        config.naming_model.openai_api_base = naming_data.get(
            "openai_api_base", "https://api.openai.com/v1"
        )
        config.naming_model.vanna_api_key = naming_data.get("vanna_api_key", "")
        config.naming_model.vanna_model_name = naming_data.get("vanna_model_name", "")

        # 设置Neo4j配置
        neo4j_data = data.get("neo4j", {})
        config.neo4j = Neo4jConfig()
        config.neo4j.uri = (
            neo4j_data.get("uri")
            or os.environ.get("NEO4J_URI")
            or "bolt://localhost:7687"
        )
        config.neo4j.user = (
            neo4j_data.get("user") or os.environ.get("NEO4J_USER") or "neo4j"
        )
        config.neo4j.password = (
            neo4j_data.get("password") or os.environ.get("NEO4J_PASSWORD") or "12345678"
        )

        # 设置语言配置
        language_data = data.get("language", {})
        config.language = LanguageConfig()
        config.language.language = language_data.get("language", "zh-CN")

        return config

    def to_dict(self):
        """将配置转换为字典"""
        result = {
            "database": {"type": self.database.type},
            "model": {"type": self.model.type},
            "store_database": {
                "path": self.store_database.path,
                "embedding_provider": self.store_database.embedding_provider,
                "embedding_function": self.store_database.embedding_function,
                "embedding_ollama_url": self.store_database.embedding_ollama_url,
                "embedding_api_key": self.store_database.embedding_api_key,
                "embedding_api_base": self.store_database.embedding_api_base,
            },
            "naming_model": {
                "enabled": self.naming_model.enabled,
                "prompt_template": self.naming_model.prompt_template,
                "use_system_model": self.naming_model.use_system_model,
                "model_type": self.naming_model.model_type,
                "ollama_url": self.naming_model.ollama_url,
                "ollama_model": self.naming_model.ollama_model,
                "openai_api_key": self.naming_model.openai_api_key,
                "openai_model": self.naming_model.openai_model,
                "openai_api_base": self.naming_model.openai_api_base,
                "vanna_api_key": self.naming_model.vanna_api_key,
                "vanna_model_name": self.naming_model.vanna_model_name,
            },
            "neo4j": {
                "uri": self.neo4j.uri,
                "user": self.neo4j.user,
                "password": self.neo4j.password,
            },
            "language": {"language": self.language.language},
        }

        # 根据数据库类型添加相应的字段
        if self.database.type == "mysql":
            result["database"].update(
                {
                    "host": self.database.host,
                    "port": self.database.port,
                    "username": self.database.username,
                    "password": self.database.password,
                    "database_name": self.database.database_name,
                }
            )
        elif self.database.type == "postgres":
            result["database"].update(
                {
                    "host": self.database.host,
                    "port": self.database.port,
                    "username": self.database.username,
                    "password": self.database.password,
                    "database_name": self.database.database_name,
                }
            )
        elif self.database.type == "sqlite":
            result["database"].update({"url": self.database.url})

        # 根据模型类型添加相应的字段
        if self.model.type == "vanna":
            result["model"].update(
                {
                    "api_key": self.model.api_key,
                    "model_name": self.model.model_name,
                    "temperature": self.model.temperature,
                }
            )
        elif self.model.type == "ollama":
            result["model"].update(
                {
                    "ollama_url": self.model.ollama_url,
                    "ollama_model": self.model.ollama_model,
                    "temperature": self.model.temperature,
                    "system_prompt": self.model.system_prompt,
                    "num_ctx": self.model.num_ctx,
                }
            )
        elif self.model.type == "openai":
            result["model"].update(
                {
                    "api_key": self.model.api_key,
                    "model_name": self.model.model_name,
                    "api_base": self.model.api_base,
                    "temperature": self.model.temperature,
                    "system_prompt": self.model.system_prompt,
                    "num_ctx": self.model.num_ctx,
                    "provider": self.model.provider,
                }
            )

        return result
