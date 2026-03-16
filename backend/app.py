from flask import Flask
import logging
from backend.utils.db_utils import init_import_history_db
from backend.services.vanna_service import vanna_manager, vanna_app
from backend.services.model_service import model_manager
from backend.routes.data_import_routes import CustomJSONEncoder
from logging.handlers import RotatingFileHandler
import os
import atexit


def setup_logging():
    log_dir = "logs"  # 可以配置日志目录
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_file = os.path.join(log_dir, "app.log")

    # 配置日志格式
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s [%(name)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 获取根 logger
    root_logger = logging.getLogger()

    # 清理已有的handlers，避免重复日志
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    log_level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_name, logging.INFO)

    # 创建一个 RotatingFileHandler，当日志文件达到一定大小时会自动分割
    # maxBytes=10MB, backupCount=5 表示最多保留5个日志文件，每个最大10MB
    file_handler = RotatingFileHandler(
        log_file, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(log_level)

    root_logger.addHandler(file_handler)
    root_logger.setLevel(log_level)

    # 可选：如果你还想在控制台看到日志
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(log_level)
    root_logger.addHandler(stream_handler)

    noisy_loggers = ["neo4j", "httpcore", "httpx", "openai", "urllib3"]
    for logger_name in noisy_loggers:
        lib_logger = logging.getLogger(logger_name)
        lib_logger.setLevel(max(log_level, logging.WARNING))

    # 注册清理函数，在程序退出时移除所有处理器
    def cleanup_handlers():
        for handler in root_logger.handlers[:]:
            handler.close()
            root_logger.removeHandler(handler)

    atexit.register(cleanup_handlers)


def create_app():
    # 在创建 Flask app 之前或之后立即设置日志
    setup_logging()

    app = Flask(__name__, template_folder="../templates", static_folder="../static")

    # 注册自定义JSON编码器，处理NumPy和Pandas数据类型
    app.json_encoder = CustomJSONEncoder

    # 确保数据库表存在
    init_import_history_db()

    # 注册路由
    from backend.routes.main_routes import main_bp
    from backend.routes.settings_routes import settings_bp
    from backend.routes.data_import_routes import data_import_bp
    from backend.routes.database_routes import database_bp
    from backend.routes.qa_routes import qa_bp
    from backend.routes.knowledge_graph_routes import kg_bp
    from backend.routes.kg_qa_routes import kgqa_bp
    from backend.routes.agent_api import agent_api

    app.register_blueprint(main_bp)
    app.register_blueprint(settings_bp)
    app.register_blueprint(data_import_bp)
    app.register_blueprint(database_bp)
    app.register_blueprint(qa_bp)
    app.register_blueprint(kg_bp)
    app.register_blueprint(kgqa_bp)
    app.register_blueprint(agent_api)
    return app


# 只在直接运行时才创建app实例，避免被导入时自动配置日志
if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)
else:
    # 当被作为模块导入时，创建一个lazy app实例
    app = None

    def get_app():
        """获取app实例，支持延迟创建"""
        global app
        if app is None:
            app = create_app()
        return app
