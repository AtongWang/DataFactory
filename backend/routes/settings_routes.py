from flask import Blueprint, render_template, request, jsonify, current_app
from backend.services.vanna_service import vanna_manager
import logging
import requests


app = current_app
settings_bp = Blueprint("settings", __name__)
logger = logging.getLogger(__name__)


@settings_bp.route("/settings")
def settings():
    config = vanna_manager.get_config()
    return render_template("settings.html", config=config)


@settings_bp.route("/api/save-settings", methods=["POST"])
def save_settings():
    try:
        config = request.json
        if not config:
            return jsonify({"status": "error", "message": "配置数据不能为空"}), 400

        # 验证必要的配置项
        if (
            "database" not in config
            or "model" not in config
            or "store_database" not in config
        ):
            return jsonify({"status": "error", "message": "配置数据格式不正确"}), 400

        vanna_manager.update_config(config, apply_async=True)

        # 获取已注册的回调数量
        callback_count = len(vanna_manager._config_update_callbacks)

        return jsonify(
            {
                "status": "success",
                "message": f"设置保存成功，已通知 {callback_count} 个系统组件更新配置",
            }
        )
    except Exception as e:
        app.logger.error(f"保存设置时发生错误: {str(e)}")
        return jsonify({"status": "error", "message": f"保存设置失败: {str(e)}"}), 500


@settings_bp.route("/api/test-database-connection", methods=["POST"])
def test_database_connection():
    db_config = request.json

    try:
        if db_config["type"] == "mysql":
            import pymysql

            conn = pymysql.connect(
                host=db_config["host"],
                user=db_config["username"],
                password=db_config["password"],
                database=db_config["database_name"],
                port=int(db_config["port"]) if db_config["port"] else 3306,
            )
            conn.close()
        elif db_config["type"] == "postgres":
            import psycopg2

            conn = psycopg2.connect(
                host=db_config["host"],
                user=db_config["username"],
                password=db_config["password"],
                dbname=db_config["database_name"],
                port=db_config["port"] if db_config["port"] else "5432",
            )
            conn.close()
        elif db_config["type"] == "sqlite":
            import sqlite3

            conn = sqlite3.connect(db_config["database_name"] or ":memory:")
            conn.close()

        return jsonify({"status": "success", "message": "连接成功！"})
    except Exception as e:
        return jsonify({"status": "error", "message": f"连接失败: {str(e)}"})


@settings_bp.route("/api/test-neo4j-connection", methods=["POST"])
def test_neo4j_connection():
    """测试Neo4j连接"""
    neo4j_config = request.json

    try:
        from neo4j import GraphDatabase

        driver = GraphDatabase.driver(
            neo4j_config["uri"], auth=(neo4j_config["user"], neo4j_config["password"])
        )

        # 测试连接
        with driver.session() as session:
            result = session.run("RETURN 1 AS test")
            test_value = result.single()[0]
            if test_value != 1:
                raise Exception("连接测试失败")

        driver.close()
        return jsonify({"status": "success", "message": "Neo4j连接成功！"})
    except Exception as e:
        return jsonify({"status": "error", "message": f"Neo4j连接失败: {str(e)}"})


@settings_bp.route("/api/naming-settings", methods=["GET"])
def get_naming_settings():
    """获取当前的命名设置"""
    try:
        config = vanna_manager.get_config()
        naming_config = config.get("naming_model", {})
        return jsonify({"status": "success", "config": naming_config})
    except Exception as e:
        app.logger.error(f"获取命名设置时发生错误: {str(e)}")
        return jsonify(
            {"status": "error", "message": f"获取命名设置失败: {str(e)}"}
        ), 500


@settings_bp.route("/api/toggle-naming", methods=["POST"])
def toggle_naming():
    """快速切换自动命名功能的启用状态"""
    try:
        data = request.json
        enabled = data.get("enabled")

        if enabled is None:
            return jsonify({"status": "error", "message": "缺少enabled参数"}), 400

        # 获取当前配置
        config = vanna_manager.get_config()

        # 确保配置中有naming_model部分
        if "naming_model" not in config:
            config["naming_model"] = {
                "enabled": False,
                "prompt_template": "根据以下对话内容，为这个对话生成一个简短的标题（不超过20个字符）：{conversation}",
            }

        # 更新启用状态
        config["naming_model"]["enabled"] = enabled

        vanna_manager.update_config(config, apply_async=True)

        return jsonify(
            {
                "status": "success",
                "message": "自动命名功能已" + ("启用" if enabled else "禁用"),
                "config": config["naming_model"],
            }
        )
    except Exception as e:
        app.logger.error(f"切换命名功能时发生错误: {str(e)}")
        return jsonify({"status": "error", "message": f"操作失败: {str(e)}"}), 500


@settings_bp.route("/api/save-language-setting", methods=["POST"])
def save_language_setting():
    """保存语言设置"""
    try:
        data = request.json
        language = data.get("language")

        if not language:
            return jsonify({"status": "error", "message": "缺少language参数"}), 400

        # 验证语言代码
        valid_languages = ["zh-CN", "en-US"]
        if language not in valid_languages:
            return jsonify(
                {"status": "error", "message": f"不支持的语言: {language}"}
            ), 400

        # 获取当前配置（字典格式）
        config = vanna_manager.get_config()

        # 确保配置中有language部分
        if "language" not in config:
            config["language"] = {"language": "zh-CN"}

        # 更新语言设置
        config["language"]["language"] = language

        vanna_manager.update_config(config, apply_async=True)

        return jsonify(
            {
                "status": "success",
                "message": f"语言设置已保存: {language}",
                "language": language,
            }
        )
    except Exception as e:
        current_app.logger.error(f"保存语言设置时发生错误: {str(e)}")
        return jsonify(
            {"status": "error", "message": f"保存语言设置失败: {str(e)}"}
        ), 500


@settings_bp.route("/api/get-language-setting", methods=["GET"])
def get_language_setting():
    """获取当前语言设置"""
    try:
        config = vanna_manager.get_config()
        language = "zh-CN"  # 默认值

        if "language" in config and "language" in config["language"]:
            language = config["language"]["language"]

        return jsonify({"status": "success", "language": language})
    except Exception as e:
        current_app.logger.error(f"获取语言设置时发生错误: {str(e)}")
        return jsonify(
            {"status": "error", "message": f"获取语言设置失败: {str(e)}"}
        ), 500


@settings_bp.route("/api/ollama-models", methods=["GET"])
def get_ollama_models():
    """获取Ollama可用模型列表"""
    ollama_url = request.args.get("url")

    if not ollama_url:
        return jsonify({"status": "error", "message": "未提供Ollama URL"}), 400

    try:
        # 确保URL格式正确
        if not ollama_url.startswith(("http://", "https://")):
            ollama_url = "http://" + ollama_url

        # 请求Ollama API获取模型列表
        response = requests.get(f"{ollama_url}/api/tags", timeout=5)

        if response.status_code != 200:
            return jsonify(
                {
                    "status": "error",
                    "message": f"Ollama服务返回错误状态码: {response.status_code}",
                }
            ), 500

        models_data = response.json()
        model_names = [model["name"] for model in models_data.get("models", [])]

        return jsonify({"status": "success", "models": model_names})
    except requests.exceptions.RequestException as e:
        app.logger.error(f"连接Ollama服务失败: {str(e)}")
        return jsonify(
            {"status": "error", "message": f"连接Ollama服务失败: {str(e)}"}
        ), 500
    except Exception as e:
        app.logger.error(f"获取Ollama模型列表时发生错误: {str(e)}")
        return jsonify(
            {"status": "error", "message": f"获取模型列表失败: {str(e)}"}
        ), 500
