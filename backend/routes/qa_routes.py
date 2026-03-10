from flask import (
    Blueprint,
    render_template,
    request,
    jsonify,
    Response,
    stream_with_context,
)
from backend.services.vanna_service import vanna_manager
from backend.services.model_service import model_manager
from backend.manager.qa_manager import qa_manager, QAManager
from backend.utils.db_utils import init_qa_db
import logging
import json
import pandas as pd

qa_bp = Blueprint("qa", __name__)
logger = logging.getLogger(__name__)

# 初始化问答数据库
init_qa_db()


@qa_bp.route("/database-qa")
def database_qa():
    return render_template("database_qa.html")


@qa_bp.route("/data-decision")
def data_decision():
    return render_template("data_decision.html")


# 会话管理接口
@qa_bp.route("/api/chat-sessions", methods=["GET"])
def get_chat_sessions():
    try:
        sessions = qa_manager.get_all_sessions()
        return jsonify({"status": "success", "sessions": sessions})
    except Exception as e:
        logger.error(f"获取聊天会话列表失败: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500


@qa_bp.route("/api/chat-sessions", methods=["POST"])
def create_chat_session():
    try:
        data = request.json
        name = data.get("name")
        database_name = data.get("database_name")
        table_name = data.get("table_name")  # 添加表名参数
        model_name = data.get("model_name")
        temperature = data.get("temperature", 0.7)

        session = qa_manager.create_session(
            name=name,
            database_name=database_name,
            table_name=table_name,  # 传入表名
            model_name=model_name,
            temperature=temperature,
        )
        return jsonify({"status": "success", "session": session})
    except Exception as e:
        logger.error(f"创建聊天会话失败: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500


@qa_bp.route("/api/chat-sessions/<int:session_id>", methods=["GET"])
def get_chat_session(session_id):
    try:
        session = qa_manager.get_session(session_id)
        if not session:
            return jsonify({"status": "error", "message": "会话不存在"}), 404
        return jsonify({"status": "success", "session": session})
    except Exception as e:
        logger.error(f"获取聊天会话失败: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500


@qa_bp.route("/api/chat-sessions/<int:session_id>", methods=["PUT"])
def update_chat_session(session_id):
    try:
        data = request.json
        session = qa_manager.update_session(session_id, **data)
        if not session:
            return jsonify({"status": "error", "message": "会话不存在"}), 404
        return jsonify({"status": "success", "session": session})
    except Exception as e:
        logger.error(f"更新聊天会话失败: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500


@qa_bp.route("/api/chat-sessions/<int:session_id>", methods=["DELETE"])
def delete_chat_session(session_id):
    try:
        success = qa_manager.delete_session(session_id)
        if not success:
            return jsonify({"status": "error", "message": "会话不存在"}), 404
        return jsonify({"status": "success", "message": "会话已删除"})
    except Exception as e:
        logger.error(f"删除聊天会话失败: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500


# 消息管理接口
@qa_bp.route("/api/chat-sessions/<int:session_id>/messages", methods=["GET"])
def get_messages(session_id):
    try:
        messages = qa_manager.get_messages(session_id)
        return jsonify({"status": "success", "messages": messages})
    except Exception as e:
        logger.error(f"获取消息失败: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500


@qa_bp.route("/api/chat-sessions/<int:session_id>/messages", methods=["DELETE"])
def clear_messages(session_id):
    try:
        # 获取所有消息
        messages = qa_manager.get_messages(session_id)

        # 删除所有消息
        for message in messages:
            qa_manager.delete_message(message["id"])

        return jsonify({"status": "success", "message": "消息已清空"})
    except Exception as e:
        logger.error(f"清空消息失败: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500


@qa_bp.route(
    "/api/chat-sessions/<int:session_id>/messages/<int:message_id>", methods=["DELETE"]
)
def delete_message(session_id, message_id):
    try:
        success = qa_manager.delete_message(message_id)
        if not success:
            return jsonify({"status": "error", "message": "消息不存在"}), 404
        return jsonify({"status": "success", "message": "消息已删除"})
    except Exception as e:
        logger.error(f"删除消息失败: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500


# 问答接口
@qa_bp.route("/api/ask", methods=["POST"])
def ask_question():
    """处理用户的问题"""
    try:
        data = request.json
        question = data.get("question")
        session_id = data.get("session_id")
        database_name = data.get("database_name")
        table_name = data.get("table_name")  # 表名参数

        if not question:
            return jsonify({"status": "error", "message": "问题不能为空"}), 400

        if not session_id:
            return jsonify({"status": "error", "message": "会话ID不能为空"}), 400

        # 处理问题并获取回答
        response = qa_manager.ask_question(
            session_id, question, database_name, table_name
        )

        return jsonify({"status": "success", "data": response})
    except Exception as e:
        logger.error(f"处理问题失败: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500


@qa_bp.route("/api/ask/stream", methods=["POST"])
def ask_question_stream():
    data = request.json or {}
    question = data.get("question")
    session_id = data.get("session_id")
    database_name = data.get("database_name")
    table_name = data.get("table_name")

    if not question:
        return jsonify({"status": "error", "message": "问题不能为空"}), 400

    if not session_id:
        return jsonify({"status": "error", "message": "会话ID不能为空"}), 400

    def generate():
        try:
            for event in qa_manager.ask_question_stream(
                session_id, question, database_name, table_name
            ):
                yield json.dumps(event, ensure_ascii=False) + "\n"
        except Exception as e:
            logger.error(f"流式处理问题失败: {str(e)}")
            yield (
                json.dumps({"type": "error", "message": str(e)}, ensure_ascii=False)
                + "\n"
            )
            yield json.dumps({"type": "end"}, ensure_ascii=False) + "\n"

    return Response(stream_with_context(generate()), mimetype="application/x-ndjson")


@qa_bp.route("/api/train", methods=["POST"])
def train():
    try:
        vanna_manager.train(**request.json)
        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@qa_bp.route("/api/ollama-models", methods=["GET"])
def get_ollama_models():
    try:
        models = model_manager.get_available_models()
        return jsonify({"status": "success", "models": models})
    except Exception as e:
        logger.error(f"获取Ollama模型列表失败: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500


@qa_bp.route("/api/config", methods=["GET"])
def get_config():
    """获取系统配置信息"""
    try:
        config = vanna_manager.get_config()
        return jsonify({"status": "success", "config": config})
    except Exception as e:
        logger.error(f"获取系统配置失败: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500


# 保存查询相关接口
@qa_bp.route("/api/saved-queries", methods=["GET"])
def get_saved_queries():
    try:
        queries = qa_manager.get_all_saved_queries()
        return jsonify({"status": "success", "queries": queries})
    except Exception as e:
        logger.error(f"获取保存的查询失败: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500


@qa_bp.route("/api/saved-queries", methods=["POST"])
def save_query():
    try:
        data = request.json
        title = data.get("title")
        question = data.get("question")
        sql = data.get("sql")
        result = data.get("result")
        visualization = data.get("visualization")
        description = data.get("description")

        if not title or not question or not sql:
            return jsonify(
                {"status": "error", "message": "标题、问题和SQL不能为空"}
            ), 400

        saved_query = qa_manager.save_query_result(
            title=title,
            question=question,
            sql=sql,
            result=result,
            visualization=visualization,
            description=description,
        )

        return jsonify({"status": "success", "query": saved_query})
    except Exception as e:
        logger.error(f"保存查询失败: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500


@qa_bp.route("/api/saved-queries/<int:query_id>", methods=["GET"])
def get_saved_query(query_id):
    try:
        query = qa_manager.get_query(query_id)
        if not query:
            return jsonify({"status": "error", "message": "查询不存在"}), 404
        return jsonify({"status": "success", "query": query})
    except Exception as e:
        logger.error(f"获取保存的查询失败: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500


@qa_bp.route("/api/saved-queries/<int:query_id>", methods=["DELETE"])
def delete_saved_query(query_id):
    try:
        success = qa_manager.delete_query(query_id)
        if not success:
            return jsonify({"status": "error", "message": "查询不存在"}), 404
        return jsonify({"status": "success", "message": "查询已删除"})
    except Exception as e:
        logger.error(f"删除保存的查询失败: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500


@qa_bp.route("/api/toggle-table-lock/<int:session_id>", methods=["POST"])
def toggle_table_lock(session_id):
    """切换会话数据表的锁定状态"""
    try:
        data = request.json
        lock_status = data.get("lock_status", True)

        # 调用管理器方法切换锁定状态
        session = qa_manager.toggle_table_lock(session_id, lock_status)

        if not session:
            return jsonify({"status": "error", "message": "会话不存在或操作失败"}), 404

        return jsonify(
            {"status": "success", "session": session, "message": "表锁定状态已更新"}
        )
    except Exception as e:
        logger.error(f"切换表锁定状态失败: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500


@qa_bp.route("/api/generate-session-name/<int:session_id>", methods=["POST"])
def generate_session_name(session_id):
    """手动触发生成会话名称"""
    try:
        data = request.json
        custom_prompt = data.get("prompt")  # 可选的自定义提示词

        # 获取会话信息
        session = qa_manager.get_session(session_id)
        if not session:
            return jsonify({"status": "error", "message": "会话不存在"}), 404

        # 获取会话的第一个用户消息
        messages = qa_manager.get_messages(session_id)
        user_messages = [msg for msg in messages if msg["role"] == "user"]

        if not user_messages:
            return jsonify(
                {"status": "error", "message": "会话中没有用户消息，无法生成名称"}
            ), 400

        first_message = user_messages[0]["content"]

        # 调用生成名称方法
        new_name = qa_manager.generate_session_name(session_id, first_message)

        if new_name:
            return jsonify(
                {
                    "status": "success",
                    "message": "会话名称生成成功",
                    "session_id": session_id,
                    "new_name": new_name,
                }
            )
        else:
            return jsonify(
                {
                    "status": "error",
                    "message": "生成会话名称失败，请检查命名模型设置是否正确",
                }
            ), 500

    except Exception as e:
        logger.error(f"生成会话名称失败: {str(e)}")
        return jsonify(
            {"status": "error", "message": f"生成会话名称失败: {str(e)}"}
        ), 500


@qa_bp.route("/api/training-data", methods=["GET"])
def get_training_data_api():
    """获取所有训练数据"""
    try:
        df = vanna_manager.get_training_data()
        # 如果是空DataFrame
        if df.empty:
            logger.info("训练数据为空")
            return jsonify({"status": "success", "data": []})

        # Ensure None values are handled correctly for JSON serialization
        df = df.replace({pd.NA: None, pd.NaT: None})

        # 处理数据类型问题
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = df[col].astype(str)

        # Convert DataFrame to list of dictionaries
        data = df.to_dict(orient="records")

        # 调试打印
        logger.info(f"成功获取{len(data)}条训练数据")

        return jsonify({"status": "success", "data": data})
    except Exception as e:
        logger.error(f"获取训练数据失败: {str(e)}")
        return jsonify(
            {"status": "error", "message": f"获取训练数据失败: {str(e)}"}
        ), 500


@qa_bp.route("/api/training-data/<item_id>", methods=["DELETE"])
def delete_training_data_api(item_id):
    """删除指定的训练数据"""
    try:
        logger.info(f"尝试删除训练数据: {item_id}")
        success = vanna_manager.remove_training_data(id=item_id)
        if success:
            logger.info(f"成功删除训练数据: {item_id}")
            return jsonify({"status": "success", "message": "训练数据删除成功"})
        else:
            # remove_training_data might return False if ID format is wrong or not found
            logger.warning(f"尝试删除训练数据失败，ID可能未找到或格式错误: {item_id}")
            return jsonify(
                {"status": "error", "message": "删除失败，未找到指定ID或格式错误"}
            ), 404
    except Exception as e:
        logger.error(f"删除训练数据时出错 (ID: {item_id}): {str(e)}")
        return jsonify(
            {"status": "error", "message": f"删除训练数据时出错: {str(e)}"}
        ), 500


@qa_bp.route("/api/add-sql-qa", methods=["POST"])
def add_sql_qa_api():
    """添加SQL问答对训练数据"""
    try:
        data = request.json
        question = data.get("question")
        sql = data.get("sql")

        if not question or not sql:
            return jsonify({"status": "error", "message": "问题和SQL语句不能为空"}), 400

        # 使用vanna_manager添加SQL问答对
        vanna_manager.train(sql=sql, question=question)

        return jsonify({"status": "success", "message": "SQL问答对添加成功"})
    except Exception as e:
        logger.error(f"添加SQL问答对失败: {str(e)}")
        return jsonify(
            {"status": "error", "message": f"添加SQL问答对失败: {str(e)}"}
        ), 500


@qa_bp.route("/api/add-documentation", methods=["POST"])
def add_documentation_api():
    """添加知识文档训练数据"""
    try:
        data = request.json
        documentation = data.get("documentation")

        if not documentation:
            return jsonify({"status": "error", "message": "文档内容不能为空"}), 400

        # 使用vanna_manager添加文档
        vanna_manager.train(documentation=documentation)

        return jsonify({"status": "success", "message": "知识文档添加成功"})
    except Exception as e:
        logger.error(f"添加知识文档失败: {str(e)}")
        return jsonify(
            {"status": "error", "message": f"添加知识文档失败: {str(e)}"}
        ), 500


@qa_bp.route("/api/naming-settings", methods=["GET"])
def get_naming_settings():
    # This route is mentioned in the code but not implemented in the provided file
    # It's assumed to exist as it's called in the code block
    return jsonify({"status": "error", "message": "This route is not implemented"}), 500
