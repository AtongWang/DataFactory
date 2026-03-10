from flask import Blueprint, request, jsonify, Response, stream_with_context
from backend.manager.agent_task_manager import agent_task_manager
import json
import logging
import asyncio # For running async stream
import logging
import random


logger = logging.getLogger(__name__)

agent_api = Blueprint('agent_api', __name__, url_prefix='/api/agent')

@agent_api.route('/tasks', methods=['POST'])
def create_agent_task():
    """创建新的 Agent 任务"""
    data = request.json
    name = data.get('name')
    user_goal = data.get('user_goal')
    sql_database_name = data.get('sql_database_name')
    sql_table_name = data.get('sql_table_name')
    kg_graph_name = data.get('kg_graph_name')
    model_name = data.get('model_name')
    temperature = data.get('temperature')
    max_iterations = data.get('max_iterations', 20) # Get max_iterations, default 20

    if not user_goal:
        return jsonify({"error": "user_goal is required"}), 400

    try:
        # Validate max_iterations briefly here too, or let manager handle it
        try:
            max_iterations_int = int(max_iterations) if max_iterations is not None else 20
            if max_iterations_int <= 0: max_iterations_int = 20 # Basic validation
        except (ValueError, TypeError):
             max_iterations_int = 20

        session_id = agent_task_manager.create_task_session(
            name=name,
            user_goal=user_goal,
            sql_database_name=sql_database_name,
            sql_table_name=sql_table_name,
            kg_graph_name=kg_graph_name,
            model_name=model_name,
            temperature=temperature,
            max_iterations=max_iterations_int # Pass validated value
        )
        return jsonify({"session_id": session_id, "message": "Agent task created successfully"}), 201
    except Exception as e:
        logging.exception(f"Failed to create agent task")
        return jsonify({"error": f"Failed to create agent task: {str(e)}"}), 500

@agent_api.route('/tasks', methods=['GET'])
def get_agent_tasks():
    """获取所有 Agent 任务列表"""
    try:
        sessions = agent_task_manager.get_all_task_sessions()
        return jsonify(sessions), 200
    except Exception as e:
        logging.exception("Failed to get agent tasks")
        return jsonify({"error": f"Failed to get agent tasks: {str(e)}"}), 500

@agent_api.route('/tasks/<int:session_id>', methods=['GET'])
def get_agent_task(session_id):
    """获取指定 Agent 任务详情"""
    try:
        session = agent_task_manager.get_task_session(session_id)
        if session:
            # Ensure stop_requested flag is included if present
            session_data = dict(session) # Convert Row object to dict if needed
            session_data.setdefault('stop_requested', False) # Default if not stored
            return jsonify(session_data), 200
        else:
            return jsonify({"error": "Agent task not found"}), 404
    except Exception as e:
        logging.exception(f"Failed to get agent task {session_id}")
        return jsonify({"error": f"Failed to get agent task {session_id}: {str(e)}"}), 500

@agent_api.route('/tasks/<int:session_id>', methods=['DELETE'])
def delete_agent_task(session_id):
    """删除 Agent 任务"""
    try:
        success = agent_task_manager.delete_task_session(session_id)
        if success:
            return jsonify({"message": f"Agent task {session_id} deleted successfully"}), 200
        else:
            # Check if deletion failed because task was running? Optional enhancement.
            return jsonify({"error": "Failed to delete agent task or task not found"}), 404 # Original was 404, keep it
    except Exception as e:
         logging.exception(f"Failed to delete agent task {session_id}")
         return jsonify({"error": f"Failed to delete agent task {session_id}: {str(e)}"}), 500

@agent_api.route('/tasks/<int:session_id>/messages', methods=['GET'])
def get_agent_task_steps(session_id):
    """获取 Agent 任务的所有步骤/消息"""
    try:
        messages = agent_task_manager.get_task_messages(session_id)
        return jsonify(messages), 200
    except Exception as e:
        logging.exception(f"Failed to get messages for agent task {session_id}")
        return jsonify({"error": f"Failed to get messages for agent task {session_id}: {str(e)}"}), 500

@agent_api.route('/tasks/<int:session_id>/run', methods=['GET'])
def run_agent_task(session_id):
    """运行 Agent 任务并流式返回步骤 (使用 GET 请求)"""
    async def async_generate():
        # 使用异步超时控制
        STREAM_TIMEOUT = 300  # 5分钟总超时
        ITEM_TIMEOUT = 60.0   # 单个项目的超时时间 (1分钟)
        
        try:
            # 获取生成器
            gen = agent_task_manager.run_task(session_id)
            
            # 设置整体超时跟踪
            start_time = asyncio.get_event_loop().time()
            end_time = start_time + STREAM_TIMEOUT
            
            # 追踪最后一次活动时间
            last_activity = start_time
            
            # 使用直接的异步迭代
            try:
                async for chunk in gen:
                    # 更新活动时间
                    now = asyncio.get_event_loop().time()
                    last_activity = now
                    
                    # 检查总时间限制
                    if now > end_time:
                        logger.warning(f"Task {session_id}: Total stream time exceeded {STREAM_TIMEOUT}s")
                        raise asyncio.TimeoutError("Total stream time exceeded")
                    
                    # 成功获取到数据，传递给客户端
                    yield f"data: {json.dumps(chunk)}\n\n"
                    
                    # 每隔几个消息发送一次心跳
                    if random.random() < 0.2:  # 约20%的消息后发送心跳
                        yield f"event: heartbeat\ndata: {{}}\n\n"
            except asyncio.TimeoutError:
                logger.warning(f"Task {session_id}: Stream generation timed out")
                raise  # 重新抛出以便外层捕获
            except Exception as stream_err:
                logger.error(f"Task {session_id}: Error during streaming: {stream_err}", exc_info=True)
                raise  # 重新抛出以便外层捕获
            
            # 检查最终状态
            final_session = agent_task_manager.get_task_session(session_id)
            final_status = final_session.get('status') if final_session else 'unknown'
            logger.info(f"SSE stream for task {session_id} naturally ending. Final status: {final_status}")
            # 发送结束事件
            yield f"event: end\ndata: {{\"status\": \"{final_status}\"}}\n\n"
            
        except asyncio.TimeoutError:
            # 超时处理
            logger.warning(f"Task {session_id}: Stream generation timed out")
            # 更新任务状态为超时
            try:
                current_session = agent_task_manager.get_task_session(session_id)
                if current_session and current_session.get('status') == 'running':
                    agent_task_manager.update_task_session(
                        session_id, 
                        status='failed',
                        error_message="Task execution timed out"
                    )
            except Exception as update_err:
                logger.error(f"Task {session_id}: Failed to update status after timeout: {update_err}")
            
            # 发送超时错误和结束事件
            error_chunk = {"type": "error", "content": "任务执行超时，已被终止"}
            yield f"event: error\ndata: {json.dumps(error_chunk)}\n\n"
            yield f"event: end\ndata: {{\"status\": \"failed\"}}\n\n"
            return
        except Exception as e:
            # 处理生成器本身的意外错误
            logging.exception(f"Error during agent task async stream setup/yield for session {session_id}")
            error_chunk = {"type": "error", "content": f"Stream generation error: {str(e)}"}
            yield f"event: error\ndata: {json.dumps(error_chunk)}\n\n"
            # 发送结束事件
            yield f"event: end\ndata: {{\"status\": \"failed\"}}\n\n"
            
            # 确保在流异常时更新任务状态
            try:
                current_session = agent_task_manager.get_task_session(session_id)
                if current_session and current_session.get('status') == 'running':
                    agent_task_manager.update_task_session(
                        session_id, 
                        status='failed',
                        error_message=f"Stream error: {str(e)}"
                    )
            except Exception as update_err:
                logger.error(f"Task {session_id}: Failed to update status after stream error: {update_err}")

    def generate_wrapper():
        """同步包装器，管理事件循环和异常处理"""
        # 每个请求使用独立的事件循环
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        gen = async_generate()
        
        # 设置更严格的关闭标志
        is_loop_closing = False
        
        try: # 包装整个循环驱动以确保清理
            while True:
                try:
                    # 驱动异步生成器获取下一个块
                    chunk = loop.run_until_complete(gen.__anext__())
                    # 在Flask上下文中同步产出块
                    yield chunk
                except StopAsyncIteration:
                    # 生成器正常结束
                    logger.info(f"generate_wrapper: StopAsyncIteration received for task {session_id}. Stream finished.")
                    break # 退出while循环
                except Exception as e:
                    # 处理块生成或处理期间的错误
                    logging.error(f"Error in generate_wrapper main loop for task {session_id}: {e}", exc_info=True)
                    error_chunk = {"type": "error", "content": f"Stream wrapper error: {str(e)}"}
                    try:
                        yield f"event: error\ndata: {json.dumps(error_chunk)}\n\n"
                        # 错误后发送结束事件
                        yield f"event: end\ndata: {{\"status\": \"failed\"}}\n\n"
                        
                        # 确保在包装器异常时更新任务状态
                        try:
                            current_session = agent_task_manager.get_task_session(session_id)
                            if current_session and current_session.get('status') == 'running':
                                logger.warning(f"Task {session_id}: Wrapper error occurred, updating status to failed")
                                agent_task_manager.update_task_session(
                                    session_id, 
                                    status='failed',
                                    error_message=f"Wrapper error: {str(e)}"
                                )
                        except Exception as update_err:
                            logger.error(f"Task {session_id}: Failed to update status after wrapper error: {update_err}")
                            
                    except Exception as yield_err:
                        logger.error(f"Task {session_id}: Failed to yield error/end chunk in wrapper: {yield_err}")
                    break # 退出while循环
        finally:
            # --- 适当的Asyncio循环关闭序列 ---
            # 这个块无论循环是正常结束还是因异常而中断都会运行
            logger.info(f"Task {session_id}: Starting event loop shutdown process...")
            
            # 设置关闭标志以避免重复关闭
            is_loop_closing = True
            
            try:
                # 在关闭事件循环前检查任务状态
                try:
                    current_session = agent_task_manager.get_task_session(session_id)
                    if current_session and current_session.get('status') == 'running':
                        logger.warning(f"Task {session_id}: Stream terminated but task still running, updating status to failed")
                        agent_task_manager.update_task_session(
                            session_id, 
                            status='failed',
                            error_message="Stream terminated prematurely"
                        )
                except Exception as update_err:
                    logger.error(f"Task {session_id}: Failed to update status during event loop shutdown: {update_err}")
                    
                # --- 关键修复：确保所有异步资源被正确关闭 ---
                try:
                    # 1. 正确关闭异步生成器
                    logger.info(f"Task {session_id}: Properly cleaning up async generator resources")
                    
                    # 创建关闭任务但设置短超时，避免阻塞
                    try:
                        cleanup_task = loop.create_task(gen.aclose())
                        loop.run_until_complete(asyncio.wait_for(cleanup_task, timeout=5.0))
                        logger.info(f"Task {session_id}: Async generator cleanup completed")
                    except asyncio.TimeoutError:
                        logger.warning(f"Task {session_id}: Async generator cleanup timed out, proceeding with other cleanups")
                    
                    # 2. 清理所有未完成任务
                    pending = asyncio.all_tasks(loop=loop)
                    if pending:
                        logger.info(f"Task {session_id}: Cancelling {len(pending)} pending tasks")
                        for task in pending:
                            if not task.done():
                                task.cancel()
                        # 给任务少量时间来处理取消，但不要等待太久
                        try:
                            loop.run_until_complete(asyncio.wait(pending, timeout=2.0))
                            logger.info(f"Task {session_id}: All pending tasks cancelled")
                        except asyncio.TimeoutError:
                            logger.warning(f"Task {session_id}: Some tasks did not respond to cancellation in time")
                except Exception as cleanup_err:
                    logger.error(f"Task {session_id}: Error during resource cleanup: {cleanup_err}", exc_info=True)
                    
            except Exception as shutdown_err:
                logger.error(f"Task {session_id}: Error during event loop shutdown: {shutdown_err}", exc_info=True)
            finally:
                # 3. 关闭循环：现在应该安全地关闭了
                if not loop.is_closed():
                    logger.info(f"Task {session_id}: Closing event loop.")
                    try:
                        loop.close()
                        logger.info(f"Task {session_id}: Event loop closed.")
                    except Exception as close_err:
                        logger.error(f"Task {session_id}: Error closing event loop: {close_err}")
                else:
                    logger.info(f"Task {session_id}: Event loop was already closed.")

    return Response(stream_with_context(generate_wrapper()), mimetype='text/event-stream')

@agent_api.route('/tasks/<int:session_id>/stop', methods=['POST'])
def stop_agent_task(session_id):
    """请求停止正在运行的 Agent 任务"""
    try:
        success, message = agent_task_manager.request_task_stop(session_id)
        if success:
            return jsonify({"message": message}), 200
        else:
             # Determine appropriate status code based on message
             status_code = 404 if "not found" in message else 409 if "not running" in message else 500
             return jsonify({"error": message}), status_code
    except Exception as e:
        logging.exception(f"Failed to process stop request for agent task {session_id}")
        return jsonify({"error": f"Failed to stop agent task {session_id}: {str(e)}"}), 500

# Ensure Blueprint registration in app.py
