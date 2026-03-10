import os
import sqlite3
import json
import logging
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from backend.models.qa_models import (
    Base as QABase,
    ChatSession,
    ChatMessage,
    SavedQuery,
)
from backend.models.kg_qa_models import (
    Base as KGQABase,
    KGQA_ChatSession,
    KGQA_ChatMessage,
    KGQA_SavedQuery,
)

logger = logging.getLogger(__name__)
# 数据库文件路径
DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "db_file",
    "app.db",
)

# 确保数据库目录存在
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

# 创建SQLAlchemy引擎和会话
engine = create_engine(f"sqlite:///{DB_PATH}")
Session = sessionmaker(bind=engine)


def init_qa_db():
    """初始化数据库表结构"""
    # 创建所有表
    QABase.metadata.create_all(engine)
    KGQABase.metadata.create_all(engine)


def init_import_history_db():
    """初始化导入历史表"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 创建导入历史表
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS import_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_name TEXT NOT NULL,
            file_path TEXT NOT NULL,
            table_name TEXT NOT NULL,
            database_type TEXT NOT NULL,
            total_rows INTEGER NOT NULL,
            import_time TIMESTAMP NOT NULL,
            column_count INTEGER NOT NULL,
            column_info TEXT NOT NULL,
            status TEXT NOT NULL,
            error_message TEXT
        )
    """)

    conn.commit()
    conn.close()


# 添加导入历史记录
def add_import_history(
    file_name,
    file_path,
    table_name,
    database_type,
    total_rows,
    column_count,
    column_info,
    status,
    error_message=None,
):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO import_history (
            file_name, file_path, table_name, database_type, total_rows,
            import_time, column_count, column_info, status, error_message
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """,
        (
            file_name,
            file_path,
            table_name,
            database_type,
            total_rows,
            datetime.now().isoformat(),
            column_count,
            json.dumps(column_info),
            status,
            error_message,
        ),
    )
    conn.commit()
    conn.close()


# 获取导入历史记录
def get_import_history(limit=10):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT * FROM import_history 
        ORDER BY import_time DESC 
        LIMIT ?
    """,
        (limit,),
    )
    columns = [description[0] for description in cursor.description]
    records = [dict(zip(columns, row)) for row in cursor.fetchall()]
    conn.close()
    return records


def get_chat_sessions():
    """获取所有聊天会话"""
    session = Session()
    try:
        sessions = (
            session.query(ChatSession).order_by(ChatSession.updated_at.desc()).all()
        )
        return [session.to_dict() for session in sessions]
    finally:
        session.close()


def get_kgqa_chat_sessions():
    """获取所有KGQA聊天会话"""
    session = Session()
    try:
        sessions = (
            session.query(KGQA_ChatSession)
            .order_by(KGQA_ChatSession.updated_at.desc())
            .all()
        )
        return [session.to_dict() for session in sessions]
    finally:
        session.close()


def get_chat_session(session_id):
    """获取指定ID的聊天会话"""
    session = Session()
    try:
        chat_session = (
            session.query(ChatSession).filter(ChatSession.id == session_id).first()
        )
        return chat_session.to_dict() if chat_session else None
    finally:
        session.close()


def get_kgqa_chat_session(session_id):
    """获取指定ID的KGQA聊天会话"""
    session = Session()
    try:
        chat_session = (
            session.query(KGQA_ChatSession)
            .filter(KGQA_ChatSession.id == session_id)
            .first()
        )
        return chat_session.to_dict() if chat_session else None
    finally:
        session.close()


def create_chat_session(
    name=None,
    database_name=None,
    table_name=None,
    model_name=None,
    temperature=0.7,
    auto_train=True,
):
    """创建新的聊天会话"""
    session = Session()
    try:
        # 判断是否需要锁定表
        is_table_locked = False
        if database_name and table_name:
            is_table_locked = True

        chat_session = ChatSession(
            name=name or "新会话",
            database_name=database_name,
            table_name=table_name,
            model_name=model_name,
            temperature=temperature,
            is_table_locked=is_table_locked,
            auto_train=auto_train,
        )
        session.add(chat_session)
        session.commit()
        return chat_session.to_dict()
    finally:
        session.close()


def create_kgqa_chat_session(
    name=None,
    database_name=None,
    table_name=None,
    model_name=None,
    temperature=0.7,
    auto_train=True,
):
    """创建新的KGQA聊天会话"""
    session = Session()
    try:
        chat_session = KGQA_ChatSession(
            name=name or "新会话",
            database_name=database_name,
            table_name=table_name,
            model_name=model_name,
            temperature=temperature,
            auto_train=auto_train,
        )
        session.add(chat_session)
        session.commit()
        return chat_session.to_dict()
    finally:
        session.close()


def update_chat_session(session_id, **kwargs):
    """更新聊天会话信息"""
    session = Session()
    try:
        chat_session = (
            session.query(ChatSession).filter(ChatSession.id == session_id).first()
        )
        if chat_session:
            # 如果会话的数据表已锁定，则不允许更改数据库和表名
            if chat_session.is_table_locked:
                # 允许修改 auto_train
                allowed_keys = {"name", "model_name", "temperature", "auto_train"}
                keys_to_remove = set(kwargs.keys()) - allowed_keys
                for key in keys_to_remove:
                    del kwargs[key]

            # 更新其他允许的字段
            for key, value in kwargs.items():
                if hasattr(chat_session, key):
                    setattr(chat_session, key, value)

            chat_session.updated_at = datetime.utcnow()

            session.commit()
            return chat_session.to_dict()
        return None
    finally:
        session.close()


def update_kgqa_chat_session(session_id, **kwargs):
    """更新KGQA聊天会话信息"""
    session = Session()
    try:
        chat_session = (
            session.query(KGQA_ChatSession)
            .filter(KGQA_ChatSession.id == session_id)
            .first()
        )
        if chat_session:
            # 如果会话的数据表已锁定，则不允许更改数据库和表名
            if chat_session.is_table_locked:
                # 允许修改 auto_train
                allowed_keys = {"name", "model_name", "temperature", "auto_train"}
                keys_to_remove = set(kwargs.keys()) - allowed_keys
                for key in keys_to_remove:
                    del kwargs[key]

            # 更新其他允许的字段
            for key, value in kwargs.items():
                if hasattr(chat_session, key):
                    setattr(chat_session, key, value)

            chat_session.updated_at = datetime.utcnow()

            session.commit()
            return chat_session.to_dict()
        return None
    finally:
        session.close()


def delete_chat_session(session_id):
    """删除聊天会话及其所有消息"""
    session = Session()
    try:
        chat_session = (
            session.query(ChatSession).filter(ChatSession.id == session_id).first()
        )
        if chat_session:
            session.delete(chat_session)
            session.commit()
            return True
        return False
    finally:
        session.close()


def delete_kgqa_chat_session(session_id):
    """删除KGQA聊天会话及其所有消息"""
    session = Session()
    try:
        chat_session = (
            session.query(KGQA_ChatSession)
            .filter(KGQA_ChatSession.id == session_id)
            .first()
        )
        if chat_session:
            session.delete(chat_session)
            session.commit()
            return True
        return False
    finally:
        session.close()


def get_chat_messages(session_id):
    """获取指定会话的所有消息"""
    session = Session()
    try:
        messages = (
            session.query(ChatMessage)
            .filter(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.created_at)
            .all()
        )
        return [message.to_dict() for message in messages]
    finally:
        session.close()


def get_kgqa_chat_messages(session_id):
    """获取指定会话的所有消息"""
    session = Session()
    try:
        messages = (
            session.query(KGQA_ChatMessage)
            .filter(KGQA_ChatMessage.session_id == session_id)
            .order_by(KGQA_ChatMessage.created_at)
            .all()
        )
        return [message.to_dict() for message in messages]
    finally:
        session.close()


def add_chat_message(
    session_id,
    role,
    content,
    sql=None,
    result=None,
    visualization=None,
    reasoning=None,
    thinking=None,
):
    """添加聊天消息"""
    session = Session()
    try:
        # 确保会话存在
        chat_session = (
            session.query(ChatSession).filter(ChatSession.id == session_id).first()
        )
        if not chat_session:
            return None

        # 更新会话最后更新时间
        chat_session.updated_at = datetime.utcnow()

        # 创建新消息
        message = ChatMessage(
            session_id=session_id,
            role=role,
            content=content,
            sql=sql,
            result=json.dumps(result) if result else None,
            visualization=json.dumps(visualization) if visualization else None,
            reasoning=reasoning,
            thinking=thinking,
        )

        session.add(message)
        session.commit()
        return message.to_dict()
    finally:
        session.close()


def add_kgqa_chat_message(
    session_id,
    role,
    content,
    cypher=None,
    result=None,
    visualization=None,
    reasoning=None,
    thinking=None,
):
    """添加KGQA聊天消息"""
    session = Session()
    try:
        # 确保会话存在
        chat_session = (
            session.query(KGQA_ChatSession)
            .filter(KGQA_ChatSession.id == session_id)
            .first()
        )
        if not chat_session:
            return None

        # 更新会话最后更新时间
        chat_session.updated_at = datetime.utcnow()

        # 创建新消息
        message = KGQA_ChatMessage(
            session_id=session_id,
            role=role,
            content=content,
            cypher=cypher,
            result=json.dumps(result) if result else None,
            visualization=json.dumps(visualization) if visualization else None,
            reasoning=reasoning,
            thinking=thinking,
        )

        session.add(message)
        session.commit()
        return message.to_dict()
    finally:
        session.close()


def delete_chat_message(message_id):
    """删除聊天消息"""
    session = Session()
    try:
        message = (
            session.query(ChatMessage).filter(ChatMessage.id == message_id).first()
        )
        if message:
            session.delete(message)
            session.commit()
            return True
        return False
    finally:
        session.close()


def delete_kgqa_chat_message(message_id):
    """删除KGQA聊天消息"""
    session = Session()
    try:
        message = (
            session.query(KGQA_ChatMessage)
            .filter(KGQA_ChatMessage.id == message_id)
            .first()
        )
        if message:
            session.delete(message)
            session.commit()
            return True
        return False
    finally:
        session.close()


def save_query(title, question, sql, result=None, visualization=None, description=None):
    """保存查询到数据库"""
    session = Session()
    try:
        saved_query = SavedQuery(
            title=title,
            description=description,
            question=question,
            sql=sql,
            result=json.dumps(result) if result else None,
            visualization=json.dumps(visualization) if visualization else None,
        )
        session.add(saved_query)
        session.commit()
        return saved_query.to_dict()
    finally:
        session.close()


def save_kgqa_query(
    title, question, cypher, result=None, visualization=None, description=None
):
    """保存KGQA查询到数据库"""
    session = Session()
    try:
        saved_query = KGQA_SavedQuery(
            title=title,
            description=description,
            question=question,
            cypher=cypher,
            result=json.dumps(result) if result else None,
            visualization=json.dumps(visualization) if visualization else None,
        )
        session.add(saved_query)
        session.commit()
        return saved_query.to_dict()
    finally:
        session.close()


def get_saved_queries():
    """获取所有保存的查询"""
    session = Session()
    try:
        queries = session.query(SavedQuery).order_by(SavedQuery.updated_at.desc()).all()
        return [query.to_dict() for query in queries]
    finally:
        session.close()


def get_saved_kgqa_queries():
    """获取指定ID的保存KGQA查询"""
    session = Session()
    try:
        queries = (
            session.query(KGQA_SavedQuery)
            .order_by(KGQA_SavedQuery.updated_at.desc())
            .all()
        )
        return [query.to_dict() for query in queries]
    finally:
        session.close()


def get_saved_query(query_id):
    """获取指定ID的保存查询"""
    session = Session()
    try:
        query = session.query(SavedQuery).filter(SavedQuery.id == query_id).first()
        return query.to_dict() if query else None
    finally:
        session.close()


def get_saved_kgqa_query(query_id):
    """获取指定ID的保存KGQA查询"""
    session = Session()
    try:
        query = (
            session.query(KGQA_SavedQuery)
            .filter(KGQA_SavedQuery.id == query_id)
            .first()
        )
        return query.to_dict() if query else None
    finally:
        session.close()


def delete_saved_query(query_id):
    """删除保存的查询"""
    session = Session()
    try:
        query = session.query(SavedQuery).filter(SavedQuery.id == query_id).first()
        if query:
            session.delete(query)
            session.commit()
            return True
        return False
    finally:
        session.close()


def delete_saved_kgqa_query(query_id):
    """删除保存的KGQA查询"""
    session = Session()
    try:
        query = (
            session.query(KGQA_SavedQuery)
            .filter(KGQA_SavedQuery.id == query_id)
            .first()
        )
        if query:
            session.delete(query)
            session.commit()
            return True
        return False
    finally:
        session.close()


# 处理列名，将空格替换为下划线，并移除特殊字符
def clean_column_name(col):
    # 首先检查输入是否为有效字符串
    if not col or not isinstance(col, str):
        return "unnamed_column"

    # 将中文括号替换为英文括号
    col = col.replace("（", "(").replace("）", ")")
    # 将空格替换为下划线
    col = col.replace(" ", "_")
    # 移除特殊字符，但保留中文字符
    col = "".join(
        c
        for c in col
        if c.isalnum() or c == "_" or c == "(" or c == ")" or "\u4e00" <= c <= "\u9fff"
    )

    # 如果清理后列名为空，返回默认名称
    if not col:
        return "unnamed_column"

    # 确保列名不以数字开头
    if col[0].isdigit():
        col = "col_" + col
    return col.lower()


def get_database_connection_string():
    """获取数据库连接字符串

    返回:
        str: 数据库连接字符串
    """
    # 读取配置或使用默认SQLite连接
    # 注意：此处返回的是SQLite数据库的路径，可以根据需要修改为其他数据库连接字符串
    return f"sqlite:///{get_db_path()}"


def get_db_path():
    """获取数据库文件路径

    返回:
        str: 数据库文件绝对路径
    """
    return DB_PATH


def get_db_connection():
    """建立并返回一个 SQLite 数据库连接"""
    conn = sqlite3.connect(DB_PATH)
    return conn


# --- Agent Task Session Management ---


def create_agent_task_sessions_table():
    """创建 agent_task_sessions 表"""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        # Add max_iterations and stop_requested columns
        # NOTE: This only works if the table doesn't exist. For existing tables,
        # manual ALTER TABLE statements are needed or a migration script.
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS agent_task_sessions (
                session_id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                user_goal TEXT,
                sql_database_name TEXT,
                sql_table_name TEXT,
                kg_graph_name TEXT,
                model_name TEXT,
                temperature REAL DEFAULT 0.7,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                status TEXT DEFAULT 'created', -- e.g., created, running, completed, failed
                sql_session_id_ref INTEGER, -- Reference to the original SQL session ID
                kg_session_id_ref INTEGER,  -- Reference to the original KG session ID
                max_iterations INTEGER DEFAULT 20, -- Added max iterations
                stop_requested BOOLEAN DEFAULT FALSE -- Added stop request flag
            )
        """)
        conn.commit()
        logger.info("Table 'agent_task_sessions' creation check complete.")

        # --- Attempt to add columns if they don't exist (for existing databases) ---
        # This is a common pattern but might vary slightly based on SQLite version capabilities
        existing_columns = [
            col[1]
            for col in cursor.execute(
                "PRAGMA table_info(agent_task_sessions)"
            ).fetchall()
        ]
        if "max_iterations" not in existing_columns:
            logger.info(
                "Adding missing 'max_iterations' column to agent_task_sessions."
            )
            cursor.execute(
                "ALTER TABLE agent_task_sessions ADD COLUMN max_iterations INTEGER DEFAULT 20"
            )
            conn.commit()
        if "stop_requested" not in existing_columns:
            logger.info(
                "Adding missing 'stop_requested' column to agent_task_sessions."
            )
            cursor.execute(
                "ALTER TABLE agent_task_sessions ADD COLUMN stop_requested BOOLEAN DEFAULT FALSE"
            )
            conn.commit()
        # --- End column addition check ---

    except sqlite3.Error as e:
        logger.error(f"Error during agent_task_sessions table setup: {e}")
    finally:
        conn.close()


def create_agent_task_messages_table():
    """创建 agent_task_messages 表"""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS agent_task_messages (
                message_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                role TEXT NOT NULL, -- 'user', 'agent', 'tool', 'system'
                content TEXT,
                message_type TEXT NOT NULL, -- 'goal', 'thought', 'action', 'observation', 'final_answer', 'error', 'system'
                tool_name TEXT,
                tool_input TEXT,
                tool_output TEXT, -- Store tool output separately if needed, or within content
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES agent_task_sessions (session_id) ON DELETE CASCADE
            )
        """)
        # 创建索引以提高查询效率
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_agent_task_messages_session_id ON agent_task_messages (session_id)"
        )
        conn.commit()
        logger.info("Table 'agent_task_messages' created or already exists.")
    except sqlite3.Error as e:
        logger.error(f"Error creating agent_task_messages table: {e}")
    finally:
        conn.close()


# 初始化时调用创建表函数
create_agent_task_sessions_table()
create_agent_task_messages_table()


def create_agent_task_session(
    name,
    user_goal=None,
    sql_database_name=None,
    sql_table_name=None,
    kg_graph_name=None,
    model_name=None,
    temperature=0.7,
    sql_session_id_ref=None,
    kg_session_id_ref=None,
    max_iterations=20,
    stop_requested=False,
):
    """创建新的 Agent 任务会话"""
    conn = get_db_connection()
    now = datetime.now()  # Get current local time once
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO agent_task_sessions (name, user_goal, sql_database_name, sql_table_name, kg_graph_name, model_name, temperature, sql_session_id_ref, kg_session_id_ref, created_at, updated_at, max_iterations, stop_requested)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                name,
                user_goal,
                sql_database_name,
                sql_table_name,
                kg_graph_name,
                model_name,
                temperature,
                sql_session_id_ref,
                kg_session_id_ref,
                now,
                now,
                max_iterations,
                stop_requested,
            ),
        )  # Use 'now' for both created_at and updated_at
        session_id = cursor.lastrowid
        conn.commit()
        logger.info(f"Created new agent task session with ID: {session_id}")
        # 创建任务时，添加用户的目标作为第一条消息
        if user_goal:
            add_agent_task_message(session_id, "user", user_goal, message_type="goal")
        return session_id
    except sqlite3.Error as e:
        logger.error(f"Error creating agent task session: {e}")
        raise
    finally:
        conn.close()


def get_agent_task_session(session_id):
    """获取指定的 Agent 任务会话信息"""
    conn = get_db_connection()
    try:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM agent_task_sessions WHERE session_id = ?", (session_id,)
        )
        session = cursor.fetchone()
        # Convert Row to dict, ensuring all columns (including new ones) are present
        return dict(session) if session else None
    except sqlite3.Error as e:
        logger.error(f"Error getting agent task session {session_id}: {e}")
        return None
    finally:
        conn.close()


def get_agent_task_sessions():
    """获取所有 Agent 任务会话"""
    conn = get_db_connection()
    try:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM agent_task_sessions ORDER BY created_at DESC")
        sessions = cursor.fetchall()
        return [dict(session) for session in sessions]
    except sqlite3.Error as e:
        logger.error(f"Error getting all agent task sessions: {e}")
        return []
    finally:
        conn.close()


def update_agent_task_session(session_id, **kwargs):
    """更新 Agent 任务会话信息"""
    conn = get_db_connection()
    fields = []
    values = []
    # Define allowed fields explicitly for safety
    allowed_fields = [
        "name",
        "user_goal",
        "sql_database_name",
        "sql_table_name",
        "kg_graph_name",
        "model_name",
        "temperature",
        "status",
        "sql_session_id_ref",
        "kg_session_id_ref",
        "max_iterations",
        "stop_requested",  # Add new fields here
    ]

    for key, value in kwargs.items():
        # 确保只更新允许的字段
        if key in allowed_fields:
            fields.append(f"{key} = ?")
            values.append(value)

    # Ensure updated_at is always part of the update
    # Only add if other fields are being updated to avoid unnecessary timestamp bumps on no-op calls
    if fields:  # Only add timestamp if other fields are changing
        if "updated_at = ?" not in fields:  # Check if already added (unlikely but safe)
            fields.append("updated_at = ?")
            values.append(datetime.now())
    else:
        # If called with no valid kwargs, just update timestamp
        # (e.g., when add_agent_task_message calls it)
        fields.append("updated_at = ?")
        values.append(datetime.now())

    # Add session_id for the WHERE clause
    values.append(session_id)
    sql = f"UPDATE agent_task_sessions SET {', '.join(fields)} WHERE session_id = ?"

    try:
        cursor = conn.cursor()
        cursor.execute(sql, tuple(values))
        rows_affected = cursor.rowcount  # Check if the row was actually updated
        conn.commit()
        if rows_affected > 0:
            logger.info(
                f"Updated agent task session {session_id} for fields: {list(kwargs.keys())}"
            )
            return True
        else:
            logger.warning(
                f"Update agent task session {session_id} matched no rows (or values were the same). Fields: {list(kwargs.keys())}"
            )
            # Consider if this should return False or True. If the intent was met (values are already set), True might be okay.
            # Let's return False to indicate no change was written.
            return False  # Indicate no change occurred
    except sqlite3.Error as e:
        logger.error(f"Error updating agent task session {session_id}: {e}")
        return False
    finally:
        conn.close()


def delete_agent_task_session(session_id):
    """删除 Agent 任务会话及其所有消息"""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        # 使用外键的 ON DELETE CASCADE 会自动删除消息，但显式删除更安全
        cursor.execute(
            "DELETE FROM agent_task_messages WHERE session_id = ?", (session_id,)
        )
        cursor.execute(
            "DELETE FROM agent_task_sessions WHERE session_id = ?", (session_id,)
        )
        conn.commit()
        logger.info(f"Deleted agent task session {session_id} and its messages.")
        return True
    except sqlite3.Error as e:
        logger.error(f"Error deleting agent task session {session_id}: {e}")
        return False
    finally:
        conn.close()


def add_agent_task_message(
    session_id,
    role,
    content,
    message_type,
    tool_name=None,
    tool_input=None,
    tool_output=None,
):
    """添加 Agent 任务消息/步骤"""
    conn = get_db_connection()
    now = datetime.now()  # Get current local time
    try:
        cursor = conn.cursor()
        # 确保 tool_input 和 tool_output 是字符串或 None
        tool_input_str = json.dumps(tool_input) if tool_input is not None else None
        tool_output_str = json.dumps(tool_output) if tool_output is not None else None

        cursor.execute(
            """
            INSERT INTO agent_task_messages (session_id, role, content, message_type, tool_name, tool_input, tool_output, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                session_id,
                role,
                content,
                message_type,
                tool_name,
                tool_input_str,
                tool_output_str,
                now,
            ),
        )  # Explicitly set timestamp to 'now'
        message_id = cursor.lastrowid
        conn.commit()
        # 更新会话的 updated_at 时间戳
        update_agent_task_session(session_id)  # 只更新时间戳
        logger.debug(
            f"Added agent task message {message_id} to session {session_id} (Type: {message_type})"
        )
        return message_id
    except sqlite3.Error as e:
        logger.error(f"Error adding agent task message to session {session_id}: {e}")
        raise
    finally:
        conn.close()


def get_agent_task_messages(session_id):
    """获取指定 Agent 任务会话的所有消息/步骤"""
    conn = get_db_connection()
    try:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM agent_task_messages WHERE session_id = ? ORDER BY timestamp ASC",
            (session_id,),
        )
        messages = cursor.fetchall()
        result = []
        for msg in messages:
            msg_dict = dict(msg)
            # 尝试解析 JSON 字符串回 Python 对象
            try:
                if msg_dict["tool_input"]:
                    msg_dict["tool_input"] = json.loads(msg_dict["tool_input"])
            except json.JSONDecodeError:
                logger.warning(
                    f"Could not decode tool_input JSON for message {msg_dict['message_id']}"
                )
                # Keep as string
            try:
                if msg_dict["tool_output"]:
                    msg_dict["tool_output"] = json.loads(msg_dict["tool_output"])
            except json.JSONDecodeError:
                logger.warning(
                    f"Could not decode tool_output JSON for message {msg_dict['message_id']}"
                )
                # Keep as string
            result.append(msg_dict)
        return result
    except sqlite3.Error as e:
        logger.error(f"Error getting agent task messages for session {session_id}: {e}")
        return []
    finally:
        conn.close()


# --- End Agent Task Session Management ---
