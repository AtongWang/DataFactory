from flask import Blueprint, render_template, request, jsonify, current_app as app
from backend.services.vanna_service import vanna_manager
from backend.services.model_service import model_manager
import logging
from datetime import datetime, date
import re
import json
from typing import List, Dict


database_bp = Blueprint('database', __name__)
logger = logging.getLogger(__name__)

# --- Helper Functions --- Moved to module level ---
def map_type(db_type_str):
    """Maps specific DB type strings to broader categories."""
    if not db_type_str: return 'UNKNOWN'
    db_type_str = db_type_str.lower()
    if 'int' in db_type_str: return 'INT'
    if 'char' in db_type_str or 'text' in db_type_str or 'varchar' in db_type_str: return 'VARCHAR'
    if 'float' in db_type_str or 'double' in db_type_str or 'decimal' in db_type_str or 'numeric' in db_type_str: return 'FLOAT'
    if 'date' in db_type_str or 'time' in db_type_str: return 'DATETIME'
    if 'bool' in db_type_str: return 'BOOLEAN'
    return db_type_str.split('(')[0].upper() # Return base type if specific mapping not found

def sanitize_identifier(identifier):
    """Basic sanitization to prevent obvious SQL injection in identifiers."""
    if not identifier:
        return None
    # Allow alphanumeric, underscore, period, and Chinese characters (Unicode range U+4E00 to U+9FFF)
    return re.sub(r'[^a-zA-Z0-9_\.\u4e00-\u9fff]', '', identifier)

def get_safe_quoted_identifier(identifier, db_type):
    """Quotes identifiers based on DB type AFTER basic sanitization."""
    if not identifier:
        raise ValueError("Invalid identifier provided.")
    if db_type == 'postgres':
        return '"' + identifier.replace('"', '""') + '"'
    elif db_type == 'mysql':
        return '`' + identifier.replace('`', '``') + '`'
    elif db_type == 'sqlite':
        return '"' + identifier.replace('"', '""') + '"'
    else:
        return '"' + identifier.replace('"', '""') + '"'

# --- Helper function to get sample data --- ADDED
def get_column_samples(db_config: Dict, table_name: str, column_name: str, limit: int = 5) -> List:
    """Fetches distinct sample values for a column."""
    db_type = db_config['type']
    samples = []
    try:
        # Use existing helpers for safety
        safe_table = get_safe_quoted_identifier(table_name, db_type)
        safe_column = get_safe_quoted_identifier(column_name, db_type)
        # Basic query, consider adding ORDER BY RANDOM() or similar if DB supports for better variety
        sql_query = f"SELECT DISTINCT {safe_column} FROM {safe_table} WHERE {safe_column} IS NOT NULL LIMIT {limit}"

        if db_type == 'mysql':
            import pymysql
            conn = pymysql.connect(host=db_config['host'], user=db_config['username'], password=db_config['password'], database=db_config['database_name'], port=int(db_config.get('port', 3306)))
            with conn.cursor() as cursor:
                cursor.execute(sql_query)
                samples = [row[0] for row in cursor.fetchall()]
            conn.close()
        elif db_type == 'postgres':
            import psycopg2
            conn = psycopg2.connect(host=db_config['host'], user=db_config['username'], password=db_config['password'], dbname=db_config['database_name'], port=db_config.get('port', '5432'))
            with conn.cursor() as cursor:
                cursor.execute(sql_query)
                samples = [row[0] for row in cursor.fetchall()]
            conn.close()
        elif db_type == 'sqlite':
            import sqlite3
            conn = sqlite3.connect(db_config['database_name'] or ':memory:')
            with conn.cursor() as cursor:
                # Need to handle potential quoting differences for SQLite identifiers if not already handled by get_safe_quoted_identifier
                cursor.execute(sql_query)
                samples = [row[0] for row in cursor.fetchall()]
            conn.close()
        # Ensure samples are JSON serializable (convert datetimes, bytes, decimals etc.)
        serializable_samples = []
        for s in samples:
            if isinstance(s, (datetime, date)): # Example for datetime
                serializable_samples.append(s.isoformat())
            elif isinstance(s, bytes): # Example for bytes
                 try:
                     serializable_samples.append(s.decode('utf-8'))
                 except UnicodeDecodeError:
                     serializable_samples.append(f"[bytes:{len(s)}]") # Placeholder if decode fails
            elif isinstance(s, (int, float, bool, str, type(None))):
                 serializable_samples.append(s)
            else: # Fallback for other types (like Decimal)
                 serializable_samples.append(str(s))
        return serializable_samples
    except Exception as e:
        # Log the error but don't fail the whole process, just return empty samples
        app.logger.error(f"Failed to get samples for {table_name}.{column_name}: {e}")
        return []

@database_bp.route('/database-management')
def database_management():
    return render_template('database_management.html')

@database_bp.route('/api/database-info', methods=['GET'])
def get_database_info():
    try:
        # 获取数据库配置
        config = vanna_manager.get_config()
        if not config or 'database' not in config:
            return jsonify({"status": "error", "message": "未配置数据库连接"}), 400
        
        db_config = config['database']
        info = {
            "database_name": db_config['database_name'],
            "database_type": db_config['type'],
            "status": "online",
            "table_count": 0,
            "size": "--",
            "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # 获取表数量及数据库大小信息
        if db_config['type'] == 'mysql':
            import pymysql
            try:
                conn = pymysql.connect(
                    host=db_config['host'],
                    user=db_config['username'],
                    password=db_config['password'],
                    database=db_config['database_name'],
                    port=int(db_config['port']) if db_config['port'] else 3306
                )
                cursor = conn.cursor()
                
                # 获取表数量
                cursor.execute("SHOW TABLES")
                info["table_count"] = len(cursor.fetchall())
                
                # 获取数据库大小
                cursor.execute(f"""
                    SELECT ROUND(SUM(data_length + index_length) / 1024 / 1024, 2) AS size
                    FROM information_schema.tables
                    WHERE table_schema = '{db_config['database_name']}'
                """)
                size = cursor.fetchone()[0]
                info["size"] = f"{size} MB" if size else "--"
                
                cursor.close()
                conn.close()
            except Exception as e:
                app.logger.error(f"获取MySQL数据库信息失败: {str(e)}")
                info["status"] = "连接错误"
        
        elif db_config['type'] == 'postgres':
            import psycopg2
            try:
                conn = psycopg2.connect(
                    host=db_config['host'],
                    user=db_config['username'],
                    password=db_config['password'],
                    dbname=db_config['database_name'],
                    port=db_config['port'] if db_config['port'] else '5432'
                )
                cursor = conn.cursor()
                
                # 获取表数量
                cursor.execute("""
                    SELECT COUNT(*) FROM information_schema.tables 
                    WHERE table_schema = 'public'
                """)
                info["table_count"] = cursor.fetchone()[0]
                
                # 获取数据库大小
                cursor.execute("""
                    SELECT pg_size_pretty(pg_database_size(current_database()))
                """)
                info["size"] = cursor.fetchone()[0]
                
                cursor.close()
                conn.close()
            except Exception as e:
                app.logger.error(f"获取PostgreSQL数据库信息失败: {str(e)}")
                info["status"] = "连接错误"
        
        elif db_config['type'] == 'sqlite':
            import sqlite3
            import os
            try:
                # 检查文件是否存在
                if db_config['database_name'] != ':memory:' and os.path.exists(db_config['database_name']):
                    # 获取文件大小
                    size_bytes = os.path.getsize(db_config['database_name'])
                    size_mb = size_bytes / (1024 * 1024)
                    info["size"] = f"{size_mb:.2f} MB"
                    
                    # 获取表数量
                    conn = sqlite3.connect(db_config['database_name'])
                    cursor = conn.cursor()
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                    tables = cursor.fetchall()
                    info["table_count"] = len(tables)
                    
                    cursor.close()
                    conn.close()
                else:
                    if db_config['database_name'] == ':memory:':
                        info["size"] = "内存数据库"
                    else:
                        info["status"] = "数据库不存在"
            except Exception as e:
                app.logger.error(f"获取SQLite数据库信息失败: {str(e)}")
                info["status"] = "连接错误"
        
        return jsonify({"status": "success", "data": info})
    except Exception as e:
        app.logger.error(f"获取数据库信息失败: {str(e)}")
        return jsonify({"status": "error", "message": f"获取数据库信息失败: {str(e)}"}), 500

@database_bp.route('/api/tables', methods=['GET'])
def get_tables():
    try:
        # 获取数据库配置
        config = vanna_manager.get_config()
        if not config or 'database' not in config:
            return jsonify({"status": "error", "message": "未配置数据库连接"}), 400
        
        db_config = config['database']
        tables = []
        
        if db_config['type'] == 'mysql':
            import pymysql
            try:
                conn = pymysql.connect(
                    host=db_config['host'],
                    user=db_config['username'],
                    password=db_config['password'],
                    database=db_config['database_name'],
                    port=int(db_config['port']) if db_config['port'] else 3306
                )
                cursor = conn.cursor()
                
                # 获取表列表
                cursor.execute("SHOW TABLES")
                for table in cursor.fetchall():
                    tables.append({"name": table[0]})
                
                cursor.close()
                conn.close()
            except Exception as e:
                app.logger.error(f"获取MySQL表列表失败: {str(e)}")
                return jsonify({"status": "error", "message": f"获取表列表失败: {str(e)}"}), 500
        
        elif db_config['type'] == 'postgres':
            import psycopg2
            try:
                conn = psycopg2.connect(
                    host=db_config['host'],
                    user=db_config['username'],
                    password=db_config['password'],
                    dbname=db_config['database_name'],
                    port=db_config['port'] if db_config['port'] else '5432'
                )
                cursor = conn.cursor()
                
                # 获取表列表
                cursor.execute("""
                    SELECT table_name FROM information_schema.tables 
                    WHERE table_schema = 'public'
                """)
                for table in cursor.fetchall():
                    tables.append({"name": table[0]})
                
                cursor.close()
                conn.close()
            except Exception as e:
                app.logger.error(f"获取PostgreSQL表列表失败: {str(e)}")
                return jsonify({"status": "error", "message": f"获取表列表失败: {str(e)}"}), 500
        
        elif db_config['type'] == 'sqlite':
            import sqlite3
            try:
                conn = sqlite3.connect(db_config['database_name'] or ':memory:')
                cursor = conn.cursor()
                
                # 获取表列表
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                for table in cursor.fetchall():
                    tables.append({"name": table[0]})
                
                cursor.close()
                conn.close()
            except Exception as e:
                app.logger.error(f"获取SQLite表列表失败: {str(e)}")
                return jsonify({"status": "error", "message": f"获取表列表失败: {str(e)}"}), 500
        
        return jsonify({"status": "success", "data": tables})
    except Exception as e:
        app.logger.error(f"获取表格列表失败: {str(e)}")
        return jsonify({"status": "error", "message": f"获取表格列表失败: {str(e)}"}), 500
    

@database_bp.route('/api/table-data/<table_name>', methods=['GET'])
def get_specific_table_data(table_name):
    try:
        limit = request.args.get('limit', default=100, type=int)
        
        if not table_name:
            return jsonify({"status": "error", "message": "缺少表名参数"}), 400
        
        # 获取数据库配置
        config = vanna_manager.get_config()
        if not config or 'database' not in config:
            return jsonify({"status": "error", "message": "未配置数据库连接"}), 400
        
        db_config = config['database']
        data = []
        columns = []
        
        if db_config['type'] == 'mysql':
            import pymysql
            conn = pymysql.connect(
                host=db_config['host'],
                user=db_config['username'],
                password=db_config['password'],
                database=db_config['database_name'],
                port=int(db_config['port']) if db_config['port'] else 3306
            )
            try:
                with conn.cursor() as cursor:
                    # 获取表结构
                    cursor.execute(f"DESCRIBE `{table_name}`")
                    columns = [col[0] for col in cursor.fetchall()]
                    
                    # 获取数据
                    cursor.execute(f"SELECT * FROM `{table_name}` LIMIT {limit}")
                    rows = cursor.fetchall()
                    
                    # 将数据转换为字典列表
                    data = [dict(zip(columns, row)) for row in rows]
            finally:
                conn.close()
        
        elif db_config['type'] == 'postgres':
            import psycopg2
            conn = psycopg2.connect(
                host=db_config['host'],
                user=db_config['username'],
                password=db_config['password'],
                dbname=db_config['database_name'],
                port=db_config['port'] if db_config['port'] else '5432'
            )
            try:
                with conn.cursor() as cursor:
                    # 获取表结构
                    cursor.execute(f"""
                        SELECT column_name FROM information_schema.columns 
                        WHERE table_name = '{table_name}'
                    """)
                    columns = [col[0] for col in cursor.fetchall()]
                    
                    # 获取数据
                    cursor.execute(f'SELECT * FROM "{table_name}" LIMIT {limit}')
                    rows = cursor.fetchall()
                    
                    # 将数据转换为字典列表
                    data = [dict(zip(columns, row)) for row in rows]
            finally:
                conn.close()
        
        elif db_config['type'] == 'sqlite':
            import sqlite3
            conn = sqlite3.connect(db_config['database_name'] or ':memory:')
            try:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                # 获取表结构
                cursor.execute(f"PRAGMA table_info({table_name})")
                columns = [row['name'] for row in cursor.fetchall()]
                
                # 获取数据
                cursor.execute(f"SELECT * FROM {table_name} LIMIT {limit}")
                rows = cursor.fetchall()
                
                # 将数据转换为字典列表
                data = [dict(row) for row in rows]
            finally:
                cursor.close()
                conn.close()
        
        return jsonify({
            "status": "success", 
            "data": {
                "columns": columns,
                "rows": data
            }
        })
    except Exception as e:
        app.logger.error(f"获取表数据失败: {str(e)}")
        return jsonify({"status": "error", "message": f"获取表数据失败: {str(e)}"}), 500

@database_bp.route('/api/execute-sql', methods=['POST'])
def execute_sql():
    try:
        data = request.json
        sql_query = data.get('sql')
        
        if not sql_query:
            return jsonify({"status": "error", "message": "缺少SQL查询语句"}), 400
        
        # 获取数据库配置
        config = vanna_manager.get_config()
        if not config or 'database' not in config:
            return jsonify({"status": "error", "message": "未配置数据库连接"}), 400
        
        db_config = config['database']
        result = {
            "columns": [],
            "rows": []
        }
        
        if db_config['type'] == 'mysql':
            import pymysql
            try:
                conn = pymysql.connect(
                    host=db_config['host'],
                    user=db_config['username'],
                    password=db_config['password'],
                    database=db_config['database_name'],
                    port=int(db_config['port']) if db_config['port'] else 3306
                )
                cursor = conn.cursor()
                
                # 执行查询
                cursor.execute(sql_query)
                
                # 对于SELECT查询，返回结果集
                if sql_query.strip().lower().startswith('select'):
                    # 获取列名
                    result["columns"] = [col[0] for col in cursor.description]
                    
                    # 获取数据
                    rows = cursor.fetchall()
                    result["rows"] = [dict(zip(result["columns"], row)) for row in rows]
                else:
                    # 对于非SELECT查询，返回受影响的行数
                    conn.commit()
                    result["columns"] = ["affected_rows"]
                    result["rows"] = [{"affected_rows": cursor.rowcount}]
                
                cursor.close()
                conn.close()
            except Exception as e:
                app.logger.error(f"执行MySQL查询失败: {str(e)}")
                return jsonify({"status": "error", "message": f"执行查询失败: {str(e)}"}), 500
        
        elif db_config['type'] == 'postgres':
            import psycopg2
            try:
                conn = psycopg2.connect(
                    host=db_config['host'],
                    user=db_config['username'],
                    password=db_config['password'],
                    dbname=db_config['database_name'],
                    port=db_config['port'] if db_config['port'] else '5432'
                )
                cursor = conn.cursor()
                
                # 执行查询
                cursor.execute(sql_query)
                
                # 对于SELECT查询，返回结果集
                if sql_query.strip().lower().startswith('select'):
                    # 获取列名
                    result["columns"] = [desc[0] for desc in cursor.description]
                    
                    # 获取数据
                    rows = cursor.fetchall()
                    result["rows"] = [dict(zip(result["columns"], row)) for row in rows]
                else:
                    # 对于非SELECT查询，返回受影响的行数
                    conn.commit()
                    result["columns"] = ["affected_rows"]
                    result["rows"] = [{"affected_rows": cursor.rowcount}]
                
                cursor.close()
                conn.close()
            except Exception as e:
                app.logger.error(f"执行PostgreSQL查询失败: {str(e)}")
                return jsonify({"status": "error", "message": f"执行查询失败: {str(e)}"}), 500
        
        elif db_config['type'] == 'sqlite':
            import sqlite3
            try:
                conn = sqlite3.connect(db_config['database_name'] or ':memory:')
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                # 执行查询
                cursor.execute(sql_query)
                
                # 对于SELECT查询，返回结果集
                if sql_query.strip().lower().startswith('select'):
                    # 获取列名
                    result["columns"] = [desc[0] for desc in cursor.description]
                    
                    # 获取数据
                    rows = cursor.fetchall()
                    result["rows"] = [dict(row) for row in rows]
                else:
                    # 对于非SELECT查询，返回受影响的行数
                    conn.commit()
                    result["columns"] = ["affected_rows"]
                    result["rows"] = [{"affected_rows": cursor.rowcount}]
                
                cursor.close()
                conn.close()
            except Exception as e:
                app.logger.error(f"执行SQLite查询失败: {str(e)}")
                return jsonify({"status": "error", "message": f"执行查询失败: {str(e)}"}), 500
        
        return jsonify({"status": "success", "data": result})
    except Exception as e:
        app.logger.error(f"执行SQL查询失败: {str(e)}")
        return jsonify({"status": "error", "message": f"执行SQL查询失败: {str(e)}"}), 500

@database_bp.route('/api/delete-table', methods=['POST'])
def delete_table():
    try:
        data = request.json
        table_name = data.get('table_name')
        
        if not table_name:
            return jsonify({"status": "error", "message": "缺少表名参数"}), 400
        
        # 获取数据库配置
        config = vanna_manager.get_config()
        if not config or 'database' not in config:
            return jsonify({"status": "error", "message": "未配置数据库连接"}), 400
        
        db_config = config['database']
        
        if db_config['type'] == 'mysql':
            import pymysql
            try:
                conn = pymysql.connect(
                    host=db_config['host'],
                    user=db_config['username'],
                    password=db_config['password'],
                    database=db_config['database_name'],
                    port=int(db_config['port']) if db_config['port'] else 3306
                )
                cursor = conn.cursor()
                
                # 删除表
                cursor.execute(f"DROP TABLE IF EXISTS `{table_name}`")
                conn.commit()
                
                cursor.close()
                conn.close()
            except Exception as e:
                app.logger.error(f"删除MySQL表失败: {str(e)}")
                return jsonify({"status": "error", "message": f"删除表失败: {str(e)}"}), 500
        
        elif db_config['type'] == 'postgres':
            import psycopg2
            try:
                conn = psycopg2.connect(
                    host=db_config['host'],
                    user=db_config['username'],
                    password=db_config['password'],
                    dbname=db_config['database_name'],
                    port=db_config['port'] if db_config['port'] else '5432'
                )
                cursor = conn.cursor()
                
                # 删除表
                cursor.execute(f'DROP TABLE IF EXISTS "{table_name}"')
                conn.commit()
                
                cursor.close()
                conn.close()
            except Exception as e:
                app.logger.error(f"删除PostgreSQL表失败: {str(e)}")
                return jsonify({"status": "error", "message": f"删除表失败: {str(e)}"}), 500
        
        elif db_config['type'] == 'sqlite':
            import sqlite3
            try:
                conn = sqlite3.connect(db_config['database_name'] or ':memory:')
                cursor = conn.cursor()
                
                # 删除表
                cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
                conn.commit()
                
                cursor.close()
                conn.close()
            except Exception as e:
                app.logger.error(f"删除SQLite表失败: {str(e)}")
                return jsonify({"status": "error", "message": f"删除表失败: {str(e)}"}), 500
        
        return jsonify({"status": "success", "message": f"表 {table_name} 已成功删除"})
    except Exception as e:
        app.logger.error(f"删除表失败: {str(e)}")
        return jsonify({"status": "error", "message": f"删除表失败: {str(e)}"}), 500

@database_bp.route('/api/table-schema/<table_name>', methods=['GET'])
def get_table_schema(table_name):
    """获取指定表的结构信息（列名、数据类型、约束等）"""
    try:
        app.logger.info(f"Received request for table schema: '{table_name}'") # Log original
        safe_table_name_param = sanitize_identifier(table_name)
        app.logger.info(f"Sanitized table name: '{safe_table_name_param}'") # Log sanitized
        if not safe_table_name_param:
            return jsonify({"status": "error", "message": "无效的表名"}), 400
        
        config = vanna_manager.get_config()
        if not config or 'database' not in config:
            return jsonify({"status": "error", "message": "未配置数据库连接"}), 400
        
        db_config = config['database']
        db_type = db_config['type']
        schema = []
        
        if db_type == 'mysql':
            import pymysql
            conn = pymysql.connect(
                host=db_config['host'], user=db_config['username'], password=db_config['password'],
                database=db_config['database_name'], port=int(db_config.get('port', 3306)),
                cursorclass=pymysql.cursors.DictCursor # Use DictCursor
            )
            try:
                with conn.cursor() as cursor:
                    # Use information_schema for more details
                    sql = """
                    SELECT
                        COLUMN_NAME,
                        COLUMN_TYPE,
                        IS_NULLABLE,
                        COLUMN_KEY,
                        COLUMN_DEFAULT,
                        EXTRA
                    FROM information_schema.COLUMNS
                    WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s
                    ORDER BY ORDINAL_POSITION;
                    """
                    cursor.execute(sql, (db_config['database_name'], safe_table_name_param))
                    for row in cursor.fetchall():
                        schema.append({
                            "name": row['COLUMN_NAME'],
                            "type": map_type(row['COLUMN_TYPE']),
                            "description": "", # Placeholder
                            "is_nullable": row['IS_NULLABLE'] == 'YES',
                            "is_primary_key": "PRI" in row['COLUMN_KEY'],
                            "is_unique": "UNI" in row['COLUMN_KEY'], # Note: PRI implies UNI in MySQL schema
                            "default_value": row['COLUMN_DEFAULT'],
                            # Add more fields like 'auto_increment': 'auto_increment' in row['EXTRA'] if needed
                        })
            finally:
                conn.close()

        elif db_type == 'postgres':
            import psycopg2
            import psycopg2.extras
            conn = psycopg2.connect(
                host=db_config['host'], user=db_config['username'], password=db_config['password'],
                dbname=db_config['database_name'], port=db_config.get('port', '5432')
            )
            try:
                with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:
                    # Query columns and constraints separately
                    sql_columns = """
                    SELECT
                        column_name,
                        data_type,
                        is_nullable,
                        column_default
                    FROM information_schema.columns
                    WHERE table_schema = 'public' AND table_name = %s
                    ORDER BY ordinal_position;
                    """
                    cursor.execute(sql_columns, (safe_table_name_param,))
                    columns_info = {row['column_name']: row for row in cursor.fetchall()}

                    # Get Primary Key and Unique constraints
                    sql_constraints = """
                    SELECT
                        kcu.column_name,
                        tc.constraint_type
                    FROM information_schema.table_constraints AS tc
                    JOIN information_schema.key_column_usage AS kcu
                      ON tc.constraint_name = kcu.constraint_name
                      AND tc.table_schema = kcu.table_schema
                    WHERE tc.table_schema = 'public'
                      AND tc.table_name = %s
                      AND tc.constraint_type IN ('PRIMARY KEY', 'UNIQUE');
                    """
                    cursor.execute(sql_constraints, (safe_table_name_param,))
                    constraints = {}
                    for row in cursor.fetchall():
                        col_name = row['column_name']
                        if col_name not in constraints: constraints[col_name] = set()
                        constraints[col_name].add(row['constraint_type'])

                    # Combine info
                    for col_name, col_data in columns_info.items():
                        col_constraints = constraints.get(col_name, set())
                        schema.append({
                            "name": col_name,
                            "type": map_type(col_data['data_type']),
                            "description": "",
                            "is_nullable": col_data['is_nullable'] == 'YES',
                            "is_primary_key": "PRIMARY KEY" in col_constraints,
                            "is_unique": "UNIQUE" in col_constraints,
                            "default_value": col_data['column_default']
                        })
            finally:
                conn.close()

        elif db_type == 'sqlite':
            import sqlite3
            conn = sqlite3.connect(db_config['database_name'] or ':memory:')
            # Enable foreign keys if needed for future reference, though PRAGMA doesn't show them easily
            # conn.execute("PRAGMA foreign_keys = ON;")
            conn.row_factory = sqlite3.Row # Use row_factory for dict-like access
            try:
                cursor = conn.cursor()
                cursor.execute(f"PRAGMA table_info({get_safe_quoted_identifier(safe_table_name_param, db_type)})")
                # cid, name, type, notnull, dflt_value, pk
                table_info = cursor.fetchall()
                if not table_info:
                     raise ValueError(f"Table '{safe_table_name_param}' not found or has no columns.")

                # Check unique constraints separately using PRAGMA index_list and index_info
                # This is more involved, for simplicity we'll skip detailed unique constraint detection for now
                # but mark PK as unique.
                for row in table_info:
                     is_pk = row['pk'] > 0
                     schema.append({
                         "name": row['name'],
                         "type": map_type(row['type']),
                         "description": "",
                         "is_nullable": not row['notnull'],
                         "is_primary_key": is_pk,
                         "is_unique": is_pk, # Simplification: Treat PK as unique, need PRAGMA index_list for others
                         "default_value": row['dflt_value']
                     })
            finally:
                 conn.close()
        else:
            return jsonify({"status": "error", "message": "不支持的数据库类型"}), 400

        return jsonify({"status": "success", "data": schema})

    except ValueError as ve: # Catch specific errors like table not found from PRAGMA
        app.logger.warning(f"Value error getting schema ({table_name}): {str(ve)}")
        return jsonify({"status": "error", "message": str(ve)}), 404
    except Exception as e:
        # Log the full traceback for debugging
        app.logger.exception(f"获取表结构失败 ({table_name}): {str(e)}")
        # Check for common connection or table not found errors
        err_str = str(e).lower()
        if "doesn't exist" in err_str or "does not exist" in err_str or "no such table" in err_str:
             return jsonify({"status": "error", "message": f"表 '{table_name}' 不存在或无法访问。"}), 404
        if "access denied" in err_str or "authentication failed" in err_str:
             return jsonify({"status": "error", "message": "数据库连接认证失败。"}), 403
        return jsonify({"status": "error", "message": f"获取表结构时发生内部错误。"}), 500

@database_bp.route('/api/column-unique-values/<table_name>/<column_name>', methods=['GET'])
def get_column_unique_values(table_name, column_name):
    """获取指定列的唯一值 (限制数量)"""
    try:
        limit = request.args.get('limit', default=20, type=int) # Limit unique values fetched

        if not table_name or not column_name:
            return jsonify({"status": "error", "message": "缺少表名或列名参数"}), 400

        config = vanna_manager.get_config()
        if not config or 'database' not in config:
            return jsonify({"status": "error", "message": "未配置数据库连接"}), 400

        db_config = config['database']
        unique_values = []

        # Basic sanitization (consider more robust SQL injection prevention if needed)
        # For simple identifiers, quoting should be sufficient for most standard DBs
        safe_table_name = f'"{table_name}"' if db_config['type'] == 'postgres' else f'`{table_name}`' if db_config['type'] == 'mysql' else table_name
        safe_column_name = f'"{column_name}"' if db_config['type'] == 'postgres' else f'`{column_name}`' if db_config['type'] == 'mysql' else column_name
        
        # Check if column exists first (to provide better error)
        # This adds overhead but improves UX. Could be optional.
        # ... (implementation for checking column existence) ...

        sql_query = f"SELECT DISTINCT {safe_column_name} FROM {safe_table_name} WHERE {safe_column_name} IS NOT NULL LIMIT {limit}"

        if db_config['type'] == 'mysql':
            import pymysql
            conn = pymysql.connect(
                host=db_config['host'], user=db_config['username'], password=db_config['password'],
                database=db_config['database_name'], port=int(db_config.get('port', 3306))
            )
            try:
                with conn.cursor() as cursor:
                    cursor.execute(sql_query)
                    unique_values = [row[0] for row in cursor.fetchall()]
            finally:
                conn.close()
        
        elif db_config['type'] == 'postgres':
            import psycopg2
            conn = psycopg2.connect(
                host=db_config['host'], user=db_config['username'], password=db_config['password'],
                dbname=db_config['database_name'], port=db_config.get('port', '5432')
            )
            try:
                with conn.cursor() as cursor:
                    cursor.execute(sql_query)
                    unique_values = [row[0] for row in cursor.fetchall()]
            finally:
                conn.close()
        
        elif db_config['type'] == 'sqlite':
            import sqlite3
            conn = sqlite3.connect(db_config['database_name'] or ':memory:')
            try:
                cursor = conn.cursor()
                cursor.execute(sql_query)
                unique_values = [row[0] for row in cursor.fetchall()]
            finally:
                 conn.close()
        else:
            return jsonify({"status": "error", "message": "不支持的数据库类型"}), 400

        return jsonify({"status": "success", "data": unique_values})

    except Exception as e:
        app.logger.error(f"获取唯一值失败 ({table_name}.{column_name}): {str(e)}")
        # Add specific error checks if needed (e.g., column not found)
        return jsonify({"status": "error", "message": f"获取唯一值失败: {str(e)}"}), 500

@database_bp.route('/api/generate-column-description', methods=['POST'])
def generate_column_description():
    """使用LLM为指定列生成描述"""
    try:
        data = request.json
        table_name = sanitize_identifier(data.get('table_name'))
        column_name = sanitize_identifier(data.get('column_name'))

        if not table_name or not column_name:
            return jsonify({"status": "error", "message": "缺少表名或列名"}), 400

        # 1. Get DB config
        config = vanna_manager.get_config()
        if not config or 'database' not in config:
            return jsonify({"status": "error", "message": "未配置数据库连接"}), 400
        db_config = config['database']
        db_type = db_config['type']

        # 2. Get Column Type and Sample Data
        column_type_raw = 'UNKNOWN' # Use a different var name to avoid confusion
        sample_data = []
        limit = 10 # Number of unique samples to fetch

        safe_table = get_safe_quoted_identifier(table_name, db_type)
        safe_column = get_safe_quoted_identifier(column_name, db_type)

        try:
            # Fetch type first (reuse parts of get_table_schema logic if possible)
            if db_type == 'mysql':
                import pymysql
                conn = pymysql.connect(host=db_config['host'], user=db_config['username'], password=db_config['password'], database=db_config['database_name'], port=int(db_config.get('port', 3306)))
                with conn.cursor() as cursor:
                    cursor.execute(f"SELECT COLUMN_TYPE FROM information_schema.COLUMNS WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s AND COLUMN_NAME = %s", (db_config['database_name'], table_name, column_name))
                    result = cursor.fetchone()
                    if result: column_type_raw = result[0]
                    # Fetch sample data
                    cursor.execute(f"SELECT DISTINCT {safe_column} FROM {safe_table} WHERE {safe_column} IS NOT NULL ORDER BY RAND() LIMIT {limit}") # Use RAND() for variety if table is large
                    sample_data = [row[0] for row in cursor.fetchall()]
                conn.close()
            elif db_type == 'postgres':
                 import psycopg2
                 conn = psycopg2.connect(host=db_config['host'], user=db_config['username'], password=db_config['password'], dbname=db_config['database_name'], port=db_config.get('port', '5432'))
                 with conn.cursor() as cursor:
                     cursor.execute(f"SELECT data_type FROM information_schema.columns WHERE table_schema = 'public' AND table_name = %s AND column_name = %s", (table_name, column_name))
                     result = cursor.fetchone()
                     if result: column_type_raw = result[0]
                     # Fetch sample data - Use TABLESAMPLE for large tables if supported and needed
                     cursor.execute(f"SELECT DISTINCT {safe_column} FROM {safe_table} WHERE {safe_column} IS NOT NULL LIMIT {limit}")
                     sample_data = [row[0] for row in cursor.fetchall()]
                 conn.close()
            elif db_type == 'sqlite':
                 import sqlite3
                 conn = sqlite3.connect(db_config['database_name'] or ':memory:')
                 conn.row_factory = sqlite3.Row
                 with conn.cursor() as cursor:
                     cursor.execute(f"PRAGMA table_info({safe_table})")
                     for row in cursor.fetchall():
                         if row['name'] == column_name:
                             column_type_raw = row['type']
                             break
                     # Fetch sample data
                     cursor.execute(f"SELECT DISTINCT {safe_column} FROM {safe_table} WHERE {safe_column} IS NOT NULL LIMIT {limit}")
                     sample_data = [row[0] for row in cursor.fetchall()]
                 conn.close()
            else:
                 return jsonify({"status": "error", "message": "不支持的数据库类型，无法获取列信息"}), 400

            # Call module-level map_type here
            column_type_mapped = map_type(column_type_raw)

        except Exception as e:
             app.logger.error(f"获取列信息失败 ({table_name}.{column_name}): {str(e)}")
             return jsonify({"status": "error", "message": f"获取列信息失败: {str(e)}"}), 500

        # 3. Check if model_manager and its provider are initialized
        if not model_manager or not model_manager.provider:
             app.logger.warning("Model manager or provider not initialized")
             return jsonify({"status": "error", "message": "模型服务未正确初始化"}), 501

        # 4. Construct Prompt
        # 获取当前语言设置
        config = vanna_manager.get_config()
        language = config.get('language', {}).get('language', 'zh-CN')
        
        if language == 'zh-CN':
            prompt = f"""请为数据库列 '{column_name}' 生成一个简洁、准确的中文描述。
列数据类型: {column_type_mapped}
样本唯一值（最多{limit}个）: {sample_data if sample_data else '[未找到唯一值或列为空]'}
请基于列名、类型和样本值推断该列的可能用途或含义。
描述:"""
        else:
            # 英文提示词
            prompt = f"""Generate a concise, one-sentence description for the database column '{column_name}' in table '{table_name}'.
Column Data Type: {column_type_mapped}
Sample distinct values (up to {limit}): {sample_data if sample_data else '[No distinct values found or column is empty]'}
Focus on the likely purpose or meaning of the data in this column based on its name, type, and sample values.
Description:"""

        # 5. Call LLM via model_manager.provider.chat
        response_content = None # Initialize for logging in case of error
        try:
            messages = [{'role': 'user', 'content': prompt}]
            # REMOVED format parameter for Ollama compatibility based on error
            # json_format_hint = {"type": "json_object"} if db_config.get('type') == 'openai' else {"type": "json"}

            app.logger.info(f"Sending request to LLM for table {safe_table} description generation.")
            # Detailed model info logging
            provider_type = model_manager.config.get('type', 'unknown')
            model_name = getattr(model_manager.provider, 'model', 'unknown')
            app.logger.info(f"使用模型提供商: {provider_type}, 模型名称: {model_name}")
            
            # Call chat without the format hint
            response_content = model_manager.provider.chat(messages=messages)

            if not response_content:
                raise Exception("LLM did not return content.")

            # --- Enhanced JSON Extraction --- START
            json_string = None
            descriptions = None
            try:
                # Find the first opening curly brace
                start_index = response_content.find('{')
                if start_index == -1:
                    raise ValueError("Could not find the start of a JSON object ('{') in the response.")

                brace_level = 0
                end_index = -1
                # Iterate through the string to find the matching closing brace
                for i, char in enumerate(response_content[start_index:]):
                    if char == '{':
                        brace_level += 1
                    elif char == '}':
                        brace_level -= 1
                        if brace_level == 0:
                            end_index = start_index + i
                            break # Found the matching brace

                if end_index == -1:
                    # Fallback: Try cleaning markdown if matching brace wasn't found cleanly
                    # This might happen if the JSON is incomplete or malformed before potential markdown
                    cleaned_response = re.sub(r'^```json\\s*|\\s*```$', '', response_content.strip(), flags=re.MULTILINE)
                    if cleaned_response.startswith('{') and cleaned_response.endswith('}'):
                         json_string = cleaned_response
                         app.logger.warning(f"JSON object boundaries possibly unclear, using regex cleaned response for table {safe_table}")
                    else:
                         raise ValueError("Could not find the matching end brace ('}') for the JSON object.")
                else:
                    # Extract the substring containing the JSON object
                    json_string = response_content[start_index : end_index + 1]

                # Attempt to parse the extracted string
                descriptions = json.loads(json_string)

            except json.JSONDecodeError as jde:
                # Re-raise with more context
                error_message = f"Failed to parse JSON: {jde}. "
                if json_string is not None:
                    error_message += f"Attempted to parse: {json_string[:200]}..." # Log beginning of attempted parse
                else: # This case should be less likely with the new logic but keep for robustness
                    error_message += f"Could not extract valid JSON boundaries. Original response: {response_content[:200]}..."
                # Log the full problematic string if available for easier debugging
                app.logger.error(f"Full string attempted for JSON parsing: {json_string}")
                raise json.JSONDecodeError(error_message, jde.doc, jde.pos) # Keep original doc and pos if possible
            # --- Enhanced JSON Extraction --- END

            if not isinstance(descriptions, dict):
                 raise ValueError(f"LLM response parsed, but it's not a dictionary. Type: {type(descriptions)}")

            app.logger.info(f"Generated descriptions for table {safe_table}. Count: {len(descriptions)}")
            return jsonify({"status": "success", "descriptions": descriptions})

        except json.JSONDecodeError as jde:
             # This will now catch the re-raised error with more context
             app.logger.error(f"解析LLM的JSON响应失败 ({safe_table}): {jde}")
             # Log the original full content only if parsing completely failed at the boundary finding stage
             if json_string is None:
                 app.logger.error(f"原始响应内容: {response_content}")
             return jsonify({"status": "error", "message": f"AI未能返回有效的JSON格式描述: {jde}"}), 500
        except Exception as e:
            app.logger.exception(f"调用LLM批量生成描述失败 ({safe_table}): {str(e)}")
            return jsonify({"status": "error", "message": f"调用AI批量生成描述失败: {str(e)}"}), 500

    except Exception as e:
        app.logger.error(f"生成列描述时发生内部错误: {str(e)}")
        return jsonify({"status": "error", "message": f"生成列描述时发生内部错误: {str(e)}"}), 500

@database_bp.route('/api/add-ddl-training', methods=['POST'])
def add_ddl_training():
    """接收DDL Markdown并添加到Vanna训练数据"""
    try:
        data = request.json
        ddl_markdown = data.get('ddl_markdown')
        table_name = data.get('table_name', '')

        if not ddl_markdown:
             return jsonify({"status": "error", "message": "缺少 DDL Markdown 内容"}), 400

        # 使用 vanna_manager 添加 DDL，传递表名
        vanna_manager.train(ddl=ddl_markdown, table_name=table_name)
        app.logger.info(f"成功添加 DDL 训练数据，表名: {table_name}")

        return jsonify({"status": "success", "message": "DDL 信息已成功添加到训练数据"})

    except AttributeError:
         app.logger.error("vanna_manager 可能未正确初始化或缺少 train 方法")
         return jsonify({"status": "error", "message": "无法调用训练方法，服务配置可能存在问题。"}), 500
    except Exception as e:
        app.logger.error(f"添加 DDL 训练数据失败: {str(e)}")
        return jsonify({"status": "error", "message": f"添加 DDL 训练数据失败: {str(e)}"}), 500
    
@database_bp.route('/api/databases', methods=['GET'])
def get_databases():
    """获取已连接的数据库信息"""
    try:
        config = vanna_manager.get_config()
        if not config or 'database' not in config:
            return jsonify({"status": "error", "message": "未配置数据库连接"}), 400
        
        db_config = config['database']
        
        # 创建数据库展示名称，包含数据库类型和数据库名
        db_type = db_config.get('type', 'Unknown').upper()
        db_name = db_config.get('database_name', '')
        
        # 返回当前连接的数据库信息
        databases = [{
            'name': db_name,
            'display_name': f"{db_type} - {db_name}",
            'description': f"已连接的{db_type}数据库"
        }]
        
        return jsonify({
            'status': 'success',
            'databases': databases
        })
    except Exception as e:
        logging.error(f"获取数据库信息失败: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500
    
@database_bp.route('/api/databases/<database_name>/tables', methods=['GET'])
def get_database_tables(database_name):
    """获取指定数据库的表列表"""
    try:
        # 获取数据库配置
        config = vanna_manager.get_config()
        if not config or 'database' not in config:
            return jsonify({"status": "error", "message": "未配置数据库连接"}), 400
        
        db_config = config['database']
        tables = []
        
        if db_config['type'] == 'mysql':
            import pymysql
            try:
                conn = pymysql.connect(
                    host=db_config['host'],
                    user=db_config['username'],
                    password=db_config['password'],
                    database=db_config['database_name'],
                    port=int(db_config['port']) if db_config['port'] else 3306
                )
                cursor = conn.cursor()
                
                # 获取表列表
                cursor.execute("SHOW TABLES")
                for table in cursor.fetchall():
                    tables.append({
                        "name": table[0],
                        "display_name": table[0],
                        "description": ""
                    })
                
                cursor.close()
                conn.close()
            except Exception as e:
                app.logger.error(f"获取MySQL表列表失败: {str(e)}")
                return jsonify({"status": "error", "message": f"获取表列表失败: {str(e)}"}), 500
        
        elif db_config['type'] == 'postgres':
            import psycopg2
            try:
                conn = psycopg2.connect(
                    host=db_config['host'],
                    user=db_config['username'],
                    password=db_config['password'],
                    dbname=db_config['database_name'],
                    port=db_config['port'] if db_config['port'] else '5432'
                )
                cursor = conn.cursor()
                
                # 获取表列表
                cursor.execute("""
                    SELECT table_name FROM information_schema.tables 
                    WHERE table_schema = 'public'
                """)
                for table in cursor.fetchall():
                    tables.append({
                        "name": table[0],
                        "display_name": table[0],
                        "description": ""
                    })
                
                cursor.close()
                conn.close()
            except Exception as e:
                app.logger.error(f"获取PostgreSQL表列表失败: {str(e)}")
                return jsonify({"status": "error", "message": f"获取表列表失败: {str(e)}"}), 500
        
        elif db_config['type'] == 'sqlite':
            import sqlite3
            try:
                conn = sqlite3.connect(db_config['database_name'] or ':memory:')
                cursor = conn.cursor()
                
                # 获取表列表
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                for table in cursor.fetchall():
                    tables.append({
                        "name": table[0],
                        "display_name": table[0],
                        "description": ""
                    })
                
                cursor.close()
                conn.close()
            except Exception as e:
                app.logger.error(f"获取SQLite表列表失败: {str(e)}")
                return jsonify({"status": "error", "message": f"获取表列表失败: {str(e)}"}), 500
        
        return jsonify({
            'status': 'success',
            'database': database_name,
            'tables': tables
        })
    except Exception as e:
        logging.error(f"获取表列表失败: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500
    
@database_bp.route('/api/generate-all-descriptions/<table_name>', methods=['POST'])
def generate_all_descriptions(table_name):
    """使用LLM为指定表的所有列生成中文描述"""
    try:
        safe_table_name = sanitize_identifier(table_name)
        if not safe_table_name:
            return jsonify({"status": "error", "message": "无效的表名"}), 400

        # 1. Get DB config
        config = vanna_manager.get_config()
        if not config or 'database' not in config:
            return jsonify({"status": "error", "message": "未配置数据库连接"}), 400
        db_config = config['database']

        # 2. Get Full Schema & Sample Data
        schema_data = []
        prompt_column_data = []
        try:
            # Call the existing function to get schema data
            schema_response = get_table_schema(safe_table_name)
            # Check if the response object itself is valid before accessing status_code
            if not schema_response or not hasattr(schema_response, 'status_code'):
                 app.logger.error(f"Invalid response received from get_table_schema for {safe_table_name}")
                 return jsonify({"status": "error", "message": "获取表结构时内部错误"}), 500

            if schema_response.status_code != 200:
                 # If get_table_schema failed, return its error JSON payload directly
                 try:
                     error_payload = schema_response.get_json()
                 except: # Handle cases where get_json might fail on non-JSON response
                     error_payload = {"status": "error", "message": "获取表结构失败，无法解析错误信息。"}
                 # Ensure status code is propagated if possible, default to 500
                 status_code = schema_response.status_code if schema_response.status_code >= 400 else 500
                 return jsonify(error_payload), status_code

            schema_data = schema_response.get_json().get('data', [])
            if not schema_data:
                 return jsonify({"status": "error", "message": "无法获取表结构信息或表为空"}), 404 # 404 might be better if schema is empty

            # Get sample data for each column
            for col in schema_data:
                col_name = col.get('name')
                col_type = col.get('type', 'UNKNOWN') # Handle missing type just in case
                if not col_name: continue # Skip if column name is missing

                samples = get_column_samples(db_config, safe_table_name, col_name, limit=5)
                prompt_column_data.append({
                    "column_name": col_name,
                    "type": col_type,
                    "samples": samples
                })

        except Exception as e:
            app.logger.exception(f"获取表结构或样本数据失败 ({safe_table_name}): {str(e)}") # Use exception logger
            return jsonify({"status": "error", "message": f"获取表结构或样本数据失败: {str(e)}"}), 500

        # 3. Check Model Manager
        if not model_manager or not model_manager.provider:
            app.logger.warning("Model manager or provider not initialized")
            return jsonify({"status": "error", "message": "模型服务未正确初始化"}), 501

        # 4. Construct Comprehensive Prompt
        # 获取当前语言设置
        config = vanna_manager.get_config()
        language = config.get('language', {}).get('language', 'zh-CN')
        
        if language == 'zh-CN':
            prompt = f"""请为数据库表 '{safe_table_name}' 中的每一列生成一个简洁、准确的中文描述。
请基于列名、数据类型和少量数据样本来推断该列的用途或含义。

表结构和数据样本如下：
{json.dumps(prompt_column_data, ensure_ascii=False, indent=2)}

请严格按照以下 JSON 对象格式返回结果，其中键是列名 (column_name)，值是对应的中文描述字符串。确保 JSON 格式有效。
例如:
{{
  "column_name_1": "该列的中文描述。",
  "column_name_2": "另一列的中文描述。"
}}
只返回 JSON 对象，不要包含任何其他解释性文字或代码块标记。"""
        else:
            # 英文提示词
            prompt = f"""Please generate a concise and accurate English description for each column in the database table '{safe_table_name}'.
Please infer the purpose or meaning of each column based on its name, data type, and sample data.

Table structure and data samples:
{json.dumps(prompt_column_data, ensure_ascii=False, indent=2)}

Please return results strictly in the following JSON object format, where keys are column names (column_name) and values are corresponding English description strings. Ensure the JSON format is valid.
Example:
{{
  "column_name_1": "Description of this column in English.",
  "column_name_2": "Description of another column in English."
}}
Return only the JSON object, do not include any other explanatory text or code block markers."""

        # 5. Call LLM via model_manager.provider.chat, requesting JSON output
        response_content = None # Initialize for logging in case of error
        try:
            messages = [{'role': 'user', 'content': prompt}]
            # REMOVED format parameter for Ollama compatibility based on error
            # json_format_hint = {"type": "json_object"} if db_config.get('type') == 'openai' else {"type": "json"}

            app.logger.info(f"Sending request to LLM for table {safe_table_name} description generation.")
            # Detailed model info logging
            provider_type = model_manager.config.get('type', 'unknown')
            model_name = getattr(model_manager.provider, 'model', 'unknown')
            app.logger.info(f"使用模型提供商: {provider_type}, 模型名称: {model_name}")
            
            # Call chat without the format hint
            response_content = model_manager.provider.chat(messages=messages)

            if not response_content:
                raise Exception("LLM did not return content.")

            # --- Enhanced JSON Extraction --- START
            json_string = None
            descriptions = None
            try:
                # Find the first opening curly brace
                start_index = response_content.find('{')
                if start_index == -1:
                    raise ValueError("Could not find the start of a JSON object ('{') in the response.")

                brace_level = 0
                end_index = -1
                # Iterate through the string to find the matching closing brace
                for i, char in enumerate(response_content[start_index:]):
                    if char == '{':
                        brace_level += 1
                    elif char == '}':
                        brace_level -= 1
                        if brace_level == 0:
                            end_index = start_index + i
                            break # Found the matching brace

                if end_index == -1:
                    # Fallback: Try cleaning markdown if matching brace wasn't found cleanly
                    # This might happen if the JSON is incomplete or malformed before potential markdown
                    cleaned_response = re.sub(r'^```json\\s*|\\s*```$', '', response_content.strip(), flags=re.MULTILINE)
                    if cleaned_response.startswith('{') and cleaned_response.endswith('}'):
                         json_string = cleaned_response
                         app.logger.warning(f"JSON object boundaries possibly unclear, using regex cleaned response for table {safe_table_name}")
                    else:
                         raise ValueError("Could not find the matching end brace ('}') for the JSON object.")
                else:
                    # Extract the substring containing the JSON object
                    json_string = response_content[start_index : end_index + 1]

                # Attempt to parse the extracted string
                descriptions = json.loads(json_string)

            except json.JSONDecodeError as jde:
                # Re-raise with more context
                error_message = f"Failed to parse JSON: {jde}. "
                if json_string is not None:
                    error_message += f"Attempted to parse: {json_string[:200]}..." # Log beginning of attempted parse
                else: # This case should be less likely with the new logic but keep for robustness
                    error_message += f"Could not extract valid JSON boundaries. Original response: {response_content[:200]}..."
                # Log the full problematic string if available for easier debugging
                app.logger.error(f"Full string attempted for JSON parsing: {json_string}")
                raise json.JSONDecodeError(error_message, jde.doc, jde.pos) # Keep original doc and pos if possible
            # --- Enhanced JSON Extraction --- END

            if not isinstance(descriptions, dict):
                 raise ValueError(f"LLM response parsed, but it's not a dictionary. Type: {type(descriptions)}")

            app.logger.info(f"Generated descriptions for table {safe_table_name}. Count: {len(descriptions)}")
            return jsonify({"status": "success", "descriptions": descriptions})

        except json.JSONDecodeError as jde:
             # This will now catch the re-raised error with more context
             app.logger.error(f"解析LLM的JSON响应失败 ({safe_table_name}): {jde}")
             # Log the original full content only if parsing completely failed at the boundary finding stage
             if json_string is None:
                 app.logger.error(f"原始响应内容: {response_content}")
             return jsonify({"status": "error", "message": f"AI未能返回有效的JSON格式描述: {jde}"}), 500
        except Exception as e:
            app.logger.exception(f"调用LLM批量生成描述失败 ({safe_table_name}): {str(e)}")
            return jsonify({"status": "error", "message": f"调用AI批量生成描述失败: {str(e)}"}), 500

    except Exception as e:
        app.logger.exception(f"批量生成列描述时发生内部错误: {str(e)}")
        return jsonify({"status": "error", "message": f"批量生成列描述时发生内部错误: {str(e)}"}), 500
    