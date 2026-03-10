-- 创建数据库（如果不存在）
CREATE DATABASE IF NOT EXISTS evaluation_test_db CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

-- 使用数据库
USE evaluation_test_db;

-- 创建用户表（示例）
CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(255) NOT NULL UNIQUE,
    email VARCHAR(255) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

-- 创建会话表
CREATE TABLE IF NOT EXISTS sessions (
    id VARCHAR(255) PRIMARY KEY,
    user_id INT,
    session_name VARCHAR(255),
    database_name VARCHAR(255),
    table_name VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

-- 创建消息表
CREATE TABLE IF NOT EXISTS messages (
    id INT AUTO_INCREMENT PRIMARY KEY,
    session_id VARCHAR(255),
    message_type ENUM('user', 'assistant') NOT NULL,
    content TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

-- 创建知识图谱会话表
CREATE TABLE IF NOT EXISTS kg_sessions (
    id VARCHAR(255) PRIMARY KEY,
    user_id INT,
    session_name VARCHAR(255),
    graph_id VARCHAR(255),
    include_types JSON,
    exclude_types JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

-- 创建知识图谱消息表
CREATE TABLE IF NOT EXISTS kg_messages (
    id INT AUTO_INCREMENT PRIMARY KEY,
    session_id VARCHAR(255),
    message_type ENUM('user', 'assistant') NOT NULL,
    content TEXT,
    cypher_query TEXT,
    execution_time FLOAT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES kg_sessions(id) ON DELETE CASCADE
);

-- 创建索引
CREATE INDEX idx_sessions_user_id ON sessions(user_id);
CREATE INDEX idx_sessions_created_at ON sessions(created_at);
CREATE INDEX idx_messages_session_id ON messages(session_id);
CREATE INDEX idx_messages_timestamp ON messages(timestamp);
CREATE INDEX idx_kg_sessions_user_id ON kg_sessions(user_id);
CREATE INDEX idx_kg_messages_session_id ON kg_messages(session_id);

-- 插入默认用户（可选）
INSERT IGNORE INTO users (username, email, password_hash) 
VALUES ('admin', 'admin@wisdomindata.com', 'default_hash');

-- 设置字符集
ALTER DATABASE evaluation_test_db CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci; 