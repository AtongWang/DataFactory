# WisdominDATA Docker 部署指南

## 概述

本指南介绍如何使用 Docker Compose 部署 WisdominDATA 智能数据分析平台。

## 系统要求

- Docker Engine 20.10+
- Docker Compose 2.0+
- 8GB+ RAM
- 50GB+ 存储空间
- NVIDIA GPU（可选，用于 Ollama 加速）

## 快速开始

### 1. 克隆项目并切换到项目目录

```bash
cd /path/to/WisdominDATA
```

### 2. 配置环境变量（可选）

复制环境变量模板：
```bash
cp docker/env.template .env
```

编辑 `.env` 文件，根据需要修改配置。

### 3. 构建并启动服务

```bash
# 一键启动所有服务
docker-compose up -d

# 查看服务状态
docker-compose ps

# 查看日志
docker-compose logs -f
```

### 4. 访问服务

- **主应用**: http://localhost:5000
- **Neo4j 浏览器**: http://localhost:7474
- **MySQL**: localhost:3306
- **Ollama API**: http://localhost:11434

## 服务架构

### 服务组件

1. **app**: 主应用服务（Flask + UV环境）
2. **mysql**: MySQL 8.0 数据库
3. **neo4j**: Neo4j 5.15 图数据库（带 APOC 插件）
4. **ollama**: Ollama AI 模型服务

### 网络配置

所有服务运行在 `wisdomindata-network` 内部网络中，服务间通过容器名通信。

### 数据持久化

- **mysql-data**: MySQL 数据
- **neo4j-data**: Neo4j 数据
- **ollama-data**: Ollama 模型数据
- **chroma-data**: ChromaDB 向量数据
- **app-logs**: 应用日志

## 配置说明

### 环境变量

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| DATABASE_HOST | mysql | MySQL 主机 |
| DATABASE_PORT | 3306 | MySQL 端口 |
| NEO4J_URI | bolt://neo4j:7687 | Neo4j 连接地址 |
| OLLAMA_URL | http://ollama:11434 | Ollama 服务地址 |
| OLLAMA_EMBEDDING_MODEL | bge-m3:latest | 嵌入模型 |
| OLLAMA_CHAT_MODEL | gemma3:27b-it-q8_0 | 对话模型 |

### 模型管理

首次启动时，系统会自动检查并拉取必要的 Ollama 模型：
- `bge-m3:latest` - 嵌入模型
- `gemma3:27b-it-q8_0` - 主对话模型
- `qwen3:1.7b` - 会话命名模型

## 常用命令

### 服务管理

```bash
# 启动服务
docker-compose up -d

# 停止服务
docker-compose down

# 重启服务
docker-compose restart

# 查看服务状态
docker-compose ps

# 查看服务日志
docker-compose logs -f [service_name]
```

### 数据管理

```bash
# 备份 MySQL 数据
docker exec wisdomindata-mysql mysqldump -uroot -p123456 evaluation_test_db > backup.sql

# 恢复 MySQL 数据
docker exec -i wisdomindata-mysql mysql -uroot -p123456 evaluation_test_db < backup.sql

# 查看数据卷
docker volume ls | grep wisdomindata
```

### 调试

```bash
# 进入应用容器
docker exec -it wisdomindata-app bash

# 进入 MySQL 容器
docker exec -it wisdomindata-mysql mysql -uroot -p123456

# 进入 Neo4j 容器
docker exec -it wisdomindata-neo4j bash

# 查看 Ollama 模型
docker exec wisdomindata-ollama ollama list
```

## 故障排除

### 常见问题

1. **服务启动失败**
   ```bash
   # 检查日志
   docker-compose logs [service_name]
   
   # 检查端口占用
   netstat -tlnp | grep [port]
   ```

2. **模型加载失败**
   ```bash
   # 手动拉取模型
   docker exec wisdomindata-ollama ollama pull bge-m3:latest
   ```

3. **数据库连接失败**
   ```bash
   # 检查 MySQL 状态
   docker exec wisdomindata-mysql mysqladmin ping -uroot -p123456
   ```

4. **内存不足**
   ```bash
   # 查看资源使用
   docker stats
   
   # 清理无用镜像
   docker system prune
   ```

### 性能优化

1. **为 Ollama 配置 GPU**
   ```yaml
   # 在 docker-compose.yml 中确保 GPU 配置正确
   deploy:
     resources:
       reservations:
         devices:
           - driver: nvidia
             count: all
             capabilities: [gpu]
   ```

2. **调整内存限制**
   ```yaml
   # 为服务添加内存限制
   deploy:
     resources:
       limits:
         memory: 4G
   ```

## 离线部署

本系统设计为离线友好：

1. **构建包含所有依赖的镜像**
2. **预下载所需模型**
3. **配置内部网络通信**
4. **无需外部网络依赖**

## 生产环境部署

### 安全配置

1. **修改默认密码**
2. **配置防火墙规则**
3. **启用 HTTPS**
4. **定期备份数据**

### 监控配置

```bash
# 添加监控服务到 docker-compose.yml
# 配置日志聚合
# 设置健康检查告警
```

## 更新升级

```bash
# 拉取最新代码
git pull

# 重新构建镜像
docker-compose build --no-cache

# 重启服务
docker-compose down && docker-compose up -d
``` 