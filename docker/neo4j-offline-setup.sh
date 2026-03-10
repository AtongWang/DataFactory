#!/bin/bash

# Neo4j 离线插件安装脚本
# 当网络环境无法直接下载插件时使用

set -e

echo "🚀 Neo4j 离线插件安装脚本"
echo "=================================="

# 配置变量
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLUGINS_DIR="${SCRIPT_DIR}/plugins"
GDS_VERSION="2.16.0"
GDS_JAR="neo4j-graph-data-science-${GDS_VERSION}.jar"

# 创建插件目录
mkdir -p "${PLUGINS_DIR}"

echo "📁 插件目录: ${PLUGINS_DIR}"

# 检查是否已存在插件文件
if [ -f "${PLUGINS_DIR}/${GDS_JAR}" ]; then
    echo "✅ 发现已存在的 GDS 插件: ${GDS_JAR}"
    echo "📏 文件大小: $(ls -lh "${PLUGINS_DIR}/${GDS_JAR}" | awk '{print $5}')"
else
    echo "❌ 未发现 GDS 插件文件"
    echo ""
    echo "请手动下载以下文件到 ${PLUGINS_DIR} 目录:"
    echo "1. Graph Data Science 插件:"
    echo "   文件名: ${GDS_JAR}"
    echo "   下载地址: https://github.com/neo4j/graph-data-science/releases/download/${GDS_VERSION}/${GDS_JAR}"
    echo "   备用地址: https://dist.neo4j.org/gds/${GDS_JAR}"
    echo ""
    echo "下载完成后重新运行此脚本。"
    exit 1
fi

# 验证插件文件
echo "🔍 验证插件文件完整性..."

if file "${PLUGINS_DIR}/${GDS_JAR}" | grep -q "Java archive data"; then
    echo "✅ GDS 插件文件验证通过"
else
    echo "❌ GDS 插件文件损坏，请重新下载"
    exit 1
fi

# 创建离线版本的 Dockerfile
echo "📝 创建离线版本的 Dockerfile..."

cat > "${SCRIPT_DIR}/Dockerfile.offline" << 'EOF'
# 基于官方Neo4j镜像
FROM neo4j:2025.03.0

# 设置维护者信息
LABEL maintainer="WisdominDATA Team"
LABEL description="Neo4j with pre-installed Graph Data Science plugins (offline build)"

# 切换到root用户以安装插件
USER root

# 安装必要的工具
RUN apt-get update && apt-get install -y file && rm -rf /var/lib/apt/lists/*

# 创建插件目录
RUN mkdir -p /plugins

# 复制本地插件文件（离线安装）
COPY plugins/*.jar /plugins/

# 验证插件文件
RUN echo "验证插件文件..." && \
    ls -la /plugins/ && \
    if [ -f "/plugins/neo4j-graph-data-science-2.16.0.jar" ]; then \
        echo "检查GDS插件文件..." && \
        test -s /plugins/neo4j-graph-data-science-2.16.0.jar && \
        file /plugins/neo4j-graph-data-science-2.16.0.jar | grep -q "Java archive data" && \
        echo "✅ GDS插件文件完整"; \
    else \
        echo "⚠️ 未找到GDS插件，将在没有GDS的情况下运行"; \
    fi && \
    echo "🎉 插件验证完成"

# 设置插件目录权限
RUN chown -R neo4j:neo4j /plugins
RUN chmod 644 /plugins/*.jar 2>/dev/null || true

# 切换回neo4j用户
USER neo4j

# 设置默认环境变量，启用插件
# Neo4j 2025.03.0 内置 APOC Core，所以APOC配置应该可以直接使用
ENV NEO4J_apoc_export_file_enabled=true
ENV NEO4J_apoc_import_file_enabled=true  
ENV NEO4J_apoc_import_file_use__neo4j__config=true
ENV NEO4J_dbms_security_procedures_unrestricted='apoc.*,gds.*'
ENV NEO4J_dbms_security_procedures_allowlist='apoc.*,gds.*'

# 禁用自动插件下载
ENV NEO4J_ACCEPT_LICENSE_AGREEMENT=yes

# 暴露标准端口
EXPOSE 7474 7687

# 使用默认启动命令
CMD ["neo4j"]
EOF

echo "✅ 离线版本 Dockerfile 已创建: ${SCRIPT_DIR}/Dockerfile.offline"

# 构建离线镜像
echo ""
echo "🔨 开始构建离线镜像..."
echo "构建命令: docker build -f Dockerfile.offline -t wisdomindata-neo4j:offline ."

cd "${SCRIPT_DIR}"
docker build -f Dockerfile.offline -t wisdomindata-neo4j:offline .

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 离线镜像构建成功!"
    echo "镜像名称: wisdomindata-neo4j:offline"
    echo ""
    echo "使用方法:"
    echo "1. 修改 docker-compose.yml 中的镜像名称:"
    echo "   image: wisdomindata-neo4j:offline"
    echo "   # 注释掉 build 部分"
    echo ""
    echo "2. 或直接运行:"
    echo "   docker run -p 7474:7474 -p 7687:7687 wisdomindata-neo4j:offline"
else
    echo "❌ 镜像构建失败"
    exit 1
fi 
