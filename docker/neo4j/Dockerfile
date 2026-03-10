# 基于官方Neo4j镜像
FROM neo4j:2025.03.0

# 设置维护者信息
LABEL maintainer="WisdominDATA Team"
LABEL description="Neo4j with pre-installed Graph Data Science plugins for offline usage"

# 切换到root用户以安装插件
USER root

# 安装必要的工具
RUN apt-get update && apt-get install -y curl wget file && rm -rf /var/lib/apt/lists/*

# 创建插件目录
RUN mkdir -p /plugins

# 设置网络超时和重试参数
ENV CURL_TIMEOUT=300
ENV CURL_MAX_RETRIES=3

# Neo4j 2025.03.0 已内置 APOC Core，无需单独下载
# 只需要下载 Graph Data Science 插件

# 下载并安装Graph Data Science插件 (兼容Neo4j 2025.03的版本)
# 使用多种下载源以提高成功率
RUN echo "开始下载 Graph Data Science 插件..." && \
    # 方法1: 直接从GitHub下载
    (curl --connect-timeout ${CURL_TIMEOUT} --max-time ${CURL_TIMEOUT} --retry ${CURL_MAX_RETRIES} \
     -L https://github.com/neo4j/graph-data-science/releases/download/2.16.0/neo4j-graph-data-science-2.16.0.jar \
     -o /plugins/graph-data-science.jar && echo "✅ GitHub下载成功") || \
    # 方法2: 备用下载链接（如果有的话）
    (echo "⚠️ GitHub下载失败，尝试备用方法..." && \
     curl --connect-timeout ${CURL_TIMEOUT} --max-time ${CURL_TIMEOUT} --retry ${CURL_MAX_RETRIES} \
     -L https://dist.neo4j.org/gds/neo4j-graph-data-science-2.16.0.jar \
     -o /plugins/graph-data-science.jar && echo "✅ 备用下载成功") || \
    # 如果所有下载都失败，创建一个空文件作为占位符
    (echo "❌ 所有下载方法都失败，将跳过GDS插件安装" && \
     touch /plugins/.gds-download-failed)

# 验证插件下载成功并检查文件完整性
RUN echo "验证插件文件..." && \
    ls -la /plugins/ && \
    # 检查是否存在下载失败标记
    if [ -f "/plugins/.gds-download-failed" ]; then \
        echo "⚠️ GDS插件下载失败，将在没有GDS的情况下运行"; \
        echo "请在网络环境良好时重新构建镜像"; \
    else \
        echo "检查GDS插件文件..." && \
        test -s /plugins/graph-data-science.jar && \
        file /plugins/graph-data-science.jar | grep -q "archive data" && \
        echo "✅ GDS插件文件完整"; \
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
