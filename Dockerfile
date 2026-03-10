# 使用官方Python 3.11作为基础镜像
FROM python:3.11-slim AS base

# 设置环境变量
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # 让 uv 在 venv 中复制解释器和二进制，避免软链接失效
    UV_LINK_MODE=copy \
    # 固定使用容器内的 Python 解释器创建 venv
    UV_PYTHON=/usr/local/bin/python3.11

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    netcat-traditional \
    default-mysql-client \
    libpq-dev \
    gettext-base \
    jq \
    && rm -rf /var/lib/apt/lists/*

# 安装UV
RUN pip install uv

# 设置工作目录
WORKDIR /app

# 仅复制依赖声明文件，以最大化缓存命中
COPY pyproject.toml uv.lock ./

# 使用 uv 基于锁文件同步依赖（确定性构建，且只安装 prod 依赖）
RUN uv sync --frozen --no-dev --python=/usr/local/bin/python3.11

# 验证虚拟环境创建成功（必须存在 python 可执行文件，否则构建失败）
RUN test -x /app/.venv/bin/python && /app/.venv/bin/python --version

# 将虚拟环境添加到PATH（运行期 python 指向 venv）
ENV PATH="/app/.venv/bin:$PATH" \
    VIRTUAL_ENV="/app/.venv"

# 复制应用代码（由 .dockerignore 控制上下文体积）
COPY . .

# 复制Docker配置和启动脚本
COPY docker/scripts/wait-for-services.sh /usr/local/bin/wait-for-services.sh
COPY docker/scripts/process_config.py /usr/local/bin/process_config.py
RUN chmod +x /usr/local/bin/wait-for-services.sh /usr/local/bin/process_config.py

# 创建必要的目录
RUN mkdir -p /app/logs /app/chroma_db /app/data /app/exports

# 安装Node.js和npm（用于前端资源）
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - && \
    apt-get install -y nodejs

# 安装前端依赖并构建
RUN npm install && npm run build

# 暴露端口
EXPOSE 5000

# 健康检查（使用 venv 中的 python）
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD /app/.venv/bin/python /tmp/health_check.py || exit 1

# 使用启动脚本
CMD ["/usr/local/bin/wait-for-services.sh"]