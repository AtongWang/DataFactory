#!/bin/bash

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
}

# 等待MySQL服务
wait_for_mysql() {
    log "等待MySQL服务启动..."
    local host=${DATABASE_HOST:-mysql}
    local port=${DATABASE_PORT:-3306}
    
    for i in {1..30}; do
        if nc -z "$host" "$port"; then
            log "MySQL服务已就绪"
            return 0
        fi
        warn "MySQL服务未就绪，等待中... ($i/30)"
        sleep 2
    done
    
    error "MySQL服务启动超时"
    return 1
}

# 等待Neo4j服务
wait_for_neo4j() {
    log "等待Neo4j服务启动..."
    local uri=${NEO4J_URI:-bolt://neo4j:7687}
    local user=${NEO4J_USER:-neo4j}
    local password=${NEO4J_PASSWORD:-12345678}
    
    # 提取主机和端口
    local host=$(echo $uri | sed 's|bolt://||' | cut -d':' -f1)
    local port=$(echo $uri | sed 's|bolt://||' | cut -d':' -f2)
    
    for i in {1..30}; do
        if nc -z "$host" "$port"; then
            log "Neo4j服务已就绪"
            return 0
        fi
        warn "Neo4j服务未就绪，等待中... ($i/30)"
        sleep 2
    done
    
    error "Neo4j服务启动超时"
    return 1
}





# 处理配置文件环境变量
process_config() {
    log "处理配置文件..."
    
    # 检查config.json是否已存在
    if [ -f "/app/config.json" ]; then
        log "发现已存在的config.json文件，跳过配置处理以保持用户设置"
        return 0
    fi
    
    if [ -f "/app/config.docker.json" ]; then
        log "首次启动，从Docker模板生成配置文件"
        # 使用Python脚本进行可靠的环境变量处理
        python3 /usr/local/bin/process_config.py
        
        if [ $? -eq 0 ]; then
            log "配置文件处理成功"
        else
            error "配置文件处理失败"
            return 1
        fi
    else
        warn "Docker配置文件不存在，使用默认配置"
    fi
}

# 创建必要目录
create_directories() {
    log "创建必要目录..."
    mkdir -p /app/logs /app/chroma_db /app/data /app/exports
    log "目录创建完成"
}

# 健康检查端点
setup_health_check() {
    log "设置健康检查..."
    
    # 创建简单的健康检查脚本
    cat > /tmp/health_check.py << 'EOF'
import sys
import requests
import json

def health_check():
    try:
        # 检查应用是否响应
        response = requests.get('http://localhost:5000/', timeout=5)
        if response.status_code == 200:
            return True
    except:
        pass
    return False

if __name__ == "__main__":
    if health_check():
        sys.exit(0)
    else:
        sys.exit(1)
EOF
    
    log "健康检查脚本已创建"
}

# 主函数
main() {
    log "开始等待服务依赖..."
    
    create_directories
    process_config
    setup_health_check
    
    # 等待所有依赖服务
    wait_for_mysql
    wait_for_neo4j
    
    log "所有服务已就绪，启动应用..."
    
    # 调试：检查虚拟环境状态
    log "检查虚拟环境状态..."
    if [ -d "/app/.venv" ]; then
        log "虚拟环境目录存在"
        log "虚拟环境内容: $(ls -la /app/.venv/bin/ 2>/dev/null | wc -l) 个文件"
        
        # 检查多个可能的Python可执行文件位置
        PYTHON_PATHS=(
            "/app/.venv/bin/python"
            "/app/.venv/bin/python3"
            "/app/.venv/bin/python3.11"
        )
        
        PYTHON_FOUND=false
        for python_path in "${PYTHON_PATHS[@]}"; do
            if [ -f "$python_path" ] && [ -x "$python_path" ]; then
                log "发现Python可执行文件: $python_path"
                PYTHON_EXEC="$python_path"
                PYTHON_FOUND=true
                break
            fi
        done
        
        if [ "$PYTHON_FOUND" = true ]; then
            log "Python可执行文件存在: $PYTHON_EXEC"
        else
            warn "未找到Python可执行文件"
            log "虚拟环境bin目录内容:"
            ls -la /app/.venv/bin/ 2>/dev/null || log "无法列出bin目录内容"
        fi
    else
        warn "虚拟环境目录不存在"
    fi
    
    # 强制使用预构建虚拟环境（如果存在且有效）
    if [ -d "/app/.venv" ] && [ "$PYTHON_FOUND" = true ]; then
        log "使用预构建虚拟环境启动应用"
        # 激活虚拟环境并启动
        export PATH="/app/.venv/bin:$PATH"
        export VIRTUAL_ENV="/app/.venv"
        # 验证Python工作
        if "$PYTHON_EXEC" --version > /dev/null 2>&1; then
            log "虚拟环境Python验证成功: $("$PYTHON_EXEC" --version)"
            exec "$PYTHON_EXEC" /app/app.py
        else
            warn "虚拟环境Python验证失败，回退到UV运行"
        fi
    fi
    
    # 回退方案：使用UV运行
    log "使用UV运行应用"
    # 不要删除虚拟环境，让UV尝试使用现有的
    if [ -d "/app/.venv" ]; then
        log "虚拟环境存在但Python可执行文件有问题，让UV尝试修复"
    fi
    # 使用正确的UV命令语法
    exec uv run python /app/app.py
}

# 信号处理
trap 'error "收到终止信号，正在关闭..."; exit 1' SIGTERM SIGINT

# 运行主函数
main "$@" 