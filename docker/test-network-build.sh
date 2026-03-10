#!/bin/bash

# Neo4j Docker 构建网络测试脚本
# 用于诊断和解决网络连接问题

set -e

echo "🌐 Neo4j Docker 构建网络诊断工具"
echo "=================================="

# 配置变量
GDS_VERSION="2.16.0"
GDS_URL="https://github.com/neo4j/graph-data-science/releases/download/${GDS_VERSION}/neo4j-graph-data-science-${GDS_VERSION}.jar"
GDS_BACKUP_URL="https://dist.neo4j.org/gds/neo4j-graph-data-science-${GDS_VERSION}.jar"

# 网络连接测试
echo "📡 测试网络连接..."

# 测试基础网络连接
echo "1. 测试DNS解析..."
if nslookup github.com > /dev/null 2>&1; then
    echo "   ✅ DNS解析正常 (github.com)"
else
    echo "   ❌ DNS解析失败 (github.com)"
    echo "   建议: 检查DNS设置或使用公共DNS (8.8.8.8, 114.114.114.114)"
fi

# 测试GitHub连接
echo "2. 测试GitHub连接..."
if curl -s --connect-timeout 10 --max-time 30 -I https://github.com > /dev/null; then
    echo "   ✅ GitHub连接正常"
else
    echo "   ❌ GitHub连接失败"
    echo "   建议: 检查防火墙设置或网络代理配置"
fi

# 测试插件下载
echo "3. 测试插件下载链接..."
echo "   主要下载地址: ${GDS_URL}"

# 获取文件头信息
if curl -s --connect-timeout 10 --max-time 30 -I "${GDS_URL}" | grep -q "200 OK"; then
    echo "   ✅ 主要下载链接可访问"
    MAIN_URL_OK=true
else
    echo "   ❌ 主要下载链接不可访问"
    MAIN_URL_OK=false
fi

echo "   备用下载地址: ${GDS_BACKUP_URL}"
if curl -s --connect-timeout 10 --max-time 30 -I "${GDS_BACKUP_URL}" | grep -q "200 OK"; then
    echo "   ✅ 备用下载链接可访问"
    BACKUP_URL_OK=true
else
    echo "   ❌ 备用下载链接不可访问"
    BACKUP_URL_OK=false
fi

# Docker 环境检查
echo ""
echo "🐳 Docker 环境检查..."

if command -v docker > /dev/null 2>&1; then
    echo "   ✅ Docker 已安装: $(docker --version)"
else
    echo "   ❌ Docker 未安装"
    exit 1
fi

if docker info > /dev/null 2>&1; then
    echo "   ✅ Docker 服务运行正常"
else
    echo "   ❌ Docker 服务未运行"
    exit 1
fi

# 网络优化建议
echo ""
echo "🔧 网络优化建议..."

if [ "$MAIN_URL_OK" = false ] && [ "$BACKUP_URL_OK" = false ]; then
    echo "❌ 所有下载链接都不可访问，建议:"
    echo "   1. 检查网络代理设置"
    echo "   2. 使用离线安装方案"
    echo "   3. 配置Docker构建代理"
    echo ""
    echo "Docker 代理配置示例:"
    echo "   在 ~/.docker/config.json 中添加:"
    echo '   {'
    echo '     "proxies": {'
    echo '       "default": {'
    echo '         "httpProxy": "http://your-proxy:port",'
    echo '         "httpsProxy": "https://your-proxy:port"'
    echo '       }'
    echo '     }'
    echo '   }'
    
    # 提供离线解决方案
    echo ""
    echo "💡 推荐使用离线安装方案:"
    echo "   ./docker/neo4j-offline-setup.sh"
    
    exit 1
elif [ "$MAIN_URL_OK" = false ]; then
    echo "⚠️ 主要下载链接不可访问，将使用备用链接"
fi

# 测试 Docker 构建（可选）
echo ""
echo "🔨 Docker 构建测试..."
read -p "是否测试 Docker 构建? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "开始构建测试..."
    
    # 设置构建参数以优化网络性能
    export DOCKER_BUILDKIT=1
    export BUILDKIT_PROGRESS=plain
    
    # 使用优化的构建命令
    if docker compose build neo4j --progress=plain --no-cache; then
        echo "✅ Docker 构建测试成功!"
    else
        echo "❌ Docker 构建测试失败"
        echo ""
        echo "故障排除建议:"
        echo "1. 检查 Dockerfile 语法"
        echo "2. 增加网络超时时间"
        echo "3. 使用离线安装方案"
        echo "4. 配置Docker构建代理"
        
        exit 1
    fi
else
    echo "跳过构建测试"
fi

echo ""
echo "✅ 网络诊断完成!"
echo ""
echo "📋 总结建议:"
if [ "$MAIN_URL_OK" = true ] || [ "$BACKUP_URL_OK" = true ]; then
    echo "   🟢 网络连接正常，可以尝试重新构建"
    echo "   构建命令: docker compose build neo4j"
else
    echo "   🔴 网络连接存在问题，建议使用离线方案"
    echo "   离线安装: ./docker/neo4j-offline-setup.sh"
fi 