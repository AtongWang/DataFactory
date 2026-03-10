# Neo4j 插件兼容性修复指南

## 问题描述

在Docker构建过程中遇到网络连接超时问题，错误信息：
```
curl: (28) Failed to connect to github.com port 443: Connection timed out
ERROR: process "/bin/sh -c curl ..." did not complete successfully: exit code: 28
```

**根本原因：** 
1. 网络连接超时导致插件下载失败
2. Neo4j 2025.03.0 版本兼容性问题
3. Docker构建配置需要优化

## 修复方案

已更新 `docker/neo4j/Dockerfile`，解决网络和兼容性问题：
- **APOC插件：** 使用Neo4j 2025.03.0内置版本（无需下载）
- **GDS插件：** 2.16.0 （兼容Neo4j 2025.03）
- **网络优化：** 增加超时处理、重试机制、备用下载源
- **构建优化：** 配置Docker BuildKit和网络参数

## 重新构建步骤

### 方案A: 网络诊断和优化构建

```bash
# 1. 运行网络诊断工具
./docker/test-network-build.sh

# 2. 如果网络正常，使用优化构建
export DOCKER_BUILDKIT=1
sudo docker-compose build neo4j --progress=plain

# 3. 验证构建成功
sudo docker images | grep wisdomindata

# 4. 测试启动（可选）
sudo docker-compose up neo4j
```

### 方案B: 离线安装（推荐用于网络受限环境）

```bash
# 1. 运行离线安装脚本
./docker/neo4j-offline-setup.sh

# 2. 按提示手动下载插件文件到 docker/neo4j/plugins/ 目录

# 3. 重新运行脚本完成构建
./docker/neo4j-offline-setup.sh
```

### 2. 导出修复后的镜像

```bash
# 1. 获取镜像ID
IMAGE_ID=$(sudo docker images | grep wisdomindata | grep neo4j | awk '{print $3}')

# 2. 导出镜像
sudo docker save -o wisdomindata-neo4j-fixed.tar wisdomindata-neo4j:latest

# 3. 压缩镜像文件（可选，减少传输大小）
gzip wisdomindata-neo4j-fixed.tar
```

### 3. 在离线环境中部署

```bash
# 1. 传输镜像文件到离线机器

# 2. 停止现有服务
sudo docker-compose down

# 3. 删除损坏的Neo4j镜像
sudo docker rmi wisdomindata-neo4j:latest

# 4. 加载修复后的镜像
sudo docker load -i wisdomindata-neo4j-fixed.tar.gz

# 5. 清理Neo4j数据卷（重要：清除损坏的插件数据）
sudo docker volume rm wisdomindata-neo4j-plugins
sudo docker volume rm wisdomindata-neo4j-data

# 6. 重新启动服务
sudo docker-compose up --no-build
```

## 验证修复

启动后检查以下内容：

### 1. 检查Neo4j日志
```bash
sudo docker logs wisdomindata-neo4j
```

应该看到类似信息：
```
INFO  ✅ APOC插件文件完整
INFO  ✅ GDS插件文件完整
INFO  🎉 所有插件下载并验证完成
```

### 2. 验证插件功能
连接到Neo4j浏览器 (http://localhost:7474) 并执行：
```cypher
// 验证APOC插件
CALL apoc.help("apoc")

// 验证GDS插件  
CALL gds.version()
```

### 3. 检查服务健康状态
```bash
# 检查所有服务状态
sudo docker-compose ps

# 检查Neo4j连接
sudo docker exec wisdomindata-neo4j cypher-shell -u neo4j -p 12345678 "CALL dbms.components()"
```

## 故障排除

### 如果仍然出现插件错误：

1. **完全清理环境**
```bash
sudo docker-compose down -v
sudo docker system prune -a
sudo docker volume prune
```

2. **检查插件文件**
```bash
sudo docker exec wisdomindata-neo4j ls -la /plugins/
sudo docker exec wisdomindata-neo4j file /plugins/*.jar
```

3. **手动验证插件**
```bash
sudo docker exec wisdomindata-neo4j java -jar /plugins/apoc.jar --version
```

### 如果需要禁用插件（临时方案）：

修改 `docker-compose.yml` 中的环境变量：
```yaml
environment:
  - NEO4J_dbms_security_procedures_unrestricted=""
  - NEO4J_dbms_security_procedures_allowlist=""
```

## 版本兼容性说明

| Neo4j版本 | APOC版本 | GDS版本 | 状态 |
|-----------|----------|---------|------|
| 2025.03.0 | 5.29.0   | 2.18.0  | ✅ 兼容 |
| 2025.03.0 | 5.28.1   | 2.16.0  | ❌ 不兼容 |

## 预防措施

1. **版本锁定**：在生产环境中使用具体版本号而非`latest`
2. **镜像验证**：构建后在测试环境验证完整性
3. **备份策略**：定期备份工作镜像和数据
4. **离线部署**：维护离线插件库

## 联系支持

如果问题持续存在，请提供：
- 完整的错误日志
- Docker版本信息
- 系统环境信息
- 镜像构建日志 