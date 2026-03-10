# OpenRouter Provider Routing 集成指南

本文档详细说明了如何在项目中使用 OpenRouter 的 Provider Routing 功能来优化模型请求路由。

## 概述

OpenRouter Provider Routing 允许你控制如何将请求路由到不同的模型提供商，以实现更好的性能、价格优化和可靠性。

## 配置说明

### 1. 基本配置

在 `config.json` 文件的 `model` 部分添加 `provider` 配置：

```json
{
  "model": {
    "type": "openai",
    "api_key": "your-openrouter-api-key",
    "model_name": "gpt-4",
    "api_base": "https://openrouter.ai/api/v1",
    "provider": {
      "order": ["anthropic", "openai"],
      "allow_fallbacks": true,
      "sort": "price"
    }
  }
}
```

### 2. 完整配置选项

```json
{
  "provider": {
    "order": ["anthropic", "openai", "google"],
    "allow_fallbacks": true,
    "require_parameters": false,
    "data_collection": "allow",
    "only": [],
    "ignore": ["provider_to_skip"],
    "quantizations": ["int4", "int8"],
    "sort": "price",
    "max_price": {
      "prompt": 0.01,
      "completion": 0.03,
      "image": 0.005,
      "request": 0.05
    }
  }
}
```

## 配置字段详解

### order
- **类型**: `string[]`
- **描述**: 按优先级顺序列出的提供商列表
- **示例**: `["anthropic", "openai", "google"]`
- **用途**: 指定优先使用的提供商顺序

### allow_fallbacks
- **类型**: `boolean`
- **默认值**: `true`
- **描述**: 当主要提供商不可用时是否允许使用备份提供商
- **建议**: 设为 `true` 以提高可靠性

### require_parameters
- **类型**: `boolean`
- **默认值**: `false`
- **描述**: 仅使用支持请求中所有参数的提供商
- **用途**: 确保特定功能的兼容性

### data_collection
- **类型**: `"allow" | "deny"`
- **默认值**: `"allow"`
- **描述**: 控制是否使用可能存储数据的提供商
- **隐私**: 设为 `"deny"` 可增强隐私保护

### only
- **类型**: `string[]`
- **描述**: 仅允许使用的提供商列表
- **示例**: `["anthropic"]` - 仅使用 Anthropic
- **注意**: 与 `order` 互斥使用

### ignore
- **类型**: `string[]`
- **描述**: 要跳过的提供商列表
- **示例**: `["provider_with_issues"]`
- **用途**: 临时排除有问题的提供商

### quantizations
- **类型**: `string[]`
- **描述**: 按量化级别过滤提供商
- **示例**: `["int4", "int8"]`
- **用途**: 优化性能或精度权衡

### sort
- **类型**: `"price" | "throughput" | "latency"`
- **描述**: 提供商排序策略
- **选项**: 
  - `"price"`: 按价格排序（最便宜优先）
  - `"throughput"`: 按吞吐量排序（最快优先）
  - `"latency"`: 按延迟排序（最低延迟优先）

### max_price
- **类型**: `object`
- **描述**: 设置最大价格限制
- **字段**:
  - `prompt`: 每百万 token 提示价格上限
  - `completion`: 每百万 token 完成价格上限
  - `image`: 每张图像价格上限
  - `request`: 每次请求价格上限

## 使用场景和最佳实践

### 1. 成本优化
```json
{
  "provider": {
    "sort": "price",
    "max_price": {
      "prompt": 0.01,
      "completion": 0.03
    }
  }
}
```

### 2. 性能优化
```json
{
  "provider": {
    "sort": "throughput",
    "order": ["fastest_provider", "backup_provider"]
  }
}
```

### 3. 可靠性优化
```json
{
  "provider": {
    "order": ["primary", "secondary", "tertiary"],
    "allow_fallbacks": true
  }
}
```

### 4. 隐私保护
```json
{
  "provider": {
    "data_collection": "deny",
    "only": ["privacy_focused_provider"]
  }
}
```

### 5. 特定功能要求
```json
{
  "provider": {
    "require_parameters": true,
    "only": ["providers_with_function_calling"]
  }
}
```

## 快捷模式

### Nitro 模式 (高吞吐量)
在模型名称后添加 `:nitro`：
```json
{
  "model_name": "gpt-4:nitro"
}
```

### Floor 模式 (最低价格)
在模型名称后添加 `:floor`：
```json
{
  "model_name": "gpt-4:floor"
}
```

## 注意事项

### 1. 模型兼容性
- Gemini 模型不支持 provider 参数
- Qwen 模型支持 provider 参数但可能有限制
- 其他 OpenAI 兼容模型通常完全支持

### 2. 配置验证
- 空值配置项会被自动过滤
- 无效的提供商名称会被忽略
- 配置错误会在日志中显示警告

### 3. 性能影响
- 过多的 fallback 可能增加延迟
- `require_parameters` 可能限制可用提供商
- 价格限制可能导致请求失败

## 故障排除

### 常见问题

1. **Provider 配置不生效**
   - 检查 API base URL 是否指向 OpenRouter
   - 确认使用的是 OpenRouter API key

2. **特定提供商不可用**
   - 检查提供商名称拼写
   - 验证提供商是否支持请求的模型

3. **价格限制导致失败**
   - 调整 `max_price` 设置
   - 检查当前提供商定价

### 调试信息

启用详细日志以查看 provider 配置的使用情况：
```python
import logging
logging.getLogger('enhanced_vanna_models').setLevel(logging.INFO)
```

## 示例配置

### 生产环境配置
```json
{
  "provider": {
    "order": ["anthropic", "openai"],
    "allow_fallbacks": true,
    "sort": "price",
    "max_price": {
      "prompt": 0.02,
      "completion": 0.06
    },
    "data_collection": "deny"
  }
}
```

### 开发环境配置
```json
{
  "provider": {
    "sort": "throughput",
    "allow_fallbacks": true,
    "ignore": ["expensive_provider"]
  }
}
```

### 高可靠性配置
```json
{
  "provider": {
    "order": ["primary", "backup1", "backup2"],
    "allow_fallbacks": true,
    "require_parameters": false
  }
}
``` 