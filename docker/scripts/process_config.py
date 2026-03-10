#!/usr/bin/env python3
"""
配置文件环境变量处理脚本
使用Python进行更可靠的环境变量替换
"""

import json
import os
import sys
import re


def process_env_vars(value):
    """
    处理环境变量替换
    支持 ${VAR_NAME:-default_value} 语法
    """
    if not isinstance(value, str):
        return value
    
    # 正则匹配 ${VAR_NAME:-default_value} 模式
    pattern = r'\$\{([^}]+)\}'
    
    def replace_var(match):
        var_expr = match.group(1)
        
        # 处理默认值语法 VAR_NAME:-default_value
        if ':-' in var_expr:
            var_name, default_value = var_expr.split(':-', 1)
        else:
            var_name = var_expr
            default_value = ''
        
        # 获取环境变量值
        return os.getenv(var_name, default_value)
    
    return re.sub(pattern, replace_var, value)


def process_config_recursive(obj):
    """
    递归处理配置对象中的所有字符串值
    """
    if isinstance(obj, dict):
        return {key: process_config_recursive(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [process_config_recursive(item) for item in obj]
    elif isinstance(obj, str):
        return process_env_vars(obj)
    else:
        return obj


def main():
    """
    主函数：读取配置模板，处理环境变量，输出最终配置
    """
    input_file = '/app/config.docker.json'
    output_file = '/app/config.json'
    
    try:
        # 读取配置模板
        with open(input_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # 处理环境变量
        processed_config = process_config_recursive(config)
        
        # 写入最终配置
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(processed_config, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 配置文件处理完成: {output_file}")
        
        # 显示关键配置用于调试
        print("🔍 关键配置检查:")
        if 'database' in processed_config:
            db_config = processed_config['database']
            print(f"  数据库主机: {db_config.get('host', 'N/A')}")
            print(f"  数据库端口: {db_config.get('port', 'N/A')}")
            print(f"  数据库用户: {db_config.get('username', 'N/A')}")
        
        if 'neo4j' in processed_config:
            neo4j_config = processed_config['neo4j']
            print(f"  Neo4j URI: {neo4j_config.get('uri', 'N/A')}")
            print(f"  Neo4j 用户: {neo4j_config.get('user', 'N/A')}")
        
        return 0
        
    except FileNotFoundError:
        print(f"❌ 配置模板文件不存在: {input_file}")
        return 1
    except json.JSONDecodeError as e:
        print(f"❌ 配置文件JSON格式错误: {e}")
        return 1
    except Exception as e:
        print(f"❌ 处理配置文件时出错: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main()) 