from flask import Blueprint, render_template, request, jsonify
import logging
from backend.manager.knowledge_graph_manager import kg_manager
import os
import json
from backend.models.knowledge_graph_models import NodeType, RelationshipType
from neo4j.exceptions import ClientError
from neo4j.time import DateTime, Date, Time, Duration # 导入 Neo4j 时间类型
import datetime # 导入 Python 标准库 datetime
import traceback # 导入 traceback

kg_bp = Blueprint('kg', __name__)
logger = logging.getLogger(__name__)

@kg_bp.route('/knowledge-graph-construction')
def knowledge_graph_construction():
    return render_template('knowledge_graph_construction.html')

@kg_bp.route('/knowledge-graph-visualization')
def knowledge_graph_visualization():
    return render_template('knowledge_graph_visualization.html')


@kg_bp.route('/knowledge-graph-metrics')
def knowledge_graph_metrics():
    """渲染知识图谱指标页面"""
    return render_template('knowledge_graph_metrics.html')

@kg_bp.route('/api/kg/database-tables', methods=['GET'])
def get_database_tables():
    """获取数据库表列表"""
    try:
        # 使用vanna_manager的配置作为数据库连接参数
        # 从应用配置中获取数据库配置
        database_config = {
            'type': kg_manager.vanna_manager.config.database.type,
            'host': kg_manager.vanna_manager.config.database.host,
            'port': kg_manager.vanna_manager.config.database.port, 
            'username': kg_manager.vanna_manager.config.database.username,
            'password': kg_manager.vanna_manager.config.database.password,
            'database': kg_manager.vanna_manager.config.database.database_name
        }
        
        tables = kg_manager.get_database_tables(database_config)
        return jsonify({
            'status': 'success',
            'tables': tables
        })
    except Exception as e:
        logger.error(f"获取数据库表失败: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@kg_bp.route('/api/kg/database-config', methods=['GET'])
def get_database_config():
    """获取数据库配置信息（前端使用）"""
    try:
        return jsonify({
            'status': 'success',
            'database_name': kg_manager.vanna_manager.config.database.database_name,
            'database_type': kg_manager.vanna_manager.config.database.type,
            'host': kg_manager.vanna_manager.config.database.host,
            'port': kg_manager.vanna_manager.config.database.port
        })
    except Exception as e:
        logger.error(f"获取数据库配置失败: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@kg_bp.route('/api/kg/table-schema', methods=['GET'])
def get_table_schema():
    """获取表结构"""
    try:
        table_name = request.args.get('table')
        if not table_name:
            return jsonify({
                'status': 'error',
                'message': '缺少表名参数'
            }), 400
        
        # 使用相同的数据库配置
        database_config = {
            'type': kg_manager.vanna_manager.config.database.type,
            'host': kg_manager.vanna_manager.config.database.host,
            'port': kg_manager.vanna_manager.config.database.port, 
            'username': kg_manager.vanna_manager.config.database.username,
            'password': kg_manager.vanna_manager.config.database.password,
            'database': kg_manager.vanna_manager.config.database.database_name
        }
        
        schema = kg_manager.get_table_schema(table_name, database_config)
        return jsonify({
            'status': 'success',
            'schema': schema
        })
    except Exception as e:
        logger.error(f"获取表结构失败: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@kg_bp.route('/api/kg/graphs', methods=['GET'])
def get_knowledge_graphs():
    """获取所有知识图谱"""
    try:
        graphs = kg_manager.get_all_knowledge_graphs()
        return jsonify({
            'status': 'success',
            'graphs': graphs
        })
    except Exception as e:
        logger.error(f"获取知识图谱列表失败: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@kg_bp.route('/api/kg/graphs/<int:graph_id>', methods=['GET'])
def get_knowledge_graph(graph_id):
    """获取指定ID的知识图谱"""
    try:
        graph = kg_manager.get_knowledge_graph(graph_id)
        if not graph:
            return jsonify({
                'status': 'error',
                'message': f'知识图谱 {graph_id} 不存在'
            }), 404
        
        return jsonify({
            'status': 'success',
            'graph': graph
        })
    except Exception as e:
        logger.error(f"获取知识图谱失败: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@kg_bp.route('/api/kg/graphs', methods=['POST'])
def create_knowledge_graph():
    """创建新的知识图谱"""
    try:
        graph_data = request.json
        if not graph_data:
            return jsonify({
                'status': 'error',
                'message': '缺少图谱数据'
            }), 400
        
        # 验证必要字段
        required_fields = ['name', 'database', 'table', 'node_types', 'relationships']
        for field in required_fields:
            if field not in graph_data:
                return jsonify({
                    'status': 'error',
                    'message': f'缺少必要字段: {field}'
                }), 400
        
        # 创建知识图谱
        new_graph = kg_manager.create_knowledge_graph(graph_data)
        
        # 手动启动构建过程（恢复自动构建功能）
        if new_graph and 'id' in new_graph:
            build_result = kg_manager.start_graph_building(new_graph['id'])
            if build_result.get('status') == 'success':
                message = '知识图谱创建成功，构建过程已启动'
            else:
                message = f'知识图谱创建成功，但构建启动失败: {build_result.get("message", "未知错误")}'
        else:
            message = '知识图谱创建成功，但无法启动构建'
        
        return jsonify({
            'status': 'success',
            'message': message,
            'graph': new_graph
        })
    except Exception as e:
        logger.error(f"创建知识图谱失败: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@kg_bp.route('/api/kg/graphs/<int:graph_id>', methods=['DELETE'])
def delete_knowledge_graph(graph_id):
    """删除知识图谱"""
    try:
        success = kg_manager.delete_knowledge_graph(graph_id)
        if not success:
            return jsonify({
                'status': 'error',
                'message': f'知识图谱 {graph_id} 不存在或删除失败'
            }), 404
        
        return jsonify({
            'status': 'success',
            'message': f'知识图谱 {graph_id} 已删除'
        })
    except Exception as e:
        logger.error(f"删除知识图谱失败: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@kg_bp.route('/api/kg/graphs/<int:graph_id>/rebuild', methods=['POST'])
def rebuild_knowledge_graph(graph_id):
    """重新构建知识图谱"""
    try:
        result = kg_manager.start_graph_building(graph_id)
        
        # 适配新的返回值格式（字典而不是布尔值）
        if result.get('status') == 'success':
            return jsonify({
                'status': 'success',
                'message': f'知识图谱 {graph_id} 重建过程已启动'
            })
        elif result.get('status') == 'skipped':
            return jsonify({
                'status': 'success',
                'message': f'知识图谱 {graph_id} {result.get("message", "跳过重建")}'
            })
        else:
            return jsonify({
                'status': 'error',
                'message': f'知识图谱 {graph_id} 重建启动失败: {result.get("message", "未知错误")}'
            }), 404
        
    except Exception as e:
        logger.error(f"重建知识图谱失败: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

def _serialize_neo4j_value(value):
    """Helper function to serialize Neo4j specific types to JSON-compatible formats."""
    if isinstance(value, (DateTime, Date, Time)):
        return value.iso_format()
    elif isinstance(value, Duration):
        # Duration might need a specific string representation based on needs
        return str(value)
    elif isinstance(value, datetime.datetime): # Handle standard Python datetime too
        return value.isoformat()
    elif isinstance(value, (list, tuple)):
        return [_serialize_neo4j_value(item) for item in value]
    elif isinstance(value, dict):
        return {k: _serialize_neo4j_value(v) for k, v in value.items()}
    # Add other type checks if necessary (e.g., Point for spatial data)
    return value

@kg_bp.route('/api/kg/graphs/<int:graph_id>/visualization-data', methods=['GET'])
def get_graph_visualization_data(graph_id):
    """获取知识图谱可视化数据 (从 Neo4j)"""
    try:
        # 1. 获取图谱元数据 (来自 SQLite)
        graph_meta = kg_manager.get_knowledge_graph(graph_id)
        if not graph_meta:
            return jsonify({
                'status': 'error',
                'message': f'知识图谱元数据 {graph_id} 不存在'
            }), 404

        # 2. 检查 Neo4j Driver 是否可用
        if not kg_manager.neo4j_driver:
             return jsonify({
                 'status': 'error',
                 'message': 'Neo4j 服务连接失败，无法获取图谱数据'
             }), 500

        # 3. 从 Neo4j 查询节点和关系
        nodes_data = []
        relationships_data = []
        node_types_stats = {}
        relationship_types_stats = {}

        with kg_manager.neo4j_driver.session() as neo4j_session:
            # 查询节点
            # OPTIONAL MATCH (n)-[r]-() allows getting degree, might be slow on large graphs
            node_result = neo4j_session.run("""
                MATCH (n {graph_id: $graph_id})
                RETURN n, labels(n) as types
                LIMIT 10000
            """, graph_id=graph_id) # Add LIMIT to prevent overwhelming results initially

            for record in node_result:
                node = record['n']
                types = record['types']
                # Convert Neo4j node to dictionary suitable for frontend
                node_attributes = {k: _serialize_neo4j_value(v) for k, v in dict(node).items()} # Serialize attributes

                node_dict = {
                    'id': node.get('unique_id'), # Use the unique_id we stored
                    'type': types[0] if types else 'Unknown', # Assumes first label is primary type
                    'attributes': node_attributes, # Use serialized attributes
                    # Add more specific formatting if frontend expects it
                }
                # Clean up internal properties if needed
                node_dict['attributes'].pop('graph_id', None)
                node_dict['attributes'].pop('unique_id', None)

                nodes_data.append(node_dict)

                # Update stats
                node_type = node_dict['type']
                node_types_stats[node_type] = node_types_stats.get(node_type, 0) + 1

            # 查询关系 (限制数量)
            rel_result = neo4j_session.run("""
                MATCH (source {graph_id: $graph_id})-[r]->(target {graph_id: $graph_id})
                WHERE source.graph_id = $graph_id AND target.graph_id = $graph_id // Ensure both nodes belong to the graph
                RETURN source.unique_id as source_id,
                       target.unique_id as target_id,
                       type(r) as type,
                       properties(r) as properties
                LIMIT 20000
            """, graph_id=graph_id) # Limit relationships too

            for record in rel_result:
                 rel_type = record['type']
                 # Serialize relationship properties
                 properties = {k: _serialize_neo4j_value(v) for k, v in record['properties'].items()}
                 # Clean up internal graph_id from relationship properties
                 properties.pop('graph_id', None)

                 relationships_data.append({
                     'source': record['source_id'],
                     'target': record['target_id'],
                     'type': rel_type,
                     'properties': properties # Include serialized relationship properties
                 })
                 # Update stats
                 relationship_types_stats[rel_type] = relationship_types_stats.get(rel_type, 0) + 1

        # 4. 检查是否因为 LIMIT 而截断了数据
        warning_message = ""
        # A more robust check would involve counting nodes/rels first,
        # but that adds overhead. We can check if the returned count hits the limit.
        if len(nodes_data) >= 10000 or len(relationships_data) >= 20000:
             warning_message = "注意：返回的数据量可能因达到上限而被截断。请考虑在可视化或查询中添加更具体的过滤条件。"


        return jsonify({
            'status': 'success',
            'graph': graph_meta, # Keep meta-data from SQLite
            'nodes': nodes_data,
            'relationships': relationships_data,
            'statistics': {
                'node_types': node_types_stats,
                'relationship_types': relationship_types_stats,
                'node_count': len(nodes_data), # Count based on retrieved data
                'relationship_count': len(relationships_data) # Count based on retrieved data
            },
            'warning': warning_message if warning_message else None
        })

    except Exception as e:
        logger.error(f"获取可视化数据失败 (graph_id={graph_id}): {str(e)}")
        # Provide more specific error info if possible
        error_detail = str(e)
        if "Connection refused" in error_detail or "failed to establish connection" in error_detail:
             message = "无法连接到 Neo4j 数据库，请检查服务是否运行以及连接配置是否正确。"
        elif "not JSON serializable" in error_detail: # More specific error for serialization
             message = f"数据序列化失败，可能存在无法处理的数据类型: {error_detail}"
             logger.error(f"Serialization error details: {traceback.format_exc()}") # Log full traceback for serialization errors
        else:
             message = f"获取 Neo4j 图谱数据时出错: {error_detail}"

        return jsonify({
            'status': 'error',
            'message': message
        }), 500

@kg_bp.route('/api/kg/graphs/<int:graph_id>/node-types', methods=['GET'])
def get_graph_node_types(graph_id):
    """获取知识图谱的节点类型"""
    try:
        session = kg_manager.Session()
        node_types = session.query(NodeType).filter(NodeType.knowledge_graph_id == graph_id).all()
        
        result = []
        for nt in node_types:
            result.append({
                'id': nt.id,
                'name': nt.name,
                'identifier_columns': json.loads(nt.identifier_columns) if nt.identifier_columns else [],
                'attribute_columns': json.loads(nt.attribute_columns) if nt.attribute_columns else [],
                'split_config': json.loads(nt.split_config) if nt.split_config else {'enabled': False, 'delimiter': None}
            })
        
        return jsonify({
            'status': 'success',
            'node_types': result
        })
    except Exception as e:
        logger.error(f"获取节点类型失败: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500
    finally:
        session.close()

@kg_bp.route('/api/kg/graphs/<int:graph_id>/relationship-types', methods=['GET'])
def get_graph_relationship_types(graph_id):
    """获取知识图谱的关系类型"""
    try:
        session = kg_manager.Session()
        rel_types = session.query(RelationshipType).filter(RelationshipType.knowledge_graph_id == graph_id).all()
        
        result = []
        for rt in rel_types:
            # 获取关联的节点类型名称
            source_node = session.query(NodeType).filter(NodeType.id == rt.source_node_type_id).first()
            target_node = session.query(NodeType).filter(NodeType.id == rt.target_node_type_id).first()
            
            result.append({
                'id': rt.id,
                'type': rt.type,
                'source_node_type': source_node.name if source_node else '',
                'target_node_type': target_node.name if target_node else '',
                'direction': rt.direction,
                'matching_mode': rt.matching_mode
            })
        
        return jsonify({
            'status': 'success',
            'relationship_types': result
        })
    except Exception as e:
        logger.error(f"获取关系类型失败: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500
    finally:
        session.close()

@kg_bp.route('/api/kg/graphs/<int:graph_id>/metrics', methods=['GET'])
def get_graph_metrics(graph_id):
    """计算并返回指定知识图谱的指标 (不包括 LLM 分析)"""
    try:
        # 检查图谱元数据是否存在 (可选，但可以快速失败)
        graph_meta = kg_manager.get_knowledge_graph(graph_id)
        if not graph_meta:
            return jsonify({
                'status': 'error',
                'message': f'知识图谱元数据 {graph_id} 不存在'
            }), 404

        # 计算指标 (不再包含 LLM 分析)
        metrics = kg_manager.calculate_graph_metrics(graph_id) # 这步现在不包含 LLM 分析

        # 检查计算过程中是否有错误
        if 'errors' in metrics:
             logger.warning(f"计算图谱 {graph_id} 指标时发生错误: {metrics['errors']}")
             # 即使有错也返回部分结果

        return jsonify({
            'status': 'success',
            'metrics': metrics
        })

    except ConnectionError as e:
         logger.error(f"获取图谱 {graph_id} 指标失败: 无法连接到 Neo4j - {str(e)}")
         return jsonify({
             'status': 'error',
             'message': f"无法连接到后端服务 (Neo4j): {str(e)}" # 修改错误信息
         }), 503 # Service Unavailable
    except Exception as e:
        logger.error(f"计算图谱 {graph_id} 指标时发生一般错误: {str(e)}")
        # Log the full traceback for debugging
        logger.error(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'message': f"计算指标时发生内部错误: {str(e)}"
        }), 500

@kg_bp.route('/api/kg/graphs/<int:graph_id>/llm-analysis', methods=['POST'])
def get_llm_analysis_for_graph(graph_id):
    """根据提供的指标数据，为指定图谱生成 LLM 分析"""
    try:
        data = request.get_json()
        if not data or 'metrics' not in data:
            return jsonify({
                'status': 'error',
                'message': '请求体中缺少指标数据 (metrics)'
            }), 400

        metrics_data = data['metrics']

        # 确保传递的 metrics 数据包含 graph_id (LLM prompt 可能需要)
        if 'graph_id' not in metrics_data:
             metrics_data['graph_id'] = graph_id # 如果前端没传，补充一下

        # 调用 manager 生成 LLM 分析
        analysis = kg_manager.generate_llm_analysis(metrics_data)

        return jsonify({
            'status': 'success',
            'analysis': analysis
        })

    except ConnectionError as e:
         logger.error(f"生成图谱 {graph_id} LLM 分析失败: 无法连接到 LLM 服务 - {str(e)}")
         return jsonify({
             'status': 'error',
             'message': f"无法连接到 LLM 服务: {str(e)}"
         }), 503 # Service Unavailable
    except Exception as e:
        logger.error(f"生成图谱 {graph_id} LLM 分析时出错: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'message': f"生成分析时发生内部错误: {str(e)}"
        }), 500

@kg_bp.route('/api/kg/llm-suggest-construction', methods=['POST'])
def llm_suggest_construction():
    """使用大模型根据表结构生成知识图谱构建条件"""
    try:
        data = request.get_json()
        if not data or 'table' not in data:
            return jsonify({
                'status': 'error',
                'message': '缺少表名参数'
            }), 400
            
        table_name = data['table']
        # 始终使用配置文件中的数据库名称，不接受前端传递的database参数
        # 这样确保数据库配置的一致性
        database_name = kg_manager.vanna_manager.config.database.database_name
        logger.info(f"使用配置文件中的数据库: {database_name}")
        
        # 使用相同的数据库配置
        database_config = {
            'type': kg_manager.vanna_manager.config.database.type,
            'host': kg_manager.vanna_manager.config.database.host,
            'port': kg_manager.vanna_manager.config.database.port, 
            'username': kg_manager.vanna_manager.config.database.username,
            'password': kg_manager.vanna_manager.config.database.password,
            'database': database_name
        }
        logger.info(f"数据库配置: {database_config}")
        # 1. 获取表结构
        schema = kg_manager.get_table_schema(table_name, database_config)
        logger.info(f"表结构: {schema}")
        # 2. 调用大模型生成知识图谱构建条件
        suggestions = kg_manager.generate_kg_construction_suggestions(table_name, schema)
        
        return jsonify({
            'status': 'success',
            'suggestions': suggestions
        })
    except Exception as e:
        logger.error(f"生成知识图谱构建条件失败: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500