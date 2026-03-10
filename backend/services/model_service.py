from backend.manager.model_manager import ModelManager
from backend.services.vanna_service import vanna_manager

# 全局变量存储当前的model_manager实例
_model_manager = None

def _get_or_create_model_manager():
    """获取或创建ModelManager实例"""
    global _model_manager
    if _model_manager is None:
        config = vanna_manager.get_config()
        _model_manager = ModelManager(config.get('model', {}))
    return _model_manager

def _on_config_update(new_config_dict):
    """配置更新回调函数"""
    global _model_manager
    try:
        # 重新创建ModelManager实例
        _model_manager = ModelManager(new_config_dict.get('model', {}))
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"ModelManager已更新配置: {new_config_dict.get('model', {}).get('type', 'unknown')}")
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"更新ModelManager配置失败: {str(e)}")

# 注册配置更新回调
vanna_manager.register_config_callback(_on_config_update)

# 提供一个属性来访问model_manager，确保总是获取最新的实例
class ModelManagerProxy:
    """ModelManager代理类，确保总是返回最新的实例"""
    def __getattr__(self, name):
        return getattr(_get_or_create_model_manager(), name)

# 创建代理实例
model_manager = ModelManagerProxy()