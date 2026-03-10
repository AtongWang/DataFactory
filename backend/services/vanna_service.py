from backend.manager.vanna_manager import VannaManager



# 获取vanna_app对象
def get_vanna_app():
    from vanna.flask import VannaFlaskApp
    vanna_app = VannaFlaskApp(vanna_manager.vn)
    return vanna_app

# 初始化Vanna管理器
vanna_manager = VannaManager()
vanna_app = get_vanna_app()