from vanna.ollama import Ollama
from vanna.chromadb import ChromaDB_VectorStore
import os


class MyVanna(ChromaDB_VectorStore, Ollama):
    def __init__(self, config=None):
        ChromaDB_VectorStore.__init__(self, config=config)
        Ollama.__init__(self, config=config)


# 确保ChromaDB目录存在
chroma_db_path = "./chroma_db"
if not os.path.exists(chroma_db_path):
    os.makedirs(chroma_db_path, exist_ok=True)

# 创建Vanna实例，使用实际配置
vn = MyVanna(config={
    'ollama_host': 'http://192.168.4.211:11434',  # Ollama服务地址
    'model': 'qwq:32b',                          # 使用的模型
    'path': chroma_db_path,                      # ChromaDB存储路径
    'embedding_model': 'bge-m3:latest'           # 嵌入模型
})

# 连接到MySQL数据库，使用实际连接信息
vn.connect_to_mysql(
    host='192.168.4.211',       # 数据库主机地址
    dbname='equipment_db',      # 数据库名称
    user='root',                # 数据库用户名
    password='123456',          # 数据库密码
    port=3306                   # 数据库端口
)

# 可以添加一些训练数据（可选）
# vn.train(documentation="这是设备数据库，包含设备基本信息、状态和维护记录。")

# 创建Flask应用
from vanna.flask import VannaFlaskApp
app = VannaFlaskApp(vn)

# 启动应用（指定主机和端口）
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)