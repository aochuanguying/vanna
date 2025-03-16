from openai import OpenAI
from vanna.openai.openai_chat import OpenAI_Chat
from vanna.chromadb.chromadb_vector import ChromaDB_VectorStore
import chromadb
from sqlalchemy import create_engine, text
import os

class MyVanna(ChromaDB_VectorStore, OpenAI_Chat):
    def __init__(self, config=None):
        # 初始化 ChromaDB
        chromadb_config = {
            'client': chromadb.Client(),
            'collection_name': 'vanna_collection'
        }
        ChromaDB_VectorStore.__init__(self, config=chromadb_config)
        
        # 初始化 OpenAI
        openai_config = {
            'client': config.get('client'),
            'model': config.get('model')
        }
        OpenAI_Chat.__init__(self, config=openai_config)
        # 设置 client 属性
        self.client = config.get('client')
        self.model = config.get('model')
        
        # 设置数据库连接
        self.db_engine = config.get('db_engine')

# 创建 MySQL 数据库连接
db_url = "mysql+pymysql://root:root@www.hxfssc.com:3306/vanna"
engine = create_engine(db_url)

# 初始化数据库表和测试数据
def init_database():
    with engine.connect() as conn:
        # 删除已存在的表（如果存在）
        conn.execute(text("DROP TABLE IF EXISTS users"))
        
        # 创建用户表
        conn.execute(text("""
            CREATE TABLE users (
                id INT PRIMARY KEY,
                name VARCHAR(100),
                email VARCHAR(255),
                created_at TIMESTAMP
            )
        """))
        
        # 插入测试数据
        conn.execute(text("""
            INSERT INTO users (id, name, email, created_at) VALUES 
            (1, '张三', 'zhangsan@example.com', '2024-01-15 10:00:00'),
            (2, '李四', 'lisi@example.com', '2024-02-20 14:30:00'),
            (3, '王五', 'wangwu@example.com', '2023-12-01 09:15:00')
        """))
        
        conn.commit()

# 创建 OpenAI 客户端
client = OpenAI(
    api_key='sk-tcoagstffdthsvfowprqygyupjthicblaskkwokcndjqllat',
    base_url='https://api.siliconflow.cn/v1'
)

# 创建 Vanna 实例
vn = MyVanna(config={
    'client': client,
    'model': 'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B',  # 或其他已部署的模型名称
    'db_engine': engine  # 添加数据库引擎
})

# 初始化数据库
try:
    init_database()
    print("数据库初始化成功！")
except Exception as e:
    print("数据库初始化失败:", e)
    os._exit(1)

# 训练示例数据
vn.train(ddl="""
    CREATE TABLE users (
        id INT PRIMARY KEY,
        name VARCHAR(100),
        email VARCHAR(255),
        created_at TIMESTAMP
    )
""")

# 添加文档说明
vn.train(documentation="用户表存储了系统中所有注册用户的基本信息，包含用户ID、姓名、邮箱和注册时间。")

# 添加示例 SQL
vn.train(sql="SELECT name, email FROM users WHERE created_at >= '2024-01-01'")

# 测试查询
query = "查找所有2024年注册的用户的名字和邮箱"
result = vn.ask(query)
print(f"\n查询问题: {query}")
print("生成的 SQL:", result)

# 执行生成的 SQL 查询
if result:
    with engine.connect() as conn:
        try:
            result_set = conn.execute(text(result))
            print("\n查询结果:")
            for row in result_set:
                print(f"姓名: {row.name}, 邮箱: {row.email}")
        except Exception as e:
            print("执行 SQL 时出错:", e)
else:
    print("SQL 生成失败，请检查模型响应。") 