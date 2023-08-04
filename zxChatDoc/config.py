import os

# 知识库默认存储路径
KB_ROOT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "knowledge_base")
# 片段长度
CHUNK_SIZE = 300
# 取最相似的TOPK个片段
TOPK = 5