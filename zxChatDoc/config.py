import os
from loguru import logger

cur_path = os.path.dirname(os.path.dirname(__file__))
# 记录日志
logger.remove()
logger.add(os.path.join(cur_path,"logs/liblog.log"))

# 知识库默认存储路径
KB_ROOT_PATH = os.path.join(cur_path, "knowledge_base")
# 片段长度
CHUNK_SIZE = 300
# 取最相似的TOPK个片段
TOPK = 5