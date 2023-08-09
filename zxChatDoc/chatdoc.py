from zxChatDoc import openai_api_embedding,openai_api_chat
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import numpy as np
from zxChatDoc.util import *
from zxChatDoc.config import *
import json
import shutil
import time

# 本地化文档智能查询
class ChatDoc():
    def __init__(self,topk = TOPK,max_word_count = CHUNK_SIZE):
        self.topk = topk
        self.max_word_count = int(max_word_count)
        self.embed = openai_api_embedding.test_request
        self.chat = openai_api_chat.test_request
        self.init_vectors = False
        self.vec_stored = False
        self.simi_th = 0.7
        self.embeddings = None

    def add_vectors_base(self, kb_name, doc_file: str):
        '''
        根据文档初始化向量库
        '''
        # 加载文档
        corpus = self.load_doc_files(doc_file)
        if corpus is None or len(corpus) == 0:
            return {"status":"error","message":"文档为空！"}
        # 文档切块
        self.data = self.corpus_to_chunks(corpus)
        # 构建特征库
        self.embeddings = self.get_trucks_embeddings(self.data)
        if self.embeddings is None:
            return {"status":"error","message":"生成向量库失败！"}
        self.save_vectors_base(kb_name)
        self.init_vectors = True
        return {"status":"sucess","message":"向量库生成成功！"}

    def save_vectors_base(self, kb_name: str):
        '''
        保存向量库
        '''
        vs_path = os.path.join(KB_ROOT_PATH, kb_name)
        if not os.path.exists(vs_path):
            os.makedirs(vs_path)
        data_path = os.path.join(vs_path, "data.npy")
        np.save(data_path,self.data)
        embedding_path = os.path.join(vs_path, "embedding.npy")
        np.save(embedding_path,self.embeddings)

    def load_vectors_base(self, kb_name: str):
        '''
        加载向量库
        '''
        start = time.time()
        vs_path = os.path.join(KB_ROOT_PATH, kb_name)
        try:  
            data_path = os.path.join(vs_path, "data.npy")
            self.data = np.load(data_path).tolist()
            embedding_path = os.path.join(vs_path, "embedding.npy")
            self.embeddings = np.load(embedding_path)
            end = time.time()
            logger.info(f"load vectors cost time:{end - start}")
        except Exception as e:
            logger.error(f":{e}")
            return [],[]
        return self.data, self.embeddings
    
    def del_vectors_base(self,kb_name:str):
        file_path = os.path.join(KB_ROOT_PATH, kb_name)
        if os.path.exists(file_path):
            shutil.rmtree(file_path)

    def load_doc_files(self, doc_file: str):
        '''
        加载文档
        '''
        if doc_file.endswith('.pdf'):
            corpus = extract_text_from_pdf(doc_file)
        elif doc_file.endswith('.docx'):
            corpus = extract_text_from_docx(doc_file)
        elif doc_file.endswith('.md'):
            corpus = extract_text_from_markdown(doc_file)
        else:
            corpus = extract_text_from_txt(doc_file)
        return corpus

    def corpus_to_chunks(self,texts):
        '''
        切分文档
        '''
        text_toks = [t.split(' ') for t in texts]
        chunks = []
        # 按字数统计
        max_strlen = int(self.max_word_count)
        text_toks = sum(text_toks,[])
        chunk = ""
        last_idx = 0
        for idx, words in enumerate(text_toks):
            chunk += words
            if len(chunk) > max_strlen:
                chunk = " ".join(text_toks[last_idx:idx+1]) 
                chunks.append(chunk)
                last_idx = idx+1
                chunk = ""
        chunks.append(chunk)
        return chunks

    def get_trucks_embeddings(self, texts, batch=16):
        '''
        调用openai接口获取文档片段的向量
        '''
        embeddings = []
        if len(texts) == 0:
            return embeddings
        for i in tqdm(range(0, len(texts), batch)):
            text_batch = texts[i:(i+batch)]
            _retry = True
            _rerty_count = 0
            while _retry:
                _retry = False
                try:
                    emb_batch = self.embed(text_batch)
                    if type(emb_batch) != np.ndarray:
                        raise Exception("emb_batch is not np.ndarray")
                except Exception as e:
                    if _rerty_count < 5:
                        _retry = True
                        _rerty_count += 1
                        continue
            if type(emb_batch) == np.ndarray:
                embeddings.append(emb_batch)
        if len(embeddings) > 0: 
            embeddings = np.vstack(embeddings)
        return np.array(embeddings)
    
    def get_topk_trucks(self,question,topn,data,embeddings):
        '''
        获取最相似的topk个片段
        '''
        self.topk = topn
        _retry = True
        _rerty_count = 0
        while _retry:
            _retry = False
            try:
                inp_emb = self.embed([question])
                if type(inp_emb) != np.ndarray:
                    raise Exception("提取向量出错！")
            except Exception as e:
                logger.error(f"提取向量出错:{e}")
                if _rerty_count < 3:
                    _retry = True
                    _rerty_count += 1
                    time.sleep(1)
                    continue
        
        if type(inp_emb) == dict:
            logger.info("查询TopK失败!")
            return []
        simis = cosine_similarity(inp_emb,embeddings)
        # 相似度排序
        simis_sorted = np.sort(simis)[:,::-1]
        neighbors = np.argsort(simis)[:,::-1][0,:self.topk]
        # 前后平均(如果是top5,前后各取2，如果是top4,取前1，后2)
        # start = max(0,neighbors[0]- int(self.topk/2) if self.topk%2 != 0 else neighbors[0]-int(self.topk/2)+1)
        # end = min(len(simis[0]),neighbors[0]+int(self.topk/2)+1)

        # 偏后
        top1 = neighbors[0]
        start = max(0, top1 - 1)
        end = min(len(simis[0]),top1+int(self.topk/2)+2 if self.topk%2 != 0 else top1+int(self.topk/2)+1)

        neighbors = list(range(start, end))
        logger.info(f"topk:{top1} {neighbors}")
        logger.info(f"len_data:{len(data)},len_simis:{len(simis[0])}")
        assert len(data) == len(simis[0]),"维度不匹配"
        # 小于相似度阈值的，不返回结果
        if simis_sorted[0][0] < self.simi_th and len(simis[0]) > self.topk:
            return []
        topk_trucks = [data[i] for i in neighbors]
        return topk_trucks

    def query(self,kb_name,question,topn):
        data,embeddings = self.load_vectors_base(kb_name)
        if len(data)==0 or len(embeddings)==0 or len(data) != len(embeddings):
            logger.error(f"加载的知识库信息：data:{len(data)},embeddings:{len(embeddings)}")
            return "","当前知识库文件存在问题，请重新创建！"
        logger.info(f"查询内容:{question}")
        try:
            topn_chunks = self.get_topk_trucks(question,topn,data,embeddings)
            if len(topn_chunks) == 0:
                return "","查询TopK失败，可能原因：\n1. 提取向量失败\n2. 根据已有资料，无法查询到答案，您可以尝试换一种提问方式！"
        except Exception as e:
            logger.error(e)
            return "","查询结果出错！"
        prompt = ""
        prompt += f"问题：{question}"
        # 长文本
        paragraph = f"\n###\n我提供的长文本："
        topn_results = ""
        for i,c in enumerate(topn_chunks):
            paragraph += c
            topn_results += f"【{i+1}】"+c+"\n"
        try:
            logger.info("chatGPT组织答案...")
            _retry = True
            _rerty_count = 0
            while _retry:
                _retry = False
                try:
                    answer = self.chat(prompt,paragraph)
                except Exception as e:
                    logger.error(e)
                    if _rerty_count < 10:
                        _retry = True
                        _rerty_count += 1
                        continue
            answer_text = json.loads(answer)['choices'][0]['message']['content']
            prefix = "输出："
            index = answer_text.find(prefix)
            index = index+len(prefix) if index > -1 else 0 
            answer_text = answer_text[index:].replace("提取的关键词是：","").strip()
            return topn_results,answer_text
        except Exception as e:
            logger.error(e)
            return "","请求ChatGPT错误"
    