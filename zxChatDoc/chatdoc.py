from zxChatDoc import openai_api_embedding,openai_api_chat
from sklearn.metrics.pairwise import cosine_similarity
from typing import Union, List
from tqdm import tqdm
import numpy as np
from zxChatDoc.util import *
import json

# 本地化文档智能查询
class ChatDoc():
    def __init__(self,topk = 5,max_word_count = 300):
        self.topk = topk
        self.max_word_count = int(max_word_count)
        self.embed = openai_api_embedding.test_request
        self.chat = openai_api_chat.test_request
        self.init_vectors = False
        self.vec_stored = False
        self.simi_th = 0.7

    def init_vectors_base(self,  doc_files: Union[str, List[str]]):
        '''
        根据文档初始化向量库
        '''
        # 加载文档
        corpus = self.load_doc_files(doc_files)
        # 文档切块
        self.data = self.corpus_to_chunks(corpus)
        # 构建特征库
        self.embeddings = self.get_trucks_embeddings(self.data)
        self.init_vectors = True

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
        max_strlen = self.max_word_count
        text_toks = sum(text_toks,[])
        chunk = ""
        last_idx = 0
        for idx, words in enumerate(text_toks):
            chunk += words
            if len(chunk) > max_strlen:
                chunk = " ".join(text_toks[last_idx:idx+1]) 
                # print(len(chunk))
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
                except Exception as e:
                    if _rerty_count < 5:
                        _retry = True
                        _rerty_count += 1
                        continue
            embeddings.append(emb_batch)
        embeddings = np.vstack(embeddings)
        return embeddings
    
    def get_topk_trucks(self,question,topn):
        '''
        获取最相似的topk个片段
        '''
        self.topk = topn
        inp_emb = self.embed([question])
        simis = cosine_similarity(inp_emb,self.embeddings)
        # 相似度排序
        simis_sorted = np.sort(simis)[:,::-1]
        neighbors = np.argsort(simis)[:,::-1][0,:self.topk]
        start = max(0,neighbors[0]- int(self.topk/2))
        end = min(len(simis[0]),neighbors[0]+int(self.topk/2) if self.topk%2 == 0 else neighbors[0]+int(self.topk/2)+1)
        neighbors = list(range(start, end))
        # 小于相似度阈值的，不返回结果
        if simis_sorted[0][0] < self.simi_th and len(simis[0]) > self.topk:
            return []
        topk_trucks = [self.data[i] for i in neighbors]
        return topk_trucks

    def query(self,question,topn):
        print("查询内容...")
        print(question)
        try:
            topn_chunks = self.get_topk_trucks(question,topn)
            if len(topn_chunks) == 0:
                return "根据已有资料，无法查询到答案，您可以尝试换一种提问方式！"
        except Exception as e:
            print(e)
            return "查询结果出错！"
        prompt = ""
        prompt += f"问题：{question}"
        # 长文本
        paragraph = f"\n###\n我提供的长文本："
        for c in topn_chunks:
            paragraph += c
        try:
            print("chatGPT组织答案...")
            _retry = True
            _rerty_count = 0
            while _retry:
                _retry = False
                try:
                    answer = self.chat(prompt,paragraph)
                except Exception as e:
                    print(e)
                    if _rerty_count < 10:
                        _retry = True
                        _rerty_count += 1
                        continue
            answer_text = json.loads(answer)['choices'][0]['message']['content']
            prefix = "输出："
            index = answer_text.find(prefix)
            index = index+len(prefix) if index > -1 else 0 
            answer_text = answer_text[index:].replace("提取的关键词是：","").strip()
            return answer_text
        except Exception as e:
            print(e)
            return "请求ChatGPT错误"
    