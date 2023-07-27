import urllib.request
import fitz
import re
import numpy as np
# import tensorflow_hub as hub
from textFeatExtractor import TextFeatureExtractor
import openai
import gradio as gr
import os
from sklearn.neighbors import NearestNeighbors
from test_azure_openai_api_chatcompletions import OPENAI_API_KEY1,test_request
import json
import test_embeddings
from sklearn.metrics.pairwise import cosine_similarity

def preprocess(text):
    text = text.replace('\n', ' ')
    text = re.sub('\s+', ' ', text)
    return text

def pdf_to_text(path, start_page=1, end_page=None):
    doc = fitz.open(path)
    total_pages = doc.page_count

    if end_page is None:
        end_page = total_pages

    text_list = []

    for i in range(start_page-1, end_page):
        text = doc.load_page(i).get_text("text")
        text = preprocess(text)
        text_list.append(text)

    doc.close()
    return text_list


def text_to_chunks(texts, word_length=150, start_page=1):
    text_toks = [t.split(' ') for t in texts]
    page_nums = []
    chunks = []
    
    for idx, words in enumerate(text_toks):
        for i in range(0, len(words), word_length):
            chunk = words[i:i+word_length]
            if (i+word_length) > len(words) and (len(chunk) < word_length) and (
                len(text_toks) != (idx+1)):
                text_toks[idx+1] = chunk + text_toks[idx+1]
                continue
            chunk = ' '.join(chunk).strip()
            chunks.append(chunk)
    return chunks


class SemanticSearch:
    
    def __init__(self):
        # self.predictor = TextFeatureExtractor(task_name="simbert-base-chinese",device="cpu")
        self.predictor = test_embeddings.test_request
        self.fitted = False
    
    
    def fit(self, data, batch=16, n_neighbors=5):
        self.data = data
        self.embeddings = self.get_text_embedding(data, batch=batch)
        self.fitted = True
    
    
    def __call__(self, text, return_data=True):
        # inp_emb = self.predictor.predict([text])
        inp_emb = self.predictor([text])
        simis = cosine_similarity(inp_emb,self.embeddings)
        # 相似度排序
        top_k = 3
        simis_sorted = np.sort(simis)[:,::-1]
        neighbors = np.argsort(simis)[:,::-1][0,:top_k]
        simi_th = 0.8
        if simis_sorted[0,:top_k].max() < simi_th:
            return np.array([0]*top_k)
        else:
            if return_data:
                return [self.data[i] for i in neighbors]
            else:
                return neighbors
    
    def get_text_embedding(self, texts, batch=16):
        embeddings = []
        for i in range(0, len(texts), batch):
            text_batch = texts[i:(i+batch)]
            # emb_batch = self.predictor.predict(text_batch)
            _retry = True
            _rerty_count = 0
            while _retry:
                _retry = False
                try:
                    emb_batch = self.predictor(text_batch)
                except Exception as e:
                    if _rerty_count < 10:
                        _retry = True
                        _rerty_count += 1
                        continue
            embeddings.append(emb_batch)
        embeddings = np.vstack(embeddings)
        return embeddings

def load_recommender(path, start_page=1):
    global recommender
    texts = pdf_to_text(path, start_page=start_page)
    chunks = text_to_chunks(texts, start_page=start_page)
    recommender.fit(chunks)
    return 'Corpus Loaded.'

def generate_answer(question):
    print("查询内容...")
    print(question)
    topn_chunks = recommender(question)
    prompt = ""
    if topn_chunks.sum() > 0:

        prompt += '查询结果:\n\n'
        for c in topn_chunks:
            prompt += c + '\n\n'
            
        # prompt += f"说明：使用给定的查询结果重新组织回复内容,"\
        #         "答案如果是有序列表，就保持查询结果中的排列顺序，以列表的方式列举出来，每一条前面加上阿拉伯数字序号。"\
        #         "答案尽量从查询结果中摘取，不要添加任何其他信息，也不要漏掉任何信息。确保答案百分百正确，不要输出虚假内容。"\
        #         "如果文本与查询无关，只需声明‘未发现任何内容’。忽略异常值以及与问题无关的答案，只回答问题。"\
        #         "\n\n"   
        prompt += f"说明：使用给定的查询结果重新组织回复内容,"\
                "根据查询结果中的客户表达问题以及客户来信，给出当前问题属于客户表达问题中的哪一个，并且直接返回写在对应问题后面的模板，不要做任何改动。格式如下："\
                "问题类型：{从查询结果中获取}\n 回复模板：{从查询结果中获取}"\
                "答案必须从查询结果中直接取出，不要添加任何其他信息，也不要漏掉任何信息。确保答案百分百正确，不要输出虚假内容。"\
                "如果文本与查询无关，只需声明‘未发现任何内容’。忽略异常值以及与问题无关的答案，只回答问题。"\
                "\n\n"   
        prompt += f"查询: {question}\n答案:"
    else:
        prompt += "根据上下文按照电商客服的话术规范来回答，尽量做到温馨，有礼貌，回复的语种需要保持与查询文本的语种一致。"

    try:
        print("chatGPT组织答案...")
        _retry = True
        _rerty_count = 0
        while _retry:
            _retry = False
            try:
                answer = test_request(prompt)
            except Exception as e:
                print(e)
                if _rerty_count < 10:
                    _retry = True
                    _rerty_count += 1
                    continue
        answer_text = json.loads(answer)['choices'][0]['message']['content']
        return answer_text
    except Exception as e:
        print(e)
        return "请求ChatGPT错误"
    
def question_answer(file, question):

    old_file_name = file.name
    file_name = file.name
    file_name = file_name[:-12] + file_name[-4:]
    if not os.path.exists(file_name):
        os.rename(old_file_name, file_name)
    print("处理文档...")
    load_recommender(file_name)

    if question.strip() == '':
        return '[ERROR]: Question field is empty'
    
    return generate_answer(question)


recommender = SemanticSearch()

