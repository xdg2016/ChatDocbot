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
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

templates = {}
event_types = []

def excel_to_text(path):
    global templates
    global event_types

    templates = {}
    event_types = []
    df = pd.read_excel(path)
    height,width = df.shape
    print(height,width,type(df))
    event_list = []
    for i in range(height):
        event_type = df.iloc[i,0]
        template = df.iloc[i,1]
        letter_sample = df.iloc[i,2]
        if letter_sample is None:
            letter_sample = ""
        letter_sample2 = df.iloc[i,3]
        if letter_sample2 is None:
            letter_sample2 = ""
        event_list.append([event_type,letter_sample, letter_sample2])
        templates[event_type] = template
        event_types.append(event_type)
    return event_list

def cosine_distance(v1,v2):
    
    # 计算余弦距离
    cosine_distance = cosine_similarity(v1, v2)
    return cosine_distance

init_info = False

use_simbert = True

class SemanticSearch:
    
    def __init__(self):
        if use_simbert:
            self.predictor = TextFeatureExtractor(task_name="simbert-base-chinese",device="cpu")
        else:
            self.predictor = test_embeddings.test_request
        self.fitted = False
    
    def __call__(self, text):
        if use_simbert:
            inp_emb = self.predictor.predict([text])
        else:
            inp_emb = self.predictor([text])
        neighbors = cosine_distance(inp_emb,self.all_embeddings)
        idx = np.argmax(neighbors)
        # 合并特征时处理
        # event = event_types[idx]
        # print(event)
        # template = templates[event]
        # result = event+"\n"+template
        # return result
        # 不合并特征时处理
        for i,(key,v) in enumerate(self.all_embeddings_idxs.items()):
            if idx in key:
                print(event_types[i])
                result = event_types[i]+"\n" +templates[v]
                return result
        
    def get_embeddings(self,texts):
        self.all_embeddings = []
        self.all_embeddings_idxs = {}
        last_idx = 0
        self.all_texts = []
        for i,text in enumerate(texts):
            count = len(text)
            self.all_texts.extend(text)
            key = tuple([last_idx+v for v in list(range(count))])
            self.all_embeddings_idxs[key] = event_types[i]
            last_idx += count
        
        # 按类别相加
        self.text_embeddings = self.get_text_embedding(self.all_texts)
        # 合并特征时打开
        # counts = list(self.all_embeddings_idxs.keys())
        # for i in range(len(counts)):
        #     tmp_text_embeddings = self.text_embeddings[counts[i][0]:counts[i][-1]+1].sum(axis=0,keepdims=True)
        #     self.all_embeddings.append(tmp_text_embeddings)
        # self.all_embeddings = np.vstack(self.all_embeddings)
        # 不合并特征时打开
        self.all_embeddings = self.text_embeddings

    def get_text_embedding(self, texts, batch=10):
        embeddings = []
        for i in range(0, len(texts), batch):
            text_batch = texts[i:(i+batch)]
            
            _retry = True
            _rerty_count = 0
            while _retry:
                _retry = False
                try:
                    if use_simbert:
                        emb_batch = self.predictor.predict(text_batch)
                    else:
                        emb_batch = self.predictor(text_batch)
                except Exception as e:
                    if _rerty_count < 10:
                        _retry = True
                        _rerty_count += 1
                        continue
            embeddings.append(emb_batch)
        embeddings = np.vstack(embeddings)
        return embeddings

def file_changed(file):
    old_file_name = file.name
    file_name = file.name
    file_name = file_name[:-12] + file_name[-4:]
    if not os.path.exists(file_name):
        os.rename(old_file_name, file_name)
    print("处理文档1...")
    status = load_recommender(file_name)
    global init_info
    init_info = status

def load_recommender(path):
    global recommender
    try:
        texts = excel_to_text(path)
        print("问题类型:\n",event_types)
        recommender.get_embeddings(texts)
        return True
    except Exception as e:
        return False

def generate_answer(question):
    print("查询内容...")
    answer = recommender(question)
    return answer
        
    
def template_gen(file, letters):
    '''
    生成模板
    '''
    if not init_info:
        return "文档初始化失败，请重新上传！"
    if len(templates.keys()) == 0 or len(event_types)==0: 
        print("处理文档2...")
        file_changed(file)
    if letters.strip() == '':
        return '[ERROR]: letters field is empty'
    
    return generate_answer(letters)


recommender = SemanticSearch()

title = '客户反馈信息分类'
description = """上传分类表格"""


with gr.Blocks() as demo:

    gr.Markdown(f'<center><h1>{title}</h1></center>')
    gr.Markdown(description)

    with gr.Row():
        with gr.Group():
            file = gr.File(label='上传表格', file_types=['.xlsx',".xls"])
            letters = gr.Textbox(label='输入来信')
            btn = gr.Button(value='提交')
            btn.style(full_width=True)

        with gr.Group():
            answer = gr.Textbox(label='查询结果 :')
        file.upload(file_changed,inputs=[file])
        btn.click(template_gen, inputs=[file, letters], outputs=[answer])
#openai.api_key = os.getenv('Your_Key_Here') 
demo.launch(server_name="0.0.0.0",server_port=8888)
