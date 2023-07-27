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


def text_to_chunks(texts, word_length=50, start_page=1):
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
        # n_neighbors = min(n_neighbors, len(self.embeddings))
        # self.nn = NearestNeighbors(n_neighbors=n_neighbors)
        # self.nn.fit(self.embeddings)
        self.fitted = True
        global init_embedding
        init_embedding = True
    
    
    def __call__(self, text, return_data=True):
        # inp_emb = self.predictor.predict([text])
        inp_emb = self.predictor([text])
        # neighbors = self.nn.kneighbors(inp_emb, return_distance=False,n_neighbors=3)[0]
        # neighbors = np.sort(neighbors)
        # print(neighbors)
        # if return_data:
        #     return [self.data[i] for i in neighbors]
        # else:
        #     return neighbors
        simis = cosine_similarity(inp_emb,self.embeddings)
        # 相似度排序
        top_k = 3
        simis_sorted = np.sort(simis)[:,::-1]
        neighbors = np.argsort(simis)[:,::-1][0,:top_k]
        neighbors = [max(neighbors[0]-1,0),neighbors[0],min(neighbors[0]+1,len(simis[0])-1)]
        neighbors = list(set(neighbors))
        simi_th = 0.8
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
                    if _rerty_count < 5:
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
    prompt += f"问题：'{question}\n\n'，"
    # 长文本
    paragraph = f"###\n我提供的长文本："
    for c in topn_chunks:
        paragraph += c
    try:
        print("chatGPT组织答案...")
        _retry = True
        _rerty_count = 0
        while _retry:
            _retry = False
            try:
                answer = test_request(prompt,paragraph)
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
    
def question_answer(file, question):

    old_file_name = file.name
    file_name = file.name
    file_name = file_name[:-12] + file_name[-4:]
    if not os.path.exists(file_name):
        os.rename(old_file_name, file_name)
    if not init_embedding:
        print("处理文档...")
        load_recommender(file_name)

    if question.strip() == '':
        return '[ERROR]: Question field is empty'
    
    return generate_answer(question)

init_embedding = False
recommender = SemanticSearch()



if __name__ == '__main__':
    title = 'PDF信息查询'
    description = """上传PDF,查询其中的知识"""
    with gr.Blocks() as demo:
        gr.Markdown(f'<center><h1>{title}</h1></center>')
        gr.Markdown(description)

        with gr.Row():
            with gr.Group():
                file = gr.File(label='上传pdf', file_types=['.pdf'])
                question = gr.Textbox(label='输入问题')
                btn = gr.Button(value='提交')
                btn.style(full_width=True)

            with gr.Group():
                answer = gr.Textbox(label='查询结果 :')
            question.submit(question_answer,inputs=[file,question],outputs=[answer],queue=False)
            btn.click(question_answer, inputs=[file, question], outputs=[answer])
    #openai.api_key = os.getenv('Your_Key_Here') 
    demo.launch(server_name="0.0.0.0",server_port=8888)
