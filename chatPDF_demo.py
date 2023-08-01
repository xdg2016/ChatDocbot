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
from tqdm import tqdm

def preprocess(text):
    text = text.replace('\n', ' ')
    text = re.sub('\s+', ' ', text)
    text = re.sub(r'\.{4,}', '', text)
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

def text_to_chunks(texts,split_rule, word_length=100,max_word_count = 300):

    text_toks = [t.split(' ') for t in texts]
    chunks = []
    split_rule = split_rules[split_rule]
    # 按字数统计
    if split_rule == "word_count":
        max_strlen = int(max_word_count)
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
    else:
        for idx, words in enumerate(text_toks):
            for i in range(0, len(words), word_length):
                chunk = words[i:i+word_length]
                if (i+word_length) > len(words) and (len(chunk) < word_length) and (len(text_toks) != (idx+1)):
                    text_toks[idx+1] = chunk + text_toks[idx+1]
                    continue
                chunk = ' '.join(chunk).strip()
                # print("chunk_len: ",len(chunk))
                chunks.append(chunk)
    return chunks


class SemanticSearch:
    
    def __init__(self):
        # self.feature_extractor = TextFeatureExtractor(task_name="simbert-base-chinese",device="cpu")
        self.predictor = test_embeddings.test_request
        self.fitted = False
    
    def fit(self, data, batch=16, n_neighbors=5):
        self.data = data
        print("构建特征库...")
        self.embeddings = self.get_text_embedding(data, batch=batch)
        # n_neighbors = min(n_neighbors, len(self.embeddings))
        # self.nn = NearestNeighbors(n_neighbors=n_neighbors)
        # self.nn.fit(self.embeddings)
        self.fitted = True
        global init_embedding
        init_embedding = True
    
    
    def __call__(self, text,top_k, return_data=True):
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
        simis_sorted = np.sort(simis)[:,::-1]
        neighbors = np.argsort(simis)[:,::-1][0,:top_k]
        start = max(0,neighbors[0]- int(top_k/2))
        end = min(len(simis[0]),neighbors[0]+int(top_k/2) if top_k%2 == 0 else neighbors[0]+int(top_k/2)+1)
        neighbors = list(range(start, end))
        simi_th = 0.8
        if return_data:
            return [self.data[i] for i in neighbors]
        else:
            return neighbors
    
    
    def get_text_embedding(self, texts, batch=16):
        embeddings = []
        for i in tqdm(range(0, len(texts), batch)):
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

def add_file(file,chatbot,split_rule_radio,max_word_count):
    '''
    上传文件
    '''
    old_file_name = file.name
    file_name = file.name
    file_name = file_name[:-12] + file_name[-4:]
    if not os.path.exists(file_name):
        os.rename(old_file_name, file_name)
    print("处理文档...")
    load_recommender(file_name,split_rule_radio,max_word_count)
    info = "文档处理结束，请开始提问！"
    print(info)
    chatbot.pop()
    chatbot+= [(None,info)]
    return chatbot,gr.update(value="", interactive=True)

def load_recommender(path,split_rule,max_word_count, start_page=1):
    global recommender
    texts = pdf_to_text(path,start_page=start_page)
    chunks = text_to_chunks(texts,split_rule,max_word_count= max_word_count)
    print(f"-------------总片段数：{len(chunks)}  -----------")
    recommender.fit(chunks)
    return 'Corpus Loaded.'

def generate_answer(question,topn):
    print("查询内容...")
    print(question)
    try:
        topn_chunks = recommender(question,topn)
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
    
def question_answer(file, question,chatbot, split_rule,max_word_count,topn):

    if len(question) == 0:
        chatbot += [(None,'[错误]: 输入的问题为空！')]
        return chatbot,gr.update(value="", interactive=True)
    old_file_name = file.name
    file_name = file.name
    file_name = file_name[:-12] + file_name[-4:]
    if not os.path.exists(file_name):
        os.rename(old_file_name, file_name)
    if not init_embedding:
        print("处理文档...")
        load_recommender(file_name,split_rule,int(max_word_count),start_page=1)
    answer = generate_answer(question,topn)
    chatbot += [(None,answer)]
    return chatbot,gr.update(value="", interactive=True)

def reset_chat(chatbot, state):
    chatbot = [(None,"请输入问题...")]
    return chatbot, gr.update(value="", interactive=True)

def add_text2(chatbot, text):
    '''
    拼接聊天记录
    '''
    chatbot = chatbot + [(text, None)]
    return chatbot, text

def add_text1(chatbot):
    '''
    拼接聊天记录
    '''
    chatbot = chatbot + [(None,"正在处理文档...")]
    return chatbot

init_embedding = False
recommender = SemanticSearch()

if __name__ == '__main__':
    title = 'PDF信息查询'
    description = """上传PDF,根据文档内容查询答案"""
    split_rules = {"按字数切分":"word_count", "按行切分":"line"}
    init_message = """欢迎使用 ChatPDF，请在上传文件后提问 """
    with gr.Blocks() as demo:
        gr.Markdown(f'<center><h1>{title}</h1></center>')
        gr.Markdown(f"<h5>{description}</h5>")

        with gr.Row():
            with gr.Column(scale=1):
                file = gr.File(label='上传pdf', file_types=['.pdf'])
                split_rule_radio = gr.Radio(["按字数切分", "按行切分"],value="按字数切分", label="切分规则")
                max_word_count = gr.Textbox(label='最大分割字数',value=300)
                topn = gr.Slider(3, 10, step=1,value=5,label="搜索数量")
            with gr.Column(scale=2):
                chatbot = gr.Chatbot([[None, init_message], [None, None]],
                                 elem_id="chat-box",
                                 show_label=False).style(height=680)
                query = gr.Textbox(show_label=False,
                               placeholder="请输入提问内容，按回车进行提交",
                               ).style(container=False)
                clear_btn = gr.Button('清空会话', elem_id='clear').style(full_width=True)
            # 触发事件
            file.upload(add_text1,inputs=[chatbot],outputs=[chatbot],queue=False).then(add_file,inputs = [file,chatbot,split_rule_radio,max_word_count],outputs=[chatbot,query])
            query.submit(add_text2,inputs=[chatbot,query],outputs=[chatbot,query],queue=False).then(question_answer,inputs=[file,query,chatbot,split_rule_radio,max_word_count,topn],outputs=[chatbot,query],queue=False)
            clear_btn.click(reset_chat, [chatbot, query], [chatbot, query])
    #openai.api_key = os.getenv('Your_Key_Here') 
    demo.launch(server_name="0.0.0.0",server_port=8888)
