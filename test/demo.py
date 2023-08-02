import gradio as gr
import os
import re
import docx2txt
import PyPDF2
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
import fitz
from zxChatDoc.openai_api_chat import OPENAI_API_KEY1,test_request
import json

split_rules = {"段落切分":"paragraph", "按字数切分":"word_count", "按行切分":"line"}

def preprocess_text(text):
    # 去除重复的标点符号
    text = re.sub(r'\.{2,}', '', text)

    return text

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

def split_document(file_path, split_rule):
    # 根据给定的分割规则切分文档并保持语义连贯性
    if file_path.endswith('.pdf'):
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ''
            for page in reader.pages:
                text += page.extract_text()
    elif file_path.endswith('.docx'):
        text = docx2txt.process(file_path)
    elif file_path.endswith('.txt') or file_path.endswith('.md'):
        with open(file_path, 'r') as file:
            text = file.read()
    document_parts = []
    split_rule = split_rules[split_rule]
    # 根据切割规则切分文档
    if split_rule == 'paragraph':
        document_parts = text.split('\n\n')
    elif split_rule == 'word_count':
        # 以每100个单词为一段进行切分
        words = text.split()
        document_parts = [' '.join(words[i:i+100]) for i in range(0, len(words), 100)]
    elif split_rule == 'line':
        document_parts = text.split('\n')
    
    # 对每个部分进行预处理
    document_parts = [preprocess_text(part) for part in document_parts]
    # 使用---break---连接连贯的内容
    document_parts = ['---break---' if len(part) == 0 else part for part in document_parts]

    return document_parts

def generate_document_vectors(document_parts):
    # 使用BERT模型生成文档的语义向量
    model = SentenceTransformer('distilbert-base-nli-mean-tokens')
    document_vectors = model.encode(document_parts)

    return document_vectors

def retrieve_top_matches(question_vector, document_vectors, document_parts):
    # 根据问题向量与文档向量计算余弦相似度，取出最相似的三个段落
    similarities = cosine_similarity([question_vector], document_vectors)[0]
    top_indices = np.argsort(similarities)[-3:][::-1]
    top_paragraphs = [document_parts[i] for i in top_indices]

    return top_paragraphs

def extract_answer(question, paragraphs):
    # 使用ChatGPT模型从段落中提取问题的答案
    prompt = ""
    prompt += f"问题：'{question}\n\n'，"
    # 长文本
    paragraph = f"###\n我提供的长文本："
    for c in paragraphs:
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
    
    return answers

def process_document(file_path, split_rule, question):
    # 文档处理的整体流程
    document_parts = split_document(file_path, split_rule)
    document_vectors = generate_document_vectors(document_parts)
    question_vector = generate_document_vectors([preprocess_text(question)])[0]
    top_paragraphs = retrieve_top_matches(question_vector, document_vectors, document_parts)
    answers = extract_answer(question, top_paragraphs)

    return answers

def process_files(file, split_rule, question):
    # 处理上传的文件
    old_file_name = file.name
    file_name = file.name
    file_name = file_name[:-12] + file_name[-4:]
    if not os.path.exists(file_name):
        os.rename(old_file_name, file_name)
    answers = process_document(file_name, split_rule, question)

    return answers

def main(file, split_rule, question):
    # 主函数，用于构建Gradio界面
    answers = process_files(file, split_rule, question)
    return answers

iface = gr.Interface(
    fn=main, 
    inputs=[
        gr.inputs.File(label="上传文件", type="file"),
        gr.inputs.Radio(["段落切分", "按字数切分", "按行切分"], label="切分规则"),
        gr.inputs.Textbox("输入问题")
    ],
    outputs="text",
    title="文档检索功能演示",
    description="上传公司资料文档，并提问相关问题，获取答案。",
    allow_flagging=False
)
iface.launch()


'''
chatGPT生成的示例代码
'''