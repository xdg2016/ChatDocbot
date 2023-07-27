import pandas as pd
import numpy as np
import time
import json

import requests

OPENAI_API_KEY1='e83cc1e656cf4c9090e0b6b3e13fcde3'
OPENAI_API_REGION='eastus'
OPENAI_API_ENDPOINT='https://openai-10-eu-01.openai.azure.com/'
OPENAI_API_VERSION='2022-12-01'  # 已删除
OPENAI_API_VERSION='2023-03-15-preview'
OPENAI_API_VERSION='2023-05-15'


event_types = {}
df = None
def excel_to_text(path):
    '''
    解析excel文件
    '''
    global event_types
    global df
    event_types = {}
    df = pd.read_excel(path)
    height,width = df.shape
    print(height,width,type(df))
    letters = []
    for i in range(height):
        event_type = df.iloc[i,0]
        event_types[event_type] == i
        letter_sample = df.iloc[i,2]
        if letter_sample is None:
            letter_sample = ""
        letter_sample2 = df.iloc[i,3]
        if letter_sample2 is None:
            letter_sample2 = ""
        letter_sample3 = df.iloc[i,4]
        if letter_sample3 is None:
            letter_sample3 = ""
        letters.extend([letter_sample, letter_sample2,letter_sample3])
    return letters

excel_path = "F:/Datasets/AIGC/kefu/AI可回复的客户常见问题(1).xlsx"
letters = excel_to_text(excel_path)

MODELS=['ai-lab-gpt-35-turbo','gpt-35-turbo-16k','ai-lab-gpt-4', 'ai-lab-gpt-4-32k']
url= f'{OPENAI_API_ENDPOINT}openai/deployments/{MODELS[1]}/chat/completions?api-version={OPENAI_API_VERSION}'
print(url)

headers_data={}
headers_data['Content-Type']='application/json; charset=UTF-8'
headers_data['api-key']=OPENAI_API_KEY1

default_prompt = '''请根据客户来信判断客户的意图。客户的意图主要有列表中的这几种["不充电/不工作（数显款充电宝）","不能被充（数显款充电宝）","不能充手机（数显款）","充电宝hold不住电","不工作（数据线）"].'''\
                '''客户意图如果可以匹配到列表中列出的，请输出和列表中完全一样的意图字符串，如果没有匹配到类别中的意图，则输出'无法判断'。'''\
                '''结果输出到json中的content字段。以下是客户来信：'''\
                '''Hi I was given a iniu B1-B5 Xmas 2021 I understand that it has a 3 year warranty, unfortunately the item will not charge anymore, can you assist with this'''

data={}
messages = []
messages.append({'role':'user', 'content':})
for i in range(len(letters)):
    prompt = ""
    try:
        print("chatGPT组织答案...")
        _retry = True
        _rerty_count = 0
        while _retry:
            _retry = False
            try:
                messages.append({'role':'user', 'content':})
                data['messages'] = messages

                startt = time.time()
                res = requests.post(url, json=data, headers=headers_data, timeout=9999, stream=True)
                endt = time.time()
                print('use time {} s'.format(endt-startt))
                if res.headers['content-type'] == 'application/json':
                    print(res.text)
                    answer_text = json.loads(res.text)['choices'][0]['message']['content']

                else:
                    for line in res.iter_lines():
                        print(line)
            except Exception as e:
                if _rerty_count < 10:
                    _retry = True
                    _rerty_count += 1
                    continue
        
    except Exception as e:
        print(e)
        return "请求ChatGPT错误"
