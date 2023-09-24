import time
import json
import os
import requests
from zxChatDoc.config import logger,USERINFO_TEMPLATE,SYSINFO_TEMPLATE


OPENAI_API_KEY1='e83cc1e656cf4c9090e0b6b3e13fcde3'
OPENAI_API_REGION='eastus'
OPENAI_API_ENDPOINT='https://openai-10-eu-01.openai.azure.com/'
OPENAI_API_VERSION='2022-12-01'  # 已删除
OPENAI_API_VERSION='2023-03-15-preview'
OPENAI_API_VERSION='2023-05-15'
#OPENAI_API_VERSION='2023-06-13'

# # 2023.08.08 美东区
# OPENAI_API_KEY1 = "42f7c51ab46e45a0902f112ac89820d9"
# OPENAI_API_ENDPOINT = "https://openai-10-eu-02.openai.azure.com/"

# 2023.08.08 法国区
OPENAI_API_KEY1 = "4818c4caf78e405f9c26491a646bfdc2"
OPENAI_API_ENDPOINT = "https://openai-10-fc-01.openai.azure.com/"

# ai-lab-gpt35-text  ai-lab-gpt-4 ai-lab-gpt-4-32k
MODELS=['ai-lab-gpt-35-turbo','gpt-35-turbo-16k','ai-lab-gpt-4', 'ai-lab-gpt-4-32k']

def test_request(text,paragraph):
    logger.info('请求chatgpt...')
    url= f'{OPENAI_API_ENDPOINT}openai/deployments/{MODELS[1]}/chat/completions?api-version={OPENAI_API_VERSION}'
    # logger.info(url)

    headers_data={}
    headers_data['Content-Type']='application/json; charset=UTF-8'
    headers_data['api-key']=OPENAI_API_KEY1

    data={}
    messages = []
    # 系统消息
    sysinfo = SYSINFO_TEMPLATE.format(paragraph)
    userinfo =  USERINFO_TEMPLATE + paragraph + text
    # messages.append({'role':'system', 'content':sysinfo})
    messages.append({'role':'user', 'content':userinfo})
    
    data['messages'] = messages
    startt = time.time()
    res = requests.post(url, json=data, headers=headers_data, timeout=9999, stream=True)
    endt = time.time()
    logger.info('请求chatgpt结束!')
    logger.info('ChatGPT use time {} s'.format(endt-startt))
    if res.headers['content-type'] == 'application/json':
        # logger.info(res.text)
        return res.text
    else:
        for line in res.iter_lines():
          line = line.decode('utf-8')
          logger.info(line)
          continue