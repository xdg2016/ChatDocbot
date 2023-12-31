import time
import json
import numpy as np
import requests
from zxChatDoc.config import logger

OPENAI_API_KEY1='e83cc1e656cf4c9090e0b6b3e13fcde3'
OPENAI_API_ENDPOINT='https://openai-10-eu-01.openai.azure.com/'

# # 2023.08.08 美东区
# OPENAI_API_KEY1 = "42f7c51ab46e45a0902f112ac89820d9"
# OPENAI_API_ENDPOINT = "https://openai-10-eu-02.openai.azure.com/"

# 2023.08.08 法国区
OPENAI_API_KEY1 = "4818c4caf78e405f9c26491a646bfdc2"
OPENAI_API_ENDPOINT = "https://openai-10-fc-01.openai.azure.com/"

def test_request(input):
    url= f'{OPENAI_API_ENDPOINT}openai/deployments/text-embedding-ada-002/embeddings?api-version=2022-12-01'
    headers_data={}
    headers_data['Content-Type']='application/json; charset=UTF-8'
    headers_data['api-key']=OPENAI_API_KEY1

    data={}
    data['input'] = input
    startt = time.time()
    res = requests.post(url, json=data, headers=headers_data, timeout=(10,30), stream=False)
    endt = time.time()
    logger.info('ada embedding use time {} s'.format(endt-startt))
    embeddings = []
    if res.headers['content-type'] == 'application/json':
        res_json = json.loads(res.text)
        if 'data' not in res_json:
          return {'error':{'message':'出错'}}
        data = res_json['data']
        if not isinstance(data, list):
          return {'error':{'message':'出错'}}
        if not data:
          return {'error':{'message':'出错'}}
        for i in range(len(data)):
            datai = data[i]
            if 'embedding' not in datai:
              return {'error':{'message':'出错'}}
            embedding = datai['embedding']
            embedding = np.asarray(embedding, dtype=np.float32)
            embedding = embedding.reshape(1,-1)
            embeddings.append(embedding)
        embeddings = np.vstack(embeddings)
        return embeddings
    else:
        for line in res.iter_lines():
          line = line.decode('utf-8')
          logger.info(line)
          continue
