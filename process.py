import pandas as pd
import os
from PIL import Image
from bs4 import BeautifulSoup
import requests
from io import BytesIO
from tqdm import tqdm
from loguru import logger
import re


# 设置这个可以不打印日志
logger.remove()
# logger.add("logs/mylog.log")
def parse_page(html):
    '''
    解析html页面
    '''
    soup = BeautifulSoup(html, 'html.parser')
    parsed_text = ""
    all_text = []
    for tag in soup.find_all(['p','div','ul']):
        text = tag.get_text(strip=True).strip()
        if text in all_text:
            continue
        if len(text)> 0:
            parsed_text += text+"\n"
            all_text.append(text)
            logger.info(f"文本：{text}")
        # 判断有效图片
        for img in tag.find_all('img', src=True):
            img_link = img['src']
            if img["alt"] != "show":
                continue
            try:
                response = requests.get(img_link)
                img = Image.open(BytesIO(response.content))
            except:
                continue
            w,h = img.size
            min_size = 300
            if w < min_size or h < min_size:
                continue
            parsed_text += f"![]({img_link})\n"
            logger.info(f"图片：{img_link}")
        # 判断有效视频
        for video in tag.find_all('video'):
            video_link = video.find('source')['src']
            parsed_text += f"<video src={video_link}></video>"
            logger.info(f"视频：{video_link}")
        for a in tag.find_all('a', href=True):
            link = a['href']
            parsed_text += f"[{a.text}]({link})"
            logger.info(f"超链接：{link}")
    
    return parsed_text

# 读取紫鸟帮助中心excel数据
def read_data(path):
    """
    Reads the data from the given path.
    :param path: path to the data
    :return: dataframe
    """
    ext = os.path.splitext(path)[-1]
    if ext == '.csv':
        df = pd.read_csv(path)
    elif ext == '.xlsx':
        df = pd.read_excel(path)
    else:
        raise Exception('Unknown file extension')
    
    # 最终的文本
    corpus = ""
    # 解析前面的三列数据
    title = df.iloc[:,0]
    detail = df.iloc[:,1]
    for i in tqdm(range(len(title))):
        corpus += "问题："+title[i]+"\n"
        detail_i = detail[i]
        parsed_text = parse_page(detail_i)
        corpus += "答案："+parsed_text
        corpus += "\n\n"
    return corpus

# 解析客户来信excel数据
def excel_to_text(path):
    global templates
    global event_types

    templates = {}
    event_types = []
    df = pd.read_excel(path)
    height,width = df.shape
    print(height,width,type(df))
    event_list = []
    save_path = os.path.join(os.path.dirname(path),os.path.basename(path)[:-5]+"_corpus.txt")
    f = open(save_path,"w",encoding="utf-8")
    for i in range(height):
        event_type = df.iloc[i,0]
        template = df.iloc[i,1]
        template = re.sub('\s+', ' ', template)
        f.write(f"问题类型{i}：{event_type}\n")
        f.write(f"答案：{template.strip()}\n")
        for j in range(2,5):
            letter_sample = df.iloc[i,j]
            if letter_sample is None:
                letter_sample = ""     
            f.write("客户问题{}：{}\n".format(j-1,(letter_sample.strip()).replace('\n',' ')))
            f.write(f"答案：{template.strip()}\n")
            f.write("\n")
    f.close()     
    return event_list

if __name__ == "__main__":

    ########################################################################
    # data_path = "F:/Datasets/AIGC/kefu/紫鸟帮助中心数据.xlsx"
    # corpus = read_data(data_path)

    # with open("temp.txt", "w", encoding="utf-8") as f:
    #     f.write(corpus)

    ########################################################################
    data_path = "F:/Datasets/AIGC/智能问答/客户来信/AI可回复的客户常见问题(1) - 副本.xlsx"
    corpus = excel_to_text(data_path)
