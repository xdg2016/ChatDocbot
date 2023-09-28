import os
from zxChatDoc.config import *
import re
import chardet
import hashlib

def extract_text_from_pdf(file_path: str):
    """提取pdf文档的文本内容"""
    import PyPDF2
    contents = []
    with open(file_path, 'rb') as f:
        pdf_reader = PyPDF2.PdfReader(f)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            contents.append(page_text+"\n")
    return contents

def extract_text_from_txt(file_path: str):
    """提取txt内容"""
    contents = []
    doc_file_bytes = open(file_path, 'rb').read()
    _r = chardet.detect(doc_file_bytes)
    _e = _r.get('encoding').lower()
    if _e == "gb2312":
        _e = "gb18030"
    with open(file_path, 'r', encoding=_e) as f:
        f.seek(0,0)
        contents = [text for text in f.readlines() if text.strip()]
    return contents

def extract_text_from_docx(file_path: str):
    """提取DOCX文档的内容"""
    import docx
    document = docx.Document(file_path)
    contents = [paragraph.text+"\n" for paragraph in document.paragraphs if paragraph.text.strip()]
    return contents

def extract_text_from_markdown(file_path: str):
    """提取Markdown文件内容"""
    import markdown
    from bs4 import BeautifulSoup
    with open(file_path, 'r', encoding='utf-8') as f:
        markdown_text = f.read()
    html = markdown.markdown(markdown_text)
    soup = BeautifulSoup(html, 'html.parser')
    contents = [text+"\n" for text in soup.get_text().splitlines() if text.strip()]
    return contents

def get_kb_embedding_path(file_name):
    vs_name = os.path.splitext(os.path.basename(file_name))[0]
    vs_name = vs_name.replace(" ","_")
    vs_name = vs_name.replace("-","_")
    vs_name = vs_name.replace(".","_")
    vs_path = os.path.join(KB_ROOT_PATH, vs_name)
    return vs_path

def split_string(string):
    '''按照各种可能的标点，将长句切分为多个短句'''
    string = re.sub(r'([,，.。？?；;：:！!——......、\n\r\t]["’”」』]{0,2})([^,，.])', r'\1#\2', string)
    string = string.split("#")
    return string

def calculate_md5(file_path):
    '''计算文件的md5值'''
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()