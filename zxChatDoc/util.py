import os
from zxChatDoc.config import *

def extract_text_from_pdf(file_path: str):
    """提取pdf文档的文本内容"""
    import PyPDF2
    contents = []
    with open(file_path, 'rb') as f:
        pdf_reader = PyPDF2.PdfReader(f)
        for page in pdf_reader.pages:
            page_text = page.extract_text().strip()
            raw_text = [text.strip() for text in page_text.splitlines() if text.strip()]
            new_text = ''
            for text in raw_text:
                new_text += " " +text
                if text[-1] in ['.', '!', '?', '。', '！', '？', '…', ';', '；', ':', '：', '”', '’', '）', '】', '》', '」',
                                '』', '〕', '〉', '》', '〗', '〞', '〟', '»', '"', "'", ')', ']', '}']:
                    contents.append(new_text)
                    new_text = ''
            if new_text:
                contents.append(new_text)
    return contents

def extract_text_from_txt(file_path: str):
    """提取txt内容"""
    contents = []
    with open(file_path, 'r', encoding='utf-8') as f:
        contents = [text.strip() for text in f.readlines() if text.strip()]
    return contents

def extract_text_from_docx(file_path: str):
    """提取DOCX文档的内容"""
    import docx
    document = docx.Document(file_path)
    contents = [paragraph.text.strip() for paragraph in document.paragraphs if paragraph.text.strip()]
    return contents

def extract_text_from_markdown(file_path: str):
    """提取Markdown文件内容"""
    import markdown
    from bs4 import BeautifulSoup
    with open(file_path, 'r', encoding='utf-8') as f:
        markdown_text = f.read()
    html = markdown.markdown(markdown_text)
    soup = BeautifulSoup(html, 'html.parser')
    contents = [text.strip() for text in soup.get_text().splitlines() if text.strip()]
    return contents

def get_kb_embedding_path(file_name):
    vs_name = os.path.splitext(os.path.basename(file_name))[0]
    vs_name = vs_name.replace(" ","_")
    vs_name = vs_name.replace("-","_")
    vs_name = vs_name.replace(".","_")
    vs_path = os.path.join(KB_ROOT_PATH, vs_name)
    return vs_path