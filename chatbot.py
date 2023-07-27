import gradio as gr
import random
import time
from chatPdf import load_recommender,recommender,generate_answer
import os
# Chatbot demo with multimodal input (text, markdown, LaTeX, code blocks, image, audio, & video). Plus shows support for streaming text.

def add_text(history, text):
    '''
    拼接聊天记录
    '''
    history = history + [(text, None)]
    return history, gr.update(value="", interactive=True)


def add_file(file,history):
    '''
    上传文件
    '''
    old_file_name = file.name
    file_name = file.name
    file_name = file_name[:-12] + file_name[-4:]
    if not os.path.exists(file_name):
        os.rename(old_file_name, file_name)
    print("处理文档...")
    load_recommender(file_name)
    print("处理文档结束！")
    history = history + [("文档上传成功,开始对话", None)]
    return history


def bot(history):
    '''
    生成回复
    '''
    response = generate_answer(history[-1][0])
    history[-1][1] = ""
    for character in response:
        history[-1][1] += character
        # time.sleep(0.0001)
        yield history


title = "智能客服demo"
with gr.Blocks() as demo:
    gr.Markdown(f'<center><h1>{title}</h1></center>')
    chatbot = gr.Chatbot([], elem_id="chatbot").style(height=720)

    with gr.Row():
        with gr.Column(scale=0.85):
            txt = gr.Textbox(
                show_label=False,
                placeholder="Enter text and press enter, or upload an image",
            ).style(container=False)
        with gr.Column(scale=0.15, min_width=0):
            btn = gr.UploadButton("📁", file_types=["pdf"])

    txt_msg = txt.submit(add_text, [chatbot, txt], [chatbot, txt], queue=False).then(
        bot, chatbot, chatbot
    )
    txt_msg.then(lambda: gr.update(interactive=True), None, [txt], queue=False)
    file_msg = btn.upload(add_file, [btn,chatbot],[chatbot], queue=False)
demo.queue()
demo.launch(server_port=7777)