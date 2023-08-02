from zxChatDoc.chatdoc import ChatDoc
import gradio as gr
import os

def add_file(file,chatbot):
    '''
    上传文件
    '''
    print("处理文档...")
    doc_chatter.init_vectors_base(file.name)
    info = "文档处理结束，请开始提问！"
    print(info)
    chatbot.pop()
    chatbot+= [(None,info)]
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

def question_answer(file, question,chatbot,topn):
    if file is None or (not os.path.exists(file.name)):
        chatbot += [(None,'没有文档信息，请上传文档后再查询！')]
        return chatbot,gr.update(value="", interactive=True)
    if len(question) == 0:
        chatbot += [(None,'[错误]: 输入的问题为空！')]
        return chatbot,gr.update(value="", interactive=True)
    if not doc_chatter.init_vectors:
        print("处理文档...")
        doc_chatter.init_vectors_base(file.name)
    answer = doc_chatter.query(question,topn)
    chatbot += [(None,answer)]
    return chatbot,gr.update(value="", interactive=True)

if __name__ == '__main__':

    # 初始化文档查询器
    doc_chatter = ChatDoc()

    title = '智能问答'
    description = """上传资料文档,根据文档内容查询答案"""
    split_rules = {"按字数切分":"word_count", "按行切分":"line"}
    init_message = """欢迎使用智能问答，请在上传文件后提问 """
    with gr.Blocks() as demo:
        gr.Markdown(f'<center><h1>{title}</h1></center>')
        gr.Markdown(f"<h5>{description}</h5>")

        with gr.Row():
            with gr.Column(scale=1):
                file = gr.File(label='上传文档，当前支持：txt,pdf,docx,markdown格式', file_types=['.txt', '.md', '.docx', '.pdf'])
                # split_rule_radio = gr.Radio(["按字数切分", "按行切分"],value="按字数切分", label="切分规则")
                max_word_count = gr.Textbox(label='最大分割字数(只在上传文档初始化时有用)',value=300)
                topn = gr.Slider(3, 10, step=1,value=5,label="搜索数量")
            with gr.Column(scale=2):
                chatbot = gr.Chatbot([[None, init_message], [None, None]],
                                 elem_id="chat-box",
                                 show_label=False).style(height=660)
                query = gr.Textbox(show_label=False,
                               placeholder="请输入提问内容，按回车进行提交",
                               ).style(container=False)
                clear_btn = gr.Button('清空会话', elem_id='clear').style(full_width=True)

        # 触发事件
        file.upload(add_text1,inputs=[chatbot],outputs=[chatbot],queue=False).then(add_file,inputs = [file,chatbot],outputs=[chatbot,query])
        query.submit(add_text2,inputs=[chatbot,query],outputs=[chatbot,query],queue=False).then(question_answer,inputs=[file,query,chatbot,topn],outputs=[chatbot,query],queue=False)
        clear_btn.click(reset_chat, [chatbot, query], [chatbot, query])
    #openai.api_key = os.getenv('Your_Key_Here') 
    demo.launch(server_name="0.0.0.0",server_port=8888)
    