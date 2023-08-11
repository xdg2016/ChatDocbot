from zxChatDoc.chatdoc import ChatDoc
from zxChatDoc.config import *
import gradio as gr
import os
import uuid
import numpy as np

def reset_chat(chatbot, state):
    chatbot = [(None,"请输入问题...")]
    return chatbot, gr.update(value="", interactive=True),gr.update(value="", interactive=False)

def add_text1(chatbot, text):
    '''
    拼接聊天记录
    '''
    chatbot = chatbot + [(text, None)]
    return chatbot, text

def get_last_question(chatbot):
    '''
    获取最近一次的问题
    '''
    text = None
    for his in chatbot[::-1]:
        if his[0] is not None:
            text = his[0]
            break
    return text

def add_text2(chatbot):
    '''
    拼接聊天记录
    '''
    text = get_last_question(chatbot)
    chatbot = chatbot + [(text, None)]
    return chatbot, text

def question_answer(kb_name, question,chatbot,topn):
    if kb_name is None or kb_name.strip() == "":
        chatbot += [(None,'当前没有选择知识库！')]
        return chatbot,gr.update(value="", interactive=True),kb_name
    if len(question) == 0:
        chatbot += [(None,'输入的问题为空！')]
        return chatbot,gr.update(value="", interactive=True),kb_name
    topn_result,answer = doc_chatter.query(kb_name,question,topn)
    chatbot += [(None,answer)]
    return chatbot,gr.update(value="", interactive=True),topn_result,kb_name

def requery(kb_name,chatbot,topn):
    '''
    重新生成
    '''
    if kb_name is None or kb_name.strip() == "":
        chatbot += [(None,'当前没有选择知识库！')]
        return chatbot,gr.update(value="", interactive=True),"",kb_name
    question = get_last_question(chatbot)
    if question is None:
        status = "上一个问题为None，请手动输入！"
        logger.info(status)
        chatbot += [(None,status)]
        return chatbot,gr.update(value="", interactive=True),"",kb_name
    topn_result,answer = doc_chatter.query(kb_name,question,topn)
    chatbot += [(None,answer)]
    return chatbot,gr.update(value="", interactive=True),topn_result,kb_name

def get_vs_list():
    lst_default = []
    if not os.path.exists(KB_ROOT_PATH):
        return lst_default
    lst = os.listdir(KB_ROOT_PATH)
    if not lst:
        return lst_default
    lst.sort()
    return lst_default + lst

def refresh_vs_list():
    return gr.update(choices=get_vs_list(),value= get_vs_list()[0] if len(get_vs_list()) > 0 else None)

def add_new_knowledge_base(kb_name,file,chunk_size,chatbot,progress = gr.Progress(track_tqdm=True)):
    '''
    新建知识库
    '''
    if kb_name is None or kb_name.strip() == "":
        vs_status = "知识库名称不能为空，请重新填写知识库名称"
        
    elif kb_name in get_vs_list():
        vs_status = "与已有知识库名称冲突，请重新选择其他名称后提交"
    elif file is None or file.name.strip() == "":
        vs_status = "请上传文件，当前支持格式：txt,pdf,docx,markdown"
    else:
        file_name = file.name
        logger.info("处理文档...")
        info = doc_chatter.add_vectors_base(kb_name,file.name,chunk_size)
        if info['status'] == "sucess":
            vs_status = f"知识库《{kb_name}》已创建，请开始提问！"
        else:
            vs_status = f"知识库《{kb_name}》创建失败，错误信息：{info['message']}，请重新创建！"

        logger.info(vs_status)
        chatbot+= [(None,vs_status)]
        return chatbot, gr.update(choices=get_vs_list(),value=kb_name),kb_name,None
    chatbot += [[None, vs_status]]
    return chatbot,refresh_vs_list(),kb_name,None

def del_knowledge_base(kb_name,chatbot):
    '''
    删除知识库
    '''
    if kb_name is None or kb_name.strip() == "":
        chatbot+= [(None,"知识库为空，请创建新的知识库！")]
        return chatbot,refresh_vs_list()
    info = doc_chatter.del_vectors_base(kb_name)
    if info['status'] == "success":
        vs_status = f"已删除知识库:《{kb_name}》"
    else:
        vs_status = f"{info['message']}"
    logger.info(vs_status)
    chatbot+= [(None,vs_status)]
    return chatbot,refresh_vs_list()

def change_knowledge_base(kb_name,chatbot):
    '''
    切换知识库
    '''
    if kb_name is None or kb_name.strip() == "":
        chatbot+= [(None,"已有知识库为空，请创建新的知识库！")]
        return chatbot
    # doc_chatter.load_vectors_base(kb_name)
    vs_status = f"知识库切换为:《{kb_name}》"
    logger.info(vs_status)
    chatbot+= [(None,vs_status)]
    return chatbot,kb_name

def expand_knowledge_base(kb_name,chatbot,file,max_word,progress = gr.Progress(track_tqdm=True)):
    '''
    扩充指定知识库
    '''
    
    if kb_name is None or kb_name.strip() == "":
        vs_status = "已有知识库为空，请创建新的知识库！"
    elif file is None or file.name.strip() == "":
        vs_status = "请上传文件，当前支持格式：txt,pdf,docx,markdown"
    else:
        file_name = file.name
        info = doc_chatter.expand_vectors_base(kb_name,file_name,max_word)
        if info['status'] == "sucess":
                vs_status = f"知识库《{kb_name}》已扩充完成，请开始提问！"
        else:
            vs_status = f"知识库《{kb_name}》扩充失败，原因：{info['message']}，请重新上传！"
            logger.info(vs_status)
            chatbot+= [(None,vs_status)]
            return chatbot, gr.update(choices=get_vs_list(),value=kb_name),None,kb_name
    chatbot += [[None, vs_status]]
    return chatbot,refresh_vs_list(),None,kb_name

def check_vs(chatbot):
    '''
    检查知识库文件是否正确
    '''
    vs_list = get_vs_list()
    null_vs = [] # 空文件夹
    for vs in vs_list:
        vs_path = os.path.join(KB_ROOT_PATH,vs)
        if len(os.listdir(vs_path)) < 1:
            null_vs.append(vs)
    if len(null_vs)==0:
        vs_status = "已完成知识库检查！"
    else:
        vs_status = "以下知识库为空：\n"
        for vs in null_vs:
            vs_status += f"《{vs}》\n"
        vs_status += "请选择其他知识库，或重新创建该知识库！"

    logger.info(vs_status)
    chatbot+= [(None,vs_status)]
    return chatbot

def load(chatbot):
    '''
    初始化信息
    '''
    vs_list = ["《"+vs+"》" for vs in get_vs_list()] 
    if len(vs_list) > 0:
        init_message = """欢迎使用智能问答，请选择已有知识库提问：\n"""
        for vs in vs_list:
            init_message += f"{vs}\n"
        chatbot += [(None,init_message)]
        # 检查知识库
        chatbot = check_vs(chatbot)
        chatbot += [(None,f"当前选择的知识库是：{vs_list[0]}")]
    else:
        init_message = "欢迎使用智能问答，当前没有知识库，请先创建知识库后进行提问\n"
        chatbot += [(None,init_message)]
    return uuid_num,refresh_vs_list(),chatbot

# 初始化文档查询器
doc_chatter = ChatDoc()
title = '智能问答'
description = """上传资料文档,根据文档内容查询答案"""
description2 = "1.页面刷新后会重置聊天记录\n2.如果当前问题无法回答或者回答结果不完整，可以点击【重新回答】来重新生成答案"

with gr.Blocks() as demo:
    gr.Markdown(f'<center><h1>{title}</h1></center>')
    gr.Markdown(f"<h4>{description}</h4>")
    gr.Markdown(f"{description2}")
    vs_name = gr.State(get_vs_list()[0] if len(get_vs_list()) > 0 else None)   # 记录当前用户选择是哪个知识库
    uuid_num = gr.State()
    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot([[None, None]],
                                elem_id="chat-box",
                                show_label=False).style(height=660)
            query = gr.Textbox(show_label=False,
                            placeholder="请输入提问内容，按回车进行提交",
                            ).style(container=False)
            with gr.Row():
                clear_btn = gr.Button('清空会话🗑️', elem_id='clear').style()
                requery_btn = gr.Button('重新回答🔄', elem_id='regen').style()

        with gr.Column(scale=1):
            with gr.Tab("已有知识库"):
                select_vs = gr.Dropdown(get_vs_list(),
                                            label="选择知识库",
                                            interactive=True,
                                            value=get_vs_list()[0] if len(get_vs_list()) > 0 else None
                                            )
                topn = gr.Slider(3, 10, step=1,value=TOPK,label="搜索数量")
                with gr.Row():
                    with gr.Column():
                        refresh_kb = gr.Button("刷新知识库")
                    with gr.Column():
                        del_kb = gr.Button("删除当前知识库")
                refresh_kb.click(refresh_vs_list,inputs=[],outputs=[select_vs])
                del_kb.click(fn=del_knowledge_base,
                                inputs=[select_vs, chatbot],
                                outputs=[chatbot,select_vs])
                select_vs.change(fn=change_knowledge_base,
                                        inputs=[select_vs,chatbot],
                                        outputs=[chatbot,vs_name])
                with gr.Accordion("扩充知识库",open=False) as vs_setting:
                    file_expand = gr.File(label='上传文档，当前支持：txt,pdf,docx,markdown格式', file_types=['.txt', '.md', '.docx', '.pdf'])
                    max_word_count_expand = gr.Textbox(label='最大分割字数',value=CHUNK_SIZE)
                    expand_kb = gr.Button("扩充当前知识库")
                    expand_kb.click(fn=expand_knowledge_base,
                                        inputs=[select_vs, chatbot,file_expand,max_word_count_expand],
                                        outputs=[chatbot,select_vs,file_expand,vs_name])
                    
                with gr.Accordion("查询的topn片段",open= False):
                    topn_result = gr.TextArea(show_label=False).style(container=False)
            with gr.Tab("创建知识库"):
                kb_name = gr.Textbox(label="知识库名称")
                file = gr.File(label='上传文档，当前支持：txt,pdf,docx,markdown格式', file_types=['.txt', '.md', '.docx', '.pdf'])
                # split_rule_radio = gr.Radio(["按字数切分", "按行切分"],value="按字数切分", label="切分规则")
                max_word_count = gr.Textbox(label='最大分割字数',value=CHUNK_SIZE)
                new_kb_btn = gr.Button('新建知识库')
                new_kb_btn.click(add_new_knowledge_base,inputs=[kb_name,file,max_word_count,chatbot],outputs=[chatbot,select_vs,vs_name,file])

        
    # 触发事件
    query.submit(add_text1,inputs=[chatbot,query],outputs=[chatbot,query],queue=False).then(question_answer,inputs=[select_vs,query,chatbot,topn],outputs=[chatbot,query,topn_result,vs_name],queue=False)
    clear_btn.click(reset_chat, [chatbot, query], [chatbot, query,topn_result])
    requery_btn.click(add_text2,inputs=[chatbot],outputs=[chatbot,query],queue=False).then(requery,inputs=[select_vs,chatbot,topn],outputs=[chatbot,query,topn_result,vs_name],queue=False)
    demo.load(
        fn=load,
        inputs=[chatbot],
        outputs=[uuid_num,select_vs,chatbot])
    
if __name__ == '__main__':
    #openai.api_key = os.getenv('Your_Key_Here') 
    demo.queue(concurrency_count=3).launch(server_name="0.0.0.0",server_port=8888,share=True)
    