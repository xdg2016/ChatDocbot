from zxChatDoc.chatdoc import ChatDoc
from zxChatDoc.config import *
import gradio as gr
import os

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
        return chatbot,gr.update(value="", interactive=True)
    if len(question) == 0:
        chatbot += [(None,'输入的问题为空！')]
        return chatbot,gr.update(value="", interactive=True)
    topn_result,answer = doc_chatter.query(kb_name,question,topn)
    chatbot += [(None,answer)]
    return chatbot,gr.update(value="", interactive=True),topn_result

def requery(kb_name,chatbot,topn):
    '''
    重新生成
    '''
    if kb_name is None or kb_name.strip() == "":
        chatbot += [(None,'当前没有选择知识库！')]
        return chatbot,gr.update(value="", interactive=True),""
    question = get_last_question(chatbot)
    if question is None:
        status = "上一个问题为None，请手动输入！"
        logger.info(status)
        chatbot += [(None,status)]
        return chatbot,gr.update(value="", interactive=True),""
    topn_result,answer = doc_chatter.query(kb_name,question,topn)
    chatbot += [(None,answer)]
    return chatbot,gr.update(value="", interactive=True),topn_result

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
    return gr.update(choices=get_vs_list())

def add_new_knowledge_base(kb_name,file,chunk_size,chatbot):
    '''
    新建知识库
    '''
    file_name = file.name
    if kb_name is None or kb_name.strip() == "":
        vs_status = "知识库名称不能为空，请重新填写知识库名称"
        
    elif kb_name in get_vs_list():
        vs_status = "与已有知识库名称冲突，请重新选择其他名称后提交"
    elif file_name is None or file_name.strip() == "":
        vs_status = "请上传文件，当前支持格式：txt,pdf,docx,markdown"
    else:
        logger.info("处理文档...")
        doc_chatter.max_word_count = int(chunk_size)
        info = doc_chatter.add_vectors_base(kb_name,file.name)
        if info['status'] == "sucess":
            vs_status = f"知识库《{kb_name}》已创建，请开始提问！"
        else:
            vs_status = f"知识库《{kb_name}》创建失败，错误信息：{info['message']}，请重新创建！"

        logger.info(vs_status)
        chatbot.pop()
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
    doc_chatter.del_vectors_base(kb_name)
    vs_status = f"已删除知识库:《{kb_name}》"
    chatbot+= [(None,vs_status)]
    return chatbot,gr.update(choices=get_vs_list(),value= get_vs_list()[0] if len(get_vs_list()) > 0 else None)

def change_knowledge_base(kb_name,chatbot):
    '''
    切换知识库
    '''
    if kb_name is None or kb_name.strip() == "":
        chatbot+= [(None,"已有知识库为空，请创建新的知识库！")]
        return chatbot
    # doc_chatter.load_vectors_base(kb_name)
    vs_status = f"知识库切换为:《{kb_name}》"
    chatbot+= [(None,vs_status)]
    return chatbot,kb_name

# 提前定义，后面就不会出现前面控件调用后面控件时出现没有定义的情况，需要配合xxx.render来使用。相当于先定义组件，后面再实时组装成界面
new_kb_btn = gr.Button('新建知识库')

# 初始化文档查询器
doc_chatter = ChatDoc()

vs_list = ["《"+vs+"》" for vs in get_vs_list()] 

title = '智能问答'
description = """上传资料文档,根据文档内容查询答案"""
split_rules = {"按字数切分":"word_count", "按行切分":"line"}
init_message = """欢迎使用智能问答，请选择已有知识库提问：\n""" if len(vs_list) > 0 else "欢迎使用智能问答，当前没有知识库，请先创建知识库后进行提问"
for vs in vs_list:
    init_message += f"{vs}\n"

with gr.Blocks() as demo:
    gr.Markdown(f'<center><h1>{title}</h1></center>')
    gr.Markdown(f"<h5>{description}</h5>")
    vs_name = gr.State(get_vs_list()[0] if len(get_vs_list()) > 0 else None)   # 记录当前用户选择是哪个知识库
    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot([[None, init_message]],
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
                    refresh_kb = gr.Button("刷新已有知识库")
                    del_kb = gr.Button("删除当前知识库")
                refresh_kb.click(refresh_vs_list,inputs=[],outputs=[select_vs])
                del_kb.click(fn=del_knowledge_base,
                                inputs=[select_vs, chatbot],
                                outputs=[chatbot,select_vs])
                select_vs.change(fn=change_knowledge_base,
                                    inputs=[select_vs,chatbot],
                                    outputs=[chatbot,vs_name])
                topn_result = gr.TextArea(label="查询的topn片段")

            with gr.Tab("创建知识库"):
                kb_name = gr.Textbox(label="知识库名称")
                file = gr.File(label='上传文档，当前支持：txt,pdf,docx,markdown格式', file_types=['.txt', '.md', '.docx', '.pdf'])
                # split_rule_radio = gr.Radio(["按字数切分", "按行切分"],value="按字数切分", label="切分规则")
                max_word_count = gr.Textbox(label='最大分割字数',value=CHUNK_SIZE)
                new_kb_btn.render()
                new_kb_btn.click(add_new_knowledge_base,inputs=[kb_name,file,max_word_count,chatbot],outputs=[chatbot,select_vs,vs_name,file])

    # 触发事件
    query.submit(add_text1,inputs=[chatbot,query],outputs=[chatbot,query],queue=False).then(question_answer,inputs=[vs_name,query,chatbot,topn],outputs=[chatbot,query,topn_result],queue=False)
    clear_btn.click(reset_chat, [chatbot, query], [chatbot, query,topn_result])
    requery_btn.click(add_text2,inputs=[chatbot],outputs=[chatbot,query],queue=False).then(requery,inputs=[vs_name,chatbot,topn],outputs=[chatbot,query,topn_result],queue=False)
    demo.load(
        fn=refresh_vs_list,
        inputs=None,
        outputs=[select_vs])
    
if __name__ == '__main__':
    #openai.api_key = os.getenv('Your_Key_Here') 
    demo.launch(server_name="0.0.0.0",server_port=8888)
    