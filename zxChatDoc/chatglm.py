from transformers import AutoModel, AutoTokenizer
import os
import json

sysinfo_path = os.path.join(os.path.dirname(__file__),"sysinfo.json")
with open(sysinfo_path, 'r',encoding="utf-8") as f:
    sysinfo = json.load(f)['info']

class ChatGlm():
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("/hy-tmp/chatGLM2_6B", trust_remote_code=True)
        self.model = AutoModel.from_pretrained("/hy-tmp/chatGLM2_6B", trust_remote_code=True).cuda()
        self.model = self.model.eval()
        self.template = sysinfo

    def predict(self,input,paragraph):
        self.template += paragraph
        input = self.template +"\n\n"+input
        response, history, = self.model.chat(self.tokenizer, input, history=[],max_length = 8192)

        return response



