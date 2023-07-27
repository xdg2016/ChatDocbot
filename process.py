import pandas as pd


df = None
def excel_to_text(excel_path,save_path):
    '''
    解析excel文件
    '''
    global event_types
    global df
    event_types = {}
    df = pd.read_excel(excel_path)
    height,width = df.shape
    print(height,width,type(df))
    letters = []
    f = open(save_path,"w",encoding="utf-8")
    for i in range(height):
        event_type = df.iloc[i,0]
        template = df.iloc[i,1]
        letter_sample = df.iloc[i,2]
        if letter_sample is None:
            letter_sample = ""
        letter_sample2 = df.iloc[i,3]
        if letter_sample2 is None:
            letter_sample2 = ""
        letter_sample3 = df.iloc[i,4]
        if letter_sample3 is None:
            letter_sample3 = ""
        letters.extend([event_type,template,letter_sample, letter_sample2,letter_sample3])
        f.write("\n\n客户表达问题："+event_type+"\n")
        f.write("客户来信1："+letter_sample +"\n")
        f.write("客户来信2："+letter_sample2+"\n")
        f.write("模板: "+template+"\n")
    f.close()
    return letters

excel_path = "F:/Datasets/AIGC/kefu/客诉 - 副本.xlsx"
save_path = "corpus3.txt"
letters = excel_to_text(excel_path,save_path)
