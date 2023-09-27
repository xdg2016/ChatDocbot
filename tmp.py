import pandas as pd
import os

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
        f.write(event_type+"\n")
        template = df.iloc[i,1]
        for j in range(2,5):
            letter_sample = df.iloc[i,j]
            if letter_sample is None:
                letter_sample = ""     
            f.write("客户问题{}：{}\n".format(j-1,(letter_sample.strip()).replace('\n',' ')))
            f.write(f"回复模板：{template.strip()}\n")
            f.write("\n")
    f.close()     
    return event_list


if __name__ == "__main__":
    excel_path = "F:/Datasets/AIGC/kefu/AI可回复的客户常见问题(1) - 副本.xlsx"

    excel_to_text(excel_path)