import tiktoken

tiktoken_encode = tiktoken.encoding_for_model('text-embedding-ada-002')

import tiktoken

def concatenate_segments(string_list):
    concatenated_segments = []
    current_segment = ""
    total_token_count = 0

    # 加载预训练的GPT模型
    tokenizer = tiktoken.Tokenizer()

    for text in string_list:
        # 计算当前字符串的token数目
        token_count = len(tiktoken_encode(text))
        
        # 如果加上当前字符串的token数目超过300，则将当前段加入结果列表，并重新开始新的段
        if total_token_count + token_count > 300:
            concatenated_segments.append(current_segment)
            current_segment = ""
            total_token_count = 0

        # 将当前字符串添加到当前段中
        current_segment += text
        total_token_count += token_count

        # 如果当前段的长度超过100个token，将当前段的前部分去掉，以满足相邻段之间的交叉需求
        while total_token_count > 100:
            split_index = current_segment.find(" ")
            current_segment = current_segment[split_index+1:]
            total_token_count -= 1

    # 添加最后一段到结果列表
    if current_segment != "":
        concatenated_segments.append(current_segment)

    return concatenated_segments

# 测试示例
string_list = [...]  # 输入你的字符串列表
segments = concatenate_segments(string_list)
print(segments)