# coding: utf-8
import os
import sys
import argparse
import platform
import numpy as np
import torch
import pandas as pd
import re
import json


import transformers

from transformers import LlamaTokenizer, LlamaForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM


generation_config = dict(
    temperature=0.7,
    top_k=40,
    top_p=0.6,
    do_sample=False,
    num_beams=1,
    repetition_penalty=1.0,
    # no_repeat_ngram_size=4,
    # encoder_no_repeat_ngram_size=4,
    max_new_tokens=1024
)


# model_path = '/home/do/ssd/iohub/finetuner/qwen2.5-r1/ft_step2400_epoch0.7'
model_path = '/home/do/ssd/modelscope/hub/models/Qwen/Qwen2.5-3B-Instruct'

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    torch_dtype=torch.bfloat16,
    device_map='auto',
    trust_remote_code=True,
)

model.eval()

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <reasoning> </reasoning> and <answer> </answer> tags, respectively, i.e., "
    "<reasoning> reasoning process here </reasoning><answer> answer here </answer>"
    "Notice that: Respond directly using numerical values only within the <answer>...</answer> block. Do not include any textual descriptions inside the answer block."
)


#######################################################################################
# 使用pyarrow引擎
df = pd.read_parquet('test-00000-of-00001.parquet', engine='pyarrow')
question_list = df["question"]
answer_list = df["answer"]
print(len(question_list))

def extract_last_number(text: str):
    """
    使用正则表达式提取字符串中最后一个数值（浮点数或整数）。

    :param text: 待处理的字符串。
    :return: 最后一个数值的字符串形式，如果未找到则返回 None。
    """
    number_pattern = r'[+-]?\d+(?:\.\d+)?|[+-]?\.\d+'
    
    matches = re.findall(number_pattern, text)

    if matches:
        return matches[-1] # 返回找到的最后一个匹配项
    else:
        return None
    
def num_eq(astr, bstr):
    try:
        if int(astr.strip()) == int(bstr.strip()):
            return True
    except Exception as e:
        pass

    try:
        if float(astr.strip()) == float(bstr.strip()):
            return True
    except Exception as e:
        pass
    return False

total = 0
true_total = 0
fw = open("temp/result.jsonl", 'w', encoding='utf-8')
for question, answer in zip(question_list, answer_list):
    if total > 200: break
    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': question},  # q1
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    model_inputs = tokenizer([text], return_tensors="pt").to('cuda')
    length = model_inputs.input_ids.shape[1]
    generation_output = model.generate(
        input_ids = model_inputs.input_ids,
        **generation_config
    )


    output_ids = generation_output.cpu().numpy()[0][length:].tolist()
    output = tokenizer.decode(output_ids, skip_special_tokens=True)

    gold = answer.strip().split('####')[1].replace(',', '').replace('$', '').strip()
    match = re.search(r'<answer>(.*?)</answer>', output)
    flag = False
    print("="*50, f"expect: {gold}")
    if match:
        answer_content = match.group(1).strip()  # 提取并去除前后空格
        last_number = extract_last_number(answer_content)
        if answer_content == gold or str(last_number) == gold or num_eq(last_number, gold):
            true_total = true_total + 1
            flag = True
        else:
            pass
    print("num:", total)
    print("true_total:", true_total)
    print(output)
    if flag:
        print(f'[ok] |{answer_content}| |{gold}|')
    total = total + 1

    data = {"question": question, "answer": answer, "output": output, "good":flag}
    fw.write(json.dumps(data, ensure_ascii = False) + '\n')
    fw.flush()

print("*"*100)
print(true_total/total)