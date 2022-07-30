import json
import os
import re
import glob
import torch
from tqdm import tqdm

# 感兴趣的标点符号列表：
punctuations_list = list("，。：；")


def preprocess(split, file):  # 忽略逗号和句号
    data_root = "./"  # TODO: change back!
    sentences = []
    with open(file, "r", encoding="utf-8") as f:
        for l in f:
            sentences.append(l)
    output_root = os.path.join(data_root, "json_files")
    output_filename = split + ".json"
    os.makedirs(output_root, exist_ok=True)
    punc_cnt = torch.zeros((2,))

    with open(os.path.join(output_root, output_filename), 'w') as f:
        f.write('{\n    "data": [\n')
        for i, s in enumerate(tqdm(sentences)):
            f.write('        {\n')
            puncless_text = []
            punc = []
            for c in s:
                if c in punctuations_list:
                    continue
                elif c != '#':
                    puncless_text.append(c)
                    punc.append(0)
                    punc_cnt[0] += 1
                else:
                    punc[-1] = 1
                    punc_cnt[0] -= 1
                    punc_cnt[1] += 1

            f.write('            "tokens": ' + json.dumps(list(puncless_text)) + ',\n')
            f.write('            "punc": ' + json.dumps(punc) + '\n')
            f.write('        }')

            if i != len(sentences) - 1:
                f.write(',')
            f.write('\n')
        f.write('    ]\n}\n')

    print("punc_cnt:", punc_cnt)


def preprocess2(split, file):  # 不忽略逗号和句号，将未切分的逗号句号等标为“0”
    data_root = "./"  # TODO: change back!
    sentences = []
    with open(file, "r", encoding="utf-8") as f:
        for l in f:
            sentences.append(l)
    output_root = os.path.join(data_root, "json_files")
    output_filename = split + ".json"
    os.makedirs(output_root, exist_ok=True)
    punc_cnt = torch.zeros((2,))

    with open(os.path.join(output_root, output_filename), 'w') as f:
        f.write('{\n    "data": [\n')
        for i, s in enumerate(tqdm(sentences)):
            f.write('        {\n')
            puncless_text = []
            punc = []
            for c in s:
                if c != '#':
                    puncless_text.append(c)
                    punc.append(0)
                    punc_cnt[0] += 1
                else:
                    punc[-1] = 1
                    punc_cnt[0] -= 1
                    punc_cnt[1] += 1

            f.write('            "tokens": ' + json.dumps(list(puncless_text)) + ',\n')
            f.write('            "punc": ' + json.dumps(punc) + '\n')
            f.write('        }')

            if i != len(sentences) - 1:
                f.write(',')
            f.write('\n')
        f.write('    ]\n}\n')

    print("punc_cnt:", punc_cnt)


if __name__ == '__main__':
    # preprocess('train', "./data_train.txt")#忽略逗号和句号
    # preprocess('val', "./data_val.txt")  #忽略逗号和句号
    preprocess2('train2', "Data/data_train.txt")
    preprocess2('val2', "Data/data_val.txt")
    # preprocess('val')
