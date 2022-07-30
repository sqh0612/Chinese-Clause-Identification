# torch
# coding:utf-8
import torch
from torch.utils.data import DataLoader
# huggingface
from transformers import AutoTokenizer, AutoModelForTokenClassification, AdamW, get_scheduler
from datasets import load_dataset
# miscellaneous
from tqdm import tqdm
import os
# mine
from dataset import PuncDataset

def punctuate_text(text, label, task_type="punc"):
    # marks = ["", "/"] if task_type == "punc" else ["", ] + list("，。、？！：；")
    marks = ["", "/"]
    return "".join([c + marks[lab] for c, lab in zip(text, label)])


def demo(model, tokenizer, text, task_type="punc"):
    inputs = tokenizer(text, padding=True, return_tensors='pt')
    inputs = {k:inputs[k].to(device) for k in inputs}
    logits = model(**inputs)['logits']
    preds = logits.argmax(dim=-1)
    output = []
    if type(text) is list:
        for s, pred in zip(text, preds):
            output.append(punctuate_text(s, pred[1:-1].cpu().tolist(), task_type=task_type))
        return output
    else:
        return punctuate_text(text, preds[0, 1:-1].cpu().tolist(), task_type=task_type)


if __name__ == '__main__':
    text = [
        "浦东开发开放是一项振兴上海建设现代化经济贸易金融中心的跨世纪工程因此大量出现的是以前不曾遇到过的新情况新问题。对此，浦东不是简单的采取“干一段时间，等积累了经验以后再制定法规条例”的做法，而是借鉴发达国家和深圳等特区的经验教训，聘请国内外有关专家学者，积极、及时地制定和推出法规性文件，使这些经济活动一出现就被纳入法制轨道。去年初浦东新区诞生的中国第一家医疗机构药品采购服务中心，正因为一开始就比较规范，运转至今，成交药品一亿多元，没有发现一例回扣。",
    ]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
    task_type = "punc" # "punc" or "punc_type"
    model = torch.load('./training_logs/model_11_punc_F1/checkpoints/best.pth')
    result = demo(model, tokenizer, text, task_type=task_type)
    print(result)
    