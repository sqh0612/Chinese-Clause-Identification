# -*- coding: UTF-8 -*-
import math
import re

import nltk
import numpy as np
import re
import xml.dom.minidom

# 1 从xml中导出标注文本，写入到mark里，更改文件名获得不同文档
txt1 = open('Data/mark_val.txt', "r", encoding="utf-8")
txt1 = open('Data/mark_val.txt', "w", encoding="utf-8")


def writecontent(file):
    DOMTree = xml.dom.minidom.parse(file)
    Data = DOMTree.documentElement
    pras = Data.getElementsByTagName("pragraph")
    for pra in pras:
        sens = pra.getElementsByTagName('relative')
        content = pra.getAttribute("content")
        print(content)
        txt1.write(content + '\n')

## main code ##
for i in range(70, 100):  # 调整range获得训练集和测试集
    if len(str(i + 1)) < 2:
        file = "h2-data/00" + str(i + 1) + ".xml"
    elif len(str(i + 1)) == 2:
        file = "h2-data/0" + str(i + 1) + ".xml"
    else:
        file = "h2-data/" + str(i + 1) + ".xml"
    writecontent(file)
    print("第", i + 1, "篇文档输出完毕")


# 2 生成训练文本1---小句1#小句2
def splitcontent(file):
    txt2 = open('data_short.txt', "w", encoding="utf-8")
    num = 0
    with open(file, "r", encoding="utf-8") as f:
        for l in f:
            l = re.sub('\n', '', l)
            lines = re.split('[\d]+、', l)
            s = ""
            for i in range(len(lines)):
                if lines[i] == "":
                    continue
                num = num + 1
                if s == "":
                    s = lines[i]
                    continue
                output = s + "#" + lines[i] + '\n'
                output = re.sub("(#*[；。，：]#)", "#", output)
                txt2.write(output)
                s = lines[i]
    print(num)


# 3 生成训练文本2---按照句号划分的文本，一个句子包含多个小句切分点
def splitcontent2(file):
    txt2 = open('Data/data_val.txt', "w", encoding="utf-8")
    num = 0
    with open(file, "r", encoding="utf-8") as f:
        for l in f:
            l = re.sub('\n', '', l)
            lines = re.split('[\d]+、', l)
            s = ""
            for i in range(len(lines)):
                if lines[i] == "":
                    continue
                num = num + 1
                s = s + "#" + lines[i]
            s = re.sub("(#*[；，：”]#)", "#", s)
            ss = re.split("。", s)
            for ls in ss:
                if len(ls) - 1 > 0:
                    txt2.write(ls[1:] + '\n')
    print(num)

## main code ##
# splitcontent2('mark_train.txt')
splitcontent2('Data/mark_val.txt')
