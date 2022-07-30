import csv
import re

f = open("Data/train.csv", "w", encoding="utf-8", newline="")

# 1. 基于文件对象构建 csv写入对象
data = csv.writer(f)

# 2. 构建列表头
data.writerow(["data", "label"])
with open("Data/mark.txt", 'r', encoding='utf-8') as f:
    for l in f:
        pattern = re.compile(r'([^(，|。)]+“.*”[^(，|。)]+[，|。]|[^(，|。)]+[^“”，。][^(，|。)]+[，|。])')
        lines = pattern.findall(l)
        for i in range(len(lines)):
            newline = re.sub('[\d]+、', "", lines[i])
            if i != len(lines) - 1:
                x = str(re.search('[\d]+、', lines[i + 1]))
                if x == "None":
                    data.writerow([newline[0:-1], "0"])
                    # data.writerow([lines[i], "1"])
                else:
                    data.writerow([newline[0:-1], "1"])
                    # data.writerow([lines[i], "0"])
            else:
                data.writerow([newline[0:-1], "1"])

