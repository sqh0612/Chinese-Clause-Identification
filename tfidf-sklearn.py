import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, recall_score, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import jieba
import warnings
warnings.filterwarnings('ignore')

# 1 数据准备
# csv文件的数据用逗号分隔, 中文encoding='gb2312'
data_train = pd.read_csv('data_process/Data/train.csv', sep=',', encoding='utf-8')
print()
print("(Training Data，Attribute) = ", data_train.shape)
print()
df1 = data_train.iloc[:, 0:]
print("Original Training Data : No. of 0 samples = {}, No. of 1 samples = {}".format(
    df1.loc[df1.iloc[:, -1] == 0, :].shape[0], df1.loc[df1.iloc[:, -1] == 1, :].shape[0]))
print()

# 提取训练集中的文本内容
train_sentences = data_train['data']
# 提取训练集中的划分标签
label = data_train['label']
# 把中文转换成英文格式以便构建特征向量（分词处加空格）
df_train = pd.DataFrame({'text': train_sentences})
df_train['train_split'] = df_train['text'].apply(lambda x: " ".join(jieba.cut(x, cut_all=False)))
# 导入停词库，含标点
stop = open('data_process/Data/stop_words.txt', encoding='utf-8').read().splitlines()


# 2 文本特征工程 —— 把文本转换成向量 —— TF-IDF模型
tf = TfidfVectorizer(
    analyzer='word',            # 以词为单位进行分析
    ngram_range=(1, 4),
    stop_words=stop,            # TF-IDF的原理相当于自动去除停用词
    max_features=150000
)
tf.fit(df_train['train_split'])
# 将训练集随机拆分为新的训练集和验证集
x_train, x_val, y_train, y_val = train_test_split(df_train['train_split'], label, test_size=0.30, random_state=50)
# 把训练集和验证集中的每一个词都进行特征工程，变成向量
x_train = tf.transform(x_train)
x_val = tf.transform(x_val)


# 3 构建分类器算法，对词袋模型处理后的文本进行机器学习和数据挖掘
# 网格搜索功能进行超参数的批量试验后，从所有参数中挑出能够使模型在验证集上预测准确率最高的
# 3.1 朴素贝叶斯
param_grid_bay = {'alpha': [0.1, 0.5, 1.0]}
skf_bay = StratifiedKFold(n_splits=5, random_state=50, shuffle=True)
bay = MultinomialNB()
grid_bay = GridSearchCV(estimator=bay, param_grid=param_grid_bay, cv=skf_bay)
grid_bay.fit(x_train, y_train)
bay_final = grid_bay.best_estimator_
y_pred_bay = bay_final.predict(x_val)
tn_bay, fp_bay, fn_bay, tp_bay = confusion_matrix(y_val, y_pred_bay).ravel()
print("TF-IDF -- Naive Bayes")
print("Best paramaters = {}".format(grid_bay.best_params_))
print("True negative = {}, False Positive = {}, False Negative = {}, True Positive = {}".format(tn_bay, fp_bay, fn_bay, tp_bay))
print("Accuracy score = {}".format(accuracy_score(y_val, y_pred_bay)))
print("Recall score = {}".format(recall_score(y_val, y_pred_bay)))
print("AUC ROC score = {}".format(roc_auc_score(y_val, y_pred_bay)))
print("F1 score = {}".format(f1_score(y_val, y_pred_bay)))
print()
# print("词袋特征提取--朴素贝叶斯分类器验证集上的预测准确率 = {}".format(bay_final.score(x_val, y_val)))

# 3.2 逻辑回归
param_grid_lg = {'C': range(1, 10), 'dual': [True, False]}
skf_lg = StratifiedKFold(n_splits=5, random_state=50, shuffle=True)
lgGS = LogisticRegression()
grid_lg = GridSearchCV(estimator=lgGS, param_grid=param_grid_lg, cv=skf_lg, n_jobs=-1)   # cv=5为5折交叉验证
grid_lg.fit(x_train, y_train)
lg_final = grid_lg.best_estimator_
y_pred_lg = lg_final.predict(x_val)
tn_lg, fp_lg, fn_lg, tp_lg = confusion_matrix(y_val, y_pred_lg).ravel()
print("TF-IDF -- Logical Regression")
print("Best paramaters = {}".format(grid_lg.best_params_))
print("True negative = {}, False Positive = {}, False Negative = {}, True Positive = {}".format(tn_lg, fp_lg, fn_lg, tp_lg))
print("Accuracy score = {}".format(accuracy_score(y_val, y_pred_lg)))
print("Recall score = {}".format(recall_score(y_val, y_pred_lg)))
print("AUC ROC score = {}".format(roc_auc_score(y_val, y_pred_lg)))
print("F1 score = {}".format(f1_score(y_val, y_pred_lg)))
print()
# print('词袋特征提取--逻辑回归分类器验证集上的预测准确率 = {}'.format(lg_final.score(x_val, y_val)))


# # 输出结果
# # 使用TF-IDF对测试集中的文本进行特征工程
# test_X = tf.transform(data_test['Phrase'])
# # 对测试集中的文本，使用lg_final逻辑回归分类器进行预测
# predictions = lg_final.predict(test_X)
# # 将预测结果加在测试集中
# data_test.loc[:, 'Sentiment'] = predictions
# # 按Kaggle比赛官网上的要求整理格式
# final_data = data_test.loc[:, ['PhraseId', 'Sentiment']]
# final_data.to_csv('Result/submission_lg.csv', index=None)