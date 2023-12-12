import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import re
import string


def __list_files__(path: str) -> [str]:
    """
    列出文件夹下所有文件
    :param path: 文件路径
    :return: 文件名列表
    """
    tmp = os.listdir(path)
    files = list(map(lambda x: path + "/" + x, tmp))
    return files


def text_labels(path: str) -> {str: int}:
    """
    构造文本和标签的键值对
    :param path: 文件路径
    :return: 文本和标签键值对
    """
    text_labels = {}
    files = __list_files__(path)
    file_cnt = len(files)
    for i in range(file_cnt):
        sub_file = os.listdir(files[i])
        new_path = list(map(lambda x: files[i] + "/" + x, sub_file))
        for file in new_path:
            with open(file, 'r', encoding='utf-8', errors='ignore') as f:
                tmp = ""
                for line in f:
                    if not line == '':
                        tmp += line.strip('\n') + ' '
                text_labels[tmp] = i

    return text_labels


def __remove_stop_words__(text: str, stop_words: set) -> str:
    """
    去除停用词
    :param text: 需要去除的文本
    :param stop_words: 停用词集合
    :return: 去除停用词后的文本
    """
    words = text.split()  # 得到句子中的单词列表
    filtered_words = [word for word in words if word.lower() not in stop_words]  # 得到不属于停用词的单词
    filtered_text = ' '.join(filtered_words)  # 将得到的词拼接成句子
    return filtered_text


def __text_preprocessing__(text: str, stop_words: set, lemmatizer: WordNetLemmatizer()) -> str:
    """
    文本预处理
    :param text: 需要预处理的文本
    :param stop_words: 停用词集合
    :param lemmatizer: 词根
    :return:
    """
    text = __remove_stop_words__(text, stop_words)
    text = re.sub(r'[^(a-zA-Z)\s]', '', text)  # 去除特殊字符
    text = text.translate(str.maketrans('', '', string.punctuation))  # 去除标点符号
    text = text.lower()  # 变为小写字母

    text = lemmatizer.lemmatize(text, pos='v')  # fastText不需要
    return text

def process_all_text(text_label: dict) -> ([str], [int]):
    """
    提取文本和标签
    :param text_label: 文本和标签的键值对
    :return: 字符串数组和标签数组
    """
    # nltk.download('stopwords') # 第一次运行时需要下载
    # nltk.download('wordnet') # 第一次运行时需要下载
    # nltk.download('omw-1.4') # 第一次运行时需要下载
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    X = []
    y = []
    for text, label in text_label.items():
        tmp_text = __text_preprocessing__(text, stop_words, lemmatizer)
        X.append(tmp_text)
        y.append(label)
    y = np.array(y)
    return X, y


def tfidf(text: [str]) -> np.array:
    """
    使用tfidf将文本向量化
    :param text: 文本数组
    :return: 向量化后内容
    """
    tfidf_vectorizer = TfidfVectorizer()
    # 将文本转换为TF-IDF编码
    tfidf_matrix = tfidf_vectorizer.fit_transform(text)
    tfidf_matrix = tfidf_matrix.toarray()
    return tfidf_matrix
