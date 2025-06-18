import jieba
import gensim
from gensim import corpora
import numpy as np
import json
import random
import re
from openai import OpenAI
from tqdm import tqdm
from typing import List, Dict, Any

# 模拟 GPT API 的密钥和URL
MOCK_GPT_API_KEY =  "sk-Ss10Now9JrhuGXaeF959CfD08bA546E28c7d60537a82FcE5"
MOCK_GPT_API_URL = "http://47.88.65.188:8405/v1/chat/completions"
MOCK_EMBEDDING_API_URL = "http://47.88.65.188:8405/v1"


def generate_embeddings_openai(
    texts: List[str],
    batch_size: int = 16,
    api_key: str = MOCK_GPT_API_KEY,
    base_url: str = MOCK_EMBEDDING_API_URL,  # 设置你的代理地址
    model: str = "text-embedding-ada-002"
) -> np.ndarray:
    """
    使用自定义 API 地址调用 OpenAI 接口生成嵌入。

    参数:
        texts: 要编码的文本列表。
        batch_size: 每批请求的文本条数。
        api_key: OpenAI API 密钥。
        base_url: 自定义 OpenAI API 地址（如代理服务器）。
        model: 嵌入模型名称（如 text-embedding-3-small）。

    返回:
        np.ndarray: shape=(len(texts), embedding_dim)
    """
    if not texts:
        return np.array([])

    client = OpenAI(base_url=base_url, api_key=api_key)
    embeddings = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding batches"):
        batch = texts[i:i + batch_size]
        batch = [text.replace("\n", " ") for text in batch]

        try:
            response = client.embeddings.create(
                input=batch,
                model=model
            )
            batch_embeddings = [np.array(data.embedding) for data in response.data]
            embeddings.append(np.vstack(batch_embeddings))
        except Exception as e:
            print(f"API 调用失败（第 {i // batch_size + 1} 批）: {e}")
            return np.array([])

    all_embeddings = np.vstack(embeddings)
    return all_embeddings

def generate_ngrams(words, n):
    """生成N-gram"""
    ngrams = []
    for i in range(len(words) - n + 1):
        ngrams.append(tuple(words[i:i + n]))
    return ngrams

def preprocess_text_for_topic(text):
    """分词并可选择性地加入停用词过滤"""
    words = jieba.cut(text)
    return list(words)

def create_dictionary_and_corpus(texts):
    """创建词典和语料库"""
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    return dictionary, corpus

def train_lda_model(corpus, dictionary, num_topics):
    """训练LDA模型"""
    lda = gensim.models.ldamodel.LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        random_state=100,
        update_every=1,
        passes=10
    )
    return lda

def get_topic_distribution(lda_model, bow):
    """获取主题分布"""
    num_topics = lda_model.num_topics
    distribution = [0.0] * num_topics
    for topic_id, prob in lda_model.get_document_topics(bow):
        distribution[topic_id] = prob
    return distribution

def calculate_kl_deviation(base_distribution, window_distribution):
    """计算KL散度作为偏离度"""
    if len(base_distribution) != len(window_distribution):
        raise ValueError(f"分布长度不匹配: {len(base_distribution)} vs {len(window_distribution)}")

    epsilon = 1e-10
    return np.sum([base_distribution[i] * np.log((base_distribution[i] + epsilon) / (window_distribution[i] + epsilon))
                   for i in range(len(base_distribution))])

import os

def _read_text_file(file_path):
    """辅助函数：读取文本文件内容"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        return text
    except Exception as e:
        print(f"读取文件 {file_path} 时发生错误：{e}")
        return None

def call_gpt_api(text_content: str, system_prompt_path: str, query_prompt_path: str, temperature=0.2):
    """模拟调用 GPT API，返回内容逻辑性和解释详细性评分"""
    # 在实际应用中，这里会进行HTTP请求到GPT API
    # 例如：
    import requests

    system_prompt_content = _read_text_file(system_prompt_path)
    query_prompt_content = _read_text_file(query_prompt_path)

    if system_prompt_content is None or query_prompt_content is None:
        print("无法读取提示文件内容，GPT API 调用失败。")
        return None

    headers = {"Authorization": f"Bearer {MOCK_GPT_API_KEY}", "Content-Type": "application/json"}
    data = {
        "model": "gpt-4o", # 或其他模型
        "temperature": temperature, 
        "messages": [
            {"role": "system", "content": system_prompt_content},
            {"role": "user", "content": f"{query_prompt_content}\n\n文本内容：{text_content}"}
        ]
     }
    response = requests.post(MOCK_GPT_API_URL, headers=headers, json=data)
    content_json = response.json()["choices"][0]["message"]["content"]
    clean = re.sub(r"^```[\w]*\n?|```$", "", content_json.strip(), flags=re.MULTILINE)
    return json.loads(clean)
