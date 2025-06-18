import os
import jieba
import jieba.posseg as pseg
from collections import Counter
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import json

# 从 utils.py 导入辅助函数
from utils import (
    generate_ngrams,
    preprocess_text_for_topic,
    create_dictionary_and_corpus,
    train_lda_model,
    get_topic_distribution,
    calculate_kl_deviation,
    generate_embeddings_openai,  # 导入新的 OpenAI Embedding 生成函数
    call_gpt_api # 导入新的 GPT API 调用函数
)


class ContentMetrics:
    def __init__(self, text_path: str):
        if not os.path.exists(text_path):
            raise FileNotFoundError(f"文件未找到: {text_path}")
        self.text_path = text_path
        self.text_content = self._read_text_file(text_path)
        if self.text_content:
            # 移除换行符和空格，以便进行N-gram和填充词检测
            self.processed_text_for_ngram = self.text_content.replace('\n', '').replace(' ', '')
        else:
            self.processed_text_for_ngram = ""

    def _read_text_file(self, file_path):
        """读取文本文件内容"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            return text
        except Exception as e:
            print(f"读取文件时发生错误：{e}")
            return None

    def analyze_ngrams(self, n_values=[1, 2, 3, 4, 5]):
        """分析文本的N-gram"""
        if not self.processed_text_for_ngram:
            return {}
        words = list(jieba.cut(self.processed_text_for_ngram))  # 使用jieba进行分词
        ngram_results = {}
        for n in n_values:
            # 调用 utils 中的 generate_ngrams
            ngrams = generate_ngrams(words, n)
            ngram_counts = Counter(ngrams)
            ngram_results[f'{n}-gram'] = ngram_counts.most_common(10)  # 返回最常见的10个N-gram

            if n == 5:
                total_5_grams = len(ngrams)
                # 计算重复的5-gram的总个数 (出现次数大于1的5-gram的计数总和)
                repeated_5_grams_count = sum(count for count in ngram_counts.values() if count > 1)
                
                if total_5_grams > 0:
                    repeated_5_gram_ratio = repeated_5_grams_count / total_5_grams
                else:
                    repeated_5_gram_ratio = 0.0
                ngram_results['5-gram_repeated_ratio'] = f"{repeated_5_gram_ratio:.2%}"

        return ngram_results

    def detect_filler_words(self):
        """检测填充词"""
        if not self.processed_text_for_ngram:
            return {'filler_word_counts': [], 'filler_word_percentage': "0.00%"}
        filler_words = [
        '嗯', '啊', '哦', '额', '呃', '唔', '啊这', '那个', '这个',
        '怎么说呢', '等一下', '我想想', '怎么讲'
        ]

        words = list(jieba.cut(self.processed_text_for_ngram))
        filler_counts = Counter(word for word in words if word in filler_words)
        total_words = len(words)

        filler_percentage = (sum(filler_counts.values()) / total_words * 100) if total_words > 0 else 0

        return {
            'filler_word_counts': filler_counts.most_common(),
            'filler_word_percentage': f"{filler_percentage:.2f}%"
        }

    def calculate_information_density(self):
        """计算信息密度（实词比例）"""
        if not self.text_content:
            return {'real_words_count': 0, 'total_words': 0, 'information_density_percentage': "0.00%"}
        real_word_pos_tags = {'n', 'v', 'a', 'd', 'm', 'q', 'b'}

        words_with_pos = pseg.cut(self.text_content)
        total_words = 0
        real_words_count = 0

        for word, flag in words_with_pos:
            if word.strip():  # 排除空字符串
                total_words += 1
                if flag in real_word_pos_tags:
                    real_words_count += 1

        information_density = (real_words_count / total_words * 100) if total_words > 0 else 0

        return {
            'real_words_count': real_words_count,
            'total_words': total_words,
            'information_density_percentage': f"{information_density:.2f}%"
        }

    def calculate_elaboration_density(self):
        """计算阐释词的密度"""
        if not self.processed_text_for_ngram:
            return {'elaboration_word_counts': [], 'elaboration_word_percentage': "0.00%"}

        # 定义阐释词列表
        elaboration_words = [
        '也就是说', '换句话说', '说白了', '其实就是', '也即是', '即', '就是说', '简单来说', '归根到底',
        '举个例子', '比如说', '例如', '比如', '举例来说',
        '本质上', '从根本上', '归根结底', '实际上', '从某种意义上说', '实际来说', '说到底',
        '总的来说', '换个角度看', '归纳一下', '综合来看', '总结一下',
        '因此', '所以', '于是', '因而', '因此可以看出', '由此说明'
        ]


        words = list(jieba.cut(self.processed_text_for_ngram))
        elaboration_counts = Counter(word for word in words if word in elaboration_words)
        total_words = len(words)

        elaboration_percentage = (sum(elaboration_counts.values()) / total_words * 100) if total_words > 0 else 0

        return {
            'elaboration_word_counts': elaboration_counts.most_common(),
            'elaboration_word_percentage': f"{elaboration_percentage:.2f}%"
        }

    # ------------------ 1) 连贯性 ------------------
    def calculate_text_coherence(
            self,
            window_size: int = 5,
            step_size: int = 3,
            *,
            threshold: float = 0.60):
        """返回 {'windows': [...], 'low_ratio': xx}"""
        if not self.text_content:
            return {"windows": [], "low_ratio": 0.0}

        model = SentenceTransformer('aspire/acge_text_embedding')
        lines = [ln.strip() for ln in self.text_content.split('\n') if ln.strip()]

        segs = [' '.join(lines[i:i+window_size])
                for i in range(0, len(lines)-window_size+1, step_size)]
        if not segs:
            return {"windows": [], "low_ratio": 0.0}

        embs = model.encode(segs, normalize_embeddings=True)
        windows = []
        for i in range(len(embs)-1):
            score = float(np.dot(embs[i], embs[i+1]))
            windows.append({'start_line': i*step_size+1, 'score': score})

        low_ratio = sum(w['score'] < threshold for w in windows) / len(windows)
        return {"windows": windows, "low_ratio": low_ratio}
    def calculate_text_coherence_gpt(
            self,
            window_size: int = 5,
            step_size: int = 3,
            *,
            threshold: float = 0.85
        ):
        """
        使用 GPT embedding 计算文本连贯性。
    
        返回:
            dict: {'windows': [...], 'low_ratio': xx}
        """
        if not self.text_content:
            return {"windows": [], "low_ratio": 0.0}
    
        lines = [ln.strip() for ln in self.text_content.split('\n') if ln.strip()]
        segs = [' '.join(lines[i:i + window_size])
                for i in range(0, len(lines) - window_size + 1, step_size)]
    
        if not segs:
            return {"windows": [], "low_ratio": 0.0}
    
        embs = generate_embeddings_openai(
            texts=segs
        )
    
        windows = []
        for i in range(len(embs) - 1):
            a, b = embs[i], embs[i + 1]
            score = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
            windows.append({
                'start_line': i * step_size + 1,
                'score': score
            })
    
        low_ratio = sum(w['score'] < threshold for w in windows) / len(windows)
        return {"windows": windows, "low_ratio": low_ratio}

    # ------------------ 2) Embedding 偏离 ------------------
    def calculate_embedding_deviation(
            self,
            *,
            window_size: int = 20,
            slide_step: int = 10,
            threshold: float = 0.50):
        """返回 {'windows': [...], 'high_ratio': xx}"""
        if not self.text_content:
            return {"windows": [], "high_ratio": 0.0}

        model = SentenceTransformer('aspire/acge_text_embedding')
        lines = [ln.strip() for ln in self.text_content.split('\n') if ln.strip()]
        full_emb = model.encode([' '.join(lines)], normalize_embeddings=True)[0]

        windows = []
        for i in range(0, len(lines)-window_size+1, slide_step):
            seg_emb = model.encode(
                [' '.join(lines[i:i+window_size])], normalize_embeddings=True)[0]
            dev = 1 - cosine_similarity([full_emb], [seg_emb])[0][0]
            windows.append({'start_line': i+1,
                            'end_line': i+window_size,
                            'deviation': dev})

        high_ratio = sum(w['deviation'] > threshold for w in windows) / len(windows)
        return {"windows": windows, "high_ratio": high_ratio}
    
    def calculate_embedding_deviation_gpt(
            self,
            *,
            window_size: int = 20,
            slide_step: int = 10,
            threshold: float = 0.15
        ):
        """
        使用 GPT embedding 计算嵌入偏离度。
    
        返回:
            dict: {'windows': [...], 'high_ratio': xx}
        """
        if not self.text_content:
            return {"windows": [], "high_ratio": 0.0}
    
        lines = [ln.strip() for ln in self.text_content.split('\n') if ln.strip()]
        full_text = ' '.join(lines)
    
        # 生成整体文本的嵌入
        full_emb = generate_embeddings_openai(
            texts=[full_text],
            batch_size=1
        )[0]
    
        windows = []
        segment_texts = [' '.join(lines[i:i+window_size])
                         for i in range(0, len(lines)-window_size+1, slide_step)]
    
        # 批量生成 segment 嵌入
        seg_embs = generate_embeddings_openai(
            texts=segment_texts
        )
    
        for idx, seg_emb in enumerate(seg_embs):
            dev = 1 - float(np.dot(full_emb, seg_emb) /
                            (np.linalg.norm(full_emb) * np.linalg.norm(seg_emb)))
            windows.append({
                'start_line': idx * slide_step + 1,
                'end_line': idx * slide_step + window_size,
                'deviation': dev
            })
    
        high_ratio = sum(w['deviation'] > threshold for w in windows) / len(windows)
        return {"windows": windows, "high_ratio": high_ratio}

    # ------------------ 3) Topic 偏离 ------------------
    def analyze_topic_deviation(
            self,
            *,
            window_size: int = 20,
            slide_step: int = 10,
            num_topics: int = 2,
            threshold: float = 0.30):
        """返回 {'windows': [...], 'high_ratio': xx}"""
        if not self.text_content:
            return {"windows": [], "high_ratio": 0.0}

        lines = self.text_content.split('\n')
        proc = [preprocess_text_for_topic(l) for l in lines if l.strip()]
        if not proc:
            return {"windows": [], "high_ratio": 0.0}

        dictionary, corpus = create_dictionary_and_corpus(proc)
        if not corpus:
            return {"windows": [], "high_ratio": 0.0}

        lda = train_lda_model(corpus, dictionary, num_topics)
        base_dist = get_topic_distribution(
            lda, dictionary.doc2bow([w for ln in proc for w in ln]))

        windows = []
        for i in range(0, len(lines)-window_size+1, slide_step):
            win_proc = [preprocess_text_for_topic(l)
                        for l in lines[i:i+window_size] if l.strip()]
            if not win_proc:
                continue
            win_bow = dictionary.doc2bow([w for ln in win_proc for w in ln])
            dev = calculate_kl_deviation(
                base_dist, get_topic_distribution(lda, win_bow))
            windows.append({'start_line': i+1, 'end_line': i+window_size, 'deviation': dev})

        high_ratio = sum(w['deviation'] > threshold for w in windows) / len(windows) if windows else 0.0
        return {"windows": windows, "high_ratio": high_ratio}


    def evaluate_with_gpt(self, system_prompt: str, query_prompt: str):
        if not self.text_content:
            print("文本内容为空，无法调用 GPT API。")
            return None

        resp = call_gpt_api(self.text_content, system_prompt, query_prompt)
        print(f"[GPT] 调用成功: {resp}")
        # 假设接口返回 JSON 字符串或 dict
        try:
            data = resp.json() if hasattr(resp, "json") else resp  # 若 call_gpt_api 已直接返 dict
        except Exception as e:
            print(f"[GPT] 解析失败: {e}")
            return None
        return data




if __name__ == "__main__":
    txt_dir = "C:/Python/Learn_accompany_3/Resources/txt"   # ← 文件夹，而非单个文件
    
    for txt_path in Path(txt_dir).glob("*.txt"):
        print("\n" + "="*60)
        print(f"【分析文件】{txt_path.name}")
        print("="*60)

        try:
            metrics_analyzer = ContentMetrics(txt_path.as_posix())
        except Exception as e:
            print(f"[跳过] 初始化失败: {e}")
            continue
        
        # --- N-gram -------------------------------------------------------
        print("\n--- N-gram 分析 ---")
        for n_type, res in metrics_analyzer.analyze_ngrams().items():
            print(f"{n_type}: {res}")

        # --- 填充词 --------------------------------------------------------
        print("\n--- 填充词检测 ---")
        filler = metrics_analyzer.detect_filler_words()
        print("填充词统计:", filler["filler_word_counts"])
        print("填充词占比:", filler["filler_word_percentage"])

        # --- 信息密度 ------------------------------------------------------
        print("\n--- 信息密度检测 ---")
        dens = metrics_analyzer.calculate_information_density()
        print(f"实词 {dens['real_words_count']}/{dens['total_words']}  "
              f"→ {dens['information_density_percentage']}")

        # --- 阐释词 --------------------------------------------------------
        print("\n--- 阐释词密度检测 ---")
        elab = metrics_analyzer.calculate_elaboration_density()
        print("阐释词统计:", elab["elaboration_word_counts"])
        print("阐释词占比:", elab["elaboration_word_percentage"])

        # --- 连贯性 --------------------------------------------------------
        coh = metrics_analyzer.calculate_text_coherence_gpt()
        print("\n--- 连贯性分析 ---")
        print(f"连贯性窗口得分:{coh['windows']}")
        print(f"★ 连贯性低(<0.85) 比率: {coh['low_ratio']:.2%}")

        # --- Embedding 偏离 -------------------------------------------------
        emb = metrics_analyzer.calculate_embedding_deviation_gpt()
        print("\n--- Embedding 偏离分析 ---")
        print(f"偏离度窗口: {emb['windows']}")
        print(f"★ Embedding 偏离高(>0.15) 比率: {emb['high_ratio']:.2%}")



        #--- Topic 偏离 -----------------------------------------------------
        topic = metrics_analyzer.analyze_topic_deviation()
        print(f"★ Topic 偏离高(>0.30) 比率: {topic['high_ratio']:.2%}")
        
        # --- GPT 评估（可选，避免额度浪费可注释）---------------------------
        system_prompt = "C:/Python/Learn_accompany_3/Metrics/Content_Metrics/system_prompt_logic.txt"
        query_prompt  = "C:/Python/Learn_accompany_3/Metrics/Content_Metrics/query_prompt_logic.txt"
        gpt = metrics_analyzer.evaluate_with_gpt(system_prompt, query_prompt)
        if gpt:
           print("\n--- GPT 评估 ---")
           print("内容逻辑性评分:", gpt.get("content_logic_score"))
           print("解释详细性评分:", gpt.get("explanation_detail_score"))
        
