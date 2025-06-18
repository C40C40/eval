
#!/usr/bin/env python
"""
retrieval_pipeline.py
---------------------
• 输入:  sample.csv （第一列为知识点名称；可追加描述列做语义检索）
• 用法:
    python retrieval_pipeline.py sample.csv "查询文本"
• 输出:
    - Top‑10 相关知识点
    - 每一步耗时（秒）

依赖:
    pip install sentence-transformers faiss-cpu rank-bm25 rapidfuzz
"""

import csv, sys, time, math, os, numpy as np

from rank_bm25 import BM25Okapi
from rapidfuzz import process
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss


# ---------- 预处理 ----------
def load_names(csv_path):
    names = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if row and row[0].strip():
                names.append(row[0].strip())
    if not names:
        raise RuntimeError("CSV 为空或未找到第一列数据")
    return names


def build_bm25(names):
    # 中文可直接按字符切分；若有分词器可改为 list(token)
    corpus = [list(name) for name in names]
    return BM25Okapi(corpus)


def build_faiss(names, model):
    embeds = model.encode(names, batch_size=256, show_progress_bar=False, normalize_embeddings=True)
    dim = embeds.shape[1]
    index = faiss.IndexHNSWFlat(dim, 32)          # HNSW, M=32
    index.hnsw.efConstruction = 200
    faiss.normalize_L2(embeds)
    index.add(embeds.astype(np.float32))
    return index


# ---------- 拼写纠错 ----------
def spell_correct(query, names, cutoff=70):
    # 取最相近知识点名作为纠错参考
    cand, score, _ = process.extractOne(query, names, score_cutoff=cutoff)
    return cand if cand else query


# ---------- RRF 融合 ----------
def rrf(bm_ids, ann_ids, k=60, topn=50):
    scores = {}
    for r, idx in enumerate(bm_ids[:topn]):
        scores[idx] = scores.get(idx, 0) + 1.0 / (k + r + 1)
    for r, idx in enumerate(ann_ids[:topn]):
        scores[idx] = scores.get(idx, 0) + 1.0 / (k + r + 1)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [idx for idx, _ in ranked]


# ---------- 主流程 ----------
def main():
    if len(sys.argv) < 3:
        print("用法: python retrieval_pipeline.py sample.csv \"查询文本\"")
        sys.exit(1)

    csv_path, query = sys.argv[1], " ".join(sys.argv[2:]).strip()

    # — Step0: 载入与建模 —
    names = load_names(csv_path)

    # Sentence-BERT (多语模型，视需求替换中文 bge-base-zh-v1)
    sbert = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    bm25 = build_bm25(names)
    faiss_index = build_faiss(names, sbert)

    # Cross‑Encoder 精排
    cross_enc = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    timings = {}

    # — Step1: 拼写纠错 —
    t0 = time.perf_counter()
    corrected = spell_correct(query, names)
    timings["spell_correct"] = time.perf_counter() - t0

    # — Step2: BM25 召回 —
    t1 = time.perf_counter()
    bm_scores = bm25.get_scores(list(corrected))
    bm_top = np.argsort(bm_scores)[::-1][:50]       # 取前50
    timings["bm25_search"] = time.perf_counter() - t1

    # — Step3: 向量检索 —
    t2 = time.perf_counter()
    q_emb = sbert.encode([query], normalize_embeddings=True)
    faiss.normalize_L2(q_emb)
    ann_dist, ann_idx = faiss_index.search(q_emb.astype(np.float32), 50)
    ann_top = ann_idx[0]
    timings["ann_search"] = time.perf_counter() - t2

    # — Step4: RRF 融合 —
    t3 = time.perf_counter()
    fused = rrf(bm_top.tolist(), ann_top.tolist())
    top_for_rerank = fused[:50]
    timings["rrf_fusion"] = time.perf_counter() - t3

    # — Step5: Cross‑Encoder 精排 —
    t4 = time.perf_counter()
    pairs = [[query, names[idx]] for idx in top_for_rerank]
    ce_scores = cross_enc.predict(pairs)
    final_idx = [idx for idx, _ in sorted(zip(top_for_rerank, ce_scores), key=lambda x: x[1], reverse=True)][:10]
    timings["cross_encoder"] = time.perf_counter() - t4

    # — 输出 —
    print(f"Query: {query}")
    print(f"Corrected query: {corrected}")
    print("\nTop‑10 results:")
    for i, idx in enumerate(final_idx, 1):
        print(f"{i:2d}. {names[idx]}")

    print("\nTimings (seconds):")
    for k, v in timings.items():
        print(f"{k:<15}: {v:0.4f}")


if __name__ == "__main__":
    main()
