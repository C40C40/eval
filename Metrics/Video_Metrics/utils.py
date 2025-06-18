import random
# utils.py 追加
# ---------------------------------------------------------------------
# 多模态 VLM 调用（基于阿里云 DashScope 的 OpenAI-compatible 接口）
# ---------------------------------------------------------------------
import os, base64, mimetypes, re
from pathlib import Path
from openai import OpenAI
import json


_DASHSCOPE_URL  = "https://dashscope.aliyuncs.com/compatible-mode/v1"
_DASHSCOPE_KEY  = "sk-464b1aac3c244f949d2bfcc925d99f82"      # 请提前在环境变量里配置
_VLM_MODEL_NAME = "qwen-vl-max-latest"                 # 可替换成别的模型

MOCK_GPT_API_KEY =  "sk-Ss10Now9JrhuGXaeF959CfD08bA546E28c7d60537a82FcE5"
MOCK_GPT_API_URL = "http://47.88.65.188:8405/v1/chat/completions"

_client = OpenAI(api_key=_DASHSCOPE_KEY, base_url=_DASHSCOPE_URL)


# utils.py
# ---------------------------------------------------------------------
# 多模态 VLM 调用：传入【本地路径 或 http(s) url】，统一搞定
# ---------------------------------------------------------------------

_client = OpenAI(api_key=_DASHSCOPE_KEY, base_url=_DASHSCOPE_URL)


def _to_data_uri(local_path: str) -> str:
    """把本地图片转成 data URI"""
    p = Path(local_path)
    if not p.is_file():
        raise FileNotFoundError(local_path)
    mime = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
    with p.open("rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f"data:{mime};base64,{b64}"

def _read_text_file(file_path):
    """辅助函数：读取文本文件内容"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        return text
    except Exception as e:
        print(f"读取文件 {file_path} 时发生错误：{e}")
        return None

def call_gpt_api(system_prompt, query_prompt, temperature=0.2):
    # 在实际应用中，这里会进行HTTP请求到GPT API
    # 例如：
    import requests

    headers = {"Authorization": f"Bearer {MOCK_GPT_API_KEY}", "Content-Type": "application/json"}
    data = {
        "model": "gpt-4o", # 或其他模型
        "temperature": temperature, 
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query_prompt}
        ]
     }
    response = requests.post(MOCK_GPT_API_URL, headers=headers, json=data)
    content_json = response.json()["choices"][0]["message"]["content"]
    clean = re.sub(r"^```[\w]*\n?|```$", "", content_json.strip(), flags=re.MULTILINE)
    return json.loads(clean)


def call_vlm_api(img_path_or_url: str,
                 prompt_text: str,
                 *,
                 system_prompt: str = "You are a helpful assistant.") -> str:
    """
    参数
    ----
    img_path_or_url : str
        • 本地路径 e.g. 'C:/ppt/slide1.png' / './slide1.jpg'<br>
        • 或已在公网可访问的 http(s) URL
    prompt_text : str
        对图片的指令，如“请用一句话总结幻灯片要点”
    """
    if not _DASHSCOPE_KEY:
        raise RuntimeError("环境变量 DASHSCOPE_API_KEY 未设置")

    # ---------- 1. 若是本地文件，自动转为 data URI ----------
    if re.match(r"^https?://", img_path_or_url, flags=re.I):
        img_payload = img_path_or_url          # http 直接用
    else:
        img_payload = _to_data_uri(img_path_or_url)

    # ---------- 2. 调用模型 ----------
    try:
        resp = _client.chat.completions.create(
            model=_VLM_MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_prompt}],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": img_payload}},
                        {"type": "text",  "text": prompt_text},
                    ],
                },
            ],
        )
        return resp.choices[0].message.content
    except Exception as e:
        print(f"[VLM] 调用失败: {e}")
        return None

    
if __name__ == "__main__":
    img = "C:/Python/Learn_accompany_3/Resources/PPT_image/1/1.png"
    summary = call_vlm_api(img, "请用中文一句话描述这张幻灯片中的的主要内容")
    print("模型返回：", summary)
