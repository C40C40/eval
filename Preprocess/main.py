"""
preprocess_pipeline_test.py
---------------------------
快速验证 5 个预处理函数能否协同工作。

目录结构示例
.
├─ tests/
│  ├─ srt/               # 演示 SRT
│  │    └─ demo.srt
│  ├─ video/             # 演示视频（名字与 SRT 同基名）
│  │    └─ demo.mp4
│  └─ ppt/               # 演示 PPT 视频 & SRT
│       ├─ lecture.mp4
│       └─ lecture.srt
└─ …你的源代码…
"""

import os

# ====== 你的函数 ======
from preprocess import *
# ---------- 测试配置 ----------
TEST_DIR = "C:/Python/Learn_accompany_3/Resources/srt"

SRT_DIR        = "C:/Python/Learn_accompany_3/Resources/srt"
VIDEO_DIR      = "C:/Python/Learn_accompany_3/Resources/video"
PPT_VIDEO      = "C:/Python/Learn_accompany_3/Resources/PPT_video/"
PPT_SRT        = "C:/Python/Learn_accompany_3/Resources/srt/"
# ---------- 临时输出目录 ----------
TXT_OUT_DIR        = "C:/Python/Learn_accompany_3/Resources/txt"
AUDIO_TEXT_OUT_DIR = "C:/Python/Learn_accompany_3/Resources/sample"
PPT_IMG_OUT_DIR    = "C:/Python/Learn_accompany_3/Resources/PPT_image"

def main():

    # 1) SRT ➜ TXT
    print("\n[1] 测试 preprocess_srt_to_txt()")
    preprocess_srt_to_txt(str(SRT_DIR), str(TXT_OUT_DIR))

    # 2) 视频 + SRT ➜ 片段抽取（音频 + 文本）
    print("\n[2] 测试 process_all_srt_and_videos()")
    process_all_srt_and_videos(
        srt_folder=str(SRT_DIR),
        video_folder=str(VIDEO_DIR),
        output_base_folder=str(AUDIO_TEXT_OUT_DIR),
    )

    # 3) PPT 翻页检测 + SRT 对齐
    print("\n[3] 测试 extract_ppt_clips_and_srt_clips()")
    process_all_ppt_srt(
        video_dir=str(PPT_VIDEO),
        srt_dir=str(PPT_SRT),
        ppt_image_output_base=str(PPT_IMG_OUT_DIR),
    )


if __name__ == "__main__":
    main()
