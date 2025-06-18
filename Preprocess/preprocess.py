from ast import Continue
import re
import chardet
import os
import av
import lameenc
import numpy as np
import os
import re
import chardet
import os, re, chardet, shutil
from pathlib import Path
from moviepy.editor import VideoFileClip
import cv2 # 导入 opencv-python
from skimage.metrics import structural_similarity as ssim # 导入 scikit-image


def ts_to_seconds(h, m, s, ms):
    return h*3600 + m*60 + s + ms/1000
def _detect_encoding(path: Path) -> str:
    raw = path.read_bytes()[:10000]
    enc = (chardet.detect(raw)["encoding"] or "utf-8").lower()
    # 处理 BOM
    if enc.startswith("utf-8"):
        enc = "utf-8-sig"
    return enc

def _safe_name(name: str) -> str:
    return re.sub(r'[\\/:*?"<>|]', "_", name)

def preprocess_srt_to_txt(input_dir: str, output_dir: str):
    """
    批量处理指定目录下的所有SRT文件，将其转换为纯文本文件并保存到另一个目录。

    参数:
    - input_dir: 包含SRT文件的输入目录路径。
    - output_dir: 保存纯文本文件的输出目录路径。
    """
    def preprocess_text(in_path: str, out_path: str):
        """
        从SRT字幕文件中提取文本内容，并保存为纯文本文件。
        过滤掉时间戳、序号和空行。

        参数:
        - in_path: 输入SRT文件的路径。
        - out_path: 输出纯文本文件的路径。
        """
        try:
            # 检测编码
            with open(in_path, "rb") as f_raw:
                raw = f_raw.read(10000) # 读取前10000字节用于编码检测
            enc = chardet.detect(raw)["encoding"]
            print(f"File encoding detected as: {enc}")

            lines = []
            with open(in_path, encoding=enc, errors="ignore") as fin:
                for line in fin:
                    s = line.strip()
                    # 过滤掉数字行（序号）、包含'-->'的行（时间戳）和空行
                    # 注意：re.fullmatch(r'\\d+', s) 中的 \\d+ 是为了匹配纯数字行，例如字幕序号
                    if re.fullmatch(r'\d+', s) or '-->' in s or not s:
                        continue
                    lines.append(s)

            # 将所有有效行拼接成一行，用换行符分隔
            text = "\n".join(lines)

            with open(out_path, "w", encoding="utf-8") as fout:
                fout.write(text)
            print(f"Wrote {len(text)} chars to {out_path}")

        except FileNotFoundError:
            print(f"错误: 文件未找到 - {in_path}")
        except Exception as e:
            print(f"处理文件时发生错误: {e}")
    if not os.path.exists(input_dir):
        print(f"错误: 输入目录不存在 - {input_dir}")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")

    print(f"\n--- 开始批量处理SRT文件 ---")
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")

    for filename in os.listdir(input_dir):
        if filename.endswith(".srt"):
            in_path = os.path.join(input_dir, filename)
            # 构建输出文件名，将 .srt 扩展名替换为 .txt
            out_filename = filename.replace(".srt", ".txt")
            out_path = os.path.join(output_dir, out_filename)

            print(f"\n正在处理文件: {filename}")
            preprocess_text(in_path, out_path)

    print(f"\n--- 批量处理完成 ---")


def extract_samples_from_video_and_srt(
    srt_path: str,
    video_path: str,
    output_path: str,
    target_clip_sec: int = 60,
    max_clips: int = 5,
) -> None:
    """
    产物层级:
      output_path/
        └─ <video_basename>/
           ├─ audiosample/clip_i.mp3
           └─ textsample/clip_i.txt
    """
    srt_path   = Path(srt_path)
    video_path = Path(video_path)
    out_root   = Path(output_path)

    # === ① 以“视频 basename”作为顶层文件夹 ===
    vid_folder = out_root 
    audio_dir  = vid_folder / "audiosample"
    text_dir   = vid_folder / "textsample"
    audio_dir.mkdir(parents=True, exist_ok=True)
    text_dir.mkdir(parents=True,  exist_ok=True)

    # ---------- 1) 解析 SRT ----------
    enc = _detect_encoding(srt_path)
    srt_data: List[Dict] = []
    idx, st, et, buf = -1, -1, -1, []
    time_re = re.compile(r"(\d{2}):(\d{2}):(\d{2}),(\d{3})")

    with srt_path.open(encoding=enc, errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if re.fullmatch(r"\d+", line):            # 序号
                if idx != -1 and buf:
                    srt_data.append({"start": st, "end": et, "text": " ".join(buf)})
                idx, buf = int(line), []
                continue
            if "-->" in line:                         # 时间轴
                t1, t2 = line.split("-->")
                h1, m1, s1, ms1 = map(int, time_re.match(t1.strip()).groups())
                h2, m2, s2, ms2 = map(int, time_re.match(t2.strip()).groups())
                st = ts_to_seconds(h1, m1, s1, ms1)
                et = ts_to_seconds(h2, m2, s2, ms2)
                continue
            buf.append(line)
        if idx != -1 and buf:
            srt_data.append({"start": st, "end": et, "text": " ".join(buf)})

    if not srt_data:
        print(f"[WARN] 无有效字幕: {srt_path.name}")
        return

    # ---------- 2) 合并字幕片段 ----------
    clips = []
    i = 0
    while len(clips) < max_clips and i < len(srt_data):
        seg_st = srt_data[i]["start"]
        seg_et = srt_data[i]["end"]
        texts  = [srt_data[i]["text"]]
        j = i + 1
        while j < len(srt_data):
            next_st, next_et = srt_data[j]["start"], srt_data[j]["end"]
            # 合并条件
            if (next_et - seg_st) <= (target_clip_sec + 10) and (next_st - seg_et) < 5:
                seg_et = next_et
                texts.append(srt_data[j]["text"])
                j += 1
            else:
                break
        dur = seg_et - seg_st
        if 0.5*target_clip_sec <= dur <= 1.5*target_clip_sec:
            clips.append({"start": seg_st, "end": seg_et, "text": " ".join(texts)})
        i = j if j > i else i+1

    # 不足 max_clips 时，补最长单句
    if len(clips) < max_clips:
        remaining = sorted(
            (d for d in srt_data),
            key=lambda x: x["end"] - x["start"],
            reverse=True
        )
        for d in remaining:
            if len(clips) >= max_clips:
                break
            clips.append({"start": d["start"], "end": d["end"], "text": d["text"]})

    # ---------- 3) 写 MP3 & TXT ----------
    with VideoFileClip(video_path.as_posix()) as vclip:
        for k, c in enumerate(clips, 1):
            subclip = vclip.subclip(c["start"], c["end"])
            mp3_path = audio_dir / f"{k}.mp3"
            txt_path = text_dir  / f"{k}.txt"

            subclip.audio.write_audiofile(
                mp3_path.as_posix(),
                fps=16000,
                codec="libmp3lame",
                bitrate="48k",
                verbose=False,
                logger=None
            )
            txt_path.write_text(c["text"], encoding="utf-8")

            # === ② 打印相对 sample/ 路径更直观 ===
            rel_mp3 = mp3_path.relative_to(out_root)
            print(f"✓  {rel_mp3}  ({c['end']-c['start']:.1f}s)")

# === 主函数 2：批量 === ------------------------------------------------------

def process_all_srt_and_videos(
    srt_folder: str,
    video_folder: str,
    output_base_folder: str,
) -> None:
    srt_folder   = Path(srt_folder)
    video_folder = Path(video_folder)
    out_base     = Path(output_base_folder)

    video_exts = (".mp4", ".mkv", ".mov", ".avi")

    for srt_file in srt_folder.glob("*.srt"):
        basename = srt_file.stem
        # 找匹配视频（忽略大小写）
        video_file = next(
            (video_folder / f"{basename}{ext}"
             for ext in video_exts
             if (video_folder / f"{basename}{ext}").exists()),
            None
        )
        if not video_file:
            print(f"[SKIP] 找不到同名视频: {basename}")
            continue

        print(f"\n=== 处理 {basename} ===")
        out_dir = out_base / _safe_name(basename)
        extract_samples_from_video_and_srt(
            srt_path=srt_file.as_posix(),
            video_path=video_file.as_posix(),
            output_path=out_dir.as_posix()
        )

# ----------------------------------------------------------------------
# 批量处理文件夹：为每对 <basename>.{mp4,mkv,…}+<basename>.srt 调 extract_
# ----------------------------------------------------------------------
from pathlib import Path

def process_all_ppt_srt(
    video_dir: str,
    srt_dir: str,
    ppt_image_output_base: str,
    *,
    start_time_sec: int = 30,
    sample_interval_sec: int = 3,
) -> None:
    """
    在 video_dir / srt_dir 中匹配同名文件（不含扩展名），为每一对调用
    extract_ppt_clips_and_srt_clips()。输出统一写进 ppt_image_output_base。
    """
    video_dir = Path(video_dir)
    srt_dir   = Path(srt_dir)
    out_base  = Path(ppt_image_output_base)
    out_base.mkdir(parents=True, exist_ok=True)

    video_exts = (".mp4", ".mkv", ".mov", ".avi")

    # —— 建立 “基名 → 视频路径” 映射 ——
    video_map = {
        f.stem.lower(): f
        for ext in video_exts
        for f in video_dir.glob(f"*{ext}")
    }

    # —— 遍历 SRT，找同名视频 ——
    for srt_file in srt_dir.glob("*.srt"):
        key = srt_file.stem.lower()
        video_file = video_map.get(key)
        if not video_file:
            print(f"[SKIP] 找不到同名视频: {srt_file.name}")
            continue

        print(f"\n=== 处理 {key} ===")
        extract_ppt_clips_and_srt_clips(
            video_path             = video_file.as_posix(),
            srt_path               = srt_file.as_posix(),
            ppt_image_output_base  = out_base.as_posix(),  # SRT 片段就放同一层
            start_time_sec         = start_time_sec,
            sample_interval_sec    = sample_interval_sec,
        )


# 新增函数：提取PPT图像和SRT口播片段
def extract_ppt_clips_and_srt_clips(
    video_path: str,
    srt_path: str,
    ppt_image_output_base: str,
    *,
    start_time_sec: int = 30,
    sample_interval_sec: int = 3
):
    """…… docstring 略 ……"""
    video_base_name = os.path.splitext(os.path.basename(video_path))[0]
    ppt_image_output_folder = os.path.join(ppt_image_output_base, video_base_name)
    os.makedirs(ppt_image_output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误: 无法打开视频文件 {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    # ---------- 定位到 30 s ----------
    start_frame = int(start_time_sec * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    interval_frames = int(sample_interval_sec * fps)   # 10 s → 多少帧
    frame_number   = start_frame                       # 从 30 s 对应帧开始计
    saved_slide_count   = 0
    last_saved_frame_number = frame_number - interval_frames
    ppt_frames_info      = []
    extracted_frames_paths = []        # ← 原先漏声明，补上

    prev_frame_gray = prev_frame_original_color = None
    prev_clarity_score = 0.0

    print(f"\n--- 开始提取（从 {start_time_sec}s，每 {sample_interval_sec}s 取 1 帧）---")

    while True:
        cap.grab()
        for _ in range(interval_frames - 1):
            if not cap.grab():
                break
        ret, frame = cap.retrieve()
        if not ret:
            break

        current_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        current_frame_gray = cv2.GaussianBlur(current_frame_gray, (21, 21), 0)

        if prev_frame_gray is not None:
            prev_clarity_score = cv2.Laplacian(prev_frame_gray, cv2.CV_64F).var()


        is_solid_color = False
        if prev_frame_original_color is not None:
            # 方法一：方差判断
            b, g, r = cv2.split(prev_frame_original_color)
            if b.var() < 100 and g.var() < 100 and r.var() < 100:
                is_solid_color = True
            
            # 方法二：颜色占比判断
            if not is_solid_color: # 如果方差判断不是纯色，再进行颜色占比判断
                # 将图像转换为HSV颜色空间，对颜色更鲁棒
                hsv_frame = cv2.cvtColor(prev_frame_original_color, cv2.COLOR_BGR2HSV)
                # 统计颜色直方图
                hist = cv2.calcHist([hsv_frame], [0, 1], None, [180, 256], [0, 180, 0, 256])
                # 找到直方图中最大的bin，即占比最高的颜色
                minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(hist)
                # 计算最大颜色占比
                max_color_percentage = maxVal / (prev_frame_original_color.shape[0] * prev_frame_original_color.shape[1])
                
                # 如果最大颜色占比超过某个阈值（例如90%），则认为是纯色背景
                if max_color_percentage > 0.65: # 阈值可根据实际情况调整
                    is_solid_color = True

        if prev_frame_gray is not None:
            frame_delta = cv2.absdiff(prev_frame_gray, current_frame_gray)
            diff_percentage = np.count_nonzero(frame_delta) / frame_delta.size

            prev = prev_frame_gray.astype(np.float32) / 255.0
            curr = current_frame_gray.astype(np.float32) / 255.0
            if prev.shape != curr.shape:
                curr = cv2.resize(curr, (prev.shape[1], prev.shape[0]))
            score, _ = ssim(prev, curr, full=True, data_range=1.0) if prev.var() or curr.var() else (1.0, None)

            # ---------- 翻页判定 & 清晰度判断 & 纯色背景页判断 ----------
            if (diff_percentage > 0.06 and score < 0.85) and \
               (prev_clarity_score > 0 and frame_number - last_saved_frame_number >= interval_frames) and \
               not is_solid_color:
                if frame_number - last_saved_frame_number >= interval_frames:  # ≥10 s 间隔
                    saved_slide_count += 1
                    image_path = os.path.join(
                        ppt_image_output_folder,
                        f"{saved_slide_count}.png"
                    )
                    cv2.imwrite(image_path, prev_frame_original_color)
                    ppt_frames_info.append((image_path, last_saved_frame_number / fps))
                    extracted_frames_paths.append(image_path)
                    last_saved_frame_number = frame_number
                    print(f"保存 {image_path} @ {frame_number/fps:.1f}s "
                          f"(Diff={diff_percentage:.3f}, SSIM={score:.3f}, Clarity={prev_clarity_score:.2f})")

        prev_frame_gray = current_frame_gray
        prev_frame_original_color = frame
        frame_number += interval_frames

    cap.release()
    print(f"提取完成，共找到 {len(extracted_frames_paths)} 页 PPT 帧。")

    # ---------- SRT 按时间戳切片 ----------
    extract_srt_clips_by_timestamps(
        srt_path=srt_path,
        ppt_frames_info=ppt_frames_info,
        output_dir=ppt_image_output_folder
    )



import os
import re

def extract_srt_clips_by_timestamps(
        srt_path: str,
        ppt_frames_info: list,
        output_dir: str,
):
    """
    从 SRT 文件中按 PPT 帧时间戳切出文本片段。
    
    参数
    ----
    srt_path : str
        字幕文件路径（UTF-8 编码）。
    ppt_frames_info : list[tuple[str, float]]
        [(image_path, timestamp_seconds), ...] 需按时间升序。
    output_dir : str
        输出目录，会在其下创建 `srt_clips/` 子目录。
    """
    # ---------- 预处理时间戳 ----------
    # 取时间戳并升序排序，最后加一个 ∞ 作哨兵
    ts_sorted = sorted([t for _, t in ppt_frames_info])
    ts_sorted.append(float("inf"))

    # 输出目录
    out_dir = output_dir
    #os.makedirs(out_dir, exist_ok=True)

    # 给每段准备缓冲区
    buffers = ["" for _ in range(len(ts_sorted) - 1)]
    seg_idx = 0                     # 当前所在的 PPT 段编号
    seg_start = ts_sorted[seg_idx]
    seg_end   = ts_sorted[seg_idx + 1]

    # ---------- 辅助函数：时间字符串 → 秒 ----------
    def _time_to_sec(t: str) -> float:
        """'HH:MM:SS,ms' → seconds (float)"""
        hh, mm, rest = t.split(":")
        ss, ms = rest.split(",")
        return int(hh) * 3600 + int(mm) * 60 + int(ss) + int(ms) / 1000

    # ---------- 读取 SRT，仅遍历一次 ----------
    time_line_re = re.compile(r"(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})")

    with open(srt_path, encoding="utf-8") as fh:
        while True:
            line = fh.readline()
            if not line:            # EOF
                break

            # 匹配时间轴行
            m = time_line_re.match(line)
            if not m:
                continue            # 跳过序号 / 空行 / 非法行

            start_sec = _time_to_sec(m.group(1))
            end_sec   = _time_to_sec(m.group(2))

            # 读取下一行开始到空行为止的正文
            text_lines = []
            while True:
                pos = fh.tell()     # 记录当前位置，方便回退
                next_line = fh.readline()
                if not next_line or next_line.strip() == "":
                    break
                if time_line_re.match(next_line):  # 意外碰到新时间轴，回退
                    fh.seek(pos)
                    break
                text_lines.append(next_line.strip())

            if not text_lines:
                continue
            text_block = " ".join(text_lines) + " "

            # ---------- 放入对应段 ----------
            # 先把所有“已结束”的段落输出，再决定当前条目属于哪段
            while start_sec >= seg_end:
                seg_idx += 1
                if seg_idx >= len(buffers):
                    break
                seg_start = ts_sorted[seg_idx]
                seg_end   = ts_sorted[seg_idx + 1]

            if seg_idx >= len(buffers):
                # SRT 后半段时间超过最后一张 PPT，直接忽略
                break

            # 若字幕条与当前段有任何重叠，就算进来
            overlaps = not (end_sec <= seg_start or start_sec >= seg_end)
            if overlaps:
                buffers[seg_idx] += text_block

   # ---------- 写出文件（带长度过滤） ----------
    MIN_CHARS = 30                  # 阈值：少于 30 字符视为无效

    for i, (img_path, _) in enumerate(ppt_frames_info):
        txt = buffers[i].strip()
        base = os.path.basename(img_path)           # ppt_frame_XXX.png

        if len(txt) < MIN_CHARS:
            # 文本太短：删除 PNG，跳过写 TXT
            if os.path.exists(img_path):
                os.remove(img_path)
                print(f"✗ Removed {base}  (subtitle < {MIN_CHARS} chars)")
            continue

        # ---------- 正常保存 ----------
        num = "".join(filter(str.isdigit, base)) or f"{i+1}"
        out_name = f"text{int(num)}.txt"
        out_path = os.path.join(out_dir, out_name)

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(txt)

        print(f"✔ {out_name:15s} ← {base}  ({len(txt)} chars)")


    print("SRT 片段提取完毕。")




if __name__ == '__main__':
    # 示例用法
    # 批量处理SRT到TXT
    # srt_input_dir = "C:/Python/Learn_accompany_3/Resources/srt"
    # txt_output_dir = "C:/Python/Learn_accompany_3/Resources/txt"
    # preprocess_srt_to_txt(srt_input_dir, txt_output_dir)

    # 批量处理SRT和视频，提取音频和文本样本
    # srt_input_folder = "C:/Python/Learn_accompany_3/Resources/srt"  
    # video_input_folder = "C:/Python/Learn_accompany_3/Resources/AI/PPT_Video"  
    # output_samples_base_folder = "C:/Python/Learn_accompany_3/Resources/sample"  
    # process_all_srt_and_videos(srt_input_folder, video_input_folder, output_samples_base_folder)

    # 新增功能示例：提取PPT图像和SRT口播片段
    ppt_video_path = "C:/Python/Learn_accompany_3/Resources/AI/PPT_Video/1.mp4" # 替换为你的PPT视频路径
    corresponding_srt_path = "C:/Python/Learn_accompany_3/Resources/srt/1.srt" # 替换为对应的SRT文件路径
    ppt_image_output_base_folder = "C:/Python/Learn_accompany_3/Resources/PPT_image" # 替换为PPT图像输出根目录
    srt_clip_output_base_folder = "C:/Python/Learn_accompany_3/Resources/sample" # 替换为SRT口播片段输出根目录

    extract_ppt_clips_and_srt_clips(
        ppt_video_path,
        corresponding_srt_path,
        ppt_image_output_base_folder,
        srt_clip_output_base_folder
    )