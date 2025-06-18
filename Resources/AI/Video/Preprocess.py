import av
import lameenc
import numpy as np

def ts_to_seconds(h, m, s, ms):
    return h*3600 + m*60 + s + ms/1000


# 输入／输出路径
in_path  = "1_2.mp4"
out_path = "1_2_4.mp3"

# 目标参数
start_s = ts_to_seconds(0,3,22,420)     # 2 分钟
end_s   = ts_to_seconds(0,4,22,288)      # 5 分钟
target_sr   = 16000 # 16 kHz
target_ch   = 1     # 单声道
bitrate_k  = 192    # 192 kbps

# 打开容器和音频流
container    = av.open(in_path)
stream      = container.streams.audio[0]

# 构造重采样器（format=s16 PCM16，mono，目标采样率）
resampler = av.audio.resampler.AudioResampler(
    format="s16",
    layout="mono",
    rate=target_sr
)

# 构造 MP3 编码器
encoder = lameenc.Encoder()
encoder.set_bit_rate(bitrate_k)
encoder.set_in_sample_rate(target_sr)
encoder.set_channels(target_ch)
encoder.set_quality(2)   # 0=best ... 9=worst

mp3_bytes = bytearray()

for packet in container.demux(stream):
    for frame in packet.decode():
        # 时间筛选
        frame_end = frame.time + frame.samples/frame.sample_rate
        if frame_end < start_s:
            continue
        if frame.time > end_s:
            break

        # 重采样：resample 返回 list
        out_frames = resampler.resample(frame)
        for of in out_frames:
            # of 是单个 AudioFrame，调用 to_ndarray()
            pcm = of.to_ndarray()    # shape = (1, N)
            data = pcm.tobytes()
            mp3_bytes.extend(encoder.encode(data))

    if frame.time > end_s:
        break

# flush 编码器
mp3_bytes.extend(encoder.flush())

# 写文件
with open(out_path, "wb") as f:
    f.write(mp3_bytes)
print("Done:", out_path)