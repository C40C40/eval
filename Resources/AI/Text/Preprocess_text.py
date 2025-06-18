import re
import chardet

in_path  = "1_2_4.srt"
out_path = "1_2_4.txt"

# 检测编码
raw = open(in_path, "rb").read(10000)
enc  = chardet.detect(raw)["encoding"]
print("File encoding detected as:", enc)

lines = []
with open(in_path, encoding=enc, errors="ignore") as fin:
    for line in fin:
        s = line.strip()
        if re.fullmatch(r'\d+', s) or '-->' in s or not s:
            continue
        lines.append(s)

# 直接拼成一行
text = "\n".join(lines)

with open(out_path, "w", encoding="utf-8") as fout:
    fout.write(text)
print("Wrote", len(text), "chars to", out_path)
