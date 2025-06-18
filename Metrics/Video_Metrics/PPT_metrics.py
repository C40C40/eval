import re, json
from pathlib import Path
from typing import List, Tuple, Dict

from utils import (
    call_vlm_api, call_gpt_api, _read_text_file
)

class PPTLectureAnalyzer:
    def __init__(self, lecture_dir: str):
        self.dir = Path(lecture_dir)
        if not self.dir.is_dir():
            raise NotADirectoryError(self.dir)
        self.slide_pairs = self._collect_pairs()
        if not self.slide_pairs:
            raise RuntimeError("未找到有效图片+字幕对")

        self.slide_summaries: Dict[str, str] = {}
        self.slide_scores:    Dict[str, Dict[str, float]] = {}

    # ---------- private ----------
    def _collect_pairs(self) -> List[Tuple[str, str]]:
        pairs = []
        for png in sorted(self.dir.glob("*.png"),
                        key=lambda p: int(re.search(r'\d+', p.stem).group())):
            # 先提取数字
            match = re.search(r'\d+', png.stem)
            if not match:
                continue
            idx = match.group()
            txt = self.dir / f"text{idx}.txt"
            if txt.exists():
                pairs.append((png.as_posix(), txt.as_posix()))
            else:
                print(f"[WARN] 缺字幕: {txt.name}")
        return pairs

    def _read(self, path: str) -> str:
        return _read_text_file(path) or ""

    # ---------- public ----------
    def run_slide_level_eval(self,
                             system_prompt_path: str,
                             query_prompt_path: str) -> Dict[str, Dict[str, float]]:
        sys_prompt = self._read(system_prompt_path)
        qry_tpl    = self._read(query_prompt_path)

        for img, txt in self.slide_pairs:
            # 1. summary
            summ = call_vlm_api(img, "请用中文一句话总结此幻灯片内容,必须涵盖PPT上所有要点")
            if not summ:
                continue
            self.slide_summaries[img] = summ

            # 2. explanation score
            subtitle = self._read(txt)
            query = (qry_tpl
                        .replace("SUMM", summ, 1)         # 只替换第一次出现，安全起见加 count=1
                        .replace("TRANSCRIPT", subtitle, 1))
            ans   = call_gpt_api(sys_prompt, query) or {}
            detail = ans.get("detail") or ans.get("detail_score", 0)
            relev  = ans.get("relevance") or ans.get("relevance_score", 0)
            self.slide_scores[img] = {"detail": detail, "relevance": relev}

        return self.slide_scores

    def get_slide_level_average(self) -> Dict[str, float]:
        if not self.slide_scores:
            raise RuntimeError("请先执行 run_slide_level_eval()")
        d_avg = sum(v["detail"]     for v in self.slide_scores.values()) / len(self.slide_scores)
        r_avg = sum(v["relevance"]  for v in self.slide_scores.values()) / len(self.slide_scores)
        return {"detail_avg": round(d_avg,3), "relevance_avg": round(r_avg,3)}

    def run_overall_logic_eval(self,
                               system_prompt_path: str,
                               query_prompt_path: str) -> float:
        if not self.slide_summaries:
            raise RuntimeError("请先执行 run_slide_level_eval()")

        ordered = [self.slide_summaries[k] for k in sorted(
                   self.slide_summaries, key=lambda p:int(re.search(r'\d+',p).group()))]
        all_summ = "\n".join(ordered)

        sys_prompt = self._read(system_prompt_path)
        qry_tpl    = self._read(query_prompt_path)
        query      = qry_tpl.replace("ALLSUM",all_summ,1)

        ans = call_gpt_api(sys_prompt, query) or {}
        return ans.get("logic_score", 0.0)

# ---------- CLI demo ----------
if __name__ == "__main__":
    analyzer = PPTLectureAnalyzer("C:/Python/Learn_accompany_3/Resources/PPT_image/1")

    # 路径示例：你自己的 system/query 模板
    SP_EXPL = "C:/Python/Learn_accompany_3/Metrics/Video_Metrics/system_prompt_consistency.txt"
    QP_EXPL = "C:/Python/Learn_accompany_3/Metrics/Video_Metrics/query_prompt_consistency.txt"
    SP_LOG  = "C:/Python/Learn_accompany_3/Metrics/Video_Metrics/system_prompt_logic.txt"
    QP_LOG  = "C:/Python/Learn_accompany_3/Metrics/Video_Metrics/query_prompt_logic.txt"

    analyzer.run_slide_level_eval(SP_EXPL, QP_EXPL)
    print("平均分:", analyzer.get_slide_level_average())

    logic_score = analyzer.run_overall_logic_eval(SP_LOG, QP_LOG)
    print("整体逻辑得分:", logic_score)
