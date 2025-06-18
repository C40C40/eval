import websocket
import datetime
import hashlib
import base64
import hmac
import json
from urllib.parse import urlencode
import time
import ssl
from wsgiref.handlers import format_date_time
from datetime import datetime
from time import mktime
import xml.etree.ElementTree as ET
import os
from typing import Dict, List, Optional, Tuple
import statistics


class SpeechQualityEvaluator:
    """语音质量评测器 - 封装版本"""
    
    def __init__(self, appid: str = "343f5b4d", 
                 api_secret: str = "ODBhYTJhMTg1YWNmM2IyMDk3NWM1NGYz",
                 api_key: str = "77d67232368eb8895d010d1aba371b4c"):
        """
        初始化评测器
        
        Args:
            appid: 科大讯飞应用ID
            api_secret: API密钥
            api_key: API Key
        """
        self.appid = appid
        self.api_secret = api_secret
        self.api_key = api_key
        self.host_url = "wss://ise-api.xfyun.cn/v2/open-ise"
        self.results = []
    
    def _load_text_with_newlines(self, path: str) -> str:
        """读取文本文件，保留换行符并添加BOM"""
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        return '\uFEFF' + "[content]\n" + content
    
    def _product_url(self) -> str:
        """生成WebSocket连接URL"""
        now_time = datetime.now()
        now_date = format_date_time(mktime(now_time.timetuple()))
        origin_base = f"host: ise-api.xfyun.cn\ndate: {now_date}\nGET /v2/open-ise HTTP/1.1"
        
        signature_sha = hmac.new(
            self.api_secret.encode('utf-8'), 
            origin_base.encode('utf-8'), 
            digestmod=hashlib.sha256
        ).digest()
        signature_sha = base64.b64encode(signature_sha).decode(encoding='utf-8')
        
        authorization_origin = f'api_key="{self.api_key}", algorithm="hmac-sha256", headers="host date request-line", signature="{signature_sha}"'
        authorization = base64.b64encode(authorization_origin.encode('utf-8')).decode(encoding='utf-8')
        
        dict_data = {
            "authorization": authorization,
            "date": now_date,
            "host": "ise-api.xfyun.cn"
        }
        return self.host_url + '?' + urlencode(dict_data)
    
    def _parse_detailed_result(self, xml_str: str, show_details: bool = False) -> Optional[Dict]:
        """
        解析评测结果，支持显示详细信息
        
        Args:
            xml_str: XML结果字符串
            show_details: 是否显示详细的字词信息
            
        Returns:
            评测结果字典或None
        """
        try:
            root = ET.fromstring(xml_str)
            chapter = root.find(".//rec_paper/read_chapter")
            
            if chapter is None:
                print("未找到 read_chapter 节点，可能是识别失败")
                return None
            
            # 解析总体分数
            result = {
                'content': chapter.attrib.get('content', ''),
                'accuracy_score': float(chapter.attrib.get('accuracy_score', '0')),
                'fluency_score': float(chapter.attrib.get('fluency_score', '0')),
                'tone_score': float(chapter.attrib.get('tone_score', '0')),
                'phone_score': float(chapter.attrib.get('phone_score', '0')),
                'integrity_score': float(chapter.attrib.get('integrity_score', '0')),
                'total_score': float(chapter.attrib.get('total_score', '0'))
            }
            
            if show_details:
                print(f"\n==== 📊 {result['content'][:20]}... 详细评测结果 ====")
                print(f"准确度: {result['accuracy_score']:.1f} | 流利度: {result['fluency_score']:.1f}")
                print(f"声调: {result['tone_score']:.1f} | 音素: {result['phone_score']:.1f} | 总分: {result['total_score']:.1f}")
                
                # 解析详细的字词信息
                result['word_details'] = self._parse_word_details(chapter)
                self._print_word_details(result['word_details'])
            else:
                print(f"✅ 评测完成 - 总分: {result['total_score']:.1f}")
            
            return result
            
        except Exception as e:
            print(f"解析XML结果时出错: {e}")
            return None
    
    def _parse_word_details(self, chapter) -> List[Dict]:
        """解析详细的字词信息"""
        word_details = []
        
        for sentence in chapter.findall('.//sentence'):
            for word in sentence.findall('.//word'):
                word_info = {
                    'content': word.attrib.get('content', ''),
                    'beg_pos': int(word.attrib.get('beg_pos', '0')),
                    'end_pos': int(word.attrib.get('end_pos', '0')),
                    'time_len': int(word.attrib.get('time_len', '0')),
                    'symbol': word.attrib.get('symbol', ''),
                    'syllables': []
                }
                
                # 解析音节信息
                for syll in word.findall('.//syll'):
                    syll_info = {
                        'content': syll.attrib.get('content', ''),
                        'symbol': syll.attrib.get('symbol', ''),
                        'time_len': int(syll.attrib.get('time_len', '0')),
                        'phones': []
                    }
                    
                    # 解析音素信息
                    for phone in syll.findall('.//phone'):
                        phone_info = {
                            'content': phone.attrib.get('content', ''),
                            'time_len': int(phone.attrib.get('time_len', '0')),
                            'is_yun': phone.attrib.get('is_yun', '0'),
                            'perr_msg': phone.attrib.get('perr_msg', '0'),
                            'mono_tone': phone.attrib.get('mono_tone', '')
                        }
                        syll_info['phones'].append(phone_info)
                    
                    word_info['syllables'].append(syll_info)
                
                word_details.append(word_info)
        
        return word_details
    
    def _print_word_details(self, word_details: List[Dict]):
        """打印详细的字词信息 - 友好格式"""
        print("\n==== 📝 字词发音分析 ====")
        
        correct_count = 0
        total_phones = 0
        
        for i, word in enumerate(word_details[:10]):  # 只显示前10个字
            word_status = "✅"
            error_phones = []
            
            for syll in word['syllables']:
                for phone in syll['phones']:
                    total_phones += 1
                    if phone['perr_msg'] != '0':
                        error_phones.append(phone['content'])
                        word_status = "❌"
                    else:
                        correct_count += 1
            
            # 简洁显示每个字的状态
            if error_phones:
                print(f"字 {i+1}: '{word['content']}' ({word['symbol']}) {word_status} - 发音问题: {', '.join(error_phones)}")
            else:
                print(f"字 {i+1}: '{word['content']}' ({word['symbol']}) {word_status}")
        
        if len(word_details) > 10:
            print(f"... 还有 {len(word_details) - 10} 个字未显示")
        
        # 统计信息
        if total_phones > 0:
            accuracy_rate = (correct_count / total_phones) * 100
            print(f"\n📊 发音准确率: {accuracy_rate:.1f}% ({correct_count}/{total_phones})")
        
        print("-" * 40)
    
    def get_pronunciation_stats(self, word_details: List[Dict]) -> Dict:
        """获取发音统计信息"""
        if not word_details:
            return {}
        
        total_words = len(word_details)
        total_phones = 0
        correct_phones = 0
        error_details = []
        
        for word in word_details:
            word_errors = []
            for syll in word['syllables']:
                for phone in syll['phones']:
                    total_phones += 1
                    if phone['perr_msg'] != '0':
                        word_errors.append(phone['content'])
                    else:
                        correct_phones += 1
            
            if word_errors:
                error_details.append({
                    'word': word['content'],
                    'symbol': word['symbol'],
                    'errors': word_errors
                })
        
        return {
            'total_words': total_words,
            'total_phones': total_phones,
            'correct_phones': correct_phones,
            'pronunciation_accuracy': (correct_phones / total_phones * 100) if total_phones > 0 else 0,
            'error_words_count': len(error_details),
            'error_words': error_details
        }
    
    def evaluate_single_audio(self, audio_path: str, text_path: str, 
                            show_details: bool = False) -> Optional[Dict]:
        """
        评测单个音频文件
        
        Args:
            audio_path: 音频文件路径
            text_path: 对应文本文件路径
            show_details: 是否显示详细的字词分析
            
        Returns:
            评测结果字典
        """
        print(f"\n🎤 开始评测: {os.path.basename(audio_path)}")
        
        try:
            # 设置WebSocket事件处理
            result_container = {'result': None}
            
            def on_message(ws, message):
                status = json.loads(message)["data"]["status"]
                if status == 2:  # 评测完成
                    xml_encoded = json.loads(message)["data"]["data"]
                    xml = base64.b64decode(xml_encoded)
                    xml_str = xml.decode("utf-8")
                    
                    # 直接解析结果，不打印原始XML
                    result_container['result'] = self._parse_detailed_result(xml_str, show_details)
                    ws.close()
            
            def on_error(ws, error):
                print(f"❌ WebSocket错误: {error}")
            
            def on_close(ws, reason, res):
                pass  # 静默关闭
            
            def on_open(ws):
                # 发送评测请求
                text = self._load_text_with_newlines(text_path)
                send_dict = {
                    "common": {"app_id": self.appid},
                    "business": {
                        "category": "read_chapter",
                        "rstcd": "utf8",
                        "sub": "ise",
                        "ent": "cn_vip",
                        "tte": "utf-8",
                        "cmd": "ssb",
                        "auf": "audio/L16;rate=16000",
                        "aue": "lame",
                        "text": text
                    },
                    "data": {"status": 0, "data": ""}
                }
                ws.send(json.dumps(send_dict))
                
                # 分块发送音频数据
                with open(audio_path, "rb") as file_flag:
                    while True:
                        buffer = file_flag.read(1280)
                        if not buffer:
                            # 发送结束标志
                            end_dict = {
                                "business": {"cmd": "auw", "aus": 4, "aue": "lame"},
                                "data": {"status": 2, "data": str(base64.b64encode(buffer).decode())}
                            }
                            ws.send(json.dumps(end_dict))
                            break
                        
                        # 发送音频块
                        send_dict = {
                            "business": {"cmd": "auw", "aus": 1, "aue": "lame"},
                            "data": {
                                "status": 1,
                                "data": str(base64.b64encode(buffer).decode()),
                                "data_type": 1,
                                "encoding": "raw"
                            }
                        }
                        ws.send(json.dumps(send_dict))
                        time.sleep(0.04)
            
            # 执行WebSocket连接
            websocket.enableTrace(False)
            ws_url = self._product_url()
            ws_entity = websocket.WebSocketApp(
                ws_url, 
                on_message=on_message, 
                on_error=on_error, 
                on_close=on_close, 
                on_open=on_open
            )
            ws_entity.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})
            
            return result_container['result']
            
        except Exception as e:
            print(f"❌ 评测过程出错: {e}")
            return None
    
    def evaluate_course_audio(self, course_folder: str, 
                            show_details: bool = False) -> Dict:
        """
        评测课程文件夹中的所有音频
        
        Args:
            course_folder: 课程文件夹路径（包含audiosample和textsample子文件夹）
            show_details: 是否显示每个文件的详细分析
            
        Returns:
            课程整体评测结果
        """
        print(f"\n📁 开始评测课程: {os.path.basename(course_folder)}")
        
        audio_folder = os.path.join(course_folder, "audiosample")
        text_folder = os.path.join(course_folder, "textsample")
        
        if not os.path.exists(audio_folder) or not os.path.exists(text_folder):
            raise ValueError(f"课程文件夹结构不正确，缺少 audiosample 或 textsample 文件夹")
        
        audio_files = [f for f in os.listdir(audio_folder) if f.endswith('.mp3')]
        text_files = [f for f in os.listdir(text_folder) if f.endswith('.txt')]
        
        results = []
        
        for audio_file in audio_files:
            base_name = os.path.splitext(audio_file)[0]
            matching_text_file = base_name + '.txt'
            
            if matching_text_file in text_files:
                audio_path = os.path.join(audio_folder, audio_file)
                text_path = os.path.join(text_folder, matching_text_file)
                
                result = self.evaluate_single_audio(audio_path, text_path, show_details)
                if result:
                    results.append(result)
            else:
                print(f"⚠️ 未找到 {audio_file} 对应的文本文件")
        
        # 计算课程整体评分
        if results:
            course_result = self._calculate_course_statistics(results)
            self._print_course_summary(course_result, os.path.basename(course_folder), show_details)
            return course_result
        else:
            print("❌ 没有成功评测的音频文件")
            return {}
    
    def _calculate_course_statistics(self, results: List[Dict]) -> Dict:
        """计算课程统计数据"""
        scores = {
            'accuracy_scores': [r['accuracy_score'] for r in results],
            'fluency_scores': [r['fluency_score'] for r in results],
            'tone_scores': [r['tone_score'] for r in results],
            'phone_scores': [r['phone_score'] for r in results],
            'total_scores': [r['total_score'] for r in results]
        }
        
        return {
            'sample_count': len(results),
            'average_accuracy': statistics.mean(scores['accuracy_scores']),
            'average_fluency': statistics.mean(scores['fluency_scores']),
            'average_tone': statistics.mean(scores['tone_scores']),
            'average_phone': statistics.mean(scores['phone_scores']),
            'average_total': statistics.mean(scores['total_scores']),
            'fluency_std': statistics.stdev(scores['fluency_scores']) if len(scores['fluency_scores']) > 1 else 0,
            'tone_std': statistics.stdev(scores['tone_scores']) if len(scores['tone_scores']) > 1 else 0,
            'phone_std': statistics.stdev(scores['phone_scores']) if len(scores['phone_scores']) > 1 else 0,
            'total_std': statistics.stdev(scores['total_scores']) if len(scores['total_scores']) > 1 else 0,
            'raw_results': results
        }
    
    def _print_course_summary(self, course_result: Dict, course_name: str, show_details: bool = False):
        """打印课程评测总结 - 修复后的唯一版本"""
        print(f"\n{'='*50}")
        print(f"📊 课程 '{course_name}' 评测总结")
        print(f"{'='*50}")
        print(f"📈 样本数量: {course_result['sample_count']} 个音频片段")
        print(f"🎯 平均总分: {course_result['average_total']:.2f}")
        print(f"🗣️  平均流利度: {course_result['average_fluency']:.2f} (标准差: {course_result['fluency_std']:.2f})")
        print(f"🎵 平均声调准确度: {course_result['average_tone']:.2f} (标准差: {course_result['tone_std']:.2f})")
        print(f"🔤 平均音素准确度: {course_result['average_phone']:.2f} (标准差: {course_result['phone_std']:.2f})")
        print(f"📐 平均准确度: {course_result['average_accuracy']:.2f}")
        
        # 稳定性评估
        if course_result['total_std'] < 5:
            stability = "🟢 优秀"
            stability_desc = "各片段评分非常一致，教学表现稳定"
        elif course_result['total_std'] < 10:
            stability = "🟡 良好"
            stability_desc = "各片段评分较为接近，偶有波动"
        else:
            stability = "🔴 需要改进"
            stability_desc = "各片段评分差异较大，表现不够稳定"
        
        print(f"📊 评分稳定性: {stability} (标准差: {course_result['total_std']:.2f}) - {stability_desc}")
        
        # 显示详细信息
        if show_details and 'raw_results' in course_result:
            print(f"\n{'='*30} 详细评分信息 {'='*30}")
            for i, result in enumerate(course_result['raw_results'], 1):
                print(f"片段 {i}: 总分={result['total_score']:.1f} | "
                      f"流利度={result['fluency_score']:.1f} | "
                      f"声调={result['tone_score']:.1f} | "
                      f"音素={result['phone_score']:.1f}")
                
                # 如果有字词详细信息，也显示
                if 'word_details' in result:
                    pronunciation_stats = self.get_pronunciation_stats(result['word_details'])
                    if pronunciation_stats:
                        print(f"       发音准确率: {pronunciation_stats['pronunciation_accuracy']:.1f}% "
                              f"({pronunciation_stats['correct_phones']}/{pronunciation_stats['total_phones']})")
                        if pronunciation_stats['error_words']:
                            error_words = [w['word'] for w in pronunciation_stats['error_words'][:3]]
                            print(f"       发音问题: {', '.join(error_words)}{'...' if len(pronunciation_stats['error_words']) > 3 else ''}")
            
            # 评分分布分析
            print(f"\n📊 评分分布分析:")
            scores = [r['total_score'] for r in course_result['raw_results']]
            print(f"   最高分: {max(scores):.1f}")
            print(f"   最低分: {min(scores):.1f}")
            print(f"   分数范围: {max(scores) - min(scores):.1f}")
        
        print("=" * 100)

    def evaluate_course_audio_simple(self, course_folder: str) -> Dict:
        """
        简化版课程评测 - 只返回核心指标，不显示详细信息
        
        Args:
            course_folder: 课程文件夹路径
            
        Returns:
            简化的评测结果
        """
        print(f"🎤 评测课程: {os.path.basename(course_folder)}")
        
        audio_folder = os.path.join(course_folder, "audiosample")
        text_folder = os.path.join(course_folder, "textsample")
        
        if not os.path.exists(audio_folder) or not os.path.exists(text_folder):
            raise ValueError(f"课程文件夹结构不正确")
        
        audio_files = [f for f in os.listdir(audio_folder) if f.endswith('.mp3')]
        text_files = [f for f in os.listdir(text_folder) if f.endswith('.txt')]
        
        results = []
        
        for audio_file in audio_files:
            base_name = os.path.splitext(audio_file)[0]
            matching_text_file = base_name + '.txt'
            
            if matching_text_file in text_files:
                audio_path = os.path.join(audio_folder, audio_file)
                text_path = os.path.join(text_folder, matching_text_file)
                
                result = self.evaluate_single_audio(audio_path, text_path, show_details=False)
                if result:
                    results.append(result)
        
        if results:
            course_result = self._calculate_course_statistics(results)
            
            # 简化输出
            print(f"✅ 完成! 平均分: {course_result['average_total']:.1f} (流利度: {course_result['average_fluency']:.1f})")
            
            return {
                'average_fluency': course_result['average_fluency'],
                'average_tone_accuracy': course_result['average_tone'],
                'average_phone_accuracy': course_result['average_phone'],
                'average_total_score': course_result['average_total'],
                'sample_count': course_result['sample_count'],
                'stability': course_result['total_std']
            }
        else:
            return {}


# 使用示例
if __name__ == '__main__':
    # 创建评测器实例
    evaluator = SpeechQualityEvaluator()
    
    # 方式1: 简化评测（最简洁，适合批量处理）
    # print("=== 方式1: 简化评测 ===")
    # course_result = evaluator.evaluate_course_audio_simple(
    #     "/Users/zhangshenao/Desktop/research/交大教育大模型测评/eval/Resources/sample/1"
    # )
    # print("简化结果:", course_result)
    
    # 方式2: 标准评测（显示总结但不显示每个片段详情）
    # print("\n=== 方式2: 标准评测 ===")
    # course_result = evaluator.evaluate_course_audio(
    #     "/Users/zhangshenao/Desktop/research/交大教育大模型测评/eval/Resources/sample/1",
    #     show_details=False  # 只显示总结，不显示每个片段的详细信息
    # )
    
    # 方式3: 详细评测（显示每个片段的详细评分和发音分析）
    # print("\n=== 方式3: 详细评测 ===")
    # course_result = evaluator.evaluate_course_audio(
    #     "/Users/zhangshenao/Desktop/research/交大教育大模型测评/eval/Resources/sample/1",
    #     show_details=True  # 显示每个音频片段的详细分析
    # )
    
    # 方式4: 单个音频深度分析
    print("\n=== 方式4: 单个音频分析 ===")
    result = evaluator.evaluate_single_audio(
        "/Users/zhangshenao/Desktop/research/交大教育大模型测评/eval/Resources/sample/1/audiosample/clip_1.mp3", 
        "/Users/zhangshenao/Desktop/research/交大教育大模型测评/eval/Resources/sample/1/textsample/clip_1.txt", 
        show_details=True  # 显示这个音频的详细发音分析和字词错误
    )