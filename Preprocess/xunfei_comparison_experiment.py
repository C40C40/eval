#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
科大讯飞语音评测API敏感性对比实验
创建针对不同评测维度的低质量音频样本，验证API的区分能力

需要安装的依赖:
pip install librosa soundfile pydub numpy

使用方法:
python xunfei_comparison_experiment.py
"""

import os
import shutil
import random
import numpy as np
import librosa
import soundfile as sf
from pydub import AudioSegment
import statistics
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings("ignore")

# 导入你的语音评测器 (需要将speech_quality_evaluator.py放在同一目录)
try:
    from speech_quality_evaluator import SpeechQualityEvaluator
except ImportError:
    print("❌ 请确保 speech_quality_evaluator.py 文件在同一目录下")
    exit(1)


class XunfeiTargetedDegrader:
    """针对科大讯飞评测维度的音频降质器"""
    
    def __init__(self):
        # 目标降质程度（相比baseline）
        self.target_drops = {
            'fluency_drop': (15, 25),      # 流利度下降15-25分
            'tone_drop': (10, 20),         # 声调准确度下降10-20分  
            'phone_drop': (10, 20),        # 音素准确度下降10-20分
            'total_drop': (15, 30),        # 总分下降15-30分
            'stability_increase': (5, 15)   # 稳定性变差(标准差增加5-15)
        }
        
        # 确保临时文件目录存在
        self.temp_dir = "temp_audio_processing"
        os.makedirs(self.temp_dir, exist_ok=True)
    
    def create_poor_samples_by_dimension(self, good_audio_folder, output_folder):
        """按维度创建低质量样本"""
        
        print("🔧 开始创建降质音频样本...")
        os.makedirs(output_folder, exist_ok=True)
        
        # 为每个维度创建专门的低质量样本
        dimensions = {
            'poor_fluency': self.degrade_fluency,
            'poor_tone': self.degrade_tone_accuracy, 
            'poor_phone': self.degrade_phone_accuracy,
            'poor_stability': self.degrade_stability,
            'poor_overall': self.degrade_overall
        }
        
        audio_files = [f for f in os.listdir(good_audio_folder) if f.endswith('.mp3')]
        
        if not audio_files:
            print(f"❌ 在 {good_audio_folder} 中未找到MP3文件")
            return
        
        for dim_name, degrade_func in dimensions.items():
            print(f"   📝 处理 {dim_name}...")
            dim_folder = os.path.join(output_folder, dim_name, "audiosample")
            os.makedirs(dim_folder, exist_ok=True)
            
            success_count = 0
            for audio_file in audio_files:
                try:
                    input_path = os.path.join(good_audio_folder, audio_file)
                    output_path = os.path.join(dim_folder, audio_file)
                    degrade_func(input_path, output_path)
                    success_count += 1
                except Exception as e:
                    print(f"      ⚠️ 处理 {audio_file} 失败: {e}")
            
            print(f"   ✅ {dim_name}: {success_count}/{len(audio_files)} 个文件处理成功")
    
    def degrade_fluency(self, input_path: str, output_path: str):
        """专门降低流利度 - 添加停顿和语速变化"""
        try:
            audio = AudioSegment.from_file(input_path)
            
            # 1. 添加不自然停顿
            segments = []
            chunk_duration = random.randint(800, 1500)  # 0.8-1.5秒切块
            pause_duration = random.randint(200, 600)   # 200-600ms停顿
            
            for i in range(0, len(audio), chunk_duration):
                chunk = audio[i:i+chunk_duration]
                segments.append(chunk)
                
                # 30%概率添加停顿
                if random.random() < 0.3 and i + chunk_duration < len(audio):
                    silence = AudioSegment.silent(duration=pause_duration)
                    segments.append(silence)
            
            choppy_audio = sum(segments) if segments else audio
            
            # 2. 随机改变语速
            speed_variations = [0.7, 0.8, 1.3, 1.4]  # 明显的语速变化
            speed_factor = random.choice(speed_variations)
            
            # 通过改变播放速度实现语速变化
            choppy_audio = choppy_audio.speedup(playback_speed=speed_factor)
            
            # 3. 导出
            choppy_audio.export(output_path, format="mp3", bitrate="128k")
            
        except Exception as e:
            print(f"      降质流利度失败 {input_path}: {e}")
            # 如果处理失败，复制原文件
            shutil.copy2(input_path, output_path)
    
    def degrade_tone_accuracy(self, input_path: str, output_path: str):
        """专门降低声调准确度 - 音调变化和颤音"""
        temp_wav = os.path.join(self.temp_dir, f"temp_tone_{random.randint(1000,9999)}.wav")
        
        try:
            # 加载音频
            y, sr = librosa.load(input_path, sr=16000)
            
            # 1. 随机音调偏移 (更大的偏移)
            pitch_shift_steps = random.uniform(-4, 4)  # ±4半音偏移
            y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=pitch_shift_steps)
            
            # 2. 添加音调抖动/颤音
            tremolo_rate = random.uniform(2.0, 5.0)  # 颤音频率 2-5Hz
            tremolo_depth = random.uniform(0.2, 0.4)  # 颤音深度
            tremolo = np.sin(2 * np.pi * tremolo_rate * np.arange(len(y_shifted)) / sr)
            y_tremolo = y_shifted * (1 + tremolo_depth * tremolo)
            
            # 3. 轻微时间拉伸
            stretch_factor = random.uniform(0.95, 1.05)
            y_stretched = librosa.effects.time_stretch(y_tremolo, rate=stretch_factor)
            
            # 4. 确保不超出范围并保存
            y_final = np.clip(y_stretched, -0.95, 0.95)
            sf.write(temp_wav, y_final, sr)
            
            # 转换为mp3
            audio = AudioSegment.from_wav(temp_wav)
            audio.export(output_path, format="mp3", bitrate="128k")
            
        except Exception as e:
            print(f"      降质声调失败 {input_path}: {e}")
            shutil.copy2(input_path, output_path)
        finally:
            # 清理临时文件
            if os.path.exists(temp_wav):
                os.remove(temp_wav)
    
    def degrade_phone_accuracy(self, input_path: str, output_path: str):
        """专门降低音素准确度 - 模糊发音和频率失真"""
        temp_wav = os.path.join(self.temp_dir, f"temp_phone_{random.randint(1000,9999)}.wav")
        
        try:
            y, sr = librosa.load(input_path, sr=16000)
            
            # 1. 添加高斯噪音 (模拟不清晰发音)
            noise_factor = random.uniform(0.03, 0.08)
            noise = np.random.normal(0, noise_factor, y.shape)
            y_noisy = y + noise
            
            # 2. 频率域处理 - 模拟发音不准
            fft = np.fft.fft(y_noisy)
            freq_bins = np.fft.fftfreq(len(fft), 1/sr)
            
            # 随机衰减重要的语音频率成分
            critical_freqs = [1000, 2000, 3000, 4000]  # 重要的语音频率
            target_freq = random.choice(critical_freqs)
            bandwidth = random.uniform(300, 700)
            
            # 创建衰减掩码
            mask = np.abs(freq_bins - target_freq) < bandwidth
            attenuation = random.uniform(0.2, 0.5)  # 衰减到20-50%
            fft[mask] *= attenuation
            
            # 同时衰减负频率部分
            mask_neg = np.abs(freq_bins + target_freq) < bandwidth
            fft[mask_neg] *= attenuation
            
            y_filtered = np.real(np.fft.ifft(fft))
            
            # 3. 添加轻微非线性失真
            distortion_level = random.uniform(1.2, 1.8)
            y_distorted = np.tanh(y_filtered * distortion_level) * 0.8
            
            # 4. 轻微低通滤波 (模拟口音)
            from scipy import signal
            nyquist = sr / 2
            cutoff_freq = random.uniform(3500, 6000)  # 截止频率
            sos = signal.butter(4, cutoff_freq/nyquist, btype='low', output='sos')
            y_lowpass = signal.sosfilt(sos, y_distorted)
            
            # 确保不超出范围
            y_final = np.clip(y_lowpass, -0.95, 0.95)
            sf.write(temp_wav, y_final, sr)
            
            # 转换为mp3
            audio = AudioSegment.from_wav(temp_wav)
            audio.export(output_path, format="mp3", bitrate="128k")
            
        except Exception as e:
            print(f"      降质音素失败 {input_path}: {e}")
            # 如果scipy不可用，使用简化版本
            try:
                y, sr = librosa.load(input_path, sr=16000)
                noise = np.random.normal(0, 0.05, y.shape)
                y_noisy = y + noise
                y_final = np.clip(y_noisy, -0.95, 0.95)
                sf.write(temp_wav, y_final, sr)
                audio = AudioSegment.from_wav(temp_wav)
                audio.export(output_path, format="mp3", bitrate="128k")
            except:
                shutil.copy2(input_path, output_path)
        finally:
            if os.path.exists(temp_wav):
                os.remove(temp_wav)
    
    def degrade_stability(self, input_path: str, output_path: str):
        """专门降低稳定性 - 让每个片段有不同程度问题"""
        try:
            audio = AudioSegment.from_file(input_path)
            
            # 为这个特定文件随机选择问题严重程度
            problem_levels = ['light', 'medium', 'severe']
            # 增加severe的概率以确保更大的稳定性差异
            weights = [0.2, 0.3, 0.5]
            problem_severity = random.choices(problem_levels, weights=weights)[0]
            
            if problem_severity == 'light':
                # 轻微问题：小幅音量变化
                volume_change = random.uniform(-4, 4)  # ±4dB
                audio = audio + volume_change
                
            elif problem_severity == 'medium': 
                # 中等问题：语速+音量组合
                speed_factor = random.choice([0.75, 0.85, 1.25, 1.35])
                audio = audio.speedup(playback_speed=speed_factor)
                
                volume_change = random.uniform(-6, 6)  # ±6dB
                audio = audio + volume_change
                
            else:  # severe
                # 严重问题：多重问题组合
                
                # 1. 添加更多停顿
                segments = []
                chunk_size = random.randint(600, 1000)  # 较小的块
                pause_duration = random.randint(300, 800)  # 较长停顿
                
                for i in range(0, len(audio), chunk_size):
                    segments.append(audio[i:i+chunk_size])
                    # 50%概率添加停顿
                    if random.random() < 0.5 and i + chunk_size < len(audio):
                        segments.append(AudioSegment.silent(duration=pause_duration))
                
                audio = sum(segments) if segments else audio
                
                # 2. 极端语速变化
                extreme_speed = random.choice([0.6, 0.7, 1.4, 1.5])
                audio = audio.speedup(playback_speed=extreme_speed)
                
                # 3. 大幅音量变化
                volume_change = random.uniform(-10, 10)  # ±10dB
                audio = audio + volume_change
            
            audio.export(output_path, format="mp3", bitrate="128k")
            
        except Exception as e:
            print(f"      降质稳定性失败 {input_path}: {e}")
            shutil.copy2(input_path, output_path)
    
    def degrade_overall(self, input_path: str, output_path: str):
        """综合降质 - 影响多个维度"""
        temp_wav = os.path.join(self.temp_dir, f"temp_overall_{random.randint(1000,9999)}.wav")
        
        try:
            y, sr = librosa.load(input_path, sr=16000)
            
            # 1. 轻微音调问题
            pitch_shift = random.uniform(-2, 2)
            y = librosa.effects.pitch_shift(y, sr=sr, n_steps=pitch_shift)
            
            # 2. 添加噪音
            noise_level = random.uniform(0.02, 0.06)
            noise = np.random.normal(0, noise_level, y.shape)
            y = y + noise
            
            # 3. 轻微失真
            distortion = random.uniform(1.1, 1.4)
            y = np.tanh(y * distortion) * 0.85
            
            # 4. 确保不超出范围
            y = np.clip(y, -0.95, 0.95)
            
            # 保存为临时wav
            sf.write(temp_wav, y, sr)
            
            # 转换为AudioSegment进行后处理
            audio = AudioSegment.from_wav(temp_wav)
            
            # 5. 添加随机停顿
            if random.random() < 0.6:  # 60%概率添加停顿
                segments = []
                chunk_size = random.randint(1000, 1500)
                pause_duration = random.randint(150, 400)
                
                for i in range(0, len(audio), chunk_size):
                    segments.append(audio[i:i+chunk_size])
                    # 25%概率添加停顿
                    if random.random() < 0.25 and i + chunk_size < len(audio):
                        segments.append(AudioSegment.silent(duration=pause_duration))
                
                audio = sum(segments) if segments else audio
            
            # 6. 轻微音量调整
            volume_change = random.uniform(-3, 3)
            audio = audio + volume_change
            
            audio.export(output_path, format="mp3", bitrate="128k")
            
        except Exception as e:
            print(f"      综合降质失败 {input_path}: {e}")
            shutil.copy2(input_path, output_path)
        finally:
            if os.path.exists(temp_wav):
                os.remove(temp_wav)
    
    def cleanup(self):
        """清理临时文件"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)


class XunfeiComparisonExperiment:
    """科大讯飞API对比实验管理器"""
    
    def __init__(self):
        self.speech_evaluator = SpeechQualityEvaluator()
        self.degrader = XunfeiTargetedDegrader()
    
    def run_comprehensive_experiment(self, baseline_course_folder: str):
        """运行完整对比实验"""
        
        print("🧪 开始科大讯飞API敏感性对比实验")
        print("="*60)
        
        # 验证输入文件夹
        if not self._validate_course_folder(baseline_course_folder):
            return None
        
        try:
            # 1. 测试baseline
            print("\n📊 第一步：测试高质量baseline...")
            baseline_results = self.speech_evaluator.evaluate_course_audio_simple(baseline_course_folder)
            
            if not baseline_results:
                print("❌ Baseline评测失败，无法继续实验")
                return None
            
            print(f"   ✅ Baseline评测成功 - 平均分: {baseline_results.get('average_total_score', 0):.1f}")
            
            # 2. 创建各维度降质样本
            print("\n🔧 第二步：创建降质样本...")
            degraded_root = f"{baseline_course_folder}_degraded_experiment"
            
            # 设置降质文件夹结构
            self._setup_degraded_folders(baseline_course_folder, degraded_root)
            
            # 创建降质音频
            audio_folder = os.path.join(baseline_course_folder, "audiosample")
            self.degrader.create_poor_samples_by_dimension(audio_folder, degraded_root)
            
            # 3. 测试各维度降质效果
            print("\n📉 第三步：测试降质样本...")
            dimension_results = self._test_all_dimensions(degraded_root)
            
            # 4. 生成对比报告
            print("\n📊 第四步：生成对比报告...")
            comparison_analysis = self._analyze_results(baseline_results, dimension_results)
            
            # 5. 生成详细报告
            self._generate_comprehensive_report(baseline_results, dimension_results, comparison_analysis)
            
            # 6. 清理
            self.degrader.cleanup()
            
            print(f"\n🎉 实验完成！降质样本保存在: {degraded_root}")
            
            return {
                'baseline': baseline_results,
                'degraded': dimension_results,
                'analysis': comparison_analysis,
                'degraded_folder': degraded_root
            }
            
        except Exception as e:
            print(f"❌ 实验过程中出现错误: {e}")
            self.degrader.cleanup()
            return None
    
    def _validate_course_folder(self, course_folder: str) -> bool:
        """验证课程文件夹结构"""
        if not os.path.exists(course_folder):
            print(f"❌ 课程文件夹不存在: {course_folder}")
            return False
        
        audio_folder = os.path.join(course_folder, "audiosample")
        text_folder = os.path.join(course_folder, "textsample")
        
        if not os.path.exists(audio_folder):
            print(f"❌ 缺少音频文件夹: {audio_folder}")
            return False
        
        if not os.path.exists(text_folder):
            print(f"❌ 缺少文本文件夹: {text_folder}")
            return False
        
        audio_files = [f for f in os.listdir(audio_folder) if f.endswith('.mp3')]
        text_files = [f for f in os.listdir(text_folder) if f.endswith('.txt')]
        
        if len(audio_files) == 0:
            print(f"❌ 音频文件夹中没有MP3文件")
            return False
        
        if len(text_files) == 0:
            print(f"❌ 文本文件夹中没有TXT文件")
            return False
        
        print(f"✅ 找到 {len(audio_files)} 个音频文件和 {len(text_files)} 个文本文件")
        return True
    
    def _setup_degraded_folders(self, baseline_folder: str, degraded_root: str):
        """设置降质实验文件夹结构"""
        print("   📁 设置文件夹结构...")
        
        text_folder = os.path.join(baseline_folder, "textsample")
        dimensions = ['poor_fluency', 'poor_tone', 'poor_phone', 'poor_stability', 'poor_overall']
        
        for dim in dimensions:
            dim_folder = os.path.join(degraded_root, dim)
            dim_text_folder = os.path.join(dim_folder, "textsample")
            
            os.makedirs(dim_text_folder, exist_ok=True)
            
            # 复制文本文件
            if os.path.exists(text_folder):
                for file in os.listdir(text_folder):
                    if file.endswith('.txt'):
                        shutil.copy2(
                            os.path.join(text_folder, file),
                            os.path.join(dim_text_folder, file)
                        )
        
        print("   ✅ 文件夹结构设置完成")
    
    def _test_all_dimensions(self, degraded_root: str) -> Dict:
        """测试所有维度的降质效果"""
        dimension_results = {}
        
        dimensions = {
            'poor_fluency': '流利度降质',
            'poor_tone': '声调降质', 
            'poor_phone': '音素降质',
            'poor_stability': '稳定性降质',
            'poor_overall': '综合降质'
        }
        
        for dim_key, dim_name in dimensions.items():
            print(f"   🔍 测试 {dim_name}...")
            dim_folder = os.path.join(degraded_root, dim_key)
            
            try:
                result = self.speech_evaluator.evaluate_course_audio_simple(dim_folder)
                dimension_results[dim_key] = result
                
                if result:
                    print(f"      ✅ 完成 - 平均分: {result.get('average_total_score', 0):.1f}")
                else:
                    print(f"      ❌ 评测失败")
                    
            except Exception as e:
                print(f"      ❌ 测试失败: {e}")
                dimension_results[dim_key] = None
        
        return dimension_results
    
    def _analyze_results(self, baseline: Dict, degraded_results: Dict) -> Dict:
        """分析实验结果"""
        analysis = {
            'changes': {},
            'sensitivity': {},
            'effectiveness': {}
        }
        
        if not baseline:
            return analysis
        
        # 分析各维度变化
        for dim, result in degraded_results.items():
            if result is None:
                continue
            
            changes = {}
            sensitivity = {}
            
            # 计算各指标变化
            metrics = ['average_fluency', 'average_tone_accuracy', 'average_phone_accuracy', 'average_total_score']
            
            for metric in metrics:
                if metric in baseline and metric in result:
                    change = baseline[metric] - result[metric]
                    changes[metric] = change
                    # 判断是否为显著变化 (>3分为显著)
                    sensitivity[metric] = abs(change) > 3
            
            # 稳定性变化 (数值越大越不稳定)
            if 'stability' in baseline and 'stability' in result:
                stability_change = result['stability'] - baseline['stability']
                changes['stability'] = stability_change
                sensitivity['stability'] = stability_change > 2  # 标准差增加>2为显著
            
            analysis['changes'][dim] = changes
            analysis['sensitivity'][dim] = sensitivity
            
            # 评估降质效果
            significant_changes = sum(sensitivity.values())
            total_metrics = len(sensitivity)
            
            if significant_changes >= total_metrics * 0.7:
                effectiveness = 'high'
            elif significant_changes >= total_metrics * 0.4:
                effectiveness = 'medium'
            else:
                effectiveness = 'low'
            
            analysis['effectiveness'][dim] = effectiveness
        
        return analysis
    
    def _generate_comprehensive_report(self, baseline: Dict, degraded_results: Dict, analysis: Dict):
        """生成综合报告"""
        
        print(f"\n{'='*70}")
        print("📊 科大讯飞API敏感性测试 - 详细报告")
        print(f"{'='*70}")
        
        if not baseline:
            print("❌ 无法生成报告：Baseline数据缺失")
            return
        
        # Baseline结果
        print(f"\n📈 Baseline（高质量样本）结果:")
        print(f"   🗣️  流利度: {baseline.get('average_fluency', 0):.2f}")
        print(f"   🎵 声调准确度: {baseline.get('average_tone_accuracy', 0):.2f}")
        print(f"   🔤 音素准确度: {baseline.get('average_phone_accuracy', 0):.2f}")
        print(f"   🎯 总分: {baseline.get('average_total_score', 0):.2f}")
        print(f"   📊 稳定性: {baseline.get('stability', 0):.2f}")
        print(f"   📋 样本数: {baseline.get('sample_count', 0)}")
        
        # 降质结果对比
        print(f"\n📉 降质样本测试结果:")
        
        dimension_names = {
            'poor_fluency': '🗣️  流利度降质',
            'poor_tone': '🎵 声调降质', 
            'poor_phone': '🔤 音素降质',
            'poor_stability': '📊 稳定性降质',
            'poor_overall': '🎯 综合降质'
        }
        
        for dim, result in degraded_results.items():
            if result is None:
                print(f"\n{dimension_names.get(dim, dim)}: ❌ 测试失败")
                continue
            
            print(f"\n{dimension_names.get(dim, dim)}:")
            
            # 显示各项指标
            metrics_display = {
                'average_fluency': '流利度',
                'average_tone_accuracy': '声调',
                'average_phone_accuracy': '音素', 
                'average_total_score': '总分',
                'stability': '稳定性'
            }
            
            for metric, display_name in metrics_display.items():
                if metric in result:
                    value = result[metric]
                    baseline_value = baseline.get(metric, 0)
                    
                    if metric == 'stability':
                        change = value - baseline_value
                        direction = "↗️" if change > 0 else "↘️" if change < 0 else "→"
                        print(f"   {display_name}: {value:.2f} {direction} (变化: {change:+.2f})")
                    else:
                        change = baseline_value - value
                        direction = "↘️" if change > 0 else "↗️" if change < 0 else "→"
                        print(f"   {display_name}: {value:.2f} {direction} (下降: {change:.2f})")
            
            # 显示效果评估
            effectiveness = analysis.get('effectiveness', {}).get(dim, 'unknown')
            effectiveness_symbols = {
                'high': '🔴 高效果',
                'medium': '🟡 中等效果', 
                'low': '🟢 低效果',
                'unknown': '❓ 未知'
            }
            print(f"   降质效果: {effectiveness_symbols.get(effectiveness, effectiveness)}")
        
        # API敏感性分析
        print(f"\n🎯 API敏感性分析:")
        
        sensitivity_summary = {}
        for dim, sensitivity_data in analysis.get('sensitivity', {}).items():
            sensitive_count = sum(sensitivity_data.values())
            total_metrics = len(sensitivity_data)
            sensitivity_ratio = sensitive_count / total_metrics if total_metrics > 0 else 0
            sensitivity_summary[dim] = sensitivity_ratio
        
        for dim, ratio in sensitivity_summary.items():
            dim_name = dimension_names.get(dim, dim)
            if ratio >= 0.7:
                level = "🔴 高敏感"
                desc = "API能很好地检测此类问题"
            elif ratio >= 0.4:
                level = "🟡 中敏感"
                desc = "API能部分检测此类问题"
            else:
                level = "🟢 低敏感"
                desc = "API对此类问题不够敏感"
            
            print(f"   {dim_name}: {level} ({ratio:.1%}) - {desc}")
        
        # 总体评估
        print(f"\n🏆 总体评估:")
        
        avg_sensitivity = statistics.mean(sensitivity_summary.values()) if sensitivity_summary else 0
        
        if avg_sensitivity >= 0.7:
            overall_rating = "🌟 优秀"
            recommendation = "科大讯飞API具有很强的区分能力，适合用于教学质量评估"
        elif avg_sensitivity >= 0.5:
            overall_rating = "👍 良好"
            recommendation = "科大讯飞API具有较好的区分能力，可以用于教学质量评估"
        elif avg_sensitivity >= 0.3:
            overall_rating = "🤔 一般"
            recommendation = "科大讯飞API的区分能力中等，需要结合其他指标使用"
        else:
            overall_rating = "⚠️ 较弱"
            recommendation = "科大讯飞API的区分能力较弱，建议谨慎使用"
        
        print(f"   API区分能力: {overall_rating} (平均敏感性: {avg_sensitivity:.1%})")
        print(f"   使用建议: {recommendation}")
        
        # 最有效的降质方法
        if analysis.get('effectiveness'):
            most_effective = max(analysis['effectiveness'].items(), key=lambda x: x[1] if x[1] != 'unknown' else '')
            if most_effective[1] != 'unknown':
                print(f"   最有效降质方法: {dimension_names.get(most_effective[0], most_effective[0])}")
        
        print("="*70)
        print("✅ 实验报告生成完成")


def main():
    """主函数 - 运行完整实验"""
    
    print("🎓 科大讯飞语音评测API敏感性验证实验")
    print("="*50)
    
    # 请修改此路径为你的高质量课程文件夹路径
    # 文件夹应包含 audiosample/ 和 textsample/ 子文件夹
    baseline_course_folder = "/Users/zhangshenao/Desktop/research/交大教育大模型测评/eval/Resources/sample/1"
    
    # 检查路径是否存在
    if not os.path.exists(baseline_course_folder):
        print(f"❌ 请修改 baseline_course_folder 路径")
        print(f"   当前路径: {baseline_course_folder}")
        print(f"   该路径不存在，请设置为你的实际课程文件夹路径")
        return
    
    # 创建实验管理器
    experiment = XunfeiComparisonExperiment()
    
    # 运行完整实验
    try:
        results = experiment.run_comprehensive_experiment(baseline_course_folder)
        
        if results:
            print(f"\n🎉 实验成功完成！")
            print(f"📁 结果保存位置: {results.get('degraded_folder', 'N/A')}")
            
            # 可选：保存结果到JSON文件
            import json
            result_file = f"{baseline_course_folder}_experiment_results.json"
            
            # 准备可序列化的结果
            serializable_results = {
                'baseline': results['baseline'],
                'degraded_summary': {k: v for k, v in results['degraded'].items() if v is not None},
                'analysis_summary': results['analysis']
            }
            
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_results, f, ensure_ascii=False, indent=2)
            
            print(f"📄 详细结果已保存到: {result_file}")
            
        else:
            print("❌ 实验失败，请检查错误信息")
            
    except KeyboardInterrupt:
        print("\n⏹️ 实验被用户中断")
    except Exception as e:
        print(f"❌ 实验过程中发生未预期的错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()