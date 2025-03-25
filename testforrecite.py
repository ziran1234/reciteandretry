"""
语音背诵评估系统
功能：语音识别 -> 文本处理 -> 语音分析 -> 生成评估报告
作者：智能助理
版本：2.1
"""

import os
import re
import json
import base64
import librosa
import soundfile as sf
import numpy as np
import requests
import torch
import jieba
import psutil
from pathlib import Path
from collections import deque
from sklearn.metrics import accuracy_score
from modelscope import AutoTokenizer, AutoModel

# ################### 配置区 ################### #
class Config:
    # 百度语音识别
    BAIDU_API_KEY = "yrwh3nyKdMae5766IpfEdbph"
    BAIDU_SECRET_KEY = "OMbwiOvpzpnx2fgrwQioCdJf9Vekb8IC"
    
    # 路径配置
    MODEL_PATH = "/home/humble/addition/paraphrase-multilingual-MiniLM-L12-v2"
    TEMP_DIR = "/tmp/recite_system"
    
    # 评估参数
    STUDENT_NAME = "学员"
    PAUSE_THRESHOLD = 0.3  # 停顿检测阈值(秒)
    VALID_PAUSE_WINDOW = 0.5  # 合理停顿区间(秒)
    MAX_SEGMENT_DURATION = 60  # 长音频分段时长(秒)
    
    # 语义处理
    DEDUP_THRESHOLD = 0.85  # 语义去重阈值

# ################### 工具函数 ################### #
def validate_path(path: str) -> Path:
    """安全路径验证"""
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"路径不存在: {path}")
    return path

def memory_monitor(func):
    """内存监控装饰器"""
    def wrapper(*args, **kwargs):
        process = psutil.Process()
        start_mem = process.memory_info().rss
        result = func(*args, **kwargs)
        delta = (process.memory_info().rss - start_mem) / 1024 / 1024
        if delta > 100:  # 100MB阈值
            print(f"[内存警告] {func.__name__} 内存增长: {delta:.2f}MB")
        return result
    return wrapper

# ################### 核心模块 ################### #
class SpeechRecognizer:
    """语音识别模块"""
    
    @staticmethod
    def get_access_token():
        """获取百度API令牌"""
        url = "https://aip.baidubce.com/oauth/2.0/token"
        params = {
            "grant_type": "client_credentials",
            "client_id": Config.BAIDU_API_KEY,
            "client_secret": Config.BAIDU_SECRET_KEY
        }
        try:
            resp = requests.post(url, params=params, timeout=10)
            return resp.json().get("access_token")
        except Exception as e:
            raise RuntimeError(f"获取Token失败: {str(e)}")

    @classmethod
    @memory_monitor
    def recognize(cls, audio_path: str) -> str:
        """语音转文本"""
        audio_path = validate_path(audio_path)
        
        # 检查音频格式
        if audio_path.suffix.lower() != ".wav":
            raise ValueError("仅支持WAV格式音频")
        
        # 获取访问令牌
        token = cls.get_access_token()
        if not token:
            raise RuntimeError("语音服务不可用")

        # 读取并编码音频
        with open(audio_path, "rb") as f:
            audio_data = base64.b64encode(f.read()).decode()

        payload = json.dumps({
            "format": "pcm",
            "rate": 16000,
            "channel": 1,
            "token": token,
            "speech": audio_data,
            "len": len(audio_data)
        })

        try:
            resp = requests.post(
                "https://vop.baidu.com/server_api",
                headers={"Content-Type": "application/json"},
                data=payload,
                timeout=15
            )
            return resp.json().get("result", [""])[0]
        except Exception as e:
            raise RuntimeError(f"识别失败: {str(e)}")

class TextProcessor:
    """文本处理模块"""
    
    def __init__(self):
        # 初始化语义模型
        self.tokenizer, self.model = self._load_model()
        
    def _load_model(self):
        """加载语义模型"""
        model_path = validate_path(Config.MODEL_PATH)
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModel.from_pretrained(model_path)
            print("✅ 本地模型加载成功")
            return tokenizer, model
        except:
            print("⚠️ 加载本地模型失败，使用在线模型")
            return AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"), \
                   AutoModel.from_pretrained("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    @staticmethod
    def _mean_pooling(model_output, attention_mask):
        """均值池化"""
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    @memory_monitor
    def deduplicate(self, text: str) -> str:
        """语义去重"""
        words = jieba.lcut(text)
        if len(words) < 2:
            return text

        cleaned = [words[0]]
        last_emb = None

        for word in words[1:]:
            if word == cleaned[-1]:
                continue

            inputs = self.tokenizer(word, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                outputs = self.model(**inputs)
            emb = self._mean_pooling(outputs, inputs["attention_mask"]).numpy().flatten()

            if last_emb is not None:
                sim = np.dot(emb, last_emb) / (np.linalg.norm(emb) * np.linalg.norm(last_emb))
                if sim >= Config.DEDUP_THRESHOLD:
                    continue

            cleaned.append(word)
            last_emb = emb

        return "".join(cleaned)

class AudioAnalyzer:
    """语音分析模块"""
    
    @staticmethod
    @memory_monitor
    def analyze(audio_path: str, reference_text: str) -> dict:
        """全面语音分析"""
        y, sr = librosa.load(str(validate_path(audio_path)), sr=16000)
        duration = librosa.get_duration(y=y, sr=sr)
        
        # 语速计算
        word_count = len(reference_text.split())
        wpm = word_count / (duration / 60) if duration > 0 else 0
        
        # 停顿分析
        valid, invalid = AudioAnalyzer._analyze_pauses(y, sr, reference_text)
        
        return {
            "duration": duration,
            "wpm": wpm,
            "valid_pauses": valid,
            "invalid_pauses": invalid
        }

    @staticmethod
    def _analyze_pauses(y, sr, reference):
        """智能停顿分析"""
        intervals = librosa.effects.split(y, top_db=20)
        total_duration = len(y) / sr
        char_duration = total_duration / len(reference) if reference else 0
        
        # 生成合理停顿窗口
        valid_windows = []
        for idx, char in enumerate(reference):
            if char in "。！？；":
                pos = idx * char_duration
                valid_windows.append( (pos - Config.VALID_PAUSE_WINDOW, pos + Config.VALID_PAUSE_WINDOW) )
        
        # 分类停顿
        valid = invalid = 0
        for start, end in intervals:
            pause_dur = (end - start) / sr
            if pause_dur < Config.PAUSE_THRESHOLD:
                continue
                
            pause_time = start / sr
            if any(lower <= pause_time <= upper for lower, upper in valid_windows):
                valid += 1
            else:
                invalid += 1
                
        return valid, invalid

# ################### 业务逻辑 ################### #
class ReciteEvaluator:
    """背诵评估系统"""
    
    def __init__(self):
        self.text_processor = TextProcessor()
        self.audio_analyzer = AudioAnalyzer()

    def process_audio(self, audio_path: str) -> str:
        """处理长音频"""
        validate_path(audio_path)
        
        # 创建临时目录
        temp_dir = Path(Config.TEMP_DIR)
        temp_dir.mkdir(exist_ok=True, parents=True)
        
        # 长音频分段处理
        if librosa.get_duration(filename=audio_path) > Config.MAX_SEGMENT_DURATION:
            return self._process_long_audio(audio_path, temp_dir)
        return SpeechRecognizer.recognize(audio_path)

    def _process_long_audio(self, audio_path: str, temp_dir: Path) -> str:
        """处理超过1分钟的音频"""
        y, sr = librosa.load(audio_path, sr=16000)
        full_text = []
        
        for i, start in enumerate(range(0, len(y), Config.MAX_SEGMENT_DURATION * sr)):
            end = min(start + Config.MAX_SEGMENT_DURATION * sr, len(y))
            seg_path = temp_dir / f"segment_{i}.wav"
            sf.write(seg_path, y[start:end], sr)
            
            try:
                text = SpeechRecognizer.recognize(str(seg_path))
                full_text.append(text)
            finally:
                seg_path.unlink()
                
        return "".join(full_text)

    def generate_report(self, audio_path: str, reference_path: str) -> str:
        """生成评估报告"""
        # 读取正确文本
        with open(validate_path(reference_path), "r", encoding="utf-8") as f:
            reference = f.read().strip()
        
        # 语音识别
        try:
            raw_text = self.process_audio(audio_path)
            cleaned_text = self.text_processor.deduplicate(raw_text)
        except Exception as e:
            return f"❌ 处理失败: {str(e)}"

        # 语音分析
        audio_stats = self.audio_analyzer.analyze(audio_path, reference)
        
        # 生成报告
        return self._format_report(
            reference=reference,
            hypothesis=cleaned_text,
            audio_stats=audio_stats
        )

    def _format_report(self, reference: str, hypothesis: str, audio_stats: dict) -> str:
        """生成格式化报告"""
        # 文本对比
        cer = self._calculate_cer(reference, hypothesis)
        wer = self._calculate_wer(reference, hypothesis)
        accuracy = (1 - cer) * 100
        
        # 报告模板
        report = f"""
        🌟 {Config.STUDENT_NAME}的背诵评估报告 🌟
        {"="*50}
        📊 综合表现：
          准确率：{accuracy:.1f}% ({self._accuracy_comment(accuracy)})
          语速：{audio_stats['wpm']:.1f} WPM ({self._speed_comment(audio_stats['wpm'])})
          停顿分析：合理停顿 {audio_stats['valid_pauses']} 次 | 多余停顿 {audio_stats['invalid_pauses']} 次
        
        📝 详细分析：
        {self._segment_analysis(reference, hypothesis)}
        
        💡 学习建议：
        {self._generate_advice(accuracy, audio_stats['invalid_pauses'])}
        {"="*50}
        """
        return report

    def _segment_analysis(self, ref: str, hyp: str) -> str:
        """分段分析"""
        segments = self._split_text(ref), self._split_text(hyp)
        analysis = []
        
        for i, (r, h) in enumerate(zip(*segments)):
            cer = self._calculate_cer(r, h)
            analysis.append(
                f"🔹 段落{i+1}:\n正确: {r}\n识别: {h}\nCER: {cer:.2%}\n"
            )
        return "\n".join(analysis)

    @staticmethod
    def _split_text(text: str) -> list:
        """智能分段"""
        sentences = re.split(r"([。！？；])", text)
        segments = []
        current = []
        
        for s in sentences:
            if len("".join(current)) + len(s) > 50:
                segments.append("".join(current))
                current = []
            current.append(s)
            
        if current:
            segments.append("".join(current))
        return segments

    @staticmethod
    def _calculate_cer(ref: str, hyp: str) -> float:
        """字符错误率"""
        d = np.zeros((len(ref)+1, len(hyp)+1))
        for i in range(len(ref)+1):
            for j in range(len(hyp)+1):
                if i == 0:
                    d[i][j] = j
                elif j == 0:
                    d[i][j] = i
                else:
                    cost = 0 if ref[i-1] == hyp[j-1] else 1
                    d[i][j] = min(d[i-1][j]+1, d[i][j-1]+1, d[i-1][j-1]+cost)
        return d[len(ref)][len(hyp)] / len(ref)

    @staticmethod
    def _calculate_wer(ref: str, hyp: str) -> float:
        """词错误率"""
        ref_words = ref.split()
        hyp_words = hyp.split()
        return 1 - accuracy_score(ref_words, hyp_words)

    @staticmethod
    def _accuracy_comment(accuracy: float) -> str:
        """准确率评语"""
        if accuracy == 100:
            return "完美背诵！🎉"
        elif accuracy >= 90:
            return "非常优秀！✨"
        elif accuracy >= 75:
            return "良好，继续努力！💪"
        return "需要更多练习~📚"

    @staticmethod
    def _speed_comment(wpm: float) -> str:
        """语速评语"""
        avg = 120
        if wpm > avg * 1.2:
            return "语速偏快"
        if wpm < avg * 0.8:
            return "语速偏慢"
        return "语速适中"

    @staticmethod
    def _generate_advice(accuracy: float, bad_pauses: int) -> str:
        """生成学习建议"""
        advice = []
        if accuracy < 75:
            advice.extend([
                "- 重点练习错误段落",
                "- 每天进行跟读训练"
            ])
        elif bad_pauses > 3:
            advice.extend([
                "- 使用'影子跟读法'改善流畅度",
                "- 录音回听寻找停顿问题"
            ])
        else:
            advice.extend([
                "- 保持良好学习节奏",
                "- 尝试情感化表达练习"
            ])
        return "\n".join(advice)

# ################### 主程序 ################### #
if __name__ == "__main__":
    # 初始化评估系统
    evaluator = ReciteEvaluator()
    
    try:
        # 输入路径
        audio_file = "/home/humble/recite/audio.wav"
        reference_file = "/home/humble/recite/correct.txt"
        
        # 生成报告
        report = evaluator.generate_report(audio_file, reference_file)
        print(report)
        
    except Exception as e:
        print(f"❌ 系统运行出错: {str(e)}")
    finally:
        # 清理临时文件
        temp_dir = Path(Config.TEMP_DIR)
        if temp_dir.exists():
            for f in temp_dir.glob("*"):
                f.unlink()
            temp_dir.rmdir()
