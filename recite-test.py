import requests
import json
import base64
import torch
import os
import numpy as np
import jieba
import librosa
import re
import soundfile as sf
from collections import deque
from modelscope import AutoTokenizer, AutoModel
from sklearn.metrics import accuracy_score

# 配置参数
API_KEY = ""
SECRET_KEY = ""
MODEL_PATH = ""
STUDENT_NAME = "同学"
PAUSE_THRESHOLD = 0.3  # 基础停顿阈值（秒）
VALID_PAUSE_WINDOW = 0.5  # 标点周围合理停顿区间（秒）

# 百度语音识别接口
def get_access_token():
    url = "https://aip.baidubce.com/oauth/2.0/token"
    params = {"grant_type": "client_credentials", "client_id": API_KEY, "client_secret": SECRET_KEY}
    try:
        response = requests.post(url, params=params)
        return response.json().get("access_token")
    except Exception as e:
        print(f"获取Token失败: {e}")
        return None

def speech_to_text(audio_path):
    token = get_access_token()
    if not token:
        return None
    
    try:
        with open(audio_path, "rb") as f:
            audio_data = f.read()
        
        payload = json.dumps({
            "format": "pcm",
            "rate": 16000,
            "channel": 1,
            "cuid": "recite_sys",
            "token": token,
            "speech": base64.b64encode(audio_data).decode("utf-8"),
            "len": len(audio_data)
        })
        
        response = requests.post(
            "https://vop.baidu.com/server_api",
            headers={"Content-Type": "application/json"},
            data=payload
        )
        return response.json().get("result", [""])[0]
    except Exception as e:
        print(f"语音识别失败: {e}")
        return None

# 语义模型初始化
if os.path.exists(MODEL_PATH):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModel.from_pretrained(MODEL_PATH)
else:
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# 文本处理模块
def semantic_deduplication(text, threshold=0.9):
    words = jieba.lcut(text)
    if len(words) < 2:
        return text
    
    cleaned = [words[0]]
    last_emb = None
    
    for word in words[1:]:
        if word == cleaned[-1]:
            continue
        
        inputs = tokenizer(word, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        emb = mean_pooling(outputs, inputs["attention_mask"]).numpy().flatten()
        
        if last_emb is not None:
            sim = np.dot(emb, last_emb) / (np.linalg.norm(emb) * np.linalg.norm(last_emb))
            if sim >= threshold:
                continue
        
        cleaned.append(word)
        last_emb = emb
    
    if len(cleaned) > 1 and cleaned[-1] == cleaned[-2]:
        cleaned.pop()
    
    return "".join(cleaned)

# 音频分析模块
def calculate_speed(audio_path):
    y, sr = librosa.load(audio_path, sr=16000)
    duration = librosa.get_duration(y=y, sr=sr)
    text = speech_to_text(audio_path)
    return len(text.split()) / (duration / 60) if text else 0

def analyze_pauses(audio_path, reference):
    y, sr = librosa.load(audio_path, sr=16000)
    intervals = librosa.effects.split(y, top_db=20)
    total_duration = librosa.get_duration(y=y, sr=sr)
    
    # 计算预期停顿位置
    sentence_ends = [i for i, c in enumerate(reference) if c in "。！？；"]
    char_duration = total_duration / len(reference) if reference else 0
    valid_windows = [
        (pos*char_duration - VALID_PAUSE_WINDOW, pos*char_duration + VALID_PAUSE_WINDOW)
        for pos in sentence_ends
    ]
    
    valid_pauses = 0
    invalid_pauses = 0
    for start, end in intervals:
        duration = (end - start)/sr
        if duration < PAUSE_THRESHOLD:
            continue
        
        pause_time = start/sr
        if any(lower <= pause_time <= upper for lower, upper in valid_windows):
            valid_pauses += 1
        else:
            invalid_pauses += 1
    
    return valid_pauses, invalid_pauses

# 文本分段处理
def split_text(text, method="semantic"):
    if method == "semantic":
        return _semantic_split(text)
    return _punctuation_split(text)

def _punctuation_split(text, max_len=50):
    segments = []
    current = []
    current_len = 0
    
    for part in re.split(r"([。！？；])", text):
        part_len = len(part)
        if current_len + part_len > max_len:
            segments.append("".join(current))
            current = []
            current_len = 0
        current.append(part)
        current_len += part_len
    
    if current:
        segments.append("".join(current))
    return segments

def _semantic_split(text):
    sentences = text.split("。")
    segments = []
    current = []
    window = deque(maxlen=2)
    
    for sent in sentences:
        if not sent.strip():
            continue
        
        inputs = tokenizer(sent, return_tensors="pt", truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        emb = mean_pooling(outputs, inputs["attention_mask"])
        
        if window:
            sim = torch.nn.functional.cosine_similarity(emb, window[-1], dim=1)
            if sim < 0.7 and len(current) > 0:
                segments.append("。".join(current) + "。")
                current = []
        
        current.append(sent)
        window.append(emb)
    
    if current:
        segments.append("。".join(current) + "。")
    return segments

# 报告生成系统
def generate_report(reference, hypothesis, audio_path):
    # 文本对齐
    ref_segs = split_text(reference)
    hyp_segs = split_text(hypothesis)
    min_len = min(len(ref_segs), len(hyp_segs))
    ref_segs = ref_segs[:min_len]
    hyp_segs = hyp_segs[:min_len]
    
    # 分段评估
    seg_details = []
    total_cer, total_wer = 0, 0
    for i, (ref, hyp) in enumerate(zip(ref_segs, hyp_segs)):
        cer = _calculate_cer(ref, hyp)
        wer = _calculate_wer(ref, hyp)
        total_cer += cer
        total_wer += wer
        seg_details.append(
            f"\n🔹 段落 {i+1}:\n正确: {ref}\n识别: {hyp}\nCER: {cer:.2%}  WER: {wer:.2%}"
        )
    
    # 综合指标
    avg_cer = total_cer / min_len
    accuracy = (1 - avg_cer) * 100
    wpm = calculate_speed(audio_path)
    valid_pauses, invalid_pauses = analyze_pauses(audio_path, reference)
    
    # 个性化反馈
    report = f"""
🌟 {STUDENT_NAME}的学习评估报告 🌟
========================================
📊 综合表现：
  准确率：{accuracy:.1f}% ({_get_accuracy_comment(accuracy)})
  语速：{wpm:.1f} WPM ({_get_speed_comment(wpm)})
  停顿分析：有效停顿 {valid_pauses} 次 | 多余停顿 {invalid_pauses} 次
  {_get_pause_comment(invalid_pauses)}

📝 详细分析：{"".join(seg_details)}

💡 学习建议：
{_get_study_advice(accuracy, invalid_pauses)}
========================================"""
    return report

def _calculate_cer(ref, hyp):
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

def _calculate_wer(ref, hyp):
    ref_words = ref.split()
    hyp_words = hyp.split()
    return 1 - accuracy_score(ref_words, hyp_words)

# 个性化反馈生成
def _get_accuracy_comment(accuracy):
    if accuracy == 100:
        return "完美表现！🎉"
    elif accuracy >= 90:
        return "非常优秀！✨"
    elif accuracy >= 75:
        return "良好，继续努力！💪"
    else:
        return "需要更多练习哦~📚"

def _get_speed_comment(wpm):
    avg = 120
    if wpm > avg*1.2:
        return "语速偏快"
    elif wpm < avg*0.8:
        return "语速偏慢"
    return "语速适中"

def _get_pause_comment(invalid_pauses):
    if invalid_pauses == 0:
        return "停顿控制非常合理！"
    elif invalid_pauses <= 2:
        return "注意不必要的停顿"
    return "请加强流畅度练习"

def _get_study_advice(accuracy, bad_pauses):
    advice = []
    if accuracy < 75:
        advice.append("- 重点练习标红错误段落")
        advice.append("- 每天进行跟读训练")
    elif bad_pauses > 3:
        advice.append("- 使用'影子跟读法'改善流畅度")
        advice.append("- 录音回听寻找停顿问题")
    else:
        advice.append("- 保持良好学习节奏")
        advice.append("- 尝试挑战更长篇章")
    return "\n".join(advice)

# 主程序
if __name__ == "__main__":
    # 配置路径
    audio_path = "/home/humble/recite/6.wav"
    correct_text_path = "/home/humble/recite/target.txt"
    
    # 读取正确文本
    try:
        with open(correct_text_path, "r", encoding="utf-8") as f:
            correct_text = f.read().strip()
    except Exception as e:
        print(f"无法读取正确文本: {e}")
        exit(1)
    
    # 长音频处理
    if librosa.get_duration(filename=audio_path) > 60:
        print("检测到长音频，进行分段处理...")
        temp_files = []
        y, sr = librosa.load(audio_path, sr=16000)
        for i, start in enumerate(range(0, len(y), 30*sr)):
            end = min(start+30*sr, len(y))
            seg_path = f"/tmp/segment_{i}.wav"
            sf.write(seg_path, y[start:end], sr)
            temp_files.append(seg_path)
        
        full_text = []
        for seg in temp_files:
            if text := speech_to_text(seg):
                full_text.append(text)
            os.remove(seg)
        input_text = "".join(full_text)
    else:
        input_text = speech_to_text(audio_path)
    
    if not input_text:
        print("语音识别失败")
        exit(1)
    
    # 处理并生成报告
    cleaned_text = semantic_deduplication(input_text)
    print("\n🔍 处理后的识别文本:", cleaned_text)
    
    report = generate_report(correct_text, cleaned_text, audio_path)
    print(report)
