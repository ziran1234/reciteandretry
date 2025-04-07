import nltk
from nltk.tokenize import word_tokenize
from collections import Counter

nltk.download('punkt')  # 确保下载分词器

def clean_text(text):
    """ 清理百度语音识别文本，检测重复和不流畅部分 """
    words = word_tokenize(text)  # 分词
    cleaned_words = []
    repetition_count = Counter()

    for i, word in enumerate(words):
        if i > 0 and word == words[i - 1]:  # 处理连续重复
            repetition_count[word] += 1
            if repetition_count[word] > 2:  # 超过2次标记为“不熟练”
                continue
        else:
            repetition_count[word] = 1

        cleaned_words.append(word)

    cleaned_text = " ".join(cleaned_words)
    return cleaned_text, repetition_count

# 示例文本（百度语音识别输出）
baidu_text = "床前 明月光 明月光 疑是地上霜 霜 举头望明月 低头 低头 思故乡"

cleaned_text, repetition_info = clean_text(baidu_text)

print("✅ 整理后的文本:", cleaned_text)
print("🔍 重复词检测:", repetition_info)
