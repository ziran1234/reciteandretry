from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import (
    Trainer,
    TrainingArguments,
    AutoModelForSequenceClassification,
    AutoTokenizer
)

# 初始化模型和分词器
model_name = "microsoft/deberta-v3-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=6)

# 加载并预处理数据
emotions = load_dataset("emotion")
tokenized_emotions = emotions.map(
    lambda x: tokenizer(x["text"], padding=True, truncation=True),
    batched=True
).remove_columns(["text"])  # 清理原始文本

# 训练参数配置
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    fp16=True,
    gradient_accumulation_steps=2,
    metric_for_best_model="f1_macro",
)

# 自定义评估指标
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    print(classification_report(labels, preds))
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
        "f1_weighted": f1_score(labels, preds, average="weighted")
    }

# 创建Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_emotions["train"],
    eval_dataset=tokenized_emotions["validation"],
    compute_metrics=compute_metrics,
)

# 执行训练
trainer.train()

# 最终评估与保存
trainer.evaluate(tokenized_emotions["test"])
trainer.save_model("deberta_emotion_classifier")
tokenizer.save_pretrained("deberta_emotion_classifier")
