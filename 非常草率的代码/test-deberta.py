from datasets import load_dataset
from sklearn.metrics import accuracy_score,f1_score
from transformers import Trainer,TrainingArguments
from transformers import AutoModelForSequenceClassification,AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("deberta-v3-large")
model = AutoModelForSequenceClassification.from_pretrained("result/checkpoint-500",num_labels=6)
emotions = load_dataset("diting")

def tokenize(batch):
    return tokenizer(batch["text"],padding=True,truncation=True)
tokenized_emotions = emotions.map(tokenize,batched=True,batch_size=None)

tokenized_emotions.set_format("torch",columns=["input_ids","attention_mask","label"])

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels,preds,average="weighted")
    acc = accuracy_score(labels,preds)
    return {"accuracy":acc,"f1":f1}

training_args = TrainingArguments(output_dir="result",)

trainer = Trainer(model=model,args=training_args,compute_metrics=compute_metrics,
                  train_dataset=tokenized_emotions["train"]
                  ,eval_dataset=tokenized_emotions["validation"])

results = trainer.evaluate()
print(results)
