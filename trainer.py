from transformers import (
    T5ForConditionalGeneration,
    Seq2SeqTrainer,
    T5Tokenizer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)
import torch
from datasets import concatenate_datasets, load_dataset, Dataset
import evaluate
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")
accuracy = evaluate.load("accuracy")

def load_and_preprocess_data():
    sentiment = load_dataset("imdb")
    sentiment = sentiment.map(lambda x: {
        "input_text": "sentiment: " + x['text'].strip().replace('<br />', ' '),
        "target_text": "positive" if x["label"] == 1 else "negative"
    }, remove_columns=["text", "label"])

    summarization = load_dataset("cnn_dailymail", "3.0.0")
    summarization = summarization.map(lambda x: {
        "input_text": "summarize: " + x['article'].strip().replace('\n', ' ')[:5000],
        "target_text": x["highlights"].strip().replace('\n', ' ')
    }, remove_columns=["article", "highlights", "id"])

    squad = load_dataset("rajpurkar/squad")
    squad = squad.map(lambda x: {
        "input_text": "question: " + x['question'].strip() + " context: " + x['context'].strip()[:2000],
        "target_text": x['answers']['text'][0].strip() if x['answers']['text'] else ""
    }, remove_columns=["question", "context", "answers", "id", "title"])

    combined = concatenate_datasets([
        sentiment["train"].shuffle(seed=42).select(range(5000)),
        summarization["train"].select(range(5000)),
        squad["train"].select(range(5000))
    ])
    
    return combined.filter(lambda x: x["input_text"] and x["target_text"])

def tokenize_data(dataset, tokenizer):
    def tokenize_fn(batch):
        inputs = tokenizer(
            text = batch["input_text"],
            max_length=128,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        
        labels = tokenizer(
            text_target = batch["target_text"],
            max_length=32,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels": labels["input_ids"]
        }
    
    return dataset.map(tokenize_fn, batched=True, batch_size=32)

def compute_metrics(eval_preds):
    preds, labels = eval_preds
  
    pred_str = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels[labels == -100] = tokenizer.pad_token_id  
    label_str = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    metrics = {}
    
    metrics["bleu"] = bleu.compute(
        predictions=pred_str, 
        references=[[ref] for ref in label_str]  
    )["bleu"]
    
    metrics["rouge"] = rouge.compute(
        predictions=pred_str,
        references=label_str,
        use_stemmer=True
    )
    
    is_sentiment = ["sentiment:" in s for s in tokenized_dataset["input_text"]]
    if any(is_sentiment):
        sentiment_preds = [1 if "positive" in p else 0 for p in pred_str]
        sentiment_labels = [1 if "positive" in l else 0 for l in label_str]
        metrics["sentiment_accuracy"] = accuracy.compute(
            predictions=sentiment_preds,
            references=sentiment_labels
        )["accuracy"]
    
    return metrics

raw_dataset = load_and_preprocess_data()
    
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small").to(device)
    
tokenized_dataset = tokenize_data(raw_dataset, tokenizer)
    
split_dataset = tokenized_dataset.train_test_split(test_size=0.2)
    
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    
args = Seq2SeqTrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=4,
        fp16=True,
        num_train_epochs=3,
        logging_dir="./logs",
        save_strategy="epoch",
        load_best_model_at_end=True,
        predict_with_generate=True
)
    
trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=split_dataset["train"],
        eval_dataset=split_dataset["test"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
)

trainer.train()
    
model.save_pretrained("./T5-Fine-Tuned")
tokenizer.save_pretrained("./T5-Fine-Tuned")