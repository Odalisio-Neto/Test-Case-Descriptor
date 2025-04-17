import torch
import numpy as np
import multiprocessing
import os
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from datasets import load_dataset
import evaluate

# Set environment variable for memory management
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Clear GPU memory
torch.cuda.empty_cache()

# Load dataset
dataset = load_dataset("cnn_dailymail", "3.0.0")

# Load tokenizer and model
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=True)  # Suppress legacy warning
model = T5ForConditionalGeneration.from_pretrained(model_name)
model.gradient_checkpointing_enable()  # Enable gradient checkpointing

# Preprocess function
def preprocess_function(examples):
    inputs = ["summarize: " + doc for doc in examples["article"]]
    targets = examples["highlights"]
    model_inputs = tokenizer(inputs, max_length=256, truncation=True, padding="max_length")  # Reduced length
    labels = tokenizer(targets, max_length=64, truncation=True, padding="max_length")      # Reduced length
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Use all CPU cores for preprocessing
num_proc = multiprocessing.cpu_count()
tokenized_dataset = dataset.map(preprocess_function, batched=True, num_proc=num_proc)

# Select subsets
train_dataset = tokenized_dataset["train"].select(range(1000))
eval_dataset = tokenized_dataset["validation"].select(range(200))

# Define training arguments
training_args = TrainingArguments(
    output_dir="./summarization_model",
    logging_dir="./logs",
    num_train_epochs=10,
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=1,        # Reduced
    per_device_eval_batch_size=1,         # Reduced
    gradient_accumulation_steps=4,        # Simulate larger batch size
    weight_decay=0.01,
    save_strategy="epoch",
    fp16=True,
    fp16_full_eval=False,                 # Disable FP16 for evaluation
    load_best_model_at_end=True                          
)

# Load ROUGE metric
rouge = evaluate.load("rouge")

# Compute metrics function
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    if predictions.ndim == 3:
        predictions = np.argmax(predictions, axis=-1)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    result = rouge.compute(predictions=decoded_preds, references=decoded_labels)
    return result

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)

# Log memory before training
print(f"Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GiB")
print(f"Cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GiB")

# Fine-tune the model
trainer.train()

# Log memory after training
print(f"Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GiB")
print(f"Cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GiB")

# Save the model
model.save_pretrained("./summarization_model/final")
tokenizer.save_pretrained("./summarization_model/final")

# Inference example
def generate_summary(text):
    
    input_text = "summarize: " + text
    inputs = tokenizer(input_text, return_tensors="pt", max_length=256, truncation=True).to("cuda")
    summary_ids = model.generate(inputs["input_ids"], max_length=64, num_beams=4, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Test the model
sample_text = """
The quick brown fox jumps over the lazy dog. This classic sentence is often used to test typewriters and keyboards because it contains every letter of the English alphabet. It has been a staple in typing exercises for decades.
"""
print("Generated Summary:", generate_summary(sample_text))