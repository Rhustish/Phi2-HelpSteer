from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
import torch

# Load the model and tokenizer
model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2")
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")

# Set the padding token
tokenizer.pad_token = tokenizer.eos_token

# Load your dataset
dataset = load_dataset("json", data_files="train.json", split="train")

# Prepare the training data
def tokenize_function(examples):
    return tokenizer(examples["prompt"], padding="max_length", truncation=True)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Define the training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    weight_decay=0.01,
)

torch.cuda.empty_cache()

# Create the trainer and fine-tune the model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

trainer.train()

# Save the fine-tuned model
model.save_pretrained("./fine_tuned_model")
