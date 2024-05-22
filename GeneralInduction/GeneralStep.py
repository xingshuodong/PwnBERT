import argparse
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from transformers import AdamW
from torch.utils.data import DataLoader, TensorDataset
import torch

# Function to tokenize the data from a given file path
def tokenize_data(file_path):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    with open(file_path, 'r') as file:
        code_snippets = file.readlines()
    tokenized_inputs = tokenizer(code_snippets, padding=True, truncation=True, return_tensors='pt')
    return tokenized_inputs

# Function to create data loaders for training and validation
def create_dataloaders(train_file, val_file, batch_size):
    train_inputs = tokenize_data(train_file)
    val_inputs = tokenize_data(val_file)
    train_dataset = TensorDataset(train_inputs['input_ids'], train_inputs['attention_mask'])
    val_dataset = TensorDataset(val_inputs['input_ids'], val_inputs['attention_mask'])
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    return train_dataloader, val_dataloader

# Function to train the model using the given data loaders
def train_model(train_dataloader, val_dataloader, epochs):
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    training_args = TrainingArguments(
        output_dir='./results',  # Directory to save the results
        num_train_epochs=epochs,  # Number of epochs to train
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,  # Warmup steps for learning rate scheduler
        weight_decay=0.01,  # Weight decay to avoid overfitting
        logging_dir='./logs',  # Directory to save logs
        logging_steps=10,
        evaluation_strategy="epoch",
        load_best_model_at_end=True,  # Load the best model at the end of training
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataloader.dataset,
        eval_dataset=val_dataloader.dataset
    )
    trainer.train()

# Main function to parse command-line arguments and execute the training
def main():
    parser = argparse.ArgumentParser(description="Fine-tune BERT model for vulnerability detection")
    parser.add_argument('--train_file', type=str, required=True, help="Path to the training data file")
    parser.add_argument('--val_file', type=str, required=True, help="Path to the validation data file")
    parser.add_argument('--batch_size', type=int, default=8, help="Batch size for training and validation")
    parser.add_argument('--epochs', type=int, default=3, help="Number of epochs to train")
    args = parser.parse_args()

    # Create data loaders for training and validation
    train_dataloader, val_dataloader = create_dataloaders(args.train_file, args.val_file, args.batch_size)
    # Train the model
    train_model(train_dataloader, val_dataloader, args.epochs)

# Entry point of the script
if __name__ == "__main__":
    main()
