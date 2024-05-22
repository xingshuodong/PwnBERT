from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from torch.nn import Dropout

"""
Why Diversifying Training Data Works

Variety in Data: Introducing diverse data reduces the model¡¯s tendency to memorize specific patterns and enhances its ability to generalize to new, unseen data. This is crucial for detecting various types of vulnerabilities in code.
Real-world Scenarios: A diverse dataset better represents real-world scenarios, making the model more robust and reliable.
Why Dropout Helps

Prevents Over-reliance: Dropout randomly "drops" units during training, forcing the model to learn redundant representations. This prevents over-reliance on particular neurons, reducing overfitting.
Why Early Stopping Works

Optimal Training Duration: Early stopping monitors the model's performance on a validation set and halts training when performance starts to degrade, ensuring the model does not overfit the training data.
Why AdamW Optimizer Works

Stable Convergence: AdamW (Adam with Weight Decay) adjusts learning rates based on each parameter¡¯s gradient and incorporates weight decay, which helps in preventing overfitting.
Efficient Training: AdamW combines the advantages of both adaptive learning rates and weight regularization, leading to faster and more stable convergence.

"""
class PwnBERTModel(BertForSequenceClassification):
    def __init__(self, config):
        super(PwnBERTModel, self).__init__(config)
        self.dropout = Dropout(p=0.3)  # Adding dropout layer with 30% probability

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=5,
    per_device_train_batch_size=16,
    evaluation_strategy="epoch",
    logging_dir='./logs',
    logging_steps=10,
    save_total_limit=3,
    load_best_model_at_end=True,  # Load best model at the end of training
    metric_for_best_model="eval_loss",
    greater_is_better=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]  # Early stopping
)

trainer.train()
