# src/training/train.py

from transformers import Trainer, TrainingArguments
from datasets import load_metric
from src.models.model import TextSummarizer
from src.data_preprocessing.preprocess import DataPreprocessor

class ModelTrainer:
    def __init__(self, model_name='facebook/bart-large'):
        self.model_name = model_name
        self.summarizer = TextSummarizer(model_name)
        self.preprocessor = DataPreprocessor()
        self.metric = load_metric('rouge')

    def load_data(self):
        self.preprocessor.load_data()
        return self.preprocessor.dataset['train'], self.preprocessor.dataset['validation']

    def compute_metrics(self, pred):
        labels_ids = pred.label_ids
        pred_ids = pred.predictions.argmax(-1)
        predictions = self.summarizer.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        references = self.summarizer.tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
        result = self.metric.compute(predictions=predictions, references=references)
        return result

    def train(self, epochs=3, batch_size=16):
        train_data, val_data = self.load_data()

        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            evaluation_strategy='steps',
            save_steps=10,
            eval_steps=10,
            logging_dir='./logs',
        )

        trainer = Trainer(
            model=self.summarizer.model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=val_data,
            compute_metrics=self.compute_metrics,
        )

        trainer.train()
        trainer.save_model('./checkpoints')

if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.train()
