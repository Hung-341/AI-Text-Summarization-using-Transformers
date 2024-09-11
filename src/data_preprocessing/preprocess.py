# src/data_preprocessing/preprocess.py

import re
from datasets import load_dataset

class DataPreprocessor:
    def __init__(self, dataset_name='cnn_dailymail', version='3.0.0'):
        self.dataset_name = dataset_name
        self.version = version
        self.dataset = None

    def load_data(self):
        self.dataset = load_dataset(self.dataset_name, self.version)

    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with one space
        text = re.sub(r'\[.*?\]', '', text)  # Remove content in square brackets
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special characters
        return text

    def preprocess(self):
        if self.dataset:
            for split in ['train', 'validation', 'test']:
                self.dataset[split] = self.dataset[split].map(lambda example: {'cleaned_article': self.clean_text(example['article'])})
                # Save processed data in the processed folder
                self.dataset[split].to_json(f"../../data/processed/{split}.json")
        else:
            raise Exception("Dataset not loaded. Run load_data() first.")
