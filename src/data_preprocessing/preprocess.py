from transformers import BarthezTokenizer
import re
import numpy as np

class TextSumPreprocess:
    def __init__(self, model = 'facebook/bart-large-cnn', max_length = 1024):
        self.tokenizer = BarthezTokenizer.from_pretrained(model)
        self.max_length = max_length

    def clean_text(seft, text):
        # remove URLs, HTML tags, non-alphanumeric character, replace multiple spaces
        text = re.sub(r'http\S+|www.\S+', '', text)  
        text = re.sub(r'<.*?>', '', text)           
        text = re.sub(r'[^A-Za-z0-9\s,.?!]', '', text)  
        text = re.sub(r'\s+', ' ', text)           
        return text.strip()

    def tokenize_text(self, text):
        """
        Tokenizes the cleaned text using the model tokenizer, ensuring proper handling of padding.
        """
        encoded_dict = self.tokenizer.encode_plus(
            text,
            add_special_tokens = True,         # Add [CLS] and [SEP] tokens
            max_length = self.max_length,      # Max length for the transformer model
            padding = 'max_length',            # Pad to max_length
            return_attention_mask = True,      # Generate attention masks
            truncation = True,                 # Truncate long sequences
            return_tensors = 'pt'              # Return as PyTorch tensors
        )
        return encoded_dict['input_ids'], encoded_dict['attention_mask']

    def preprocess(self, text):
        cleaned_text = self.cleaned_text(text)
        input_ids, attention_mask = self.tokenize_text(cleaned_text)
        return input_ids, attention_mask

    if __name__ == '__main__':
        text = ""

        preprocessor = TextSumPreprocess()

        input_ids, attention_mask = preprocessor.preprocess(text)
       


    
