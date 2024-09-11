# src/models/model.py

from transformers import BartForConditionalGeneration, T5ForConditionalGeneration, BartTokenizer, T5Tokenizer

class TextSummarizer:
    def __init__(self, model_name='facebook/bart-large'):
        self.model_name = model_name
        self.load_model()

    def load_model(self):
        if 'bart' in self.model_name:
            self.model = BartForConditionalGeneration.from_pretrained(self.model_name)
            self.tokenizer = BartTokenizer.from_pretrained(self.model_name)
        elif 't5' in self.model_name:
            self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)
            self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
        else:
            raise ValueError("Unsupported model. Please use BART or T5.")

    def summarize(self, text, max_length=150, min_length=40, do_sample=False):
        inputs = self.tokenizer([text], return_tensors='pt', truncation=True, padding='longest')
        summary_ids = self.model.generate(inputs['input_ids'], max_length=max_length, min_length=min_length, do_sample=do_sample)
        summary = self.tokenizer.batch_decode(summary_ids, skip_special_tokens=True)[0]
        return summary
