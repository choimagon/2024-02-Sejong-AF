from transformers import T5ForConditionalGeneration
import pandas as pd
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments

class myT5:
    def __init__(self, modelpath):
        self.loadmodel = T5ForConditionalGeneration.from_pretrained(modelpath)
    def predict(self, text):
        tokenizer = T5Tokenizer.from_pretrained('digit82/kolang-t5-base')
        inputs = tokenizer.encode(text, return_tensors="pt", max_length=512, truncation=True)
        summary_ids = self.loadmodel.generate(inputs, max_length=150, num_beams=2, early_stopping=True)
        summary_predicted = tokenizer.decode(summary_ids, skip_special_tokens=True)
        return summary_predicted
