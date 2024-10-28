from transformers import T5Tokenizer, T5ForConditionalGeneration

class myT5:
    def __init__(self, modelpath):
        self.loadmodel = T5ForConditionalGeneration.from_pretrained(modelpath)
        self.tokenizer = T5Tokenizer.from_pretrained('digit82/kolang-t5-base')

    def predict(self, text):
        tokens = self.tokenizer.encode(text)
        num_tokens = len(tokens)
        print(f"\033[92m{num_tokens}\033[0m") 
        inputs = self.tokenizer.encode(text, return_tensors="pt", max_length=512, truncation=True)
        summary_ids = self.loadmodel.generate(inputs, max_length=150, num_beams=2, early_stopping=True)

        # 수정된 부분: 첫 번째 결과만 디코딩
        summary_predicted = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary_predicted
