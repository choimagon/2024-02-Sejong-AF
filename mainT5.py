import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments


df123 = pd.read_csv("output.csv")
df123 = df123.dropna()

#이거 학습 데이터 많으면 오래걸림 줄여야함.
df = df123.head(2000)

train_df, val_df = train_test_split(df, test_size=0.1)

tokenizer = T5Tokenizer.from_pretrained('digit82/kolang-t5-base')
model = T5ForConditionalGeneration.from_pretrained('digit82/kolang-t5-base')

class CustomDataset(torch.utils.data.Dataset):
    # 토큰 인풋 아웃풋 좀 조절해야함. 안하면 ㅈ됌
    def __init__(self, dataframe, tokenizer, max_input_length=550, max_target_length=150):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        context = str(self.data.iloc[index]["context"])
        summary = str(self.data.iloc[index]["summary"])

        input_encoding = self.tokenizer(
            context,
            max_length=self.max_input_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        target_encoding = self.tokenizer(
            summary,
            max_length=self.max_target_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        labels = target_encoding["input_ids"]
        labels[labels == 0] = -100

        return {
            "input_ids": input_encoding["input_ids"].flatten(),
            "attention_mask": input_encoding["attention_mask"].flatten(),
            "labels": labels.flatten(),
        }

train_dataset = CustomDataset(train_df, tokenizer)
val_dataset = CustomDataset(val_df, tokenizer)

#모델에 대한 설정들은 여기서만 건들이면 됌
training_args = TrainingArguments(
    per_device_train_batch_size=4,  # 배치 크기 줄임
    per_device_eval_batch_size=4,   # 배치 크기 줄임
    output_dir="./results",
    num_train_epochs=4,  # 에폭을 4로 늘림
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=3,
    learning_rate=5e-5  # 학습률을 낮춤
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

trainer.train()

trainer.evaluate()
print("check")