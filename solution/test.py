import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from model import TextClassifier  
from datasets import create_test_dataframe, TextPairDatasetTest
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Импорт модели
model = TextClassifier()
model.load_state_dict(torch.load("best_model.pt", map_location=device))
model.to(device)
model.eval()

# Создание Dataset и DataLoader
test_dir_path = "/home/egikor/ML/Practic Project/data/test"
test_df = create_test_dataframe(test_dir_path)

# Инициализация токенизатора
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Инициализация датасета
test_dataset = TextPairDatasetTest(test_df, tokenizer)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Инициализация DataFrame в котором содержатся текста и предсказания
result_df = pd.DataFrame(columns = ["Text_1", "Text_2", "Predict"])


with torch.no_grad():
    for batch in test_loader:
        input_ids_0 = batch["input_ids_0"].to(device)
        attention_mask_0 = batch["attention_mask_0"].to(device)
        input_ids_1 = batch["input_ids_1"].to(device)
        attention_mask_1 = batch["attention_mask_1"].to(device)
        outputs = model(
            input_ids_0,
            attention_mask_0,
            input_ids_1,
            attention_mask_1
        )
        preds = torch.sigmoid(outputs).squeeze()
        preds = (preds > 0.5).type(torch.int16).to("cpu")
        for text_1, text_2, pred in zip(batch["text_1"], batch["text_2"], preds):
            result_df.loc[len(result_df)] = [text_1, text_2, int(pred)]

print(result_df)
