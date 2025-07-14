import torch
import pandas as pd
import os
import csv
from torch.utils.data import Dataset


def read_line_csv(spamreader):
    """
    Генератор для чтения строки csv-файла
    params: 
    spamreader (csvreader) : Объект csvreader считывающий файлы из csv-файла
    Yields:
    row (tuple[str]) : строка csv-файла
    """
    for row in spamreader:
        yield row


def create_train_dataframe(train_dir_path: str, train_csv_path: str):
    """
    Чтение из папки train txt файлов (объектов) и чтение из csv файла ответов (таргетов) и добавление их в DataFrame
    Params:
    train_dir_path (str) : Путь к train директории, которая содержит директории с парами txt-файлов
    train_csv_path (str) : Путь к csv-файлу, который содержит real_text_id
    Returns: 
    df (Pandas.DataFrame) : Датафрейм с колонками "Text_1", "Text_2", "Target"
    """
    train_df = pd.DataFrame(columns = ["Text_1", "Text_2", "Target"])
    csvfile = open(train_csv_path, newline='') 
    spamreader = csv.reader(csvfile, delimiter=',')
    csv_generator = read_line_csv(spamreader)
    next(csv_generator)
    for dir_id in sorted(os.listdir(train_dir_path)):
        result_row = []
        for file_name in sorted(os.listdir(os.path.join(train_dir_path, dir_id))):
            file_path = os.path.join(train_dir_path, dir_id, file_name)
            with open(file_path, "r") as f:
                file_content = f.readlines()
                result_row.append(" ".join(file_content))
        train_df.loc[len(train_df)] = [*result_row, int(next(csv_generator)[1])]
    return train_df



class TextPairDatasetTrain(Dataset):
    def __init__(self, df, tokenizer, max_length=128):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        text1 = str(self.df.iloc[idx]['Text_1'])
        text2 = str(self.df.iloc[idx]['Text_2'])
        target = int(self.df.iloc[idx]['Target'])

        encoding1 = self.tokenizer(
            text1, 
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        encoding2 = self.tokenizer(
            text2,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids_0': encoding1['input_ids'].squeeze(0),
            'attention_mask_0': encoding1['attention_mask'].squeeze(0),
            'input_ids_1': encoding2['input_ids'].squeeze(0),
            'attention_mask_1': encoding2['attention_mask'].squeeze(0),
            'target': torch.tensor(target, dtype=torch.float)
        }
    


def create_test_dataframe(test_dir_path: str):
    """
    Чтение из папки test txt файлов (объектов) и добавление их в DataFrame
    Params:
    test_dir_path (str) : Путь к test директории, которая содержит директории с парами txt-файлов
    Returns: 
    df (Pandas.DataFrame) : Датафрейм с колонками "Text_1", "Text_2"
    """
    test_df = pd.DataFrame(columns = ["Text_1", "Text_2"])
    for dir_id in sorted(os.listdir(test_dir_path)):
        result_row = []
        for file_name in sorted(os.listdir(os.path.join(test_dir_path, dir_id))):
            file_path = os.path.join(test_dir_path, dir_id, file_name)
            with open(file_path, "r") as f:
                file_content = f.readlines()
                result_row.append(" ".join(file_content))
        test_df.loc[len(test_df)] = [*result_row]
    return test_df


class TextPairDatasetTest(Dataset):
    def __init__(self, df, tokenizer, max_length=128):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        text1 = str(self.df.iloc[idx]['Text_1'])
        text2 = str(self.df.iloc[idx]['Text_2'])

        encoding1 = self.tokenizer(
            text1, 
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        encoding2 = self.tokenizer(
            text2,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids_0': encoding1['input_ids'].squeeze(0),
            'attention_mask_0': encoding1['attention_mask'].squeeze(0),
            'input_ids_1': encoding2['input_ids'].squeeze(0),
            'attention_mask_1': encoding2['attention_mask'].squeeze(0),
            'text_1' : text1,
            'text_2' : text2,
        }