import torch
import torch.nn as nn
from transformers import BertTokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import TextPairDatasetTrain, create_train_dataframe
from model import TextClassifier
from IPython.display import display


torch.cuda.empty_cache()
torch.cuda.ipc_collect()

# Инициализация токенизатора
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Создание Dataset и DataLoader
train_dir_path = "/home/egikor/ML/Practic Project/data/train"
train_csv_path = "/home/egikor/ML/Practic Project/data/train.csv"

train_df = create_train_dataframe(train_dir_path, train_csv_path)
train_df['Target'] = train_df['Target'].astype(int)
train_df['Target'] = train_df['Target'] - 1
display(train_df)
train_dataset = TextPairDatasetTrain(train_df, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# 4. Инициализация модели и оптимизатора
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TextClassifier().to(device)
# Балансировка классов для loss function
pos_weight = torch.tensor([len(train_df)/sum(train_df['Target'])]).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

<<<<<<< HEAD
num_epochs = 6 # Количество эпох
dict_train = {"avg_train_loss":[],"avg_train_acc":[]} # Словарь для хранения статистики обучения
=======
num_epochs = 5 # Количество эпох
best_val_acc = 0 # Переменная для хранения лучшего accuracy 
dict_train = {"avg_train_loss":[],"avg_train_acc":[]}
>>>>>>> 3f2e39c229a916e4cb90add799de24d24e8329cc

def train_model():
    """
    Функция для обучения модели на тренировочном датасете.
    """
    best_val_acc = 0 # Переменная для хранения лучшего accuracy 
    for epoch in range(num_epochs): # Итерация по эпохам

        model.train() # Перевод модели в режим обучения

        train_loss, train_acc = 0, 0 # Переменная для хранения текущего loss и accuracy 

        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}'): # Итерация по датасету 
            
            optimizer.zero_grad()
            
            # Получаем токены и маски с батча
            inputs = {
                'input_ids_0': batch['input_ids_0'].to(device),
                'attention_mask_0': batch['attention_mask_0'].to(device),
                'input_ids_1': batch['input_ids_1'].to(device),
                'attention_mask_1': batch['attention_mask_1'].to(device)
            }

            targets = batch['target'].to(device)
            
            # Прямой проход
            logits = model(**inputs)
            
            # Вычисляем loss
            loss = criterion(logits, targets)
            
            # Обратный проход
            loss.backward()

            # Ограничение нормы градиентов
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Обновляем параметры модели
            optimizer.step()
            
            # Вычисление accuracy
            with torch.no_grad():
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
                acc = (preds == targets).float().mean()
            
            train_loss += loss.item()
            train_acc += acc.item()
        
        # Валидация
        model.eval()
        
        # Вывод статистики
        avg_train_loss = train_loss / len(train_loader) 
        dict_train['avg_train_loss'].append(avg_train_loss) 
        avg_train_acc = train_acc / len(train_loader)
        dict_train['avg_train_acc'].append(avg_train_acc)
        
        print(f"\nEpoch {epoch+1}/{num_epochs}:")
        print(f"Train Loss: {avg_train_loss:.4f} | Acc: {avg_train_acc:.4f}")

        # Сохраняем модель, если она лучшая
        if avg_train_acc > best_val_acc:
            best_val_acc = avg_train_acc
            torch.save(model.state_dict(), 'best_model.pt')
    return best_val_acc, dict_train

if __name__ == "__main__":
    model = train_model()
    print("\nTraining complete!")
    print(f"Best Validation Accuracy: {model[1]}")
    