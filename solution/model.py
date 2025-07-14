import torch
import torch.nn as nn
from transformers import BertModel

class TextClassifier(nn.Module):
    def __init__(self, pretrained_model_name='bert-base-uncased'):
        super().__init__()

        # Слой который преобразует исходный текст (токены) в вектор (embedding)
        self.bert = BertModel.from_pretrained(pretrained_model_name)

        # Обычные слои бинарной классификации
        self.classifier = nn.Sequential(
            nn.Linear(768*4, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )
        
    def forward(self, input_ids_0, attention_mask_0, input_ids_1, attention_mask_1):
        """
        Прямой проход модели
        params:
        input_ids_0 (torch.Tensor) : Тензор ID токенов первого текста
        attention_mask_0 (torch.Tensor) : Маска внимания для первого текста 
        input_ids_1 (torch.Tensor) : Тензор ID токенов второго текста
        attention_mask_1 (torch.Tensor) : Маска внимания для второго текста 
        Returns:
        Предсказания модели — логиты формы (batch_size)
        """
        # Получаем выходы BERT для обоих текстов
        out_0 = self.bert(input_ids=input_ids_0, attention_mask=attention_mask_0)
        out_1 = self.bert(input_ids=input_ids_1, attention_mask=attention_mask_1)
        
        # Извлекаем только CLS-токен 
        emb_0 = out_0.last_hidden_state[:, 0, :]  
        emb_1 = out_1.last_hidden_state[:, 0, :]
        
        # Формируем новые признаки
        features = torch.cat([
            emb_0,
            emb_1,
            torch.abs(emb_0 - emb_1),
            emb_0 * emb_1
        ], dim=1)
        
        # Прогоняем признаки через остальные слои
        return self.classifier(features).squeeze(1)