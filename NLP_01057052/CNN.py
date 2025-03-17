import json
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd
import seaborn as sns
import re
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import numpy as np

#####################################################################################
# 加載數據
with open('train_data.json', 'r') as f:
    train_data = json.load(f)
with open('test_data.json', 'r') as f:
    test_data = json.load(f)

#####################################################################################
# 去除標點符號的函數（保留中文和英文字母）
def remove_punctuation(text):
    # 只保留字母、數字、中文字符，去除其他標點符號
    return re.sub(r'[^\w\s\u4e00-\u9fa5]', '', text)

# 文本預處理
def preprocess_data(data):
    descriptions = [' '.join(item['description_wseg_list'][0]) for item in data]
    descriptions = [remove_punctuation(desc) for desc in descriptions]  # 去除標點符號

    categories = [item['catid'] for item in data]

    pos_tags = [' '.join(item['description_wpos_list'][0]) for item in data]
    pos_tags = [remove_punctuation(pos) for pos in pos_tags]  # 去除標點符號

    return descriptions, categories, pos_tags

train_texts, train_labels, train_pos_tags = preprocess_data(train_data)
test_texts, _, test_pos_tags = preprocess_data(test_data)

# 文本向量化 (TF-IDF)
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))  # 1-gram 和 2-gram
X_train_text = vectorizer.fit_transform(train_texts).toarray()
X_test_text = vectorizer.transform(test_texts).toarray()

#####################################################################################
# 詞性標註向量化 (CountVectorizer)
count_vectorizer = CountVectorizer(max_features=5000)
X_train_pos = count_vectorizer.fit_transform(train_pos_tags).toarray()  # 使用 CountVectorizer 來處理詞性標註
X_test_pos = count_vectorizer.transform(test_pos_tags).toarray()

# 合併 TF-IDF 特徵與詞性特徵
X_train = np.hstack([X_train_text, X_train_pos])  # 合併文本和詞性特徵
X_test = np.hstack([X_test_text, X_test_pos])

# 標籤編碼
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(train_labels)

#####################################################################################
# 數據集與數據加載器
class ShopeeDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]

train_dataset = ShopeeDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

#####################################################################################
# 定義模型
import torch.nn as nn
import torch.nn.functional as F

class CNNClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_layers=2, num_filters=128, kernel_size=3, dropout_rate=0.3, activation='relu'):
        super(CNNClassifier, self).__init__()

        # 設置激活函數
        self.activation_function = {
            'relu': nn.ReLU(),
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh()
        }[activation]

        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(
            nn.Conv1d(in_channels=1, out_channels=num_filters, kernel_size=kernel_size, padding=1)
        )
        for _ in range(hidden_layers - 1):
            self.conv_layers.append(
                nn.Conv1d(in_channels=num_filters, out_channels=num_filters, kernel_size=kernel_size, padding=1)
            )

        self.pool = nn.MaxPool1d(kernel_size=2)

        # 自動計算展平大小
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, input_dim)
            dummy_output = dummy_input
            for conv in self.conv_layers:
                dummy_output = self.pool(conv(dummy_output))
            self.flattened_size = dummy_output.view(-1).size(0)

        self.fc = nn.Sequential(
            nn.Linear(self.flattened_size, 256),
            nn.Dropout(dropout_rate),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # (batch_size, 1, input_dim)

        # 卷積和池化
        for conv in self.conv_layers:
            x = self.activation_function(conv(x))
            x = self.pool(x)

        # 展平
        x = x.view(x.size(0), -1)

        # 全連接層
        x = self.fc(x)
        return x

#####################################################################################
# 訓練參數
batch_size = 32
input_dim = X_train.shape[1]  # 自動獲取 TF-IDF 特徵維度
num_classes = len(label_encoder.classes_)
hidden_layers = 2
num_filters = 128
kernel_size = 3
dropout_rate = 0.3
activation = 'relu'
learning_rate = 1e-3
num_epochs = 10

#####################################################################################
# 訓練模型
device = torch.device('mps')
model = CNNClassifier(input_dim, num_classes, hidden_layers, num_filters, kernel_size, dropout_rate, activation)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

model = model.to(device)
train_losses = []
train_accuracies = []

for epoch in range(1, num_epochs + 1):
    model.train()
    epoch_loss = 0.0
    correct = 0
    total = 0

    for features, labels in tqdm(train_loader, desc=f'Epoch {epoch}/{num_epochs}'):
        features, labels = features.to(device), labels.to(device)
        optimizer.zero_grad()

        # 前向傳播
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # 記錄 Loss 和 Accuracy
        epoch_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    # 計算 Loss 和 Accuracy
    avg_loss = epoch_loss / len(train_loader)
    accuracy = correct / total * 100
    print(f'Epoch [{epoch}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')

    train_losses.append(avg_loss)
    train_accuracies.append(accuracy)

#####################################################################################
# 繪製 Loss 和 Accuracy 圖
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs + 1), train_losses, label='Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs + 1), train_accuracies, label='Accuracy', color='orange')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Training Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

#####################################################################################
# 測試與輸出結果
test_dataset = ShopeeDataset(X_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model.eval()
predictions = []
with torch.no_grad():
    for features in test_loader:
        features = features.to(device).float()  # 確保類型是 float32
        outputs = model(features)
        _, predicted = torch.max(outputs, 1)
        predictions.extend(predicted.tolist())

# 生成結果文件
with open('prediction.csv', 'w') as f:
    f.write('itemid,catid\n')
    for item, pred in zip(test_data, predictions):
        f.write(f"{item['itemid']},{label_encoder.inverse_transform([pred])[0]}\n")

#####################################################################################
# 比較與評估
csv_df = pd.read_csv("prediction.csv")
all_predictions = csv_df['catid'].tolist()
true_labels = [int(item['catid']) for item in test_data]

accuracy = accuracy_score(true_labels, all_predictions)
print(f"\nTest Accuracy: {accuracy * 100:.2f}%")

conf_matrix = confusion_matrix(true_labels, all_predictions)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

class_names = [str(name) for name in label_encoder.classes_]

class_report = classification_report(true_labels, all_predictions, target_names=class_names)
print("\nClassification Report:")
print(class_report)
