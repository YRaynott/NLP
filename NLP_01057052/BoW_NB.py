import json
from sklearn.feature_extraction.text import CountVectorizer  # 改為使用 CountVectorizer，純 BoW
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

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
# 計算類別權重（解決類別不平衡）
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(zip(np.unique(y_train), class_weights))

# Logistic Regression
log_reg_model = LogisticRegression(max_iter=1000, class_weight='balanced', solver='liblinear')

# 交叉驗證
kf = StratifiedKFold(n_splits=5)
best_model = None
best_accuracy = 0

for train_index, val_index in kf.split(X_train, y_train):
    X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
    y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
    
    # 訓練 Logistic Regression 模型
    log_reg_model.fit(X_train_fold, y_train_fold)
    val_preds = log_reg_model.predict(X_val_fold)
    val_accuracy = accuracy_score(y_val_fold, val_preds)
    
    # 選擇最佳模型
    if val_accuracy > best_accuracy:
        best_model = log_reg_model
        best_accuracy = val_accuracy

# 最終訓練選定的最佳模型
best_model.fit(X_train, y_train)

# 在測試集上評估性能
test_preds = best_model.predict(X_test)

# 生成結果文件
with open('prediction_Bow.csv', 'w') as f:
    f.write('itemid,catid\n')
    for item, pred in zip(test_data, test_preds):
        f.write(f"{item['itemid']},{label_encoder.inverse_transform([pred])[0]}\n")

# 比較與評估
csv_df = pd.read_csv("prediction_Bow.csv")
all_predictions = csv_df['catid'].tolist()
true_labels = [int(item['catid']) for item in test_data]

accuracy = accuracy_score(true_labels, all_predictions)
print(f"\nTest Accuracy: {accuracy * 100:.2f}%")

# 混淆矩陣
conf_matrix = confusion_matrix(true_labels, all_predictions)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

# 分類報告
class_names = [str(name) for name in label_encoder.classes_]
class_report = classification_report(true_labels, all_predictions, target_names=class_names)
print("\nClassification Report:")
print(class_report)
