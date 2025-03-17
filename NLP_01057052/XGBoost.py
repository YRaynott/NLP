import xgboost as xgb
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import LabelEncoder
import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import re

# 加載數據
with open('train_data.json', 'r') as f:
    train_data = json.load(f)
with open('test_data.json', 'r') as f:
    test_data = json.load(f)

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

# 計算類別權重
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(zip(np.unique(y_train), class_weights))

# 訓練 XGBoost 模型，加入一些常用的超參數
xgb_model = xgb.XGBClassifier(
    objective='multi:softmax',
    num_class=len(label_encoder.classes_),
    use_label_encoder=False,
    max_depth=12,  # 樹的最大深度
    learning_rate=0.01,  # 學習率
    n_estimators=1000,  # 樹的數量
    subsample=0.9,  # 隨機抽樣的比例
    colsample_bytree=0.8,  # 隨機選擇特徵的比例
    scale_pos_weight=class_weight_dict  # 如果類別不均衡，可以調整
)
xgb_model.fit(X_train, y_train)

# 預測
test_preds = xgb_model.predict(X_test)

# 生成結果文件
with open('prediction_XGBoost.csv', 'w') as f:
    f.write('itemid,catid\n')
    for item, pred in zip(test_data, test_preds):
        f.write(f"{item['itemid']},{label_encoder.inverse_transform([pred])[0]}\n")

# 評估性能
csv_df = pd.read_csv("prediction_XGBoost.csv")
all_predictions = csv_df['catid'].tolist()
true_labels = [int(item['catid']) for item in test_data]

# 準確度計算
accuracy = accuracy_score(true_labels, all_predictions)
print(f"\nTest Accuracy: {accuracy * 100:.2f}%")

# 混淆矩陣
conf_matrix = confusion_matrix(true_labels, all_predictions)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()

# 顯示分類報告
class_names = [str(name) for name in label_encoder.classes_]
class_report = classification_report(true_labels, all_predictions, target_names=class_names)
print("\nClassification Report:")
print(class_report)
