import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# 讀取訓練數據
train_data = pd.read_json('train_data.json')
X_train_full = train_data['description']
y_train_full = train_data['catid']

# 數據預處理
vectorizer = TfidfVectorizer(analyzer='word', token_pattern=r'\b\w+\b')  # 修改這行
X_vectorized = vectorizer.fit_transform(X_train_full)

# 檢查標籤是否在有效範圍內
if y_train_full.max() >= 19:
    # 重映射標籤到有效範圍
    label_mapping = {old_label: new_label for new_label, old_label in enumerate(sorted(set(y_train_full)))}
    y_train_full = y_train_full.map(label_mapping)

# 分割數據集
X_train, X_val, y_train, y_val = train_test_split(X_vectorized, y_train_full, test_size=0.2, random_state=42)

# SVM 分類器
svm_model = make_pipeline(StandardScaler(with_mean=False), svm.SVC(kernel='linear'))
svm_model.fit(X_train, y_train)

# 預測驗證集
y_val_pred_svm = svm_model.predict(X_val)

# 評估準確率
svm_accuracy = accuracy_score(y_val, y_val_pred_svm)
print(f'Test Accuracy: {svm_accuracy * 100:.2f}%')

# 訓練 SVM 最終模型
svm_model.fit(X_vectorized, y_train_full)

# 讀取測試數據
test_data = pd.read_json('test_data.json')
X_test = test_data['description']
X_test_vectorized = vectorizer.transform(X_test)

# 預測測試集
y_test_pred_svm = svm_model.predict(X_test_vectorized)

# 輸出 SVM 測試結果
test_data['predicted_catid_svm'] = y_test_pred_svm
test_data.to_json('test_data_predictions_svm.json', orient='records', lines=True)
