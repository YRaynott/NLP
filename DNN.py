import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
from tensorflow.keras.callbacks import ReduceLROnPlateau  # 修改這行

# 讀取訓練數據
train_data = pd.read_json('train_data.json')
X_train_full = train_data['description']
y_train_full = train_data['catid']

# 檢查標籤值的範圍
print(f'Min label: {y_train_full.min()}, Max label: {y_train_full.max()}')
print(f'Unique labels: {sorted(set(y_train_full))}')

# 數據預處理
vectorizer = TfidfVectorizer(analyzer='word', token_pattern=r'\b\w+\b', preprocessor=lambda x: ' '.join(x))
X_vectorized = vectorizer.fit_transform(X_train_full)

# 檢查標籤是否在有效範圍內
if y_train_full.max() >= 19:
    # 重映射標籤到有效範圍
    label_mapping = {old_label: new_label for new_label, old_label in enumerate(sorted(set(y_train_full)))}
    y_train_full = y_train_full.map(label_mapping)

# 分割數據集
X_train, X_val, y_train, y_val = train_test_split(X_vectorized, y_train_full, test_size=0.2, random_state=42)

# 重新計算類別數量
num_classes = len(set(y_train_full))
print(f'Number of classes: {num_classes}')

# 定義 DNN 模型
def create_dnn_model(input_shape, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_shape,)),

        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model

# 設置動態學習率
LR_function = ReduceLROnPlateau(monitor='val_accuracy', patience=5, verbose=1, factor=0.5, min_lr=0.00001)

# 使用最佳學習率訓練最終 DNN 模型
best_lr = 0.001  # 使用初始學習率
final_model = create_dnn_model(X_train.shape[1], num_classes)
final_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=best_lr),
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])
final_model.fit(X_vectorized.toarray(), y_train_full, epochs=10, batch_size=32, validation_data=(X_val.toarray(), y_val), callbacks=[LR_function])

# 讀取測試數據
test_data = pd.read_json('test_data.json')
X_test = test_data['description']
X_test_vectorized = vectorizer.transform(X_test)

# 預測測試集
y_test_pred = final_model.predict(X_test_vectorized.toarray())
y_test_pred_classes = np.argmax(y_test_pred, axis=1)

# 輸出測試結果到 CSV
test_data['predicted_catid'] = y_test_pred_classes
reverse_label_mapping = {v: k for k, v in label_mapping.items()}  # 反向映射字典
test_data['predicted_catid'] = test_data['predicted_catid'].map(reverse_label_mapping)
test_data[['itemid', 'predicted_catid']].to_csv('prediction_DNN.csv', index=False)

# 比較與評估
test_data = pd.read_json('test_data.json')
true_labels = test_data['catid'].tolist()
csv_df = pd.read_csv("prediction_DNN.csv")
all_predictions = csv_df['predicted_catid'].tolist()

accuracy = accuracy_score(true_labels, all_predictions)
print(f"\nTest Accuracy: {accuracy * 100:.2f}%")

# 混淆矩陣和分類報告
conf_matrix = confusion_matrix(y_val, np.argmax(final_model.predict(X_val.toarray()), axis=1))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

print("\nClassification Report:")
print(classification_report(y_val, np.argmax(final_model.predict(X_val.toarray()), axis=1)))