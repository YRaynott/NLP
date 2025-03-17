import random
import json

# 讀取文本
with open('Shopee_training_ckipseg_v3_listOK.json', 'r') as file:
    data = json.load(file)

# 打亂資料順序
def shuffle_data(data):
    random.shuffle(data)
    return data

# 切割資料，依照9:1比例
def split_data(data, ratio=0.9):
    split_index = int(len(data) * ratio)
    return data[:split_index], data[split_index:]

# 儲存資料為 JSON 檔案
def save_to_json(data, file_name):
    with open(file_name, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

# 打亂資料
shuffled_data = shuffle_data(data)

# 切割資料，9:1比例
train_data, test_data = split_data(shuffled_data)

# 儲存為兩個 JSON 檔案
save_to_json(train_data, 'train_data.json')
save_to_json(test_data, 'test_data.json')

# check
print("資料已分割並儲存為 'train_data.json' 和 'test_data.json'.")

print('data：'+len(data))
print('train_data：'+len(train_data))
print('test_data：'+len(test_data))