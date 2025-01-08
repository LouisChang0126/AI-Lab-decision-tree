import pandas as pd
from sklearn.utils import shuffle

# 讀取 dataset.csv
df = pd.read_csv('dataset.csv')
df.drop(columns=['Inflight entertainment','Seat comfort'], inplace=True)
# 洗混資料
df_shuffled = shuffle(df, random_state=42)

# 設定 A 和 B 的數量
A = 20000  # 訓練資料的數量
B = 5000   # 測試資料的數量

# 確保 A + B 小於資料集總數
if A + B > len(df_shuffled):
    raise ValueError("A + B 應該小於資料集的總數")

# 分割資料集
train_data = df_shuffled[:A]
test_data = df_shuffled[A:A+B]

# 儲存為 train.csv 和 test.csv
train_data.to_csv('train.csv', index=False)
test_data.to_csv('test.csv', index=False)
