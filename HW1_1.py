import numpy as np

# 在 Python 中實作感知器
# 1. 首先將權重和偏移初始化為較小的隨機值
# 2. 對於每個訓練樣本，將輸入向量輸入感知器，並使用目前的權重和偏移量產生預測值
# 3. 將此預測值與目標值進行比較－差值可以告訴你每個權重需要增加或減少，以及需要增加或減少多少（根據學習率進行縮放）
# 4. 對所有訓練樣本重複此過程，直到模型收斂為止，迭代次數（或 epoch）需達到設定值

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

y = [0, 0, 0, 1]

np.random.seed(1)

w = np.random.randn(2) * 0.01
b = np.random.randn() * 0.01

learning_rate = 0.1
epochs = 20

print("Initial weights:", w)
print("Initial bias:", b)

# 訓練感知器
for epoch in range(epochs):

    print("\n===== Epoch", epoch + 1, "=====")

    errors = 0

    for i in range(len(X)):

        x1 = X[i][0]
        x2 = X[i][1]

        print("\nSample", i + 1)
        print("輸入資料:", X[i])

        # 計算 z
        z = w[0] * x1 + w[1] * x2 + b
        print("z =", z)

        # 預測
        if z >= 0:
            y_pred = 1
        else:
            y_pred = 0

        print("Prediction:", y_pred)
        print("Target:", y[i])

        # 計算誤差
        error = y[i] - y_pred
        print("Error:", error)

        # 更新前權重
        print("更新前 -> w:", w, "b:", b)

        # 更新權重
        w[0] = w[0] + learning_rate * error * x1
        w[1] = w[1] + learning_rate * error * x2
        b = b + learning_rate * error

        # 更新後權重
        print("更新後  -> w:", w, "b:", b)

        if error != 0:
            errors += 1

    print("\nEpoch", epoch + 1, "Total Errors:", errors)

    if errors == 0:
        print("\nTraining converged!")
        break


# 測試模型
print("\n========== Testing Result ==========")

for i in range(len(X)):

    z = w[0] * X[i][0] + w[1] * X[i][1] + b

    if z >= 0:
        y_pred = 1
    else:
        y_pred = 0

    print("Input:", X[i], "Prediction:", y_pred)