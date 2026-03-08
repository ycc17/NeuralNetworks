import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# 訓練感知器並評估其性能
#  在本練習中，您將使用鳶尾花資料集訓練和評估感知器。鳶尾花資料集是機器學習領域的經典基準資料集，包含三種鳶尾花的測量資料
#  為了簡化問題，我們將三個類別簡化為兩個：「山鳶尾」和其他所有品種
#  您將使用萼片長度和寬度作為輸入特徵，建立感知器分類器，並在部分資料集上進行訓練，然後在預留的測試集上評估其性能

# 載入 Iris 資料集
iris = load_iris()

# 只取兩個特徵 (sepal length, sepal width)
X = iris.data[:, :2]

# 目標值
y = iris.target

# 將三分類變成二分類
# setosa = 1, 其他 = 0
y = np.where(y == 0, 1, 0)

# 分割訓練集與測試集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1
)

# 初始化權重與偏移
np.random.seed(1)

w = np.random.randn(2) * 0.01
b = np.random.randn() * 0.01

learning_rate = 0.01
epochs = 30 #訓練30次

print("Initial weights:", w)
print("Initial bias:", b)
print("====================================")

# 訓練感知器
for epoch in range(epochs):

    print("\n===== Epoch", epoch + 1, "=====")

    errors = 0

    for i in range(len(X_train)):

        x1 = X_train[i][0]
        x2 = X_train[i][1]

        # 計算 z
        z = w[0] * x1 + w[1] * x2 + b

        # 預測
        if z >= 0:
            y_pred = 1
        else:
            y_pred = 0

        # 計算誤差
        error = y_train[i] - y_pred

        # 更新權重與偏移
        w[0] = w[0] + learning_rate * error * x1
        w[1] = w[1] + learning_rate * error * x2
        b = b + learning_rate * error

        if error != 0:
            errors += 1

    print("Epoch", epoch + 1, "Errors:", errors)

print("\nTraining Finished")
print("Final weights:", w)
print("Final bias:", b)

# 測試模型

print("\n========== Testing ==========")

correct = 0

for i in range(len(X_test)):

    z = w[0] * X_test[i][0] + w[1] * X_test[i][1] + b

    if z >= 0:
        y_pred = 1
    else:
        y_pred = 0

    print("Input:", X_test[i], "Prediction:", y_pred, "True:", y_test[i])

    if y_pred == y_test[i]:
        correct += 1


accuracy = correct / len(X_test)

print("\nTest Accuracy:", accuracy)