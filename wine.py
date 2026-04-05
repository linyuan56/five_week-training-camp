import numpy as np
import pandas as pd


def numpy_spilt(X, Y, test_size=0.2, val_size = 0.0, random_state=42):
    """这个函数是用来划分数据集的，相当于sklearn的train_test_split"""
    np.random.seed(random_state) # 确保可复现
    n_samples = X.shape[0] # 样本数
    indices = np.random.permutation(n_samples)
    # 划分每个集的数量
    spilt_size = int(n_samples * test_size)
    test_indices = indices[:spilt_size]
    if val_size > 0:
        val_size = int(n_samples * val_size)
        val_indices = indices[spilt_size:spilt_size+val_size]
        train_indices = indices[val_size+spilt_size:]
    else:
        train_indices = indices[spilt_size:]
    # 根据数量提取数据
    X_train = X[train_indices]
    Y_train = Y[train_indices]
    X_test = X[test_indices]
    Y_test = Y[test_indices]
    if val_size > 0:
        X_val = X[val_indices]
        Y_val = Y[val_indices]
        X_train, X_test, X_val = standardization(X_train, X_test, X_val)
        return X_train, Y_train, X_test, Y_test, X_val, Y_val
    X_train, X_test, *_ = standardization(X_train, X_test)
    return X_train, Y_train, X_test, Y_test

def MSE(y_true, y_pred):
    """用于量化线性回归的损失函数"""
    return np.mean((y_true - y_pred) ** 2)

def LogLoss(y_true, y_pred):
    """用于量化逻辑回归的损失函数"""
    # 为了防止 log(0) 报错，对预测值进行极小值的裁剪
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def standardization(X_train, X_test, *others):
    """数据标准化"""
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    std = np.where(std == 0, 1, std)  # 避免除0
    return (X_train - mean) / std, (X_test - mean) / std, *[(X - mean) / std for X in others]

def LinearRegression1(X_raw, Y_raw):
    """解析法"""
    X_raw1 = X_raw.copy()
    X_raw1['bias'] = 1.0  # 这个是给bias的量
    X_raw1 = X_raw.to_numpy()
    Y_raw = Y_raw.to_numpy()
    X_train, Y_train, X_test, Y_test= numpy_spilt(X_raw1,Y_raw)
    W1 = np.linalg.pinv(X_train.T.dot(X_train)).dot(X_train.T).dot(Y_train)
    W = W1[:-1].reshape(-1, 1) # 将二者分开来
    b = W1[-1]
    # MSE测试结果
    def predict(W1, X):
        return np.dot(X, W1)

    mse = MSE(Y_test, predict(W1, X_test))
    print("解析法结果如下：")
    print(f"w:{W}\nb:{b}")
    print(f"MSE:{mse}\n")

def LinearRegression2(X_raw, Y_raw):
    """梯度下降法"""
    X_raw = X_raw.to_numpy()
    Y_raw = Y_raw.to_numpy()
    # 划分数据集
    X_train, Y_train, X_test, Y_test, X_val, Y_val = numpy_spilt(X_raw, Y_raw, val_size=0.1)
    batch_size = 32 # 批次大小
    learning_rate = 0.01 # 学习率LR
    epochs = 1000 # 训练次数
    W = np.random.randn(X_train.shape[1], 1)
    b = 0.0

    def fit(X_train, Y_train,W, b, epochs, learning_rate, batch_size):
        Y_train = Y_train.reshape(-1, 1)
        memory = float('inf')
        for i in range(epochs):
            batch_left = 0
            batch_right = batch_size
            # 如果样本数不等于32的公倍数，那末尾最后的直接不纳入训练
            while batch_right <= X_train.shape[0]:
                X_batch = X_train[batch_left:batch_right] # 划分批次
                Y_batch = Y_train[batch_left:batch_right]
                # 前向传播
                y_predict = np.dot(X_batch, W) + b
                # 计算梯度
                dw = (1/len(X_batch)) * np.dot(X_batch.T, y_predict-Y_batch)
                db = (1/len(X_batch)) * np.sum(y_predict - Y_batch)
                W -= learning_rate * dw
                b -= learning_rate * db
                # 批次更新
                batch_left += batch_size
                batch_right += batch_size
            # 测试集测试
            loss = MSE(Y_val, predict(W, b, X_val))
            if loss > memory:
                print(f"在第{i+1}轮时，验证集的损失反弹，停止学习")
                break
            else: memory = loss
        return W, b # 调试那么久的原因是因为我没有返回这个值，无语

    def predict(W, b, X):
        return np.dot(X, W) + b

    W, b = fit(X_train, Y_train, W, b, epochs, learning_rate, batch_size)
    mse = MSE(Y_test, predict(W, b, X_test))
    print("梯度下降法结果如下：")
    print(f"w:{W}\nb:{b}")
    print(f"MSE:{mse}\n")

def Logicalregression(X_raw, Y_raw):
    """逻辑回归"""
    # 标签二值化: 质量 > 6 为好酒 (1), 否则为坏酒 (0)
    y = np.where(Y_raw > 6, 1, 0).reshape(-1, 1)

    X_raw = X_raw.to_numpy()
    # 划分数据集
    X_train, Y_train, X_test, Y_test, X_val, Y_val = numpy_spilt(X_raw, y, val_size=0.1)
    batch_size = 32  # 批次大小
    learning_rate = 0.000001  # 学习率LR
    epochs = 10000  # 训练次数
    W = np.random.randn(X_train.shape[1], 1)
    b = 0.0

    def fit(X_train, Y_train, W, b, epochs, learning_rate, batch_size):
        Y_train = Y_train.reshape(-1, 1)
        memory = float('inf')
        for i in range(epochs):
            batch_left = 0
            batch_right = batch_size
            # 如果样本数不等于32的公倍数，那末尾最后的直接不纳入训练
            while batch_right <= X_train.shape[0]:
                X_batch = X_train[batch_left:batch_right]  # 划分批次
                Y_batch = Y_train[batch_left:batch_right]
                # 前向传播
                y_predict = np.dot(X_batch, W) + b
                # 计算梯度
                dw = (1 / len(X_batch)) * np.dot(X_batch.T, y_predict - Y_batch)
                db = (1 / len(X_batch)) * np.sum(y_predict - Y_batch)
                W -= learning_rate * dw
                b -= learning_rate * db
                # 批次更新
                batch_left += batch_size
                batch_right += batch_size
            # 测试集测试
            loss = LogLoss(Y_val, predict(X_val, W, b))
            if loss > memory+1:
                print(f"在第{i + 1}轮时，验证集的损失反弹，停止学习")
                break
            else:
                memory = loss
        return W, b

    def predict(X, W, b, threshold=0.5):
        """预测函数"""
        probs = sigmoid(np.dot(X, W) + b)
        return (probs >= threshold).astype(int)

    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    W, b = fit(X_train, Y_train, W, b, epochs, learning_rate, batch_size)
    y_pred = predict(X_test, W, b)
    accuracy = np.mean(y_pred == Y_test)

    print("逻辑回归结果如下")
    print(f"w:{W}\nb:{b}")
    print(f"模型准确率: {accuracy:.4f}")

def support_v_m():
    # SVM支持向量机
    pass

data = pd.read_csv("winequality-red.csv", sep=';')
# 数据初始化
Y_raw = data['quality']
X_raw = data.drop('quality', axis=1)
np.random.seed(42)  # 确保实验可重现
LinearRegression1(X_raw, Y_raw)
LinearRegression2(X_raw, Y_raw)
Logicalregression(X_raw, Y_raw)
