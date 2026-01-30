"""
反向传播算法详解与实现
========================

本文件演示神经网络中反向传播算法的核心概念：
1. 前向传播计算
2. 损失函数计算
3. 梯度计算（链式法则）
4. 参数更新

作者：AI课程作业
"""

import numpy as np

# 设置随机种子，确保结果可复现
np.random.seed(42)


# =============================================================================
# 第一部分：激活函数及其导数
# =============================================================================

def sigmoid(x):
    """
    Sigmoid激活函数

    公式: f(x) = 1 / (1 + e^(-x))
    输出范围: (0, 1)
    """
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # clip防止溢出


def sigmoid_derivative(x):
    """
    Sigmoid函数的导数

    公式: f'(x) = f(x) * (1 - f(x))
    这个性质使得Sigmoid的导数计算非常方便
    """
    s = sigmoid(x)
    return s * (1 - s)


def relu(x):
    """
    ReLU激活函数

    公式: f(x) = max(0, x)
    优点: 计算简单，避免梯度消失
    """
    return np.maximum(0, x)


def relu_derivative(x):
    """
    ReLU函数的导数

    公式: f'(x) = 1 if x > 0 else 0
    注意: x=0处导数定义为0
    """
    return (x > 0).astype(float)


# =============================================================================
# 第二部分：损失函数
# =============================================================================

def mse_loss(y_pred, y_true):
    """
    均方误差损失函数 (Mean Squared Error)

    公式: L = (1/n) * Σ(y_pred - y_true)²
    用途: 回归问题
    """
    return np.mean((y_pred - y_true) ** 2)


def mse_loss_derivative(y_pred, y_true):
    """
    MSE损失函数的导数

    公式: dL/dy_pred = (2/n) * (y_pred - y_true)
    这是反向传播的起点
    """
    return 2 * (y_pred - y_true) / y_true.shape[0]


# =============================================================================
# 第三部分：简单的两层神经网络类
# =============================================================================

class SimpleTwoLayerNet:
    """
    简单的两层神经网络

    网络结构:
    输入层 (input_size) → 隐藏层 (hidden_size) → 输出层 (output_size)

    激活函数:
    隐藏层: ReLU
    输出层: Sigmoid (用于二分类或回归)
    """

    def __init__(self, input_size, hidden_size, output_size):
        """
        初始化网络参数

        参数:
            input_size: 输入特征维度
            hidden_size: 隐藏层神经元数量
            output_size: 输出维度
        """
        # =====================================================================
        # 权重初始化 (Xavier初始化)
        # =====================================================================
        # W1: 输入层到隐藏层的权重矩阵
        # 形状: (input_size, hidden_size)
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)

        # b1: 隐藏层的偏置向量
        # 形状: (hidden_size,)
        self.b1 = np.zeros(hidden_size)

        # W2: 隐藏层到输出层的权重矩阵
        # 形状: (hidden_size, output_size)
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)

        # b2: 输出层的偏置向量
        # 形状: (output_size,)
        self.b2 = np.zeros(output_size)

        # 用于存储前向传播的中间结果（反向传播需要用到）
        self.cache = {}

    def forward(self, X):
        """
        前向传播

        计算过程:
        z1 = X @ W1 + b1      # 隐藏层线性变换
        a1 = ReLU(z1)         # 隐藏层激活
        z2 = a1 @ W2 + b2     # 输出层线性变换
        a2 = Sigmoid(z2)      # 输出层激活

        参数:
            X: 输入数据，形状 (batch_size, input_size)

        返回:
            a2: 网络输出，形状 (batch_size, output_size)
        """
        # 保存输入
        self.cache['X'] = X

        # 第一层：线性变换 + ReLU激活
        z1 = np.dot(X, self.W1) + self.b1        # 线性变换: z1 = X @ W1 + b1
        self.cache['z1'] = z1                     # 保存z1用于反向传播
        a1 = relu(z1)                             # ReLU激活
        self.cache['a1'] = a1                     # 保存a1用于反向传播

        # 第二层：线性变换 + Sigmoid激活
        z2 = np.dot(a1, self.W2) + self.b2       # 线性变换: z2 = a1 @ W2 + b2
        self.cache['z2'] = z2                     # 保存z2用于反向传播
        a2 = sigmoid(z2)                          # Sigmoid激活
        self.cache['a2'] = a2                     # 保存输出

        return a2

    def backward(self, y_true, learning_rate=0.01):
        """
        反向传播

        核心思想: 链式法则
        从输出层到输入层，逐层计算梯度

        参数:
            y_true: 真实标签
            learning_rate: 学习率

        返回:
            loss: 当前损失值
        """
        # 获取batch大小
        batch_size = y_true.shape[0]

        # 从缓存中取出前向传播的中间结果
        X = self.cache['X']
        z1 = self.cache['z1']
        a1 = self.cache['a1']
        z2 = self.cache['z2']
        a2 = self.cache['a2']

        # =====================================================================
        # 步骤1: 计算损失
        # =====================================================================
        loss = mse_loss(a2, y_true)

        # =====================================================================
        # 步骤2: 输出层梯度计算
        # =====================================================================
        # dL/da2: 损失对输出的梯度
        da2 = mse_loss_derivative(a2, y_true)

        # dL/dz2 = dL/da2 * da2/dz2 (链式法则)
        # da2/dz2 是Sigmoid的导数
        dz2 = da2 * sigmoid_derivative(z2)

        # dL/dW2 = a1^T @ dz2 (矩阵求导)
        dW2 = np.dot(a1.T, dz2)

        # dL/db2 = sum(dz2) (对batch求和)
        db2 = np.sum(dz2, axis=0)

        # =====================================================================
        # 步骤3: 隐藏层梯度计算
        # =====================================================================
        # dL/da1 = dz2 @ W2^T (梯度从输出层反向传播到隐藏层)
        da1 = np.dot(dz2, self.W2.T)

        # dL/dz1 = dL/da1 * da1/dz1 (链式法则)
        # da1/dz1 是ReLU的导数
        dz1 = da1 * relu_derivative(z1)

        # dL/dW1 = X^T @ dz1
        dW1 = np.dot(X.T, dz1)

        # dL/db1 = sum(dz1)
        db1 = np.sum(dz1, axis=0)

        # =====================================================================
        # 步骤4: 参数更新 (梯度下降)
        # =====================================================================
        # 参数 = 参数 - 学习率 * 梯度
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

        return loss


# =============================================================================
# 第四部分：演示反向传播过程
# =============================================================================

def demonstrate_backprop():
    """
    演示反向传播的完整过程

    使用一个简单的XOR问题来展示神经网络的学习能力
    XOR问题无法用单层感知机解决，但两层网络可以学会
    """
    print("=" * 60)
    print("反向传播算法演示")
    print("=" * 60)

    # =========================================================================
    # 1. 准备数据 (XOR问题)
    # =========================================================================
    # XOR真值表:
    # 0 XOR 0 = 0
    # 0 XOR 1 = 1
    # 1 XOR 0 = 1
    # 1 XOR 1 = 0
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    y = np.array([
        [0],
        [1],
        [1],
        [0]
    ])

    print("\n数据集 (XOR问题):")
    print("输入 X:")
    print(X)
    print("\n期望输出 y:")
    print(y.T)

    # =========================================================================
    # 2. 创建网络
    # =========================================================================
    # 输入层: 2个神经元 (两个二进制输入)
    # 隐藏层: 4个神经元
    # 输出层: 1个神经元 (XOR结果)
    net = SimpleTwoLayerNet(input_size=2, hidden_size=4, output_size=1)

    print("\n网络结构:")
    print(f"  输入层: 2 个神经元")
    print(f"  隐藏层: 4 个神经元 (ReLU激活)")
    print(f"  输出层: 1 个神经元 (Sigmoid激活)")

    # =========================================================================
    # 3. 训练网络
    # =========================================================================
    print("\n" + "=" * 60)
    print("开始训练")
    print("=" * 60)

    epochs = 10000
    learning_rate = 0.5

    for epoch in range(epochs):
        # 前向传播
        output = net.forward(X)

        # 反向传播 + 参数更新
        loss = net.backward(y, learning_rate)

        # 每1000轮打印一次
        if epoch % 1000 == 0:
            print(f"Epoch {epoch:5d} | Loss: {loss:.6f}")

    # =========================================================================
    # 4. 测试结果
    # =========================================================================
    print("\n" + "=" * 60)
    print("训练完成，测试结果")
    print("=" * 60)

    final_output = net.forward(X)

    print("\n输入 -> 预测输出 -> 期望输出 -> 四舍五入后")
    print("-" * 50)
    for i in range(len(X)):
        pred = final_output[i, 0]
        true = y[i, 0]
        rounded = round(pred)
        status = "[OK]" if rounded == true else "[X]"
        print(f"{X[i]} -> {pred:.4f} -> {true} -> {rounded} {status}")

    # 计算准确率
    predictions = (final_output > 0.5).astype(int)
    accuracy = np.mean(predictions == y) * 100
    print(f"\n准确率: {accuracy:.1f}%")


# =============================================================================
# 第五部分：梯度检验（验证反向传播的正确性）
# =============================================================================

def gradient_check():
    """
    使用数值方法检验反向传播计算的梯度是否正确

    原理:
    数值梯度 ≈ (f(x+ε) - f(x-ε)) / (2ε)
    如果数值梯度与解析梯度接近，说明反向传播实现正确
    """
    print("\n" + "=" * 60)
    print("梯度检验")
    print("=" * 60)

    # 创建简单网络
    net = SimpleTwoLayerNet(input_size=3, hidden_size=4, output_size=2)

    # 生成随机数据
    X = np.random.randn(5, 3)  # 5个样本，3个特征
    y = np.random.randn(5, 2)  # 5个样本，2个输出

    # 计算解析梯度（通过反向传播）
    output = net.forward(X)

    # 保存原始参数
    W1_original = net.W1.copy()

    # 计算解析梯度
    batch_size = y.shape[0]
    da2 = mse_loss_derivative(output, y)
    dz2 = da2 * sigmoid_derivative(net.cache['z2'])
    da1 = np.dot(dz2, net.W2.T)
    dz1 = da1 * relu_derivative(net.cache['z1'])
    analytical_gradient = np.dot(X.T, dz1)

    # 计算数值梯度
    epsilon = 1e-5
    numerical_gradient = np.zeros_like(net.W1)

    for i in range(net.W1.shape[0]):
        for j in range(net.W1.shape[1]):
            # f(x + ε)
            net.W1[i, j] = W1_original[i, j] + epsilon
            output_plus = net.forward(X)
            loss_plus = mse_loss(output_plus, y)

            # f(x - ε)
            net.W1[i, j] = W1_original[i, j] - epsilon
            output_minus = net.forward(X)
            loss_minus = mse_loss(output_minus, y)

            # 数值梯度
            numerical_gradient[i, j] = (loss_plus - loss_minus) / (2 * epsilon)

            # 恢复参数
            net.W1[i, j] = W1_original[i, j]

    # 计算相对误差
    difference = np.abs(analytical_gradient - numerical_gradient)
    relative_error = difference / (np.abs(analytical_gradient) + np.abs(numerical_gradient) + 1e-8)
    max_relative_error = np.max(relative_error)

    print(f"\n最大相对误差: {max_relative_error:.2e}")

    if max_relative_error < 1e-5:
        print("梯度检验通过！反向传播实现正确。")
    else:
        print("警告：梯度误差较大，请检查反向传播实现。")


# =============================================================================
# 主函数
# =============================================================================

if __name__ == "__main__":
    # 演示反向传播
    demonstrate_backprop()

    # 梯度检验
    gradient_check()

    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)
    print("""
反向传播的核心步骤:

1. 前向传播: 计算每一层的输出，保存中间结果

2. 计算损失: 比较预测值和真实值

3. 反向传播: 使用链式法则，从输出层向输入层计算梯度
   - 输出层: dL/dz2 = dL/da2 * da2/dz2
   - 隐藏层: dL/dz1 = dL/da1 * da1/dz1

4. 参数更新: W = W - learning_rate * dL/dW

关键公式:
   - dL/dW = a_prev^T @ dz   (权重梯度)
   - dL/db = sum(dz)          (偏置梯度)
   - da_prev = dz @ W^T       (传递到上一层的梯度)
""")
