# 神经网络激活函数详解

## 目录
1. [ReLU及其变体](#relu及其变体)
2. [其他常用激活函数](#其他常用激活函数)
3. [激活函数选择指南](#激活函数选择指南)

---

## ReLU及其变体

### 1. ReLU (Rectified Linear Unit)

**公式**：
```
f(x) = max(0, x)
```

**导数**：
```
f'(x) = 1, 如果 x > 0
f'(x) = 0, 如果 x ≤ 0
```

**特点**：
- 计算简单，只需要一个阈值判断
- 解决了Sigmoid的梯度消失问题
- 收敛速度比Sigmoid/Tanh快6倍

**缺点**：
- "死亡神经元"问题：当输入为负时，梯度为0，神经元可能永远不会被激活

**Python实现**：
```python
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)
```

---

### 2. Leaky ReLU

**公式**：
```
f(x) = max(αx, x)，其中 α = 0.01（通常取值）
```

**导数**：
```
f'(x) = 1, 如果 x > 0
f'(x) = α, 如果 x ≤ 0
```

**特点**：
- 解决了ReLU的"死亡神经元"问题
- 负数区域有一个小的斜率，允许梯度流动
- α通常取0.01，确保负数区域仍有小梯度

**Python实现**：
```python
def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def leaky_relu_derivative(x, alpha=0.01):
    return np.where(x > 0, 1, alpha)
```

---

### 3. PReLU (Parametric ReLU)

**公式**：
```
f(x) = max(αx, x)，其中 α 是可学习的参数
```

**特点**：
- 与Leaky ReLU类似，但α是通过反向传播学习的
- 每个神经元可以有不同的α值
- 模型可以自适应地决定负数区域的斜率

**与Leaky ReLU的区别**：
- Leaky ReLU的α是固定的超参数（如0.01）
- PReLU的α是可训练的参数，通过梯度下降优化

---

### 4. ELU (Exponential Linear Unit)

**公式**：
```
f(x) = x,           如果 x > 0
f(x) = α(e^x - 1),  如果 x ≤ 0
```

**导数**：
```
f'(x) = 1,          如果 x > 0
f'(x) = f(x) + α,   如果 x ≤ 0
```

**特点**：
- 负数区域使用指数函数，输出均值接近0
- 减少了批归一化的需求
- 比ReLU收敛更快，准确率更高

**缺点**：
- 计算量比ReLU大（需要计算指数）

**Python实现**：
```python
def elu(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

def elu_derivative(x, alpha=1.0):
    return np.where(x > 0, 1, elu(x, alpha) + alpha)
```

---

### 5. GELU (Gaussian Error Linear Unit)

**公式**：
```
f(x) = x * Φ(x)

其中 Φ(x) 是标准正态分布的累积分布函数

近似公式：
f(x) ≈ 0.5x * (1 + tanh(√(2/π) * (x + 0.044715x³)))
```

**特点**：
- Transformer架构（如BERT、GPT）中广泛使用
- 提供了平滑的非线性变换
- 结合了dropout的随机正则化思想

**Python实现**：
```python
def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))
```

---

## 其他常用激活函数

### 1. Sigmoid

**公式**：
```
f(x) = 1 / (1 + e^(-x))
```

**导数**：
```
f'(x) = f(x) * (1 - f(x))
```

**特点**：
- 输出范围：(0, 1)
- 常用于二分类问题的输出层
- 可以将任意实数映射到概率

**缺点**：
- 梯度消失问题：当|x|很大时，梯度接近0
- 输出不以0为中心
- 计算量较大（指数运算）

**Python实现**：
```python
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)
```

---

### 2. Tanh (双曲正切)

**公式**：
```
f(x) = (e^x - e^(-x)) / (e^x + e^(-x))
     = 2 * sigmoid(2x) - 1
```

**导数**：
```
f'(x) = 1 - f(x)²
```

**特点**：
- 输出范围：(-1, 1)
- 输出以0为中心，比Sigmoid更好
- 常用于RNN和LSTM

**缺点**：
- 仍存在梯度消失问题
- 计算量较大

**Python实现**：
```python
def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x)**2
```

---

### 3. Softmax

**公式**：
```
f(x_i) = e^(x_i) / Σ(e^(x_j))  对于所有j
```

**特点**：
- 将任意实数向量转换为概率分布
- 所有输出之和为1
- 常用于多分类问题的输出层

**数值稳定性**：
为了防止指数溢出，通常减去最大值：
```python
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
```

---

### 4. Swish

**公式**：
```
f(x) = x * sigmoid(x) = x / (1 + e^(-x))
```

**导数**：
```
f'(x) = f(x) + sigmoid(x) * (1 - f(x))
```

**特点**：
- 由Google Brain提出
- 无界、平滑、非单调
- 在深层网络中表现优于ReLU
- 自门控（self-gated）机制

**Python实现**：
```python
def swish(x):
    return x * sigmoid(x)

def swish_derivative(x):
    s = sigmoid(x)
    return swish(x) + s * (1 - swish(x))
```

---

## 激活函数选择指南

| 场景 | 推荐激活函数 | 原因 |
|------|--------------|------|
| 隐藏层（通用） | ReLU | 计算快，效果好 |
| 解决死亡神经元 | Leaky ReLU / ELU | 负数区域有梯度 |
| Transformer | GELU | 平滑，效果好 |
| 二分类输出 | Sigmoid | 输出概率 |
| 多分类输出 | Softmax | 输出概率分布 |
| RNN/LSTM | Tanh | 输出以0为中心 |
| 深层网络 | Swish / GELU | 性能更好 |

---

## 总结

1. **ReLU系列**是目前最常用的激活函数家族，计算简单且效果好
2. **Sigmoid和Tanh**主要用于特定场景（输出层、RNN）
3. **GELU**在Transformer中表现优异
4. 选择激活函数时要考虑：
   - 计算效率
   - 梯度流动
   - 具体任务需求
