# BN 和 LN 的用法及 Transformer 中使用 LN 的原因

## Batch Normalization (BN)

### 用法
Batch Normalization（批归一化）是一种加速神经网络训练的技术，主要用于卷积神经网络（CNN）中。其主要步骤如下：
1. 对每个 mini-batch 的输入计算均值和方差。
2. 使用均值和方差对输入进行归一化，使其均值为 0，方差为 1。
3. 引入可学习的缩放参数 γ 和偏移参数 β，对归一化后的值进行线性变换：
   \[
   \hat{x} = \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
   \]

### 优点
- 减少了对权重初始化的敏感性。
- 加速了模型的收敛。
- 在一定程度上缓解了梯度消失和梯度爆炸问题。

### 局限性
- 对于小 batch size 的情况下效果较差，因为均值和方差的估计不准确。
- 在序列建模任务中（如 RNN），对时间步的归一化可能会 **破坏时间依赖性**。

---

## Layer Normalization (LN)

### 用法
Layer Normalization（层归一化）是一种归一化技术，主要用于序列建模任务（如 Transformer）。其主要步骤如下：
1. 对每个样本的特征维度计算均值和方差。
2. 使用均值和方差对特征进行归一化，使其均值为 0，方差为 1。
3. 同样引入可学习的缩放参数 γ 和偏移参数 β：
   \[
   \hat{x} = \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
   \]

### 优点
- 不依赖于 batch size，适用于小 batch 或 batch size 为 1 的情况。
- 更适合序列建模任务，因为归一化是在特征维度上进行的，不会破坏时间依赖性。

---

## Transformer 中使用 LN 的原因

Transformer 模型中广泛使用 Layer Normalization（LN），而不是 Batch Normalization（BN），主要原因如下：

1. **适用于小 batch size**
   Transformer 的训练通常使用小 batch size，尤其是在自然语言处理任务中。LN 不依赖于 batch size，因此在这种情况下表现更稳定。

2. **序列建模的需求**
   文本和时间序列通常是变长的，Transformer 处理序列数据（如文本、时间序列）时，LN 在特征维度上进行归一化，不会破坏序列的时间依赖性。而 BN 在 batch 维度上归一化，可能会引入不必要的噪声。

3. **更快的收敛**
   LN 能够更快地稳定训练过程，尤其是在深度模型中。Transformer 中的多头注意力机制和前馈网络层都受益于 LN 的稳定性。

4. **适应性强**
   LN 对输入分布的变化更具鲁棒性，能够更好地适应 Transformer 中复杂的注意力机制。

---

## 总结
- **BN**：适用于 CNN，依赖 batch size，主要在 batch 维度归一化。
- **LN**：适用于序列建模任务，不依赖 batch size，主要在特征维度归一化。
- **Transformer 使用 LN** 是因为其对小 batch size 的适应性、对序列建模的支持以及更快的收敛速度。