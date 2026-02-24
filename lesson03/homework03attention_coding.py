import numpy as np
def softmax(x):
    # axis=-1 表示在最内层进行softmax计算
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)
def scaled_dot_product_attention(Q, K, V, mask=None):
    # 1.计算点积相似度
    # Q: (batch_size, seq_len_q, d_k)
    # K: (batch_size, seq_len_k, d_k)
    # V: (batch_size, seq_len_v, d_v)
    d_k = Q.shape[-1]
    matmul_qk = np.matmul(Q, K.swapaxes(-2, -1)) 
    # 2.缩放
    scaled_attention_logits = matmul_qk / np.sqrt(d_k)
    # 3.应用mask（如果有）
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)  # 将mask位置的值设置为一个非常大的负数
    # 4.计算softmax
    attention_weights = softmax(scaled_attention_logits)
    # 5.加权求和
    output = np.matmul(attention_weights, V)  # (batch_size, seq_len
    return output, attention_weights


class SingleHeadAttention:
    def __init__(self, d_kmodel,d_k):
        # d_model: 输入和输出的维度q
        # d_k: Q、K、V的维度
        self.W_q = np.random.randn(d_kmodel, d_k)
        self.W_k = np.random.randn(d_kmodel, d_k)
        self.W_V = np.random.randn(d_kmodel, d_k)

    def forward(self, x, mask=None):
        # x 形状： (batch_size, seq_len, d_model)
        #1.线性变换
        Q = np.matmul(x, self.W_q)  # (batch_size, seq_len, d_k)
        K = np.matmul(x, self.W_k)  # (batch_size, seq_len, d_k)
        V = np.matmul(x, self.W_V)  # (batch_size, seq_len, d_k)
        #2.计算注意力
        output, attention_weights = scaled_dot_product_attention(Q, K, V, mask)
        return output, attention_weights
    


class MultiHeadAttention:
    def __init__(self, d_model, num_heads):
        # d_model: 输入和输出的维度
        # num_heads: 注意力头的数量
    
        self.d_model = d_model
        self.num_heads = num_heads
        # 确保d_model可以被num_heads整除
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads

        # 定义每个头的线性变换矩阵
        self.W_q = np.random.randn(d_model, d_model)
        self.W_k = np.random.randn(d_model, d_model)
        self.W_v = np.random.randn(d_model, d_model)
        self.W_o = np.random.randn(d_model, d_model)  # 输出线性
    
    def split_heads(self, x):
        # 将输入分成多个头
        # 关键步骤：（batch_size, seq_len, d_model) -> (batch_size, num_heads, seq_len, d_k)
        batch_size, seq_len, d_model = x.shape
        x = x.reshape(batch_size, seq_len, self.num_heads, self.d_k)
        return x.transpose(0, 2, 1, 3)  # (batch_size, num_heads, seq_len, d_k)
    
    def forward(self, x, mask=None):
        batch_size=x.shape[0]

        # 1.线性变换
        q= np.matmul(x, self.W_q)  # (batch_size, seq_len, d_model)
        k= np.matmul(x, self.W_k)  # (batch_size, seq_len, d_model)
        v= np.matmul(x, self.W_v)  # (batch_size, seq_len

        # 2.分头
        q = self.split_heads(q)  # (batch_size, num_heads, seq_len, d_k)
        k = self.split_heads(k)  # (batch_size, num_heads, seq_len
        v = self.split_heads(v)  # (batch_size, num_heads, seq_len, d_k)

        # 3.缩放点积注意力

        scaled_attention,weights = scaled_dot_product_attention(q, k, v,mask)  # (batch_size, num_heads, seq_len, d_k)

        # 4. 合并多头 (重塑回 3 维)
        # 先把 Heads 换回第 3 维：(batch, seq, num_heads, d_k)
        concat_attention = scaled_attention.transpose(0, 2, 1, 3)
        concat_attention = concat_attention.reshape(batch_size, -1, self.d_model)

        # 5. 最后的线性输出层
        output = np.dot(concat_attention, self.W_o)
        return output, weights
    
# --- 测试部分 ---
batch_size, seq_len, d_model = 2, 5, 128
x = np.random.randn(batch_size, seq_len, d_model)

mha = MultiHeadAttention(d_model=128, num_heads=8)
output, weights = mha.forward(x) # 现在的 forward 只需要 x

print(f"输出形状: {output.shape}") 
print(f"权重形状: {weights.shape}")