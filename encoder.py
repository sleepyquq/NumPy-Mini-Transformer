import numpy as np

def softmax(x):
    """计算 softmax，为了数值稳定性减去最大值"""
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)

def scaled_dot_product_attention(q, k, v, mask=None):
    """
    纯 NumPy 实现缩放点积注意力机制
    公式: Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
    """
    d_k = q.shape[-1]
    
    # 1. 计算点积 q * k^T
    scores = np.matmul(q, k.swapaxes(-1, -2)) / np.sqrt(d_k)
    
    # 掩码 (Masking)，如果有 Decoder
    if mask is not None:
        scores = np.where(mask == 0, -1e9, scores)
        
    # 3. 计算注意力权重 (Softmax)
    attention_weights = softmax(scores)
    
    # 4. 加权求和得到最终输出
    output = np.matmul(attention_weights, v)
    
    return output, attention_weights

class MultiHeadAttention:
    """多头注意力模块"""
    def __init__(self, d_model, num_heads):
        self.num_heads = num_heads
        self.d_model = d_model
        # 确保特征维度可以被头数整除
        assert d_model % self.num_heads == 0 
        self.d_k = d_model // num_heads
        
        # 初始化线性映射权重 (通常用 Xavier 初始化，这里为简写用标准差缩放)
        self.W_q = np.random.randn(d_model, d_model) * 0.01
        self.W_k = np.random.randn(d_model, d_model) * 0.01
        self.W_v = np.random.randn(d_model, d_model) * 0.01
        self.W_o = np.random.randn(d_model, d_model) * 0.01

    def split_heads(self, x, batch_size):
        """将张量重塑并转置，拆分出多个头"""
        # (batch_size, seq_length, d_model) -> (batch_size, seq_length, num_heads, d_k)
        x = x.reshape(batch_size, -1, self.num_heads, self.d_k)
        # 转置为 (batch_size, num_heads, seq_length, d_k) 方便后续矩阵相乘
        return x.transpose(0, 2, 1, 3)

    def forward(self, q, k, v, mask=None):
        batch_size = q.shape[0]
        
        # 1. 线性映射
        q_proj = np.matmul(q, self.W_q)
        k_proj = np.matmul(k, self.W_k)
        v_proj = np.matmul(v, self.W_v)
        
        # 2. 拆分为多头
        q_split = self.split_heads(q_proj, batch_size)
        k_split = self.split_heads(k_proj, batch_size)
        v_split = self.split_heads(v_proj, batch_size)
        
        # 3. 计算缩放点积注意力
        attention_output, weights = scaled_dot_product_attention(q_split, k_split, v_split, mask)
        
        # 4. 拼接多头 (Concat)
        # 形状恢复: (batch_size, num_heads, seq_length, d_k) -> (batch_size, seq_length, num_heads, d_k)
        attention_output = attention_output.transpose(0, 2, 1, 3)
        # 展平最后两个维度拼接: (batch_size, seq_length, d_model)
        concat_output = attention_output.reshape(batch_size, -1, self.d_model)
        
        # 5. 最终的线性映射
        final_output = np.matmul(concat_output, self.W_o)
        
        return final_output, weights
    
class FeedForward:
    """前馈神经网络模块"""
    def __init__(self, d_model, d_ff):
            # d_ff 通常是 d_model 的 4 倍
            self.W1 = np.random.randn(d_model, d_ff) * 0.01
            self.W2 = np.random.randn(d_ff, d_model) * 0.01

    def forward(self, x):
        # ReLU 激活函数
        hidden = np.maximum(0, x @ self.W1)
        return hidden @ self.W2
        
class LayerNorm:
    """层归一化模块"""
    def __init__(self, eps=1e-5):
        self.eps = eps

    def forward(self, x):
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        # 减去均值，除以标准差 (eps防止除以0)
        return (x - mean) / np.sqrt(var + self.eps)
    
class TransformerEncoderBlock:
    """完整的 Transformer 编码器块"""
    def __init__(self, d_model, num_heads, d_ff):
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = LayerNorm()
        self.norm2 = LayerNorm()

    def forward(self, x, mask=None):
        # 第一部分：多头注意力 + 残差连接 + 归一化
        attn_out, weights = self.mha.forward(x, x, x, mask)
        x = self.norm1.forward(x + attn_out)
        
        # 第二部分：前馈网络 + 残差连接 + 归一化
        ffn_out = self.ffn.forward(x)
        x = self.norm2.forward(x + ffn_out)
        
        return x, weights