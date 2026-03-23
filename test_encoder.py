import numpy as np
from encoder import TransformerEncoderBlock

def test_transformer_encoder():
    # 1. 设置模型超参数
    batch_size = 2      # 一次处理 2 句话
    seq_length = 5      # 每句话 5 个词
    d_model = 16        # 词向量维度 16
    num_heads = 4       # 4 个注意力头
    d_ff = 64           # FFN 隐藏层维度 (通常是 d_model 的 4 倍)

    # 2. 模拟输入数据 (未经处理的 Embedding)
    # 形状: (batch_size, seq_length, d_model) = (2, 5, 16)
    x = np.random.randn(batch_size, seq_length, d_model)

    print("-" * 50)
    print(f"📦 输入数据形状: {x.shape}")
    
    # 3. 实例化完整的 Encoder Block
    encoder_block = TransformerEncoderBlock(d_model, num_heads, d_ff)

    # 4. 执行前向传播
    output, attn_weights = encoder_block.forward(x)

    # 5. 验证输出
    print(f"✅ 编码器输出形状: {output.shape} (应与输入完全一致)")
    print(f"🔍 注意力权重形状: {attn_weights.shape} (batch_size, num_heads, seq_len, seq_len)")
    print("-" * 50)
    print("Transformer 编码器底层前向传播测试通过！🚀")

if __name__ == "__main__":
    test_transformer_encoder()