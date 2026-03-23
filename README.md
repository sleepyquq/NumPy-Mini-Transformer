# NumPy Transformer Encoder: Attention From Scratch 🚀

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![NumPy](https://img.shields.io/badge/NumPy-Dependency_Free-green.svg)
![License](https://img.shields.io/badge/License-MIT-blue.svg)

A lightweight, dependency-free implementation of the core Transformer Encoder architecture using **pure NumPy**. 

This project deconstructs the "black box" of modern deep learning frameworks (like PyTorch and TensorFlow). By hand-coding the forward pass, matrix multiplications, multi-head splitting logic, and layer normalization, it bridges theoretical Deep Learning concepts with raw engineering implementation.

## 🌟 Core Features

- **Pure NumPy Implementation**: Zero dependencies on heavy machine learning frameworks.
- **Scaled Dot-Product Attention**: Includes a numerically stable Softmax implementation to prevent overflow.
- **Multi-Head Attention (MHA)**: Fully implements tensor reshaping and batched matrix multiplication for parallel head computation.
- **Complete Encoder Block**: Seamlessly integrates the Feed-Forward Network (FFN) and Add & Norm (residual connections and layer normalization).
- **Dimensionality Safety**: Built-in `assert` mechanisms to ensure correct tensor shapes and division across attention heads.

## 📁 Project Structure

```text
NumPy-Transformer/
├── encoder.py         # Core source: MHA, FFN, LayerNorm, and Encoder Block
├── test_encoder.py    # Test script: Verifies forward pass and tensor dimensions
├── .gitignore         # Git ignore configurations
└── README.md          # Project documentation
```

## 🛠️ Quick Start

Clone the repository and run the test script to observe the forward pass of tensors through the Transformer encoder:
```bash
# Clone the repo
git clone [https://github.com/sleepyquq/NumPy-Mini-Transformer.git](https://github.com/your-username/NumPy-Mini-Transformer.git)

# Navigate to the directory
cd NumPy-Transformer

# Run the test script
python test_encoder.py
```

## How It Works
The test script initializes a random input tensor simulating word embeddings with shape (batch_size, seq_len, d_model). It passes this tensor through the full Encoder Block, successfully routing the data through the MHA and FFN layers while maintaining perfect dimensionality and extracting the (batch_size, num_heads, seq_len, seq_len) attention weights.