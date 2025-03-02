# **Differential nGPT: Combining Normalized Transformer with Differential Attention**

This repository implements a hybrid model that combines two innovative transformer architecture advancements:

1. **nGPT (Normalized Transformer)** from NVIDIA Research [arXiv:2410.01131](https://arxiv.org/abs/2410.01131)
2. **Differential Transformer** from Microsoft Research [arXiv:2410.05258](https://arxiv.org/abs/2410.05258)

This hybrid architecture aims to leverage the benefits of both approaches:
- **nGPT's** unit-normalized hypersphere representation learning for faster convergence
- **Differential Transformer's** attention noise cancellation for enhanced context relevance

## **Architecture Overview**

### **Key Components**

1. **Normalized Representation Learning (from nGPT)**
   - All vectors are normalized to lie on a unit hypersphere
   - Matrix-vector multiplications represent cosine similarities
   - Updates follow geodesic paths with eigen learning rates
   - No need for weight decay or warmup

2. **Differential Attention Mechanism (from Differential Transformer)**
   - Uses two separate softmax attention maps and takes their difference
   - Cancels out common-mode noise in attention
   - Enhances focus on relevant context
   - Improves retrieval capabilities and reduces hallucinations

3. **Combined Benefits**
   - Faster convergence from normalized representation
   - Enhanced attention allocation with differential mechanism
   - Better noise-to-signal ratio in context modeling
   - Reduced activation outliers (helpful for quantization)

## **Getting Started**

### **Dependencies**

- **PyTorch**: version 2.0+ recommended for best performance
- **FlashAttention**: from [Dao-AILab](https://github.com/Dao-AILab/flash-attention)
- **Data Preparation**: Follow [nanoGPT repository](https://github.com/karpathy/nanoGPT) instructions for preparing the OpenWebText dataset

### **Running the Code**

To start the training process with defined hyperparameters:

```bash
# Modify problem_name in launcher.sh to select configuration
./launcher.sh
```

Available configurations:
- `DiffNGPT_1kctx_10k_lr30e-4`: Differential nGPT with 1k context
- `DiffNGPT_4kctx_10k_lr30e-4`: Differential nGPT with 4k context
- `DiffTransformer_1kctx_10k_lr30e-4`: Standard Transformer with differential attention (no nGPT normalization)

### **Implementation Details**

- **Model Architecture**: See `model.py` for the implementation of Differential nGPT
- **Training Loop**: See `train.py` for the training procedure with normalization
- **Configuration**: Use `launcher.sh` to customize training parameters

## **Expected Benefits**

Based on the original papers, this hybrid model may exhibit:

1. **Faster Convergence**:
   - nGPT demonstrated 4-20x faster convergence depending on context length
   - The combined model should maintain this advantage

2. **Enhanced Context Modeling**:
   - Differential Transformer showed improved retrieval capabilities (30-85% better than standard Transformer)
   - Better hallucination mitigation (9-19% reduction)
   - Enhanced in-context learning robustness

3. **Quantization-Friendly**:
   - Reduced activation outliers, supporting lower bit quantization

## **Acknowledgements**

This work builds directly upon:

- **nGPT**: [arXiv:2410.01131](https://arxiv.org/abs/2410.01131) by Ilya Loshchilov, Cheng-Ping Hsieh, Simeng Sun, and Boris Ginsburg (NVIDIA)
- **Differential Transformer**: [arXiv:2410.05258](https://arxiv.org/abs/2410.05258) by Tianzhu Ye, Li Dong, Yuqing Xia, Yutao Sun, Yi Zhu, Gao Huang, and Furu Wei (Microsoft Research)
- **nanoGPT**: by Andrej Karpathy, which serves as the foundational codebase for nGPT

This implementation is a research exploration combining these two innovative approaches to transformer architecture. The codebase is meant to serve as a reference implementation to illustrate the concepts rather than a production-ready solution.

## **Repository Goals**

The main goal of this repository is to explore the potential synergies between two cutting-edge transformer innovations:

1. **Research Exploration**: This is a proof-of-concept implementation demonstrating how these approaches can be combined
2. **Educational Resource**: The code aims to illustrate the concepts of both architectures clearly
3. **Reference Implementation**: While not optimized for production use, it provides a foundation for further exploration

We welcome contributions and discussions about the hybrid approach and potential improvements.
