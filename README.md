# Project 1: Reproduce the Final Project's Image Encoder

本项目使用纯 NumPy 实现了 OpenCLIP ViT-B-32 图像编码器（Image Tower）的前向传播计算。不依赖 PyTorch，无需 GPU。

## 环境依赖
- Python 3.9 或更高版本
- NumPy
- Pillow (PIL)

如未安装依赖，请先运行：
```bash
python -m pip install -r requirements.txt

## 文件说明
image_encoder.py：核心代码。包含了完成全部 6 个 TODO (Patchify, Attention, MLP, Embed, Output) 的 NumPy 实现。
helpers.py：提供的辅助数学函数（LayerNorm, GELU, Softmax 等）。
check.py：官方提供的数值验证脚本。
run.py：用于生成最终图像特征向量的执行脚本。

## 如何复现结果 (Reproduction Steps)
1. 验证代码正确性
运行完整的测试流程，程序将逐层比对我们手写的 NumPy 输出与官方参考答案的误差：

python check.py --stage all

预期结果：终端将输出所有阶段（patches, tokens, block0, last_tokens, output, batch）均为 PASS，且最大绝对误差（max_abs）保持在 1e-5 到 1e-7 量级，远低于容差要求。

2. 生成图像输出特征
运行以下命令，使用完成后的编码器处理默认测试集，并生成保存 512 维特征向量的 .npz 文件：

python run.py --output output_tokens.npz

3. 处理自定义图片
如果您想使用该编码器处理自己的图片，可以使用 --images 参数（支持相对路径或绝对路径）：

python run.py --images path/to/your_image.jpg --output my_tokens.npz
