import os
import tarfile
import pickle
import numpy as np
import shutil

def main():
    # 1. 定义路径
    base_dir = "dataset_downloaded"
    tar_path = os.path.join(base_dir, "cifar-10-python.tar.gz")
    output_dir = os.path.join(base_dir, "cifar10_float")
    
    if not os.path.exists(tar_path):
        print(f"[Error] 找不到文件: {tar_path}")
        print("请确认 CIFAR-10 压缩包已经放置在该位置。")
        return

    # 确保输出目录干净
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"[Info] 正在从 {tar_path} 读取数据...")
    
    test_images = None
    test_labels = None
    
    # 2. 从 tar.gz 中直接提取 test_batch
    # CIFAR-10 有 5 万张训练集和 1 万张测试集。推理通常在测试集上进行。
    with tarfile.open(tar_path, 'r:gz') as tar:
        try:
            f = tar.extractfile('cifar-10-batches-py/test_batch')
            if f:
                entry = pickle.load(f, encoding='bytes')
                test_images = entry[b'data']
                test_labels = entry[b'labels']
        except KeyError:
            print("[Error] 在压缩包中找不到 test_batch 文件。")
            return

    print("[Info] 正在转换数据格式...")
    
    # 3. 数据转换 (关键步骤)
    # 原始 CIFAR-10 格式是 (10000, 3072)，表示 (N, 3通道 x 32高 x 32宽)
    # 我们需要转换为 TensorFlow 使用的 channels_last 格式: (N, 32, 32, 3)
    x_test = test_images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    
    # 转换为 float32 并归一化到 [0, 1]
    # 注意：vgg_cifar_ro_0 模型禁用了均值方差归一化(f_data_std=False)，所以这里只除以 255 即可
    x_test_float = x_test.astype(np.float32) / 255.0
    
    # 标签转换为 One-hot, float32 格式
    y_test = np.array(test_labels)
    y_test_onehot = np.zeros((y_test.size, 10), dtype=np.float32)
    y_test_onehot[np.arange(y_test.size), y_test] = 1.0

    print(f"[Info] 测试集形状: {x_test_float.shape}") # 预期: (10000, 32, 32, 3)
    print(f"[Info] 标签集形状: {y_test_onehot.shape}") # 预期: (10000, 10)

    # 4. 存储标签 (单个文件)
    label_path = os.path.join(output_dir, "label_onehot")
    y_test_onehot.tofile(label_path)
    print(f"[Success] 标签已保存至: {label_path}")

    # 5. 存储图片 (每张图片一个 .bin 文件)
    print(f"[Info] 正在保存 {x_test_float.shape[0]} 张图片...")
    for i in range(x_test_float.shape[0]):
        # 文件名例如: 0.bin, 1.bin ...
        # C/C++ 读取时只需申请 3072 * sizeof(float) 的内存
        img_path = os.path.join(output_dir, f"{i}.bin")
        # 展平为一维数组后写入
        x_test_float[i].flatten().tofile(img_path)
    
    print(f"[Success] 所有图片已成功保存至: {output_dir}")

if __name__ == "__main__":
    main()
