# 如何走到现在

## 1 复现原项目并准备必要基础

从ANN到SNN三步：

准备：训练ANN

```bash
python main.py \
    --dataset=MNIST \
    --ann_model=CNN \
    --nn_mode=ANN \
    --en_train=True \
    --epoch=500 \
    --batch_size=100 \
    --model_name=mnist_cnn_demo \
    --lr=0.001
```

第一步：收集激活统计信息

```bash
python main.py \
    --dataset=MNIST \
    --ann_model=CNN \
    --model_name=mnist_cnn_demo \
    --nn_mode=ANN \
    --en_train=False \
    --f_fused_bn=True \
    --f_write_stat=True \
    --f_w_norm_data=False \
    --f_stat_train_mode=True \
    --batch_size=100
```

第二步：训练时间常数

```bash
python main.py \
    --dataset=MNIST \
    --ann_model=CNN \
    --model_name=mnist_cnn_demo \
    --nn_mode=SNN \
    --en_train=False \
    --f_fused_bn=True \
    --f_w_norm_data=True \
    --f_write_stat=False \
    --neural_coding=TEMPORAL \
    --input_spike_mode=TEMPORAL \
    --n_type=IF \
    --n_init_vth=1.0 \
    --tc=20 \
    --time_fire_start=80 \
    --time_fire_duration=80 \
    --time_window=80 \
    --time_step=400 \
    --f_train_time_const=True \
    --epoch_train_time_const=6 \
    --time_const_save_interval=10000 \
    --f_refractory=True \
    --f_record_first_spike_time=True \
    --batch_size=100
```

第三步：加载刚刚训练好的时间常数，并在测试集上进行最终的 SNN 推理

```bash
python main.py \
    --dataset=MNIST \
    --ann_model=CNN \
    --model_name=mnist_cnn_demo \
    --nn_mode=SNN \
    --en_train=False \
    --f_fused_bn=True \
    --f_w_norm_data=True \
    --f_write_stat=False \
    --neural_coding=TEMPORAL \
    --input_spike_mode=TEMPORAL \
    --n_type=IF \
    --n_init_vth=1.0 \
    --tc=20 \
    --time_fire_start=80 \
    --time_fire_duration=80 \
    --time_window=80 \
    --time_step=400 \
    --f_train_time_const=False \
    --f_load_time_const=True \
    --time_const_num_trained_data=60000 \
    --f_refractory=True \
    --f_record_first_spike_time=True \
    --batch_size=100
```

SNN推理中的输出打印显示，准确率达到了99%以上。说明有了一个可用的权重。

接下来需要导出这个权重，创建我的自定义cuda算法，使用cuda编程重新实现此项目使用tensorflow实现的TTFS-SNN神经网络。

这个是改造原有main函数，导出原项目的权重。

```bash
python main_inject_getweight.py     --dataset=MNIST     --ann_model=CNN     --model_name=mnist_cnn_demo     --nn_mode=SNN     --en_train=False     --f_fused_bn=True     --f_w_norm_data=True     --f_write_stat=False     --neural_coding=TEMPORAL     --input_spike_mode=TEMPORAL     --n_type=IF     --n_init_vth=1.0     --tc=20     --time_fire_start=80     --time_fire_duration=80     --time_window=80     --time_step=400     --f_train_time_const=False     --f_load_time_const=True     --time_const_num_trained_data=60000     --f_refractory=True     --f_record_first_spike_time=True     --batch_size=100
```

使用导出的TTFS-SNN权重对一张图片进行推理，此处使用由`create_binary_mnist/`中的脚本导出到`dataset_downloaded/mnist_float/`的mnist数据集中的第0张图片`0.bin`。

前期工作到此结束。

## 2 CUDA实现

编写CUDA代码`snn_inference`，实现与原项目tensorflow实现的脉冲神经网络完全等价的脉冲神经网络计算。

```bash
nvcc snn_inference.cu -o snn_inference
./snn_inference 0
```

使用torch实现与cuda相同的计算，从而验证cuda的计算是否正确。

```bash
python snn_inference.py 0
```
