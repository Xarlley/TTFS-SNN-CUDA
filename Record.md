# 如何走到现在

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

```bash
nvcc snn_inference.cu -o snn_inference
./snn_inference 0
```

使用torch实现与cuda相同的计算，从而验证cuda的计算是否正确。

```bash
python snn_inference.py 0
```

迭代：
`snn_inference.cu`有如下问题
1. 遗漏了原作者的权重指数衰减
2. 处理bias时出现错误
3. tc和td与原作者的处理方式不同。

修改后，得到`snn_inference4.cu`。但是，`snn_inference4.cu`在推理时出现了一个现象：凡是有推理输出的，都能够推理正确，但是有大概30%的图片，并没有任何最终输出脉冲（打印显示所有Class都是Did not fire）。此问题暂时无法解决。由于问题未解决，暂时不修改作为比对的`snn_inference.py`代码

```bash
nvcc snn_inference4.cu -o snn_inference4
./snn_inference 0
```
