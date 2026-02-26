# VGG-CIFAR10任务

## 训练ANN

```
python main.py \

    --dataset CIFAR-10 \
    --ann_model VGG16 \
    --model_name vgg_cifar_ro_0 \
    --nn_mode ANN \
    --en_train True \
    --epoch 200 \
    --batch_size 100 \
    --use_bn True \
    --use_bias True \
    --save_interval 10
```

## 收集统计数据

```
python main.py \
    --dataset=CIFAR-10 \
    --ann_model=VGG16 \
    --model_name=vgg_cifar_ro_0 \
    --nn_mode=ANN \
    --en_train=False \
    --f_write_stat=True \
    --f_stat_train_mode=True \
    --batch_size=100 \
    --checkpoint_load_dir=./models_ckpt \
    --checkpoint_dir=./models_ckpt \
    --use_bn=True \
    --use_bias=True
```

## 训练时间常数

```
python main.py \
    --dataset=CIFAR-10 \
    --ann_model=VGG16 \
    --model_name=vgg_cifar_ro_0 \
    --nn_mode=SNN \
    --en_train=False \
    --f_train_time_const=True \
    --epoch_train_time_const=6 \
    --batch_size=100 \
    --checkpoint_load_dir=./models_ckpt \
    --checkpoint_dir=./models_ckpt \
    --path_stat=./stat/vgg_cifar_ro_0/ \
    --use_bn=True \
    --use_bias=True \
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
    --time_step=1500 \
    --time_const_save_interval=10000 \
    --f_refractory=True \
    --f_record_first_spike_time=True
```

注意：之前在做MNIST任务时，`stat/`中的统计信息被直接保存在了`stat/`目录下，而没有加以`model_name`命名的子目录。这次加上了。所以统计信息在`stat/vgg_cifar_ro_0/`目录中。使用了`--path_stat=./stat/vgg_cifar_ro_0/`参数来指定。而`--checkpoint_load_dir=./models_ckpt`和`--checkpoint_dir=./models_ckpt`在代码中会自动完成目录拼接，所以不需要手动指定以`model_name`命名的子目录。

## 最终的推理

```
python main.py \
    --dataset=CIFAR-10 \
    --ann_model=VGG16 \
    --model_name=vgg_cifar_ro_0 \
    --nn_mode=SNN \
    --en_train=False \
    --f_load_time_const=True \
    --time_const_num_trained_data=60000 \
    --batch_size=100 \
    --checkpoint_load_dir=./models_ckpt \
    --checkpoint_dir=./models_ckpt \
    --path_stat=./stat/vgg_cifar_ro_0/ \
    --use_bn=True \
    --use_bias=True \
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
    --time_step=1500 \
    --f_refractory=True \
    --f_record_first_spike_time=True
```

