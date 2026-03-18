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

修改 `configs/weight_norm.conf`

开启写统计数据的开关：

```
f_write_stat_train_data=True
#f_write_stat_train_data=False
```

修改 `run.sh`

```
# --- 模型和模式选择 ---
nn_mode='ANN'             # 必须是 ANN
exp_case='VGG16_CIFAR-10'

# --- 运行模式 ---
training_mode=False       # 必须是 False，因为我们是跑推理来收集数据
```

运行 `run.sh`

```
bash run.sh
```

触发OOM时的修复：

修改 `run.sh`

找到全量测试开关，并将其关闭

```
# full test
#f_full_test=True       <-- 注释掉 True
f_full_test=False      <-- 取消注释 False
```

修改子集的数据量

```
###############################################################
# Batch size - small test
###############################################################

# only when (f_full_test = False)
batch_size=100           <-- 建议改成 100
idx_test_dataset_s=0
num_test_dataset=1000    <-- 建议改成 1000 (跑 10 个 Batch)
```

重新运行 `run.sh`

## 训练时间常数

修改 configs/weight_norm.conf

```
Bash
# 修改前：
f_write_stat_train_data=True
#f_write_stat_train_data=False

# 修改后：
#f_write_stat_train_data=True
f_write_stat_train_data=False
```

修改 `run.sh`，找到对应行并修改为以下状态：

```
#nn_mode='ANN'
nn_mode='SNN'       <-- 确保这里是 SNN

# full test
f_full_test=True    <-- 取消注释 True，恢复全量测试
#f_full_test=False  <-- 注释掉 False

f_load_time_const=False         <-- 修改为 False (不加载旧的)
#f_load_time_const=True

# train time constant for temporal coding
#f_train_time_const=False
f_train_time_const=True         <-- 修改为 True (开启训练)

tc=20
time_fire_start=40              <-- 确保这里是 40 (而不是 80)
time_fire_duration=80
time_window=${time_fire_duration}

time_const_init_file_name='./temporal_coding/time_const'
```

运行 `run.sh`

## 进行最终推理

修改 `run.sh`

开启加载 (Load) 开关

```
# 修改前：
f_load_time_const=False
#f_load_time_const=True

# 修改后：
#f_load_time_const=False
f_load_time_const=True
```

关闭训练 (Train) 开关

```
# 修改前：
#f_train_time_const=False
f_train_time_const=True

# 修改后：
f_train_time_const=False
#f_train_time_const=True
```

确认其他参数保持原样

```
time_const_num_trained_data=60000
time_const_init_file_name='./temporal_coding/time_const'
time_fire_start=40
```

运行 `run.sh`，即可完成最终的 SNN 推理测试。




