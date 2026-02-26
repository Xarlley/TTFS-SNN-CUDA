from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import os
import tarfile
import pickle

def load_local_cifar10(filepath):
    if not os.path.exists(filepath):
        # 如果找不到本地文件，尝试回退到默认位置或报错
        print(f"Error: Local dataset file not found at {filepath}")
        # 这里你可以选择抛出异常，或者尝试去默认路径找
        raise FileNotFoundError(f"Please ensure {filepath} exists.")

    print(f"Loading CIFAR-10 from local file: {filepath}")
    
    train_images = []
    train_labels = []
    test_images = None
    test_labels = None

    with tarfile.open(filepath, 'r:gz') as tar:
        # CIFAR-10 python版压缩包内的文件夹名称通常是 'cifar-10-batches-py'
        # 训练集: data_batch_1 ~ 5
        for i in range(1, 6):
            member_name = f'cifar-10-batches-py/data_batch_{i}'
            try:
                f = tar.extractfile(member_name)
                if f:
                    entry = pickle.load(f, encoding='bytes')
                    train_images.append(entry[b'data'])
                    train_labels.extend(entry[b'labels'])
            except KeyError:
                print(f"Warning: {member_name} not found in tar file.")

        # 测试集: test_batch
        try:
            f = tar.extractfile('cifar-10-batches-py/test_batch')
            if f:
                entry = pickle.load(f, encoding='bytes')
                test_images = entry[b'data']
                test_labels = entry[b'labels']
        except KeyError:
            print("Warning: test_batch not found in tar file.")

    # 拼接并转换格式
    train_images = np.concatenate(train_images)
    train_labels = np.array(train_labels)
    test_images = np.array(test_images)
    test_labels = np.array(test_labels)

    # 原始数据格式是 (N, 3072)，即 (N, 3*32*32)，通道顺序为 RGB (Channel First)
    # 转换为 Keras 默认的 (N, 32, 32, 3) (Height, Width, Channel)
    train_images = train_images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    test_images = test_images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

    # 标签形状转换为 (N, 1) 以匹配 Keras 输出
    train_labels = train_labels.reshape(-1, 1)
    test_labels = test_labels.reshape(-1, 1)

    return (train_images, train_labels), (test_images, test_labels)

def load(conf):
    print("load CIFAR10 dataset")
    #(img_train,label_train), (img_test, label_test) = tf.keras.datasets.cifar10.load_data()
    local_data_path = 'dataset_downloaded/cifar-10-python.tar.gz'
    if os.path.exists(local_data_path):
        (img_train, label_train), (img_test, label_test) = load_local_cifar10(local_data_path)
    else:
        print(f"Warning: Local file {local_data_path} not found. Falling back to Keras automatic download...")
        (img_train,label_train), (img_test, label_test) = tf.keras.datasets.cifar10.load_data()

    img_train = img_train.astype(float)
    img_test = img_test.astype(float)

    img_train = img_train / 255.0
    img_test = img_test / 255.0

    if conf.f_data_std:
        cifar_mean = [0.485, 0.456, 0.406]
        cifar_std = [0.229, 0.224, 0.225]

        img_train[:,:,:,0] = (img_train[:,:,:,0]-cifar_mean[0])/cifar_std[0]
        img_train[:,:,:,1] = (img_train[:,:,:,1]-cifar_mean[1])/cifar_std[1]
        img_train[:,:,:,2] = (img_train[:,:,:,2]-cifar_mean[2])/cifar_std[2]

        img_test[:,:,:,0] = (img_test[:,:,:,0]-cifar_mean[0])/cifar_std[0]
        img_test[:,:,:,1] = (img_test[:,:,:,1]-cifar_mean[1])/cifar_std[1]
        img_test[:,:,:,2] = (img_test[:,:,:,2]-cifar_mean[2])/cifar_std[2]

    num_train_dataset = 45000
    num_val_dataset = 5000
    num_test_dataset = conf.num_test_dataset

    img_val = img_train[num_train_dataset:,:,:,:]
    label_val = label_train[num_train_dataset:,:]

    img_train = img_train[:num_train_dataset,:,:,:]
    label_train = label_train[:num_train_dataset,:]


    label_test=label_test[conf.idx_test_dataset_s:conf.idx_test_dataset_s+conf.num_test_dataset]
    img_test=img_test[conf.idx_test_dataset_s:conf.idx_test_dataset_s+conf.num_test_dataset,:,:,:]

    train_dataset = tf.data.Dataset.from_tensor_slices((img_train,tf.squeeze(tf.one_hot(label_train,10))))

    val_dataset = tf.data.Dataset.from_tensor_slices((img_val,tf.squeeze(tf.one_hot(label_val,10))))
    val_dataset = val_dataset.map(preprocess_test, num_parallel_calls=2)
    val_dataset = val_dataset.prefetch(10*conf.batch_size)

    test_dataset = tf.data.Dataset.from_tensor_slices((img_test,tf.squeeze(tf.one_hot(label_test,10))))
    test_dataset = test_dataset.map(preprocess_test, num_parallel_calls=2)
    test_dataset = test_dataset.prefetch(10*conf.batch_size)


    # for stat of train dataset
    if conf.f_stat_train_mode:
        test_dataset = tf.data.Dataset.from_tensor_slices((img_train,tf.squeeze(tf.one_hot(label_train,10))))
        test_dataset = test_dataset.map(preprocess_test, num_parallel_calls=2)

    if conf.f_train_time_const:
        label_train=label_train[conf.idx_test_dataset_s:conf.idx_test_dataset_s+conf.num_test_dataset]
        img_train=img_train[conf.idx_test_dataset_s:conf.idx_test_dataset_s+conf.num_test_dataset,:,:,:]

        test_dataset = tf.data.Dataset.from_tensor_slices((img_train,tf.squeeze(tf.one_hot(label_train,10))))
        test_dataset = test_dataset.map(preprocess_test, num_parallel_calls=2)


    print(train_dataset)

    val_dataset = val_dataset.batch(conf.batch_size)
    test_dataset = test_dataset.batch(conf.batch_size)


    return train_dataset, val_dataset, test_dataset, num_train_dataset, num_val_dataset, num_test_dataset


def preprocess_train(img, label):
    img_p = tf.image.convert_image_dtype(img,tf.float32)
    img_p = tf.image.resize_with_crop_or_pad(img_p,36,36)
    img_p = tf.image.random_crop(img_p,[32,32,3])
    img_p = tf.image.random_flip_left_right(img_p)
    img_p.set_shape([32,32,3])

    return img_p, label

def preprocess_test(img, label):
    img_p = tf.image.convert_image_dtype(img,tf.float32)
    img_p.set_shape([32,32,3])

    return img_p, label

def train_data_augmentation(train_dataset, batch_size):
    train_dataset_p = train_dataset.shuffle(10000)
    train_dataset_p = train_dataset_p.prefetch(2*batch_size)
    train_dataset_p = train_dataset_p.map(preprocess_train, num_parallel_calls=8)
    train_dataset_p = train_dataset_p.batch(batch_size)

    return train_dataset_p
