from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
sys.path.insert(0,'./')

from datetime import datetime
import struct

#en_gpu=False
en_gpu=True

gpu_number=0
os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu_number)

os.environ['TF_CPP_MIN_LOG_LEVEL']='1'

if en_gpu==False:
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import numpy as np
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from tensorflow.python.profiler import option_builder
builder = option_builder.ProfileOptionBuilder

import train
import test

# models
from models import cnn_mnist
from models import VGG16

# dataset
from datasets import mnist
from datasets import cifar10
from datasets import cifar100

import shutil
import pprint

now = datetime.now()

# ================= [HIJACK FUNCTION START] =================
def hijack_export_weights(model, filename="exported_models/snn_weights_vgg.bin"):
    print(f"\n[HIJACK] Exporting VGG16 weights + INPUT LAYER + SAMPLE IMAGE to {filename}...")
    try:
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
    except:
        pass
    
    n_in_tc = 20.0 # Default fallback
    n_in_td = 0.0
    
    if 'in' in model.list_neuron:
        print("  Found Input Layer: 'in'")
        n_in_tc = model.list_neuron['in'].time_const_fire.numpy()
        n_in_td = model.list_neuron['in'].time_delay_fire.numpy()
    elif 'n_in' in model.list_neuron:
        print("  Found Input Layer: 'n_in'")
        n_in_tc = model.list_neuron['n_in'].time_const_fire.numpy()
        n_in_td = model.list_neuron['n_in'].time_delay_fire.numpy()
    else:
        if hasattr(model, 'n_in'):
             print("  Found Input Layer: model.n_in")
             n_in_tc = model.n_in.time_const_fire.numpy()
             n_in_td = model.n_in.time_delay_fire.numpy()
        else:
             print("  WARNING: Could not find Input Layer params. Using defaults (TC=20, TD=0).")

    print(f"  >>> Input Layer Config: TC={n_in_tc}, TD={n_in_td}")

    # [已修改] 针对 VGG16-CIFAR10 的所有网络层
    layer_names = [
        'conv1', 'conv1_1', 
        'conv2', 'conv2_1', 
        'conv3', 'conv3_1', 'conv3_2', 
        'conv4', 'conv4_1', 'conv4_2', 
        'conv5', 'conv5_1', 'conv5_2', 
        'fc1', 'fc2', 'fc3'
    ]
    
    with open(filename, 'wb') as f:
        f.write(struct.pack('i', len(layer_names)))
        
        for name in layer_names:
            print(f"  Exporting Layer: {name}")
            k_layer = model.list_layer[name]
            n_layer = model.list_neuron[name]
            
            w = k_layer.kernel.numpy()
            b = k_layer.bias.numpy()
            
            # Kernel
            dims = w.shape
            f.write(struct.pack('i', len(dims)))
            for d in dims: f.write(struct.pack('i', d))
            f.write(w.astype(np.float32).tobytes())

            # Bias
            dims = b.shape
            f.write(struct.pack('i', len(dims)))
            for d in dims: f.write(struct.pack('i', d))
            f.write(b.astype(np.float32).tobytes())

            # TC
            tc = n_layer.time_const_fire.numpy()
            if tc.ndim == 0: tc = np.array([tc])
            dims = tc.shape
            f.write(struct.pack('i', len(dims)))
            for d in dims: f.write(struct.pack('i', d))
            f.write(tc.astype(np.float32).tobytes())

            # TD
            td = n_layer.time_delay_fire.numpy()
            if td.ndim == 0: td = np.array([td])
            dims = td.shape
            f.write(struct.pack('i', len(dims)))
            for d in dims: f.write(struct.pack('i', d))
            f.write(td.astype(np.float32).tobytes())

    print("\n[IMPORTANT] Please update your CUDA code (k_encode_image) with:")
    print(f"   float tc = {float(n_in_tc)}f;")
    print(f"   float td = {float(n_in_td)}f;")
    print("[HIJACK] Export Complete. Exiting.\n")
    os._exit(0)
# ================= [HIJACK FUNCTION END] =================

class SafeConfig:
    def __init__(self, original_conf):
        self._conf = original_conf

    def __getattr__(self, name):
        try:
            return getattr(self._conf, name)
        except AttributeError:
            return False

np.set_printoptions(threshold=np.inf, linewidth=1000, precision=4)
pp = pprint.PrettyPrinter().pprint

flags = tf.compat.v1.app.flags
tf.compat.v1.app.flags.DEFINE_string('date','','date')
tf.compat.v1.app.flags.DEFINE_integer('epoch', 300, 'Number os epochs')
tf.compat.v1.app.flags.DEFINE_string('gpu_fraction', '1/3', 'define the gpu fraction used')
tf.compat.v1.app.flags.DEFINE_integer('batch_size', 100, '')
tf.compat.v1.app.flags.DEFINE_string('nn_mode', 'SNN', 'ANN: Analog Neural Network, SNN: Spiking Neural Network')
tf.compat.v1.app.flags.DEFINE_string('output_dir', './tensorboard', 'Directory to write TensorBoard summaries')
tf.compat.v1.app.flags.DEFINE_string('checkpoint_dir', './models_ckpt', 'Directory to save checkpoints')
tf.compat.v1.app.flags.DEFINE_string('checkpoint_load_dir', './models_ckpt', 'Directory to load checkpoints')
tf.compat.v1.app.flags.DEFINE_bool('en_load_model', False, 'Enable to load model')
tf.compat.v1.app.flags.DEFINE_boolean('en_train', False, 'enable training')
tf.compat.v1.app.flags.DEFINE_boolean('use_bias', True, 'use bias')
tf.compat.v1.app.flags.DEFINE_boolean('use_bn', True, 'use batch normalization')
tf.compat.v1.app.flags.DEFINE_float('n_init_vth', 0.7, 'initial value of vth')
tf.compat.v1.app.flags.DEFINE_float('n_in_init_vth', 0.7, 'initial value of vth of n_in')
tf.compat.v1.app.flags.DEFINE_float('n_init_vinit', 0.0, 'initial value of vinit')
tf.compat.v1.app.flags.DEFINE_float('n_init_vrest', 0.0, 'initial value of vrest')
tf.compat.v1.app.flags.DEFINE_float('lr', 0.001, 'learning rate')
tf.compat.v1.app.flags.DEFINE_float('momentum', 0.9, 'momentum')
tf.compat.v1.app.flags.DEFINE_float('lamb',0.0001, 'lambda')
tf.compat.v1.app.flags.DEFINE_float('lr_decay', 0.1, '')
tf.compat.v1.app.flags.DEFINE_integer('lr_decay_step', 50, '')
tf.compat.v1.app.flags.DEFINE_integer('time_step', 10, 'time steps per sample in SNN')
tf.compat.v1.app.flags.DEFINE_integer('idx_test_dataset_s', 0, 'start index of test dataset')
tf.compat.v1.app.flags.DEFINE_integer('num_test_dataset', 10000, 'number of test datset')
tf.compat.v1.app.flags.DEFINE_integer('size_test_batch', 1, 'size of test batch')
tf.compat.v1.app.flags.DEFINE_integer('save_interval', 10, 'save interval of model')
tf.compat.v1.app.flags.DEFINE_bool('en_remove_output_dir', False, 'enable removing output dir')
tf.compat.v1.app.flags.DEFINE_string('regularizer', 'L2', 'L2 or L1 regularizer')
tf.compat.v1.app.flags.DEFINE_string('model_name', 'snn_train_mlp_mnist', 'model name')
tf.compat.v1.app.flags.DEFINE_string('n_type', 'LIF', 'LIF or IF: neuron type')
tf.compat.v1.app.flags.DEFINE_string('dataset', 'MNIST', 'dataset')
tf.compat.v1.app.flags.DEFINE_string('ann_model', 'MLP', 'neural network model')
tf.compat.v1.app.flags.DEFINE_boolean('verbose',False, 'verbose mode')
tf.compat.v1.app.flags.DEFINE_boolean('verbose_visual',False, 'verbose visual mode')
tf.compat.v1.app.flags.DEFINE_integer('time_step_save_interval',10,'snn test save interval')
tf.compat.v1.app.flags.DEFINE_bool('f_fused_bn',False,'f_fused_bn')
tf.compat.v1.app.flags.DEFINE_bool('f_stat_train_mode',False,'f_stat_train_mode')
tf.compat.v1.app.flags.DEFINE_bool('f_vth_conp',False,'f_vth_conp')
tf.compat.v1.app.flags.DEFINE_bool('f_spike_max_pool',False,'f_spike_max_pool')
tf.compat.v1.app.flags.DEFINE_bool('f_w_norm_data',False,'f_w_norm_data')
tf.compat.v1.app.flags.DEFINE_float('p_ws',8,'period of wieghted synapse')
tf.compat.v1.app.flags.DEFINE_integer('num_class',10,'number_of_class (do not touch)')
tf.compat.v1.app.flags.DEFINE_string('input_spike_mode','POISSON','input spike mode - POISSON, WEIGHTED_SPIKE, ROPOSED')
tf.compat.v1.app.flags.DEFINE_string('neural_coding','RATE','neural coding - RATE, WEIGHTED_SPIKE, PROPOSED')
tf.compat.v1.app.flags.DEFINE_bool('f_positive_vmem',False,'positive vmem')
tf.compat.v1.app.flags.DEFINE_bool('f_tot_psp',False,'accumulate total psp')
tf.compat.v1.app.flags.DEFINE_bool('f_refractory',False,'refractory mode')
tf.compat.v1.app.flags.DEFINE_bool('f_write_stat',False,'write stat')
tf.compat.v1.app.flags.DEFINE_string('act_save_mode','channel','activation save mode')
tf.compat.v1.app.flags.DEFINE_bool('f_save_result',True,'save result to xlsx file')
tf.compat.v1.app.flags.DEFINE_integer('input_size', 28, 'input image width / height')
tf.compat.v1.app.flags.DEFINE_string('path_stat','./stat/', 'path stat')
tf.compat.v1.app.flags.DEFINE_string('prefix_stat','act_n_train', 'prefix of stat file name')
tf.compat.v1.app.flags.DEFINE_bool('f_data_std', True, 'data_standardization')
tf.compat.v1.app.flags.DEFINE_string('path_result_root','./result/', 'path result root')
tf.compat.v1.app.flags.DEFINE_integer('tc',10,'time constant for temporal coding')
tf.compat.v1.app.flags.DEFINE_integer('time_window',20,'time window of each layer for temporal coding')
tf.compat.v1.app.flags.DEFINE_float('time_fire_start',20,'time fire start (integration time before starting fire) for temporal coding')
tf.compat.v1.app.flags.DEFINE_float('time_fire_duration',20,'time fire duration for temporal coding')
tf.compat.v1.app.flags.DEFINE_bool('f_record_first_spike_time',False,'flag - recording first spike time of each neuron')
tf.compat.v1.app.flags.DEFINE_bool('f_visual_record_first_spike_time',False,'flag - visual recording first spike time of each neuron')
tf.compat.v1.app.flags.DEFINE_bool('f_train_time_const',False,'flag - enable to train time constant for temporal coding')
tf.compat.v1.app.flags.DEFINE_bool('f_train_time_const_outlier',True,'flag - enable to outlier roubst train time constant for temporal coding')
tf.compat.v1.app.flags.DEFINE_bool('f_load_time_const',False,'flag - load time constant for temporal coding')
tf.compat.v1.app.flags.DEFINE_string('time_const_init_file_name','./temporal_coding/time_const','temporal coding file name - time_const, time_delay`')
tf.compat.v1.app.flags.DEFINE_integer('time_const_num_trained_data',0,'number of trained data - time constant')
tf.compat.v1.app.flags.DEFINE_integer('time_const_save_interval',10000,'save interval - training time constant')
tf.compat.v1.app.flags.DEFINE_integer('epoch_train_time_const',1,'epoch - training time constant')
tf.compat.v1.app.flags.DEFINE_enum('snn_output_type',"VMEM", ["SPIKE", "VMEM", "FIRST_SPIKE_TIME"], "snn output type")
tf.compat.v1.app.flags.DEFINE_bool("en_tensorboard_write", False, "Tensorboard write")

conf = flags.FLAGS

if conf.model_name == 'vgg_cifar_ro_0':
    conf.f_data_std = False

if conf.f_write_stat:
    en_gpu=False

def main(_):
    print('main start')

    if conf.en_remove_output_dir:
        shutil.rmtree(conf.output_dir,ignore_errors=True)

    data_format = 'channels_last'
    (device, data_format) = ('/cpu:0', data_format) if not en_gpu else ('/gpu:0', data_format)

    with tf.device('/gpu:0'):
        dataset_type = {'MNIST': mnist, 'CIFAR-10': cifar10, 'CIFAR-100': cifar100}
        dataset = dataset_type[conf.dataset]
        (train_dataset, val_dataset, test_dataset, num_train_dataset, num_val_dataset, num_test_dataset) = dataset.load(conf)

    model = None
    if conf.ann_model=='CNN':
        if conf.dataset=='MNIST':
            safe_conf = SafeConfig(conf)
            model = cnn_mnist.MNISTModel_CNN(data_format, safe_conf)
    elif conf.ann_model=='VGG16':
        if conf.dataset=='CIFAR-10':
            model = VGG16.CIFARModel_CNN(data_format,conf)
        elif conf.dataset=='CIFAR-100':
            model = VGG16.CIFARModel_CNN(data_format,conf)

    if model is None:
        print('not supported model name: '+conf.ann_model)
        os._exit(0)

    lr=tf.Variable(conf.lr)
    optimizer = tf.keras.optimizers.Adam(lr)

    checkpoint_dir = os.path.join(conf.checkpoint_dir,conf.model_name)
    checkpoint_load_dir = os.path.join(conf.checkpoint_load_dir,conf.model_name)

    global_epoch = tf.Variable(name='global_epoch', initial_value=tf.zeros(shape=[]),dtype=tf.float32,trainable=False)

    with tf.device(device):
        print(' Test Phase >')

        if conf.dataset=='ImageNet':
            images_0 = next(test_dataset.__iter__())[0]
        else:
            images_0 = next(test_dataset.__iter__())[0]
        
        # dummy run
        model(images_0,False)

        load_layer = model.load_layer_ann_checkpoint
        load_model = tf.train.Checkpoint(model=load_layer, optimizer=optimizer, global_epoch=global_epoch)

        status = load_model.restore(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()
        print('load model done')
        model(images_0, False)

        # [HIJACK START]
        if conf.nn_mode == 'SNN':
            print("\n[DEBUG] Verifying Input Spike Times...")
            n_layer = None
            if 'in' in model.list_neuron: n_layer = model.list_neuron['in']
            elif 'n_in' in model.list_neuron: n_layer = model.list_neuron['n_in']
            
            if n_layer:
                tc = n_layer.time_const_fire.numpy()
                td = n_layer.time_delay_fire.numpy()
                print(f"  Input Params from TF: TC={tc}, TD={td}")
                
                # [修改] CIFAR-10 是 (BATCH, 32, 32, 3)
                img_data = images_0.numpy()[0] 
                
                # 选取特征像素 (以R通道[..., 0]为例)
                p_center = img_data[16, 16, 0] 
                p_corner = img_data[0, 0, 0]   
                p_mid    = img_data[16, 8, 0] 
                
                print(f"  Pixel(16,16,R) [Center]: Value={p_center:.4f}")
                print(f"  Pixel(0,0,R)   [Corner]: Value={p_corner:.4f}")
                print(f"  Pixel(16,8,R)  [Mid]   : Value={p_mid:.4f}")

                def calc_spike(p, tc, td):
                    if p < 1e-5: return 9999.0
                    return td - tc * np.log(p + 1e-9)

                t_center = calc_spike(p_center, tc, td)
                t_corner = calc_spike(p_corner, tc, td)
                t_mid    = calc_spike(p_mid, tc, td)
                
                print(f"  [Python Calc] Spike Time (16,16,R): {t_center:.4f}")
                print(f"  [Python Calc] Spike Time (0,0,R)  : {t_corner:.4f}")
                print(f"  [Python Calc] Spike Time (16,8,R) : {t_mid:.4f}")
                
            hijack_export_weights(model)
        # [HIJACK END]

if __name__ == '__main__':
    tf.compat.v1.app.run()
