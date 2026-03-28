#!/bin/bash
#SBATCH -N 1
#SBATCH -n 20
##SBATCH -c 5
#SBATCH -M swarm
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --no-requeue
module load nvidia/cuda/12.9
source /project/wanglei3/anaconda3/etc/profile.d/conda.sh
conda activate snn_coding 
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export LD_LIBRARY_PATH=/home/wanglei3/project/anaconda3/envs/spikformer-new/lib:$LD_LIBRARY_PATH

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export CUDA_HOME=$CONDA_PREFIX
#echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
#echo "SLURM_PROCID: $SLURM_PROCID"
#echo "SLURM_LOCALID: $SLURM_LOCALID"
nvidia-smi  # For reference

python main_inject_getweight_vgg.py \
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
