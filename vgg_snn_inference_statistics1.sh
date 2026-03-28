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

nvcc snn_inference_vgg_statistics.cu -o snn_inference_vgg_statistics -O3
./snn_inference_vgg_statistics
