#!/bin/bash

# activate conda environment
source /home/tiger/miniconda/etc/profile.d/conda.sh
conda activate lerobot

# config
# REPO_ID=physical-intelligence/libero
REPO_ID=lerobot/pusht
TASK=PushT-v0
OUTPUT_DIR=./outputs/

# clean previous run
rm -rf $OUTPUT_DIR

# training params
STEPS=10000
BATCH_SIZE=16
EVAL_FREQ=5000
SAVE_FREQ=2000
NUM_WORKERS=8

# model params
POLICY=pi0
# USE_AMP=true
OPTIMIZER_LR=1e-4
# PEFT_METHOD=lora
# LOAD_VLM_WEIGHTS=true
# VLM_REPO_ID=None
# MAX_ACTION_DIM=32
# MAX_STATE_DIM=32

# dataset/image params
# USE_IMAGENET_STATS=false
# ENABLE_IMG_TRANSFORM=true
# MAX_NUM_IMAGES=2
# MAX_IMAGE_DIM=1024
unset LEROBOT_HOME
unset HF_LEROBOT_HOME


# launch
export WANDB_API_KEY=49ed6b98715d126dd3ff62f759ffd0d9cc7a32b5 
# For distributed training debugging and stability
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=39871 # Chosen high port to avoid EADDRINUSE
export NCCL_SOCKET_IFNAME=lo # Use loopback interface for single-node
export NCCL_SOCKET_FAMILY=AF_INET # Force IPv4 to avoid lo IPv6 mismatch
export NCCL_IB_DISABLE=1 # Disable InfiniBand if not available or configured
export NCCL_P2P_DISABLE=1 # Can be removed if the above works, helps isolating issues
export NCCL_NET=Socket # Force NCCL to use TCP sockets backend
export NCCL_COLLNET_ENABLE=0
export NCCL_CROSS_NIC=0
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_SHOW_CPP_STACKTRACES=1
export GLOO_SOCKET_IFNAME=lo
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7


export TOKENIZERS_PARALLELISM=false
# accelerate launch 
# python src/lerobot/scripts/train.py \
# Prefer torchrun to control rendezvous/master port precisely
torchrun --standalone --nnodes=1 --nproc_per_node=8 --master_addr ${MASTER_ADDR} --master_port ${MASTER_PORT} src/lerobot/scripts/train_accelerate.py \
  --policy.repo_id=$POLICY \
  --policy.path=lerobot/$POLICY \
  --dataset.repo_id=$REPO_ID \
  --env.type=pusht \
  --env.task=$TASK \
  --output_dir=$OUTPUT_DIR \
  --steps=$STEPS \
  --batch_size=$BATCH_SIZE \
  --eval_freq=0 \
  --save_freq=$SAVE_FREQ \
  --num_workers=$NUM_WORKERS \
  --wandb.enable=true