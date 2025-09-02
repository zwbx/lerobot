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
STEPS=30000
BATCH_SIZE=16
EVAL_FREQ=5000
SAVE_FREQ=5000
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
export MUJOCO_GL=egl

# launch
export WANDB_API_KEY=49ed6b98715d126dd3ff62f759ffd0d9cc7a32b5 
# accelerate launch 
python -m accelerate.commands.launch --num_processes=8 --main_process_port 47571 src/lerobot/scripts/train_accelerate.py \
  --policy.type=$POLICY \
  --dataset.repo_id=$REPO_ID \
  --env.type=pusht \
  --env.task=$TASK \
  --output_dir=$OUTPUT_DIR \
  --steps=$STEPS \
  --batch_size=$BATCH_SIZE \
  --eval_freq=$EVAL_FREQ \
  --save_freq=$SAVE_FREQ \
  --num_workers=$NUM_WORKERS \
  --env.multitask_eval=False \
  --eval.batch_size=1 \
  --eval.n_episodes=4 \
  --wandb.enable=true