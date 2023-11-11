#!/bin/bash

export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet
export CUDA_VISIBLE_DEVICES=0
epi_num=$1
log_dir=results/vln_baselines/NavGPT/${epi_num}_test_angel
export BASE_LOG_DIR=${log_dir}
export MODE=oracle
# export https_proxy=http://127.0.0.1:7895
# export http_proxy=http://127.0.0.1:7895

python run.py  --overwrite --exp-config configs/experiments/eval_vln.yaml  --run-type eval --model-dir ${log_dir}  EVAL_CKPT_PATH_DIR "data/checkpoints/zson_conf_A.pth"  EVAL.SPLIT "val_unseen" NUM_ENVIRONMENTS 1 TASK_CONFIG.DATASET.EPI_NUM ${epi_num}
