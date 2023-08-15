#!/bin/bash

export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet
export CUDA_VISIBLE_DEVICES=0

epi_num=$1
/mnt/cephfs/home/zhihongyan/anaconda3/envs/zson_vfm/bin/python run.py  --overwrite --exp-config configs/experiments/eval_vln.yaml  --run-type eval --model-dir results/vln_baselines/NavGPT/${epi_num}  EVAL_CKPT_PATH_DIR "data/checkpoints/zson_conf_A.pth"  EVAL.SPLIT "val_unseen" NUM_ENVIRONMENTS 1 TASK_CONFIG.DATASET.EPI_NUM ${epi_num}
