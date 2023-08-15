#!/usr/bin/env python3
import argparse
import os
import random
from datetime import datetime
os.environ['HABITAT_SIM_LOG'] = 'quiet'
os.environ['MAGNUM_LOG'] = 'quiet'
os.environ['GLOG_minloglevel'] = '2'

import numpy as np
import torch
from habitat import logger
from habitat.config import Config
from habitat_baselines.common.baseline_registry import baseline_registry

from zson.config import get_config
from zson.utils import make_run_dir
from zson.trainer import ZSONTrainer

from habitat_baselines.rl.ddppo.ddp_utils import get_distrib_size

from zson import sensors,action,dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-type",
        choices=["train", "eval"],
        required=True,
        help="run type of the experiment (train or eval)",
    )
    parser.add_argument(
        "--debug",
        default=False,
        action='store_true',
        help="debug using 1 scene"
    )
    parser.add_argument(
        "--exp-config",
        type=str,
        required=True,
        help="path to config yaml containing info about experiment",
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )
    parser.add_argument(
        "--local_rank",
        default=0,
        type=int,
        help="preserved args for torch.distributed.run",
    )
    parser.add_argument(
        "--seed",
        default=-1,
        type=int,
        help="control the random seed",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="path to sotrage the log and checkpoints",
    )
    parser.add_argument(
        "--overwrite",
        default=False,
        action="store_true",
        help="overwrite the exsiting folder",
    )
    parser.add_argument(
        "--note",
        default="",
        help="Add extra note for running file"
    )
    args = parser.parse_args()
    run_exp(**vars(args))


def execute_exp(config: Config, run_type: str, seed: int) -> None:
    r"""This function runs the specified config with the specified runtype
    Args:
    config: Habitat.config
    runtype: str {train or eval}
    """
    # set a random seed (from detectron2)
    if seed == -1:
        seed = (
            os.getpid()
            + int(datetime.now().strftime("%S%f"))
            + int.from_bytes(os.urandom(2), "big")
        )
        logger.info("Using a generated random seed {}".format(seed))
    else:
        logger.info("Using a specified random seed {}".format(seed))
    config.defrost()
    config.RUN_TYPE = run_type
    config.TASK_CONFIG.SEED = seed
    config.freeze()
    random.seed(config.TASK_CONFIG.SEED)
    np.random.seed(config.TASK_CONFIG.SEED)
    torch.manual_seed(config.TASK_CONFIG.SEED)
    if config.FORCE_TORCH_SINGLE_THREADED and torch.cuda.is_available():
        torch.set_num_threads(1)

    trainer_init = baseline_registry.get_trainer(config.TRAINER_NAME)
    assert trainer_init is not None, f"{config.TRAINER_NAME} is not supported"
    trainer: ZSONTrainer = trainer_init(config)

    if run_type == "train":
        trainer.train()
    elif run_type == "eval":
        trainer.eval()


def run_exp(exp_config: str, run_type: str, local_rank:int, seed: int, model_dir: str, opts=None, debug=False, overwrite=False, note=None) -> None:
    r"""Runs experiment given mode and config

    Args:
        exp_config: path to config file.
        run_type: "train" or "eval.
        opts: list of strings of additional config options.

    Returns:
        None.
    """
    config = get_config(exp_config, opts, debug, run_type)
    make_run_dir(config, model_dir, run_type, overwrite, note)
    execute_exp(config, run_type, seed)


if __name__ == "__main__":
    main()
