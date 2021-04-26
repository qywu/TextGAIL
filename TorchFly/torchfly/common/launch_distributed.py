import os
import sys
import random
import subprocess
from omegaconf import OmegaConf
from torchfly.flyconfig import GlobalFlyConfig
from typing import Callable
import logging

logger = logging.getLogger(__name__)

# TODO: We only support single machine multi gpu distributed training
#       Also, MASTER_PORT and MASTER_ADDR are fixed for now


def launch_distributed(config_path: str, worker_fn: Callable, *args, **kwargs):
    config_manager = GlobalFlyConfig(config_path=config_path, disable_chdir=True, disable_logging=True)
    config = config_manager.user_config

    num_gpus_per_node = config.training.num_gpus_per_node

    if num_gpus_per_node <= 1:
        GlobalFlyConfig._instances.clear()
        config = GlobalFlyConfig(config_path=config_path).user_config
        worker_fn(*args, **kwargs)
    elif int(os.environ.get("FLY_DISTRIBUTED_INIT", 0)) == 0:
        # Distributed Training
        current_env = os.environ.copy()

        # TODO: the port should be able to be adjusted manually
        current_env["MASTER_ADDR"] = "127.0.0.1"
        current_env["MASTER_PORT"] = f"{29500 + random.randint((1), 1000)}"

        current_env["WORLD_SIZE"] = str(num_gpus_per_node)
        current_env["FLY_DISTRIBUTED_INIT"] = str(1)

        processes = []

        if 'OMP_NUM_THREADS' not in os.environ:
            current_env["OMP_NUM_THREADS"] = str(1)
            print(
                "*****************************************\n"
                "Setting OMP_NUM_THREADS environment variable for each process "
                "to be {} in default, to avoid your system being overloaded, "
                "please further tune the variable for optimal performance in "
                "your application as needed. \n"
                "*****************************************".format(current_env["OMP_NUM_THREADS"])
            )

        for local_rank in range(0, num_gpus_per_node):
            dist_rank = local_rank
            current_env["RANK"] = str(dist_rank)
            current_env["LOCAL_RANK"] = str(local_rank)

            cmd = [
                sys.executable,
            ] + sys.argv.copy()

            process = subprocess.Popen(cmd, env=current_env)
            processes.append(process)

        for process in processes:
            process.wait()
            if process.returncode != 0:
                raise subprocess.CalledProcessError(returncode=process.returncode, cmd=cmd)
    else:
        GlobalFlyConfig._instances.clear()
        config = GlobalFlyConfig(config_path=config_path).user_config
        # report status
        print(
            "*****************************************\n"
            f"RANK {os.environ['RANK']} has been initialized!\n"
            "*****************************************"
        )
        # start worker
        worker_fn(*args, **kwargs)
