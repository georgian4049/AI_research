import torch.distributed as dist
import os

def init_distributed_mode(cfg):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        cfg["TRAINER"]["RANK"] = int(os.environ["RANK"])
        cfg["TRAINER"]["WORLD_SIZE"] = int(os.environ["WORLD_SIZE"])
        cfg["TRAINER"]["GPU"] = int(os.environ["LOCAL_RANK"])
    else:
        print("Not using distributed mode")
        cfg["TRAINER"]["DISTRIBUTED"] = False
        return

    cfg["TRAINER"]["DISTRIBUTED"] = True
    torch.cuda.set_device(cfg["TRAINER"]["GPU"])
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://{cfg['TRAINER']['MASTER_ADDR']}:{cfg['TRAINER']['MASTER_PORT']}",
        world_size=cfg["TRAINER"]["WORLD_SIZE"],
        rank=cfg["TRAINER"]["RANK"],
    )
    print(f"Initialized distributed mode on GPU {cfg['TRAINER']['GPU']}")