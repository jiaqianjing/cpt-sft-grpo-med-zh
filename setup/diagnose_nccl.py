"""Minimal NCCL collective diagnostic.

CALLING SPEC:
    torchrun --standalone --nproc-per-node=8 setup/diagnose_nccl.py

    Inputs: torchrun rank environment and visible CUDA GPUs.
    Output: rank-0 version data and the all-reduced rank sum.
    Side effects: initializes and destroys one NCCL process group.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    value = torch.tensor([float(dist.get_rank())], device="cuda")
    dist.all_reduce(value)
    if dist.get_rank() == 0:
        print(
            f"torch={torch.__version__} cuda={torch.version.cuda} "
            f"nccl={torch.cuda.nccl.version()} sum={value.item()}"
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
