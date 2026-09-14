import torch


def split_battles(battles: int, worker_count: int) -> list[int]:
    base = battles // worker_count
    remainder = battles % worker_count

    return [
        base + (1 if index < remainder else 0) for index in range(worker_count)
    ]


def worker_initializer(torch_threads: int) -> None:
    # Every rollout process already gives us CPU parallelism.
    torch.set_num_threads(torch_threads)
    torch.set_num_interop_threads(1)
