import torch
from accelerate.logging import get_logger
import random

logger = get_logger(__name__, log_level="INFO")
import os

class CombinedDataLoader:
    def __init__(self, dataloaders_list, ratios=None, total_batches=None, seed=3407):
        if not dataloaders_list:
            raise ValueError("dataloaders_list cannot be empty.")
        self.dataloaders = dataloaders_list

        if ratios is None:
            ratios = [1] * len(dataloaders_list)
        assert len(ratios) == len(
            dataloaders_list
        ), "ratios must match dataloader count"
        self.ratios = ratios

        if total_batches is None:
            self._len = sum(len(dl) for dl in self.dataloaders)
        else:
            self._len = total_batches

        if self._len == 0:
            logger.warning(
                "CombinedDataLoader has a total length of 0. All provided dataloaders might be empty."
            )

        # 采样概率
        total = sum(ratios)
        self.probs = [r / total for r in ratios]
        self.seed = seed

    def __len__(self):
        return self._len

    def __iter__(self):
        random.seed(self.seed)
        print("seed:", self.seed)
        iters = [iter(dl) for dl in self.dataloaders]
        num_dataloaders = len(self.dataloaders)
        yielded = 0

        while yielded < self._len:
            # 按概率随机选择 dataloader
            dl_idx = random.choices(range(num_dataloaders), weights=self.probs, k=1)[0]
            try:
                batch = next(iters[dl_idx])
            except StopIteration:
                # 该 dataloader 到头，重新创建一个迭代器
                iters[dl_idx] = iter(self.dataloaders[dl_idx])
                batch = next(iters[dl_idx])
                logger.info(
                    "Resetting iterator."
                )
            except Exception as e:
                logger.error(
                    f"Error fetching from DataLoader index {dl_idx}: {e}. Skipping this batch."
                )
                yielded += 1
                continue

            yield batch
            yielded += 1

