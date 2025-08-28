import numpy as np
import random
from torch.utils.data import Sampler

class PerTableBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, drop_last=True):
        self.dataset = dataset
        self.limit = dataset.limit
        self.batch_size = batch_size
        self.total_num = len(dataset)

        self.table_start = 0
        self.table_end = len(dataset.entries[dataset.split])
        self.drop_last = drop_last

    def __iter__(self):
        if self.drop_last:
            range_num = self.total_num // self.batch_size
        else:
            range_num = (self.total_num + self.batch_size - 1) // self.batch_size

        table_idx_list = []
        total = sum(self.limit)
        table_ratios = [count / total for count in self.limit]
        for idx, ratio in enumerate(table_ratios):
            count = int(round(ratio * range_num))
            table_idx_list.extend([idx] * count)

        while len(table_idx_list) < range_num:
            table_idx_list.append(random.randint(0, self.table_end - self.table_start - 1))
        table_idx_list = table_idx_list[:range_num]
        random.shuffle(table_idx_list)

        # [0, 1, ..., total_num - 1]
        all_indices = list(range(self.total_num))
        random.shuffle(all_indices)

        for i in range(range_num):
            start = i * self.batch_size
            end = start + self.batch_size
            batch_indices = all_indices[start:end]
            if self.drop_last and len(batch_indices) < self.batch_size:
                continue

            table_idx = table_idx_list[i]
            yield [(idx, table_idx) for idx in batch_indices]

    def __len__(self):
        if self.drop_last:
            return self.total_num // self.batch_size
        else:
            return (self.total_num + self.batch_size - 1) // self.batch_size  # 向上取整
