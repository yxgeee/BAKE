
from collections import defaultdict
import numpy as np
import math
import copy

import torch
from torch.utils.data.sampler import Sampler
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

__all__ = ["DistributedClassSampler"]


class DistributedClassSampler(Sampler):
    def __init__(self, dataset, num_instances, seed=0):
        if not dist.is_available():
            raise RuntimeError("Requires distributed package to be available")
        num_replicas = dist.get_world_size()
        rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.seed = seed
        self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas

        self.num_instances = num_instances
        self.pid_index = defaultdict(list)
        for idx, item in enumerate(self.dataset):
            self.pid_index[item['class']].append(idx)


    def __len__(self):
        return self.num_samples * self.num_instances

    def __iter__(self):
        # deterministically shuffle based on epoch and seed
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        ret = []
        for i in indices:
            ret.append(i)
            select_indexes = [j for j in self.pid_index[self.dataset[i]['class']] if j!=i]
            if (not select_indexes):
                continue
            if (len(select_indexes)>=self.num_instances-1):
                ind_indexes = np.random.choice(select_indexes, size=self.num_instances-1, replace=False)
            else:
                ind_indexes = np.random.choice(select_indexes, size=self.num_instances-1, replace=True)
            ret.extend(ind_indexes)
        return iter(ret)

    def set_epoch(self, epoch):
        self.epoch = epoch
