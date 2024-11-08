import torch
from torch.utils.data import DataLoader, WeightedRandomSampler, RandomSampler, BatchSampler
import numpy as np

def proportional_collate_fn(batch, dataset, proportion):
    
    year_data = dataset.datasets[dataset.current_time][dataset.mode]
    images = year_data['images']
    labels = year_data['labels']
    
    num_samples = len(images)
    sample_size = int(num_samples * proportion)
    
    sampled_indices = np.random.choice(num_samples, sample_size, replace=False)
    
    sampled_images = images[sampled_indices]
    sampled_labels = labels[sampled_indices]
    image_tensors = torch.FloatTensor(sampled_images).permute(0, 3, 1, 2)
    label_tensors = torch.LongTensor(sampled_labels)
    
    return image_tensors, label_tensors

class _InfiniteSampler(torch.utils.data.Sampler):
    """Wraps another Sampler to yield an infinite stream."""

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            for batch in self.sampler:
                yield batch


class InfiniteDataLoader:
    def __init__(self, dataset, weights, batch_size, num_workers, collate_fn=None):
        super().__init__()

        if weights is not None:
            sampler = torch.utils.data.WeightedRandomSampler(weights,
                                                             replacement=True,
                                                             num_samples=batch_size)
        else:
            sampler = torch.utils.data.RandomSampler(dataset,
                                                     replacement=True)

        batch_sampler = torch.utils.data.BatchSampler(
            sampler,
            batch_size=min(batch_size, len(dataset)),
            drop_last=True)

        self._infinite_iterator = iter(DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=_InfiniteSampler(batch_sampler),
            collate_fn=collate_fn
        ))

    def __iter__(self):
        while True:
            yield next(self._infinite_iterator)

    def __len__(self):
        raise ValueError



class ProportionalDataLoader:
    def __init__(self, dataset, weights, batch_size, num_workers, collate_fn=None):
        super().__init__()

        if weights is not None:
            sampler = torch.utils.data.WeightedRandomSampler(weights,
                                                             replacement=True,
                                                             num_samples=batch_size)
        else:
            sampler = torch.utils.data.RandomSampler(dataset,
                                                     replacement=True)

        batch_sampler = torch.utils.data.BatchSampler(
            sampler,
            batch_size=min(batch_size, len(dataset)),
            drop_last=True)

        self._infinite_iterator = iter(DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=_InfiniteSampler(batch_sampler),
            collate_fn=lambda batch: proportional_collate_fn(batch, dataset, proportion)
        ))

    def __iter__(self):
        while True:
            yield next(self._infinite_iterator)

    def __len__(self):
        raise ValueError


class FastDataLoader:
    """DataLoader wrapper with slightly improved speed by not respawning worker
    processes at every epoch."""

    def __init__(self, dataset, batch_size, num_workers, collate_fn=None):
        super().__init__()

        batch_sampler = torch.utils.data.BatchSampler(
            torch.utils.data.RandomSampler(dataset, replacement=False),
            batch_size=min(batch_size, len(dataset)),
            drop_last=False
        )

        self._infinite_iterator = iter(torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=_InfiniteSampler(batch_sampler),
            collate_fn=collate_fn
        ))

        self._length = len(batch_sampler)

    def __iter__(self):
        for _ in range(len(self)):
            yield next(self._infinite_iterator)

    def __len__(self):
        return self._length
