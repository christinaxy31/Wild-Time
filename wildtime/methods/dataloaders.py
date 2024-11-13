import torch
from torch.utils.data import DataLoader, WeightedRandomSampler, RandomSampler, BatchSampler, Subset, Dataset, ConcatDataset
import numpy as np
from torch.utils.data import  Subset





'''
def proportional_collate_fn(batch, dataset, proportion):
    
    year_data = dataset.datasets[dataset.current_time][dataset.mode]
    images = year_data['images']
    labels = year_data['labels']
    
    num_samples = len(images)
    sample_size = int(num_samples * proportion)
    
    sampled_indices = np.random.choice(num_samples, sample_size, replace=False)
    
    sampled_images = images[sampled_indices]
    sampled_labels = labels[sampled_indices]
    image_tensors = torch.FloatTensor(sampled_images).permute(0,3,1,2)
    label_tensors = torch.LongTensor(sampled_labels)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    return image_tensors.to(device), label_tensors.to(device)

'''

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
    def __init__(self, dataset, weights, proportion, batch_size, num_workers, collate_fn=None):
        super().__init__()

        # Step 1: Select a subset of the dataset based on the specified proportion
        subset_size = int(len(dataset) * proportion)
        indices = torch.randperm(len(dataset))[:subset_size]  # Randomly select a proportion of the dataset
        self.subset = Subset(dataset, indices)  # Create a subset with the specified proportion of data

        # Step 2: Set up sampler based on weights or random sampling
        if weights is not None:
            # Apply weights only to the subset
            subset_weights = [weights[i] for i in indices]
            sampler = torch.utils.data.WeightedRandomSampler(subset_weights, replacement=True, num_samples=batch_size)
        else:
            sampler = torch.utils.data.RandomSampler(self.subset, replacement=True)

        # Step 3: Set up batch sampler
        batch_sampler = torch.utils.data.BatchSampler(
            sampler,
            batch_size=min(batch_size, len(self.subset)),
            drop_last=True
        )

        # Step 4: Initialize the infinite iterator
        self._infinite_iterator = iter(DataLoader(
            self.subset,
            num_workers=num_workers,
            batch_sampler=_InfiniteSampler(batch_sampler),
            collate_fn=collate_fn
        ))

    def __iter__(self):
        while True:
            yield next(self._infinite_iterator)

    def __len__(self):
        raise ValueError("Infinite data loader does not have a defined length.")




class CombinedInfiniteDataLoader:
    def __init__(self, dataset, split_year=1970, proportion=0.5, weights=None, batch_size=32, num_workers=0, collate_fn=None):
        self.split_year = split_year
        self.proportion = proportion
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.collate_fn = collate_fn
        
        # Step 1: Get indices for data before and after 1970
        for i, data in enumerate(dataset[:10]):
            print(data)
        pre_1970_indices = [i for i, data in enumerate(dataset) if data[0] < split_year]
        post_1970_indices = [i for i, data in enumerate(dataset) if data[0] >= split_year]
        
        # Step 2: Create a subset containing all data before 1970
        self.pre_1970_subset = Subset(dataset, pre_1970_indices)
        self.pre_1970_loader = InfiniteDataLoader(
            dataset=self.pre_1970_subset, 
            weights=weights, 
            batch_size=batch_size // 2,  # Half batch size for each loader
            num_workers=num_workers, 
            collate_fn=collate_fn
        )
        
        # Step 3: Create a subset of data after 1970, sampled by proportion
        subset_size = int(len(post_1970_indices) * proportion)
        sampled_post_1970_indices = torch.randperm(len(post_1970_indices))[:subset_size]
        self.post_1970_subset = Subset(dataset, [post_1970_indices[i] for i in sampled_post_1970_indices])
        self.post_1970_loader = ProportionalDataLoader(
            dataset=self.post_1970_subset, 
            weights=weights, 
            proportion=1.0,  # Already sampled a subset with specified proportion
            batch_size=batch_size // 2,  # Half batch size for each loader
            num_workers=num_workers, 
            collate_fn=collate_fn
        )

        # Initialize the infinite iterator for combined batches
        self._infinite_iterator = self._create_infinite_iterator()

    def _create_infinite_iterator(self):
        # Create iterators for each loader
        pre_1970_iter = iter(self.pre_1970_loader)
        post_1970_iter = iter(self.post_1970_loader)
        
        while True:
            # Get one batch from each loader
            pre_batch = next(pre_1970_iter)
            post_batch = next(post_1970_iter)
            
            # Combine the batches by concatenating them along the batch dimension (dim=0)
            combined_batch = {
                'images': torch.cat([pre_batch['images'], post_batch['images']], dim=0),
                'labels': torch.cat([pre_batch['labels'], post_batch['labels']], dim=0)
            }
            yield combined_batch

    def __iter__(self):
        return self._infinite_iterator

    def __len__(self):
        raise ValueError("CombinedInfiniteDataLoader does not have a defined length.")


'''
class ProportionalDataLoader:
    def __init__(self, dataset, weights, proportion, batch_size, num_workers, collate_fn=None):
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
'''

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
