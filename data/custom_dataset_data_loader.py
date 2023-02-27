import torch
import torch.utils.data
from torch.utils.data.distributed import DistributedSampler
from data.base_dataset_loader import BaseDataLoader
from data.long_video_dataset import LongVideoDataset
from data.text_to_video_dataset import text_to_video_dataset

def CreateDataset(opt):
    dataset = None

    # from data.long_video_dataset import LongVideoDataset
    # dataset = LongVideoDataset()
    
    # dataset = LongVideoDataset()
    dataset = text_to_video_dataset()

    print_args = not torch.distributed.is_initialized()
    if not print_args:
        if torch.distributed.get_rank() == 0:
            print_args = True
    if print_args:
        print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(opt)
    return dataset


class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        self.dataset = CreateDataset(opt)
        if opt.isTrain:
            self.sampler = DistributedSampler(self.dataset, shuffle=not opt.serial_batches)
            shuffle = False
        else:
            self.sampler = None
            shuffle = not opt.serial_batches
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize//len(opt.gpu_ids),
            shuffle=shuffle,
            num_workers=int(opt.nThreads),
            pin_memory=True,
            sampler=self.sampler
        )

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)
