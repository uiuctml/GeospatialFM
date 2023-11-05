from GeospatialFM.data import *
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from dataclasses import dataclass
from multiprocessing import Value

class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value('i', epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value

@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler = None
    shared_epoch: SharedEpoch = None

    def set_epoch(self, epoch):
        if self.shared_epoch is not None:
            self.shared_epoch.set_value(epoch)
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)

def get_sampler(cfg):
    return None

def get_data(cfg):
    train_ds, val_ds, test_ds = get_datasets(cfg['DATASET'])
    batch_size = cfg['TRAINER']['per_device_train_batch_size']
    num_workers = cfg['TRAINER']['dataloader_num_workers']
    pin_memory = cfg['TRAINER']['dataloader_pin_memory']
    drop_last = cfg['TRAINER']['dataloader_drop_last']
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last)
    train_dl.num_samples = len(train_ds)
    train_dl.num_batches = len(train_dl)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    val_dl.num_samples = len(val_ds)
    val_dl.num_batches = len(val_dl)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_dl.num_samples = len(test_ds)
    test_dl.num_batches = len(test_dl)
    data = dict(
        train=DataInfo(train_dl),
        val=DataInfo(val_dl),
        test=DataInfo(test_dl),
    )
    return data