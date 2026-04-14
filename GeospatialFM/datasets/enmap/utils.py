from .enmap import SpectralEarthDataset
from .enmap_bdforet import EnMAPBDForetDataset
from .enmap_corine import EnMAPCorineDataset

ENMAP_DATASET = {
    "enmap_bdforet": EnMAPBDForetDataset,
    "enmap_corine": EnMAPCorineDataset,
}

def get_enmap_metadata():
    return SpectralEarthDataset.metadata

def get_enmap_downstream_metadata(dataset_name, dataset_version=None):
    return ENMAP_DATASET[dataset_name].metadata

def get_enmap_downstream_dataset(args, train_transform, eval_transform):
    dataset_dict = {}
    splits = ["train", "val", "test"]
    for split in splits:
        transform = train_transform if split == "train" else eval_transform
        dataset_dict[split] = ENMAP_DATASET[args.dataset_name](
            root=args.data_dir,
            split=split,
            transform=transform
        )

    return dataset_dict