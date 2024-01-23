from torchgeo.datasets import BigEarthNet, So2Sat
import torch
from torch import Tensor
import numpy as np

class myBigEarthNet(BigEarthNet):
    RGB_INDEX = [3, 2, 1]
    def __init__(self, pad_s2=False, **kwargs):
        if kwargs['bands'] == 'rgb':
            self.rgb=True
            kwargs['bands'] = 's2'
        else:
            self.rgb=False
        super().__init__(**kwargs)
        self.pad_s2 = pad_s2   # pad sentinel-2 images to 13 bands

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data and label at that index
        """
        image = self._load_image(index)
        label = self._load_target(index)

        sample: dict[str, Tensor] = {"image": image, "label": label.float()}

        if self.transforms is not None:
            sample = self.transforms(sample)

        image = sample['image']

        if self.bands == 'all':
            radar = image[:2]
            image = image[2:]
        if self.pad_s2:
            assert image.shape[0] == 12
            img_size = image.shape[1:]
            image = torch.cat((image[:10], torch.zeros((1, *img_size)), image[10:]), dim=0)
        if self.rgb:
            image = image[self.RGB_INDEX]

        sample['image'] = image
        if self.bands == 'all':
            sample['radar'] = radar

        return sample


class mySo2Sat(So2Sat):
    def __init__(self, pad_s2=False, **kwargs):
        super().__init__(**kwargs)
        self.pad_s2 = pad_s2

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data and label at that index
        """
        import h5py

        with h5py.File(self.fn, "r") as f:
            s1 = f["sen1"][index].astype(np.float64)  # convert from <f8 to float64
            s1 = np.take(s1, indices=self.s1_band_indices, axis=2)
            s2 = f["sen2"][index].astype(np.float64)  # convert from <f8 to float64
            s2 = np.take(s2, indices=self.s2_band_indices, axis=2)

            # convert one-hot encoding to int64 then torch int
            label = torch.tensor(f["label"][index].argmax())

            s1 = np.rollaxis(s1, 2, 0)  # convert to CxHxW format
            s2 = np.rollaxis(s2, 2, 0)  # convert to CxHxW format

            s1 = torch.from_numpy(s1)
            s2 = torch.from_numpy(s2)

        if self.pad_s2:
            img_size = s2.shape[1:]
            s2 = torch.cat((torch.zeros((1, *img_size)), s2[:8], torch.zeros((2, *img_size)), s2[8:]), dim=0)

        sample = {"image": torch.cat([s1, s2]).float(), "label": label}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample