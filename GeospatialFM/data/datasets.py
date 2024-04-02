from torchgeo.datasets import BigEarthNet, So2Sat, OSCD, SSL4EOS12
import torch
from torch import Tensor
import numpy as np
import rasterio # CHANGE
from itertools import product # CHANGE
from PIL import Image # CHANGE
import os # CHANGE
import glob # CHANGE
from typing import Optional, Callable # CHANGE
import random # CHANGE

class myBigEarthNet(BigEarthNet):
    RGB_INDEX = [3, 2, 1]
    def __init__(self, pad_s2=False, **kwargs):
        if kwargs['bands'] == 'rgb':
            self.rgb=True
            kwargs['bands'] = 's2'
        else:
            self.rgb=False
        super().__init__(**kwargs, download=True)
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

        if self.bands == 'all':
            sample['radar'] = sample['image'][:2]
            sample['image'] = sample['image'][2:]

        # if self.rgb:
        #     image = image[self.RGB_INDEX]
        if self.transforms is not None:
            sample = self.transforms(sample)

        image = sample['image']
        if self.pad_s2:
            assert image.shape[0] == 12
            img_size = image.shape[1:]
            image = torch.cat((image[:10], torch.zeros((1, *img_size)), image[10:]), dim=0)
        sample['image'] = image

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

        sample = {'label': label}
        if s1.shape[0] > 0:
            sample['radar'] = s1
        if s2.shape[0] > 0:
            sample['image'] = s2

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

def sort_sentinel2_bands(x: str) -> str:
    """Sort Sentinel-2 band files in the correct order."""
    x = os.path.basename(x).split("_")[-1]
    x = os.path.splitext(x)[0]
    if x == "B8A":
        x = "B08A"
    return x

class myOSCD(OSCD):
    def __init__(self, patch_size, overlap=0.5, **kwargs):
        self.patch_size = patch_size
        self.overlap = overlap
        super().__init__(**kwargs)


    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data and label at that index
        """
        files = self.files[index]
        image1 = self._load_image(files["images1"], files["limit"])
        image2 = self._load_image(files["images2"], files["limit"])
        mask = self._load_target(str(files["mask"]), files["limit"])

        sample = {"image1": image1, "image2": image2, "mask": mask.unsqueeze(0)}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample


    def __len__(self) -> int:
        """Return the number of data points in the dataset.

        Returns:
            length of the dataset
        """
        return len(self.files)


    def _load_files(self):
        regions = []
        labels_root = os.path.join(
            self.root,
            f"Onera Satellite Change Detection dataset - {self.split.capitalize()} "
            + "Labels",
        )
        images_root = os.path.join(
            self.root, "Onera Satellite Change Detection dataset - Images"
        )
        folders = glob.glob(os.path.join(labels_root, "*/"))
        for folder in folders:
            region = folder.split(os.sep)[-2]
            mask = os.path.join(labels_root, region, "cm", "cm.png")

            def get_image_paths(ind: int) -> list[str]:
                return sorted(
                    glob.glob(
                        os.path.join(images_root, region, f"imgs_{ind}_rect", "*.tif")
                    ),
                    key=sort_sentinel2_bands,
                )

            images1, images2 = get_image_paths(1), get_image_paths(2)
            if self.bands == "rgb":
                images1, images2 = images1[1:4][::-1], images2[1:4][::-1]

            with open(os.path.join(images_root, region, "dates.txt")) as f:
                dates = tuple(
                    line.split()[-1] for line in f.read().strip().splitlines()
                )

            img = rasterio.open(images1[0]) # for each region, the width and height are the same across img1 and img2 and across all bands
            img_width, img_height = img.width, img.height
            img_limits = product(range(0, img_height, int(self.patch_size*self.overlap)), range(0, img_width, int(self.patch_size*self.overlap)))

            for l in img_limits:
                if l[0] + self.patch_size < img_height and l[1] + self.patch_size < img_width:
                    regions.append(
                        dict(
                            region=region,
                            images1=images1, # list of paths to 13 band's tif for img1, so len(images1)=13
                            images2=images2, # list of paths to 13 band's tif for img1, so len(images2)=13
                            mask=mask, # path to the label
                            dates=dates,
                            limit=(l[0], l[1], l[0] + self.patch_size, l[1] + self.patch_size) # crop coordinate: top left, bot right
                        )
                    )
        return regions

    def _load_image(self, paths, limit) -> Tensor:
        """Load a single image.

        Args:
            path: path to the image
            limit: the crop coordinate

        Returns:
            the image
        """
        images: list["np.typing.NDArray[np.int_]"] = []
        for path in paths:
            with Image.open(path) as img:
                # crop the image
                img_array = np.array(img)
                img_cropped = img_array[limit[0]:limit[2], limit[1]:limit[3]]
                images.append(img_cropped)
        array: "np.typing.NDArray[np.int_]" = np.stack(images, axis=0).astype(np.int_)
        tensor = torch.from_numpy(array).float()
        return tensor

    def _load_target(self, path: str, limit) -> Tensor:
        """Load the target mask for a single image.

        Args:
            path: path to the image

        Returns:
            the target mask
        """
        filename = os.path.join(path)
        with Image.open(filename) as img:
            array: "np.typing.NDArray[np.int_]" = np.array(img.convert("L"))
            array_cropped = array[limit[0]:limit[2], limit[1]:limit[3]]
            tensor = torch.from_numpy(array_cropped)
            tensor = torch.clamp(tensor, min=0, max=1)
            tensor = tensor.to(torch.long)
            return tensor

class mySSL4EO(SSL4EOS12):
    def __init__(self, 
                 root: str = "/data",
                 optical_split: str = "s2c",
                 radar_split: str = "s1",
                 transforms: Optional[Callable[[dict[str, Tensor]], dict[str, Tensor]]] = None,
                 seasons: int = 1,
                 checksum: bool = False,
                 **kwargs
    ) -> None:
        self.root = root
        self.optical_split = optical_split
        self.radar_split = radar_split
        self.transforms = transforms
        self.seasons = seasons
        assert self.seasons == 1, "Currently only support 1 season"
        self.checksum = checksum

        self.optical_bands = self.metadata[self.optical_split]["bands"]
        self.radar_bands = self.metadata[self.radar_split]["bands"]

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        file_idx = int(sorted(os.listdir(os.path.join(self.root, self.optical_split)))[index])        
        optical_root = os.path.join(self.root, self.optical_split, f"{file_idx:07}")
        radar_root = os.path.join(self.root, self.radar_split, f"{file_idx:07}")
        optical_subdirs = os.listdir(optical_root)
        radar_subdirs = os.listdir(radar_root)
        optical_time_stamp = [int(subdir.split("_")[0].split('T')[0]) for subdir in optical_subdirs]
        radar_time_stamp = [int(subdir.split("_")[4].split('T')[0]) for subdir in radar_subdirs]

        # sort the subdirs by time stamp
        optical_subdirs = [x for _, x in sorted(zip(optical_time_stamp, optical_subdirs))]
        radar_subdirs = [x for _, x in sorted(zip(radar_time_stamp, radar_subdirs))]

        # subsampe the same season from the optical and radar data
        sub_index = random.randint(0, len(optical_subdirs)-1)
        optical_subdirs = optical_subdirs[sub_index]
        radar_subdirs = radar_subdirs[sub_index]
        
        # subdirs = random.sample(subdirs, self.seasons)

        images = []
        optical_directory = os.path.join(optical_root, optical_subdirs)
        for band in self.optical_bands:
            filename = os.path.join(optical_directory, f"{band}.tif")
            with rasterio.open(filename) as f:
                image = f.read(out_shape=(1, self.size, self.size))
                images.append(torch.from_numpy(image.astype(np.float32)))

        radars = []
        radar_directory = os.path.join(radar_root, radar_subdirs)
        for band in self.radar_bands:
            filename = os.path.join(radar_directory, f"{band}.tif")
            with rasterio.open(filename) as f:
                radar = f.read(out_shape=(1, self.size, self.size))
                radars.append(torch.from_numpy(radar.astype(np.float32)))

        sample = {"image": torch.cat(images), "radar": torch.cat(radars)}
        
        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample
