import cv2
from torchvision import transforms
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
import random
from PIL import Image
import numpy as np
from torchvision.transforms.v2.functional import to_tensor, to_image, to_dtype


class RoadDataset(Dataset):
    def __init__(self, metadata, size=512, train=True):
        self.metadata = metadata
        self.train = train
        self.size = size

    def __len__(self):
        return self.metadata.shape[0]

    def __getitem__(self, index):
        image = cv2.imread(self.metadata.iloc[index]["sat_image_path"])
        mask = cv2.imread(self.metadata.iloc[index]["mask_path"])

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #only one channel for mask
        mask = mask[:, :, 0]
        mask = (mask > 0).astype(np.uint8)

        # Convert image and mask to PIL Images to apply torchvision transforms
        image = Image.fromarray(image)
        mask = Image.fromarray(mask)

        if (self.train):
            if random.random() > 0.5:
                image = F.hflip(image)
                mask = F.hflip(mask)

            if random.random() > 0.5:
                image = F.vflip(image)
                mask = F.vflip(mask)

            # Get the same crop parameters for both the image and mask
            i, j, h, w = transforms.RandomResizedCrop.get_params(image, scale=(0.5, 1.0), ratio=(1.0, 1.0))

            image = F.resized_crop(image, i, j, h, w, (self.size, self.size))
            mask = F.resized_crop(mask, i, j, h, w, (self.size, self.size))

        else:
            image = F.resize(image, (self.size, self.size))
            mask = F.resize(mask, (self.size, self.size))

        image = to_image(image)
        mask = to_image(mask)
        image = to_dtype(image, dtype=torch.float32, scale=True)
        mask = to_dtype(mask, dtype=torch.float32)

        image = F.normalize(image, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        return image, mask











