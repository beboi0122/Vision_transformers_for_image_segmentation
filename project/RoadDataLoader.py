import pytorch_lightning as pl
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from RoadDataset import RoadDataset

class RoadDataLoader(pl.LightningDataModule):
    def __init__(self, metadata, batch_size, num_workers=0, image_size=512):
        super().__init__()
        self.metadata = metadata
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.persistent_workers = num_workers > 0

    def setup(self, stage=None):
        # Split metadata into training, validation, and test sets
        train_data, val_data = train_test_split(self.metadata, test_size=0.2, random_state=42)
        self.train_data, self.test_data = train_test_split(train_data, test_size=0.1, random_state=42)
        self.val_data = val_data

    def train_dataloader(self):
        train_dataset = RoadDataset(self.train_data, train=True, size=self.image_size)
        return DataLoader(train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, persistent_workers=self.persistent_workers)

    def val_dataloader(self):
        val_dataset = RoadDataset(self.val_data, size=self.image_size)
        return DataLoader(val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=self.persistent_workers)

    def test_dataloader(self):
        test_dataset = RoadDataset(self.test_data, size=self.image_size)
        return DataLoader(test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=self.persistent_workers)




