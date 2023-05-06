import pytorch_lightning as pl
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split, TensorDataset, ConcatDataset
import torch
import numpy as np

# Datamodule Containing Imbalanced MNIST Dataset, following original design
class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, class0=4, class1=9, seed=0, train_split=0.9, test_split=0.5, ntrain=5000, nval=10, ntest=500):
        super().__init__()
        
        # User Defined Properties
        self.batch_size = batch_size
        self.class0 = class0
        self.class1 = class1
        self.seed = seed # Random seed for this run
        self.train_split = train_split # What percentage of the training data is class1
        self.test_split = test_split # What percentage of test data is class1
        self.ntrain = ntrain # How many training samples overall
        self.nval = nval # How many validation samples
        self.ntest = ntest # How many test samples
        
        # Intrinsic Properties
        self.train_workers = 4
        self.val_workers = 4
        self.test_workers = 4
    
    # Ensure that the MNIST dataset is downloaded
    def prepare_data(self):
        tfrms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        data_dir = "./datasets/"
        self.mnist_train = MNIST(data_dir, train=True, download=True, transform=tfrms)
        self.mnist_test = MNIST(data_dir, train=False, download=True, transform=tfrms)
    
    # Setup the individal datasets for the dataloaders
    def setup(self, stage=None):
        # Parse MNIST dataset
        self.x_train = self.mnist_train.data
        self.y_train = self.mnist_train.targets
        self.x_test = self.mnist_test.data
        self.y_test = self.mnist_test.targets
        
        # Get data in "Class 0"
        self.x_train_0 = self.x_train[self.y_train == self.class0]
        self.x_test_c0 = self.x_test[self.y_test == self.class0]
        
        # Get lengths of data
        class0_train = int(np.rint(self.ntrain * (1 - self.train_split)))
        class0_val = int(np.rint(self.nval * (1 - self.test_split)))
        class0_test = int(np.rint(self.ntest * (1 - self.test_split)))
        # print(class0_train, class0_val, class0_test)
        
        # Randomly sample MNIST to get data
        self.x_train_0, self.x_val_0, _ = random_split(self.x_train_0, [class0_train, class0_val, len(self.x_train_0) - class0_train - class0_val])
        self.x_test_0, _ = random_split(self.x_test_c0, [class0_test, self.x_test_c0.shape[0] - class0_test])
        # print(len(self.x_train_0), len(self.x_val_0), len(self.x_test_0))
        
        # Get data in "Class 1"
        self.x_train_1 = self.x_train[self.y_train == self.class1]
        self.x_test_1 = self.x_test[self.y_test == self.class1]
        
        # Get lengths of data
        class1_train = int(np.rint(self.ntrain * (self.train_split)))
        class1_val = int(np.rint(self.nval * (self.test_split)))
        class1_test = int(np.rint(self.ntest * (self.test_split)))
        # print(class1_train, class1_val, class1_test)
        
        # Randomly sample MNIST to get data
        self.x_train_1, self.x_val_1, _ = random_split(self.x_train_1, [class1_train, class1_val, len(self.x_train_1) - class1_train - class1_val])
        self.x_test_1, _ = random_split(self.x_test_1, [class1_test, len(self.x_test_1) - class1_test])
        # print(len(self.x_train_1), len(self.x_val_1), len(self.x_test_1))

        # Verify Data Counts before proceeding
        assert len(self.x_train_0.indices) + len(self.x_train_1.indices) == class0_train + class1_train == self.ntrain
        assert len(self.x_val_0.indices) + len(self.x_val_1.indices) == class0_val + class1_val == self.nval
        assert len(self.x_test_0.indices) + len(self.x_test_1.indices) == class0_test + class1_test == self.ntest
        
        # Construct Dataset with binary encoding
        train_feat = torch.cat((self.x_train[self.x_train_0.indices], self.x_train[self.x_train_1.indices]))
        train_tgts = torch.cat((torch.zeros((class0_train)), torch.ones((class1_train))))
        val_feat = torch.cat((self.x_train[self.x_val_0.indices], self.x_train[self.x_val_1.indices]))
        val_tgts = torch.cat((torch.zeros((class0_val)), torch.ones((class1_val))))
        test_feat = torch.cat((self.x_test[self.x_test_0.indices], self.x_test[self.x_test_1.indices]))
        test_tgts = torch.cat((torch.zeros((class0_test)), torch.ones((class1_test))))
        
        self.train_data = TensorDataset(train_feat, train_tgts)
        self.val_data = TensorDataset(val_feat, val_tgts)
        self.test_data = TensorDataset(test_feat, test_tgts)
        
        # self.train_class_0 = TensorDataset(self.x_train[self.x_train_0.indices], torch.zeros((class0_train,1)))
        # self.val_class_0 = TensorDataset(self.x_train[self.x_val_0.indices], torch.zeros((class0_val,1)))
        # self.test_class_0 = TensorDataset(self.x_test[self.x_test_0.indices], torch.zeros((class0_test,1)))
        
        # self.train_class_1 = TensorDataset(self.x_train[self.x_train_1.indices], torch.ones((class1_train,1)))
        # self.val_class_1 = TensorDataset(self.x_train[self.x_val_1.indices], torch.ones((class1_val,1)))
        # self.test_class_1 = TensorDataset(self.x_test[self.x_test_1.indices], torch.ones((class1_test,1)))
        
        # self.train_data = ConcatDataset([self.train_class_0, self.train_class_1])
        # self.val_data = ConcatDataset([self.val_class_0, self.val_class_1])
        # self.test_data = ConcatDataset([self.test_class_0, self.test_class_1])
        
    def train_dataloader(self):
        return DataLoader(dataset=self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=self.train_workers, pin_memory=True)
        
    def val_dataloader(self):
        return DataLoader(dataset=self.val_data, batch_size=self.batch_size, shuffle=True, num_workers=self.val_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(dataset=self.test_data, batch_size=self.batch_size, shuffle=False, num_workers=self.test_workers, pin_memory=True)

    def predict_dataloader(self):
        return DataLoader(dataset=self.test_data, batch_size=self.batch_size, shuffle=False, num_workers=self.test_workers, pin_memory=True)
    
if __name__ == "__main__":
    pl.seed_everything(0) # Use this to set random state of any run
    dataModule = MNISTDataModule(batch_size=32)
    dataModule.prepare_data()
    dataModule.setup()