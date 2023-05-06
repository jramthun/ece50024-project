import pytorch_lightning as pl
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10
from torch.utils.data import DataLoader, TensorDataset
import torch
import copy
import numpy as np

# Datamodule Containing Imbalanced MNIST Dataset, following original design
class UbCIFAR(pl.LightningDataModule):
    def __init__(self, batch_size=100, seed=0, train_splits=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], ntrain=10000, nval=100, ntest=1000):
        super().__init__()

        # User Defined Properties
        self.batch_size = batch_size
        self.seed = seed  # Random seed for this run
        self.train_splits = train_splits  # What percentage of the training data is class1
        # self.test_splits = test_splits  # What percentage of test data is class1
        self.ntrain = ntrain  # How many training samples overall
        self.nval = nval // 10  # How many validation samples
        self.ntest = ntest // 10  # How many test samples
        self.num_classes = 10
        self.num_channels = 3
        self.img_width = 32
        self.img_height = 32
        # self.num_channels = 1
        # self.img_width = 28
        # self.img_height = 28

        # Intrinsic Properties
        self.train_workers = 4
        self.test_workers = 4

    def prepare_data(self):
        tfrms = transforms.Compose([transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)), transforms.ConvertImageDtype(torch.float)])
        # tfrms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        data_dir = "./datasets/"
        # self.cifar_train = MNIST(data_dir, train=True, download=True, transform=tfrms)
        # self.cifar_test = MNIST(data_dir, train=False, download=True, transform=tfrms)
        self.cifar_train = CIFAR10(data_dir, train=True, download=True, transform=tfrms)
        self.cifar_test = CIFAR10(data_dir, train=False, download=True, transform=tfrms)

    def setup(self, stage=None):
        x_train = self.cifar_train.data
        y_train = self.cifar_train.targets
        x_test = self.cifar_test.data
        y_test = self.cifar_test.targets
        
        # Create Balanced Test Set - CIFAR10 is already balanced in testing
        if self.num_channels > 1:
            test_feat = np.empty((self.num_classes, self.ntest, self.img_width, self.img_height, self.num_channels), np.float32)
        else:
            test_feat = np.empty((self.num_classes, self.ntest, self.img_width, self.img_height), np.float32)
        test_tgts = np.zeros((self.num_classes * self.ntest))
        
        for i, _ in enumerate(self.cifar_test.classes):
            test_feat[i] = np.array(x_test[np.where(np.array(y_test) == i)[0]][:self.ntest])
            test_tgts[i*self.ntest:(i+1)*self.ntest] = i
        
        test_feat = torch.from_numpy(test_feat)
        test_tgts = torch.from_numpy(test_tgts).type(torch.LongTensor)
        
        test_feat = test_feat.reshape((self.num_classes * self.ntest, self.num_channels, self.img_width, self.img_height)) / 255
        # test_tgts = test_tgts.reshape((self.num_classes * self.ntest, 10))

        # self.cifar_test.data = test_feat
        # self.cifar_test.targets = test_tgts
        # self.test_dataset = self.cifar_test
        # print(test_feat.size(), test_tgts.size())
        self.test_dataset = TensorDataset(test_feat, test_tgts)
        
        # Create Imbalanced Training Set
        if sum(self.train_splits) != 1.0:
            self.train_splits = [s / sum(self.train_splits) for s in self.train_splits]
        assert 1.01 > sum(self.train_splits) > 0.99, f"Sum of Normalized weights = {sum(self.train_splits)}. Splits: {self.train_splits}"

        class_lens = [int(s * self.ntrain) for s in self.train_splits]
        total = sum(class_lens)

        if self.num_channels > 1:
            train_feat = np.empty((total, self.img_width, self.img_height, self.num_channels), np.float32)
        else:
            train_feat = np.empty((total, self.img_width, self.img_height), np.float32)
        train_tgts = np.zeros((total))

        last = 0
        for i, c in enumerate(class_lens):
            train_feat_c = np.array(x_train[np.where(np.array(y_train) == i)[0]][:c])
            # print(f"Class {i}, proportion: {c/total}, length: {c}, {train_feat_c.shape}")
            train_feat[last:last + c] = train_feat_c
            train_tgts[last:last + c] = i
            last += c
        
        train_feat = torch.from_numpy(train_feat)
        train_tgts = torch.from_numpy(train_tgts).type(torch.LongTensor)
        
        train_feat = train_feat.reshape((total, self.num_channels, self.img_width, self.img_height)) / 255
        # train_tgts = train_tgts.reshape((self.num_classes * self.ntrain))

        # print(train_feat.size(), train_tgts.size())

        self.train_dataset = TensorDataset(train_feat, train_tgts)
        
        # Create Balanced Meta-Training set for determining weights
        # self.meta_dataset = copy.deepcopy(self.train_dataset)
        
        if self.num_channels > 1:
            meta_feat = np.empty((self.num_classes, self.nval, self.img_width, self.img_height, self.num_channels), np.float32)
        else:
            meta_feat = np.empty((self.num_classes, self.nval, self.img_width, self.img_height), np.float32)
        meta_tgts = np.zeros((self.num_classes, self.nval))
        
        for i, _ in enumerate(self.cifar_train.classes):
            meta_feat[i] = np.array(x_train[np.where(np.array(y_train) == i)[0]][:int(self.nval)])
            meta_tgts[i*self.nval:(i+1)*self.nval] = i
        
        meta_feat = torch.from_numpy(meta_feat)
        meta_tgts = torch.from_numpy(meta_tgts).type(torch.LongTensor)

        meta_feat = meta_feat.reshape((self.num_classes*self.nval, self.num_channels, self.img_width, self.img_height)) / 255
        meta_tgts = meta_tgts.reshape((self.num_classes*self.nval))
        
        self.meta_dataset = TensorDataset(meta_feat, meta_tgts)

        # self.meta_dataset.data = meta_feat
        # self.meta_dataset.targets = meta_tgts
        
        # print(meta_feat.shape, meta_tgts.shape)
        

    def train_dataloader(self):
        return DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.train_workers, pin_memory=True)

    def meta_dataloader(self):
        return DataLoader(self.meta_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(dataset=self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.test_workers, pin_memory=True)

    def predict_dataloader(self):
        return DataLoader(dataset=self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.test_workers, pin_memory=True)


if __name__ == "__main__":
    pl.seed_everything(0)  # Use this to set random state of any run
    dataModule = UbCIFAR(batch_size=32)
    dataModule.prepare_data()
    dataModule.setup()
