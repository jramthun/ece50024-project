import pytorch_lightning as pl
from torchvision import transforms
from torchvision.datasets import Kitti
from torch.utils.data import DataLoader
import torch
import copy

# Datamodule Containing Imbalanced MNIST Dataset, following original design
class UbKITTI(pl.LightningDataModule):
    def __init__(self, batch_size=100, class0=4, class1=9, seed=0, train_split=0.9, test_split=0.5, ntrain=5000, nval=10, ntest=500):
        super().__init__()

        # User Defined Properties
        self.batch_size = batch_size
        self.class0 = class0
        self.class1 = class1
        self.seed = seed  # Random seed for this run
        self.train_split = train_split  # What percentage of the training data is class1
        self.test_split = test_split  # What percentage of test data is class1
        self.ntrain = ntrain  # How many training samples overall
        self.nval = nval  # How many validation samples
        self.ntest = ntest  # How many test samples

        # Intrinsic Properties
        self.train_workers = 4
        self.test_workers = 4

    def prepare_data(self):
        tfrms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        data_dir = "./datasets/"
        self.mnist_train = Kitti(data_dir, train=True, download=True, transform=tfrms)
        self.mnist_test = Kitti(data_dir, train=False, download=True, transform=tfrms)

    def setup(self, stage=None):
        x_train = self.mnist_train.data
        y_train = self.mnist_train.targets
        x_test = self.mnist_test.data
        y_test = self.mnist_test.targets
        
        # Create Balanced Test Set
        test_c0 = x_test[y_test == self.class0][:self.ntest]
        test_c1 = x_test[y_test == self.class1][:self.ntest]
        test_feat = torch.cat((test_c1, test_c0))
        test_tgts = torch.cat((torch.ones(len(test_c1)), torch.zeros(len(test_c0))))
        self.mnist_test.data = test_feat
        self.mnist_test.targets = test_tgts
        self.test_dataset = self.mnist_test
        # self.test_dataset = TensorDataset(test_feat, test_tgts)
        
        # Create Imbalanced Training Set
        num_train_c1 = int(self.ntrain * self.train_split)
        num_train_c0 = self.ntrain - num_train_c1
        train_c1 = x_train[y_train == self.class1][:num_train_c1]
        train_c0 = x_train[y_train == self.class0][:num_train_c0]
        train_feat = torch.cat((train_c1, train_c0))
        train_tgts = torch.cat((torch.ones(len(train_c1)), torch.zeros(len(train_c0))))
        # self.train_dataset = TensorDataset(train_feat, train_tgts)
        self.mnist_train.data = train_feat
        self.mnist_train.targets = train_tgts
        self.train_dataset = self.mnist_train
        
        # Create Balanced Meta-Training set for determining weights
        self.meta_dataset = copy.deepcopy(self.train_dataset)
        meta_c1 = x_train[y_train == self.class1][:(self.nval // 2)]
        meta_c0 = x_train[y_train == self.class0][:(self.nval // 2)]
        meta_feat = torch.cat((meta_c1, meta_c0))
        meta_tgts = torch.cat((torch.ones(len(meta_c1)), torch.zeros(len(meta_c0))))
        self.meta_dataset.data = meta_feat
        self.meta_dataset.targets = meta_tgts
        # self.meta_dataset = TensorDataset(meta_feat, meta_tgts)

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
    dataModule = UbKITTI(batch_size=32)
    dataModule.prepare_data()
    # dataModule.setup()
