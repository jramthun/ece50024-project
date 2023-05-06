import torch
import torch.nn as nn
import itertools
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
# from lightning.pytorch.loggers import TensorBoardLogger
import torchmetrics as tm
from ubcifar import UbCIFAR
import timm

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

class Baseline(pl.LightningModule):
    def __init__(self, lr=1e-3, momentum=0.1):
        super().__init__()
        self.lr = lr
        self.momentum = momentum
        self.model = timm.create_model('resnet18', pretrained=False, num_classes=10)
        self.train_loss = nn.CrossEntropyLoss(reduction='none')
        self.meta_loss = nn.CrossEntropyLoss(reduction='none')
        self.mean_loss = nn.CrossEntropyLoss(reduction='mean')
        self.train_acc = tm.Accuracy(task="multiclass", num_classes=10)
        self.valid_acc = tm.Accuracy(task="multiclass", num_classes=10)
        self.test_acc = tm.Accuracy(task="multiclass", num_classes=10)

    def forward(self, x):
        return self.model(x)
    
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum)
        lr_scheduler = {"scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5), "monitor": "Train/Accuracy"}
        return [optimizer], [lr_scheduler]

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        train_loss = torch.sum(self.meta_loss(y_hat, y))
        self.log('Train/Loss', train_loss, prog_bar=True, on_epoch=True)
        train_acc = self.train_acc(y_hat, y)
        self.log('Train/Accuracy', train_acc, on_epoch=True)
        return train_loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        test_loss = torch.sum(self.meta_loss(y_hat, y))
        self.log('Test/Loss', test_loss)
        test_acc = self.test_acc(y_hat, y)
        self.log('Test/Accuracy', test_acc)
        return test_loss

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)
        self.max_pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.max_pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5)
        self.fc1 = nn.Linear(in_features=120, out_features=84)
        self.fc2 = nn.Linear(in_features=84, out_features=1)
        self.act = nn.Tanh()
        self.smx = nn.Softmax(dim=0)

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.max_pool_1(x))
        x = self.act(self.conv2(x))
        x = self.act(self.max_pool_2(x))
        x = self.act(self.conv3(x))
        x = x.view(-1, 120)
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        x = x.view(-1)
        return x

class ReweightNet(pl.LightningModule):
    def __init__(self, lr=1e-3, momentum=0.1, nsteps=4000):
        super().__init__()

        # Use Custom Optimization Strategy
        self.save_hyperparameters()  # required to log hyperparameters in Tensorboard
        self.automatic_optimization = False  # required for manual optimization

        # User Defined Properties
        self.lr = lr
        self.momentum = momentum
        self.nsteps = nsteps

        # Intrinsic Properties
        # self.model = LeNet5()
        self.model = timm.create_model('resnet26', pretrained=False, num_classes=10)
        self.train_loss = nn.CrossEntropyLoss(reduction='none')
        self.meta_loss = nn.CrossEntropyLoss(reduction='none')
        self.mean_loss = nn.CrossEntropyLoss(reduction='mean')
        self.train_acc = tm.Accuracy(task="multiclass", num_classes=10)
        self.valid_acc = tm.Accuracy(task="multiclass", num_classes=10)
        self.test_acc = tm.Accuracy(task="multiclass", num_classes=10)


    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum)
        lr_scheduler = {"scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5), "monitor": "Meta-Train/Accuracy"}
        return [optimizer], [lr_scheduler]

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        opt = self.optimizers(use_pl_optimizer=False) # pytorch_lightning only supports first-order optimization, must use vanilla pytorch optim
        opt.zero_grad()
        outputs = self.model(inputs)
        
        meta_model = timm.create_model('resnet26', pretrained=False, num_classes=10).to(self.device)
        # meta_model = LeNet5().to(self.device)
        meta_model.load_state_dict(self.model.state_dict()) # increased train loss, allowed model to learn from meta-dataset, decreased test loss/increase test accuracy
        meta_opt = torch.optim.SGD(meta_model.parameters(), lr=self.lr, momentum=self.momentum)
        meta_opt.load_state_dict(opt.state_dict()) # increased train loss, allowed model to learn from meta-dataset, decreased test loss/increase test accuracy
        meta_opt.zero_grad()
        
        meta_train_outputs = meta_model(inputs)
        meta_train_loss = self.meta_loss(meta_train_outputs, labels)
        epsilon = torch.zeros(meta_train_loss.size(), requires_grad=True, device=self.device)
        meta_train_loss = torch.sum(epsilon * meta_train_loss)
        meta_train_loss.backward()
        meta_opt.step()
        meta_opt.zero_grad()
        
        meta_inputs, meta_labels = next(meta_loader)
        meta_inputs = meta_inputs.to(self.device)
        meta_labels = meta_labels.to(self.device)
        meta_val_outputs = meta_model(meta_inputs)
        meta_val_loss = self.mean_loss(meta_val_outputs, meta_labels)
        meta_val_loss.backward(retain_graph=True)
        
        self.log("Meta-Train/Loss", meta_val_loss, on_epoch=True)
        val_acc = self.test_acc(meta_val_outputs, meta_labels)
        self.log('Meta-Train/Accuracy', val_acc, on_epoch=True)
        epsilon_grads = epsilon.grad.detach() # detach gradients from meta-model's computational graph, resolves in-place error in final fully connected layer
        meta_train_loss.grad = None

        w_tilde = torch.clamp(epsilon_grads, min=0)
        weights = w_tilde / torch.sum(w_tilde) if torch.sum(w_tilde) != 0 else w_tilde

        outputs = self.model(inputs)
        minibatch_loss = self.meta_loss(outputs, labels)
        train_loss = torch.sum(weights * minibatch_loss)
        self.manual_backward(train_loss)
        opt.step()

        self.log("Train/Loss", train_loss, prog_bar=True, on_epoch=True)
        # print(outputs[0], labels[0])
        train_acc = self.train_acc(outputs, labels)
        # print(train_acc)
        self.log('Train/Accuracy', train_acc, on_epoch=True)
        return None

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        test_loss = torch.sum(self.meta_loss(y_hat, y))
        self.log('Test/Loss', test_loss)
        test_acc = self.test_acc(y_hat, y)
        self.log('Test/Accuracy', test_acc)
        return None


if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    pl.seed_everything(0)
    nsteps = 8000  # used for learning rate scheduler, this experiment completes before the learning rate can change
    batch_size = 1000

    # Part 1
    # dm = UbCIFAR(batch_size=batch_size, train_splits=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.8])
    # dm.prepare_data() # ensure CIFAR dataset is downloaded
    # dm.setup() # allow data to be extracted from datamodule
    
    # # infinite loop of balanced training data
    # meta_loader = itertools.cycle(dm.meta_dataloader())

    # model = Baseline(momentum=0.0, lr=1e-2)
    # logger = TensorBoardLogger("./", log_graph=False, default_hp_metric=False)
    # trainer = Trainer(accelerator='gpu', devices=1, min_epochs=1, max_epochs=100, log_every_n_steps=1, logger=logger)
    # trainer.fit(model=model, datamodule=dm)
    # trainer.test(model=model, datamodule=dm)

    # model = ReweightNet(nsteps=nsteps, momentum=0.0, lr=1e-2)
    # logger = TensorBoardLogger("./", log_graph=False, default_hp_metric=False)
    # trainer = Trainer(accelerator='gpu', devices=1, min_epochs=1, max_epochs=100, log_every_n_steps=1, logger=logger)
    # trainer.fit(model=model, datamodule=dm)
    # trainer.test(model=model, datamodule=dm)

    # Second Part
    dm = UbCIFAR(batch_size=batch_size, train_splits=[0.3, 0.1, 0.1, 0.3, 0.1, 0.1, 0.3, 0.1, 0.1, 0.3])
    dm.prepare_data() # ensure CIFAR dataset is downloaded
    dm.setup() # allow data to be extracted from datamodule
    
    # infinite loop of balanced training data
    meta_loader = itertools.cycle(dm.meta_dataloader())

    # model = Baseline(momentum=0.0, lr=1e-2)
    # logger = TensorBoardLogger("./", log_graph=False, default_hp_metric=False)
    # trainer = Trainer(accelerator='gpu', devices=1, min_epochs=1, max_epochs=100, log_every_n_steps=1, logger=logger)
    # trainer.fit(model=model, datamodule=dm)
    # trainer.test(model=model, datamodule=dm)

    model = ReweightNet(nsteps=nsteps, momentum=0.0, lr=1e-2)
    logger = TensorBoardLogger("./", log_graph=False, default_hp_metric=False)
    trainer = Trainer(accelerator='gpu', devices=1, min_epochs=1, max_epochs=1000, log_every_n_steps=1, logger=logger)
    trainer.fit(model=model, datamodule=dm)
    trainer.test(model=model, datamodule=dm)
