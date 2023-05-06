import torch
import torch.nn as nn
import itertools
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
import torchmetrics as tm
from ubmnist import UbMNIST

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

# Or use LeNet from https://github.com/bollakarthikeya/LeNet-5-PyTorch/blob/master/lenet5_gpu.py
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
        
class Baseline(pl.LightningModule):
    def __init__(self, lr=1e-3, momentum=0.1, nsteps=4000):
        super().__init__()
        self.lr = lr
        self.momentum = momentum
        self.nsteps = nsteps
        
        self.model = LeNet5()
        self.train_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.meta_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.mean_loss = nn.BCEWithLogitsLoss(reduction='mean')
        self.train_acc = tm.Accuracy()
        self.valid_acc = tm.Accuracy()
        self.test_acc = tm.Accuracy()
    
    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [self.nsteps//2], gamma=0.5)
        return [optimizer], [lr_scheduler]
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        train_loss = torch.sum(self.train_loss(y_hat, y.type_as(y_hat)))
        self.log('Train/Loss', train_loss)
        train_acc = self.test_acc(y_hat, y)
        self.log('Train/Accuracy', train_acc)
        return train_loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        test_loss = torch.sum(self.meta_loss(y_hat, y.type_as(y_hat)))
        self.log('Test/Loss', test_loss)
        test_acc = self.test_acc(y_hat, y)
        self.log('Test/Accuracy', test_acc)
        return test_loss

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
        self.model = LeNet5()
        self.train_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.meta_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.mean_loss = nn.BCEWithLogitsLoss(reduction='mean')
        self.train_acc = tm.Accuracy()
        self.valid_acc = tm.Accuracy()
        self.test_acc = tm.Accuracy()

        self.example_input_array = torch.randn((100, 1, 28, 28)) # used to draw computational graph (wrong but still)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [self.nsteps//2], gamma=0.5)
        return [optimizer], [lr_scheduler]

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        opt = self.optimizers(use_pl_optimizer=False) # pytorch_lightning only supports first-order optimization, must use vanilla pytorch optim
        opt.zero_grad()
        outputs = self.model(inputs)
        
        meta_model = LeNet5()
        meta_model.load_state_dict(self.model.state_dict()) # increased train loss, allowed model to learn from meta-dataset, decreased test loss/increase test accuracy
        meta_opt = torch.optim.SGD(meta_model.parameters(), lr=self.lr, momentum=self.momentum)
        meta_opt.load_state_dict(opt.state_dict()) # increased train loss, allowed model to learn from meta-dataset, decreased test loss/increase test accuracy
        meta_opt.zero_grad()
        
        meta_train_outputs = meta_model(inputs)
        meta_train_loss = self.meta_loss(meta_train_outputs, labels.type_as(outputs))
        epsilon = torch.zeros(meta_train_loss.size(), requires_grad=True)
        meta_train_loss = torch.sum(epsilon * meta_train_loss)
        meta_train_loss.backward()
        meta_opt.step()
        meta_opt.zero_grad()
        
        meta_inputs, meta_labels = next(meta_loader)
        meta_val_outputs = meta_model(meta_inputs)
        meta_val_loss = self.mean_loss(meta_val_outputs, meta_labels.type_as(outputs))
        meta_val_loss.backward(retain_graph=True)
        
        self.log("Meta-Train/Loss", meta_val_loss, on_epoch=True)
        val_acc = self.test_acc(meta_val_outputs, meta_labels)
        self.log('Meta-Train/Accuracy', val_acc, on_epoch=True)
        epsilon_grads = epsilon.grad.detach() # detach gradients from meta-model's computational graph, resolves in-place error in final fully connected layer
        meta_train_loss.grad = None

        w_tilde = torch.clamp(epsilon_grads, min=0)
        weights = w_tilde / torch.sum(w_tilde) if torch.sum(w_tilde) != 0 else w_tilde

        outputs = self.model(inputs)
        minibatch_loss = self.meta_loss(outputs, labels.type_as(outputs))
        train_loss = torch.sum(weights * minibatch_loss)
        self.manual_backward(train_loss)
        opt.step()

        self.log("Train/Loss", train_loss, prog_bar=True, on_epoch=True)
        train_acc = self.train_acc(outputs, labels)
        self.log('Train/Accuracy', train_acc, on_epoch=True)
        
        # print(min(outputs), max(outputs))
        
        mean_weights_c0 = torch.mean(weights[outputs < 0])
        std_weights_c0 = torch.std(weights[outputs < 0])
        if mean_weights_c0 == None or torch.isnan(mean_weights_c0):
            mean_weights_c0 = 0.0
        if std_weights_c0 == None or torch.isnan(std_weights_c0):
            std_weights_c0 = 0.0
        mean_weights_c1 = torch.mean(weights[outputs > 0])
        std_weights_c1 = torch.std(weights[outputs > 0])
        if mean_weights_c1 == None or torch.isnan(mean_weights_c1):
            mean_weights_c1 = 0.0
        if std_weights_c1 == None or torch.isnan(std_weights_c1):
            std_weights_c1 = 0.0
        self.log("Weights/Class0 Mean", mean_weights_c0, on_epoch=True, on_step=False)
        self.log("Weights/Class1 Mean", mean_weights_c1, on_epoch=True, on_step=False)
        self.log("Weights/Class0 Std", std_weights_c0, on_epoch=True, on_step=False)
        self.log("Weights/Class1 Std", std_weights_c1, on_epoch=True, on_step=False)
        
        return None

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        test_loss = torch.sum(self.meta_loss(y_hat, y.type_as(y_hat)))
        self.log('Test/Loss', test_loss)
        test_acc = self.test_acc(y_hat, y)
        self.log('Test/Accuracy', test_acc)
        return None


if __name__ == "__main__":
    splits = [0.5, 0.9, 0.95, 0.98, 0.99, 0.995]
    for i,s in enumerate(splits):
        pl.seed_everything(i)
        nsteps = 8000  # used for learning rate scheduler, this experiment completes before the learning rate can change
        batch_size = 100
        dm = UbMNIST(batch_size=batch_size, train_split=s)
        dm.prepare_data() # ensure MNIST dataset is downloaded
        dm.setup() # allow data to be extracted from datamodule
        
        # infinite loop of balanced training data
        meta_loader = itertools.cycle(dm.meta_dataloader())

        model = ReweightNet(nsteps=nsteps, momentum=0, lr=1e-2)
        logger = TensorBoardLogger("./mnist-rw", log_graph=True, default_hp_metric=False)
        # model = Baseline(nsteps=nsteps, momentum=0, lr=1e-2)
        # logger = TensorBoardLogger("./mnist-base", log_graph=True, default_hp_metric=False)
        trainer = Trainer(accelerator='auto', devices=1, min_epochs=1, max_epochs=30, log_every_n_steps=1, logger=logger)
        trainer.fit(model=model, datamodule=dm)
        trainer.test(model=model, datamodule=dm)
    
    # pl.seed_everything(0)
    # nsteps = 8000  # used for learning rate scheduler, this experiment completes before the learning rate can change
    # batch_size = 100
    # dm = UbMNIST(batch_size=batch_size, train_split=0.9)
    # dm.prepare_data() # ensure MNIST dataset is downloaded
    # dm.setup() # allow data to be extracted from datamodule
    
    # # infinite loop of balanced training data
    # meta_loader = itertools.cycle(dm.meta_dataloader())

    # model = ReweightNet(nsteps=nsteps, momentum=0, lr=1e-2)
    # logger = TensorBoardLogger("./", log_graph=True, default_hp_metric=False)
    # trainer = Trainer(accelerator='auto', devices=1, min_epochs=1, max_epochs=15, log_every_n_steps=1, logger=logger)
    # trainer.fit(model=model, datamodule=dm)
    # trainer.test(model=model, datamodule=dm)
