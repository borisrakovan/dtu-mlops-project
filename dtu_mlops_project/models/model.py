import torchvision
from lightning import LightningModule
from torch import optim


class SpeechSpectrogramsTransferLearning(LightningModule):
    def __init__(self, learning_rate):
        super().__init__()
        self.learning_rate = learning_rate
        self.resnet50 = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)

    def forward(self, x):
        x = self.resnet50(x)
        return x

    def training_step(self, batch, batch_idx):
        data, target = batch
        preds = self(data)
        loss = self.criterium(preds, target)
        acc = (target == preds.argmax(dim=-1)).float().mean()
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        return loss

    def validation_step(self, batch, batch_idx):
        data, target = batch
        preds = self(data)
        loss = self.criterium(preds, target)
        acc = (target == preds.argmax(dim=-1)).float().mean()
        self.log('val_loss', loss)
        self.log('val_acc', acc)
        return loss

    def test_step(self, batch, batch_idx):
        data, target = batch
        preds = self(data)
        loss = self.criterium(preds, target)
        acc = (target == preds.argmax(dim=-1)).float().mean()
        self.log('test_loss', loss)
        self.log('test_acc', acc)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)
