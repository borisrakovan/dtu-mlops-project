import torchvision
from lightning import LightningModule
from torch import optim, nn


class SpeechSpectrogramsTransferLearning(LightningModule):
    def __init__(self, learning_rate, pretrained_weights):
        super().__init__()
        self.learning_rate = learning_rate

        if pretrained_weights == 'IMAGENET1K_V2':
            self.resnet50 = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
        elif pretrained_weights == 'IMAGENET1K_V1':
            self.resnet50 = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
        else:
            self.resnet50 = torchvision.models.resnet50(weights=None)




        # Monkey patched the number of channels.
        # self.resnet50.conv1 = nn.Conv2d(1, self.resnet50.inplanes, kernel_size=7, stride=2, padding=3, bias=False)

        self.resnet50.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # self.resnet50.inplanes is overwritten within the torchvision resnet implementation deeper in the model.
        # that's why placing it back here in conv1 after the model has been initialized causes a shape issue.
        # We solved that here by passing the initial startvalue of self.resnet50.inplanes when redefining self.resnet50.conv1
        # to have only 1 channel.

        self.criterium = nn.CrossEntropyLoss()

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
