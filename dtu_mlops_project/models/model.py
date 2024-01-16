import logging
import torchvision
from lightning import LightningModule
from torch import Tensor, optim, nn
from typing import Any, List, Dict, Tuple, Union

logger = logging.getLogger(__name__)


class SpeechSpectrogramsTransferLearning(LightningModule):
    VALID_RESNET_VERSIONS: List[str] = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
    RESNET_WEIGHTS: Dict[str, Any] = {
        'resnet18': torchvision.models.ResNet18_Weights,
        'resnet34': torchvision.models.ResNet34_Weights,
        'resnet50': torchvision.models.ResNet50_Weights,
        'resnet101': torchvision.models.ResNet101_Weights,
        'resnet152': torchvision.models.ResNet152_Weights,
    }
    def __init__(self, learning_rate: float, resnet_version: str, pretrained_weights: Union[str, None]):
        super().__init__()
        self.learning_rate = learning_rate
        # pick resnet version
        if resnet_version not in self.VALID_RESNET_VERSIONS:
            raise ValueError(f'Invalid resnet_version: {resnet_version}. Valid resnet versions are: {self.VALID_RESNET_VERSIONS}')
        self.resnet_version = resnet_version
        # pick pretrained weights
        if pretrained_weights is not None:
            self.pretrained_weights = self.RESNET_WEIGHTS[resnet_version][pretrained_weights]
        else:
            self.pretrained_weights = None
        # init resnet
        logger.info(f'Initializing resnet {resnet_version} with weights {pretrained_weights}')
        self.resnet = getattr(torchvision.models, resnet_version)(weights=self.pretrained_weights)
        # Monkey patched the number of channels.
        # self.resnet.conv1 = nn.Conv2d(1, self.resnet.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        old_conv1_weights = self.resnet.conv1.weight
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.conv1.weight = nn.Parameter(old_conv1_weights.mean(dim=1, keepdim=True))
        # self.resnet.inplanes is overwritten within the torchvision resnet implementation deeper in the model.
        # that's why placing it back here in conv1 after the model has been initialized causes a shape issue.
        # We solved that here by passing the initial startvalue of self.resnet.inplanes when redefining self.resnet.conv1
        # to have only 1 channel.
        self.criterium = nn.CrossEntropyLoss()

    def forward(self, x: Tensor) -> Tensor:
        x = self.resnet(x)
        return x

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        data, target = batch
        preds = self(data)
        loss = self.criterium(preds, target)
        acc = (target == preds.argmax(dim=-1)).float().mean()
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        return loss

    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        data, target = batch
        preds = self(data)
        loss = self.criterium(preds, target)
        acc = (target == preds.argmax(dim=-1)).float().mean()
        self.log('val_loss', loss)
        self.log('val_acc', acc)
        return loss

    def test_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        data, target = batch
        preds = self(data)
        loss = self.criterium(preds, target)
        acc = (target == preds.argmax(dim=-1)).float().mean()
        self.log('test_loss', loss)
        self.log('test_acc', acc)
        return loss

    def configure_optimizers(self) -> optim.Optimizer:
        return optim.Adam(self.parameters(), lr=self.learning_rate)
