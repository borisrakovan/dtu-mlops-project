import logging
import torch
import torchaudio
import pandas as pd
from omegaconf import ListConfig
from torch.utils.data import Dataset, DataLoader
from lightning import LightningDataModule


logger = logging.getLogger(__name__)


class SpeechCommands(Dataset):
    def __init__(self, root, subset, classes, preproc, **kwargs):
        # load dataset
        self.dataset = torchaudio.datasets.SPEECHCOMMANDS(root=root, subset=subset, download=True)
        # load metadata into dataframe
        self.df = pd.DataFrame(
            [self.dataset.get_metadata(i) for i in range(len(self.dataset))],
            columns=["path", "sr", "label", "speaker_id", "utterance_number"])
        # filter data by classes and store mapping
        self.idx_to_class = list(self.df.label.sort_values().unique())
        self.idx_to_class = classes if isinstance(classes, (list, ListConfig)) else self.idx_to_class[:classes]
        self.indexes = self.df.index[self.df.label.isin(self.idx_to_class)].tolist()
        self.class_to_idx = {c: i for i, c in enumerate(self.idx_to_class)}
        # store other args
        self.n_classes = len(self.idx_to_class)
        self.preproc = preproc

    def __getitem__(self, idx):
        _idx = self.indexes[idx]
        waveform, sample_rate, label, speaker_id, utterance_number = self.dataset[_idx]
        # apply preprocessing
        audio_features = waveform
        if self.preproc is not None:
            audio_features = self.preproc(audio_features)
        return audio_features, self.class_to_idx[label]

    def __len__(self):
        return len(self.indexes)

    def collate_fn(self, batch):
        tensors, targets = zip(*batch)
        tensors = [t.squeeze(0).T for t in tensors]
        tensors = torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True, padding_value=0.)
        tensors = tensors.unsqueeze(1).permute(0, 1, 3, 2)
        targets = torch.tensor(targets)
        return tensors, targets



class SpeechCommandsDataModule(LightningDataModule):
    def __init__(self, data_dir, batch_size, n_workers, train_transforms, test_transforms, **kwargs):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.n_workers = n_workers
        self.train_transforms = torch.nn.Sequential(*train_transforms)
        self.test_transforms = torch.nn.Sequential(*test_transforms)
        self.dataset_kwargs = kwargs

    def setup(self, stage):
        if stage == "fit" or stage is None:
            self.speechcmd_train = SpeechCommands(self.data_dir, subset="training", preproc=self.train_transforms, **self.dataset_kwargs)
            self.speechcmd_val = SpeechCommands(self.data_dir, subset="validation", preproc=self.test_transforms, **self.dataset_kwargs)
        if stage == "test" or stage is None:
            self.speechcmd_test = SpeechCommands(self.data_dir, subset="testing", preproc=self.test_transforms, **self.dataset_kwargs)

    def train_dataloader(self):
        return DataLoader(
            self.speechcmd_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.n_workers,
            collate_fn=self.speechcmd_train.collate_fn)

    def val_dataloader(self):
        return DataLoader(
            self.speechcmd_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.n_workers,
            collate_fn=self.speechcmd_val.collate_fn)

    def test_dataloader(self):
        return DataLoader(
            self.speechcmd_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.n_workers,
            collate_fn=self.speechcmd_test.collate_fn)
