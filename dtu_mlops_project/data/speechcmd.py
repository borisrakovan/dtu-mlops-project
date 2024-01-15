import logging
import torch
import torchaudio
import pandas as pd
from omegaconf import ListConfig
from torch.utils.data import Dataset


logger = logging.getLogger(__name__)


class SpeechCommands(Dataset):
    def __init__(self, root, subset, max_datapoints, classes, preproc, **kwargs):
        # sourcery skip: collection-builtin-to-comprehension
        if "data_mode" in kwargs:
            logger.warning("data_mode is ignored for SpeechCommands dataset")
        # load dataset
        self.dataset = torchaudio.datasets.SPEECHCOMMANDS(root=root, subset=subset, download=True)
        # load metadata into dataframe
        self.df = pd.DataFrame(
            [self.dataset.get_metadata(i) for i in range(len(self.dataset))],
            columns=["path", "sr", "label", "speaker_id", "utterance_number"])
        # filter data by classes and store mapping
        self.idx_to_class = list(self.df.label.sort_values().unique())
        self.idx_to_class = classes if isinstance(classes, (list, ListConfig)) else self.idx_to_class[:classes]
        self.indexes = self.df.index[self.df.label.isin(self.idx_to_class)].tolist()[:max_datapoints]
        self.class_to_idx = {c: i for i, c in enumerate(self.idx_to_class)}
        # store other args
        self.n_classes = len(self.idx_to_class)
        self.max_datapoints = max_datapoints
        self.preproc = preproc

    def __getitem__(self, idx):
        _idx = self.indexes[idx]
        waveform, sample_rate, label, speaker_id, utterance_number = self.dataset[_idx]
        # apply preprocessing
        if self.preproc is not None:
            audio_features = self.preproc(waveform)
        return audio_features[0].T, self.class_to_idx[label]

    def __len__(self):
        return len(self.indexes)

    def collate_fn(self, batch):
        tensors, targets = zip(*batch)
        tensors = torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True, padding_value=0.)
        targets = torch.tensor(targets)
        return tensors, targets
