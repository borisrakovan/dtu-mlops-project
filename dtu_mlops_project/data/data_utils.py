import torch
import torchaudio
from torch import nn


# amplitude/power to db conversion
class AmplitudeToDB(nn.Module):
    def __init__(self, power=1.0, ref=1.0, amin=1e-5, top_db=80.0):
        super().__init__()
        self.power = torch.tensor(power)
        self.ref = torch.tensor(ref) ** self.power
        self.amin = torch.tensor(amin) ** self.power
        self.top_db = torch.tensor(top_db)

    def forward(self, x):
        pow = torch.pow(torch.abs(x), self.power)
        log_spec = 10.0 * torch.log10(torch.maximum(self.amin, pow))
        log_spec -= 10.0 * torch.log10(torch.maximum(self.amin, self.ref))
        if self.top_db is not None:
            log_spec = torch.maximum(log_spec, log_spec.max() - self.top_db)
        return log_spec


# pitch shift
class PitchShift(nn.Module):
    def __init__(self, sample_rate, n_steps, bins_per_octave=12, **kwargs):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_steps = n_steps
        self.bins_per_octave = bins_per_octave
        self.kwargs = kwargs

    def forward(self, waveform):
        return torchaudio.functional.pitch_shift(
            waveform, self.sample_rate, self.n_steps, self.bins_per_octave, **self.kwargs)


# random time stretch
class RandomTimeStretch(nn.Module):
    def __init__(self, rate_range, return_complex, **kwargs):
        super().__init__()
        self.rate_min, self.rate_max = rate_range
        self.return_complex = return_complex
        self.time_stretch = torchaudio.transforms.TimeStretch(
            fixed_rate=None,
            **kwargs,
        )

    def forward(self, spec):
        rate = torch.rand(1).item() * (self.rate_max - self.rate_min) + self.rate_min
        out = self.time_stretch(spec, rate)
        out = out if self.return_complex else torch.abs(out)
        return out
