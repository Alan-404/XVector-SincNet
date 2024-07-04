
import torch
from scipy.io import wavfile
import librosa
import numpy as np

from typing import Union, List

MAX_AUDIO_VALUE = 32768.0

class XVectorSincNetProcessor:
    def __init__(self, sample_rate: int) -> None:
        self.sample_rate = sample_rate

    def load_audio(self, path: str, return_tensor: bool = False) -> Union[np.ndarray, torch.Tensor]:
        sr, signal = wavfile.read(path)
        signal = signal / MAX_AUDIO_VALUE
        if sr != self.sample_rate:
            signal = librosa.resample(signal, orig_sr=sr, target_sr=self.sample_rate)
        return signal if return_tensor == False else torch.tensor(signal)
    
    def pad_signals(self, signals: Union[List[np.ndarray], List[torch.Tensor]]) -> Union[np.ndarray, torch.Tensor]:
        max_length = 0
        lengths = []

        is_tensor = None

        for signal in signals:
            if is_tensor is None:
                if isinstance(signal, torch.Tensor):
                    is_tensor = True
                else:
                    is_tensor = False
            length = len(signal)
            if length > max_length:
                max_length = length
            lengths.append(length)

        padded_signals = []
        for index, signal in enumerate(signals):
            if is_tensor:
                padded_item = torch.nn.functional.pad(
                    signal, (0, max_length - lengths[index]), value=0.0
                )
            else:
                padded_item = np.pad(
                    signal, (0, max_length - lengths[index]), constant_values=0.0
                )

            padded_signals.append(padded_item)

        if is_tensor:
            return torch.stack(padded_signals)
        return np.arange(padded_signals)