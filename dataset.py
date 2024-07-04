import os

import torch
from torch.utils.data import Dataset
import pandas as pd

from scipy.io import wavfile
import numpy as np
import librosa

from typing import Optional, List, Tuple

MAX_AUDIO_VALUE = 32768.0

class XVectorSincNetDataset(Dataset):
    def __init__(self, manifest_path: str, sample_rate: int = 16000, max_duration: Optional[int] = None, num_examples: Optional[int] = None) -> None:
        super().__init__()
        assert os.path.exists(manifest_path)

        self.prompts = pd.read_csv(manifest_path)
        if num_examples is not None:
            self.prompts = self.prompts[:num_examples]
        
        self.max_length = None
        if max_duration is not None:
            self.max_length = max_duration * sample_rate
        
        assert isinstance(self.prompts['speaker'].dtype, int)
        self.num_speakers = self.get_num_speakers()

        self.sample_rate = sample_rate

    def __len__(self) -> int:
        return len(self.prompts)
    
    def get_num_speakers(self) -> int:
        return len(list(set(self.prompts['speaker'].to_list())))
    
    def load_audio(self, path: str) -> np.ndarray:
        sr, signal = wavfile.read(path)
        signal = signal / MAX_AUDIO_VALUE
        if sr != self.sample_rate:
            signal = librosa.resample(signal, orig_sr=sr, target_sr=self.sample_rate)
        if self.max_length is not None:
            signal = signal[:self.max_length]
        return signal
    
    def __getitem__(self, index: int):
        index_df = self.prompts.iloc[index]

        path = index_df['path']
        sid = index_df['speaker']

        signal = torch.FloatTensor(self.load_audio(path))

        return signal, sid
    
class XVectorSincNetCollate:
    def __init__(self) -> None:
        pass

    def __call__(self, batch: Tuple[List[torch.Tensor], List[int]]) -> Tuple[torch.Tensor, torch.Tensor]:
        signals, ids = zip(*batch)
        signals = torch.nn.utils.rnn.pad_sequence(signals, batch_first=True, padding_value=0.0)
        ids = torch.tensor(ids)
        return signals, ids