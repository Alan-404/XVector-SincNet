import os

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from .processing.processor import XVectorSincNetProcessor
import pandas as pd
from scipy.io import wavfile
import librosa
import numpy as np

MAX_AUDIO_VALUE = 32768.0

from typing import Optional, Tuple, List, Union

class XVectorSincNetDataset(Dataset):
    def __init__(self, manifest_path: str, processor: XVectorSincNetProcessor, training: bool = False, num_examples: Optional[int] = None) -> None:
        super().__init__()
        assert os.path.exists(manifest_path)

        self.prompts = pd.read_csv(manifest_path)
        if num_examples is not None:
            self.prompts = self.prompts[:num_examples]

        self.processor = processor
        self.training = training

    def __len__(self) -> int:
        return len(self.prompts)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor]:
        index_df = self.prompts.iloc[index]
        path = index_df['path']
        label = index_df['speaker']

        if self.training:
            return self.processor.load_audio(path), label
        else:   
            ref_path = index_df['ref_path']
            return self.processor.load_audio(path), label, self.processor.load_audio(ref_path)
    
class XVectorSincNetCollate:
    def __init__(self, processor: XVectorSincNetProcessor, training: bool = False) -> None:
        self.processor = processor
        self.training = training

    def __call__(self, batch: Tuple[List[torch.Tensor], List[str]]) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        if self.training:
            signals, speakers = zip(*batch)

            signals = self.processor(signals).unsqueeze(1)
            speakers = self.processor.as_target(speakers)

            return signals, speakers
        else:
            signals, speakers, ref_signals = zip(*batch)

            signals = self.processor(signals).unsqueeze(1)
            speakers = self.processor.as_target(speakers)
            ref_signals = self.processor(ref_signals)

            return signals, speakers, ref_signals
        
class InferenceDataset(Dataset):
    def __init__(self, manifest_path: str, sample_rate: int = 16000, num_examples: Optional[int] = None) -> None:
        super().__init__()
        self.prompts = pd.read_csv(manifest_path)
        if num_examples is not None:
            self.prompts = self.prompts[:num_examples]

        self.sample_rate = sample_rate
        
    def __len__(self) -> int:
        return len(self.prompts)
    
    def handle_audio(self, signal: np.ndarray, sr: int) -> np.ndarray:
        if sr != self.sample_rate:
            signal = librosa.resample(signal, orig_sr=sr, target_sr=self.sample_rate)
        return signal / MAX_AUDIO_VALUE
    
    def __getitem__(self, index: int):
        index_df = self.prompts.iloc[index]

        path1 = index_df['path1']
        path2 = index_df['path2']

        sr_1, audio_1 = wavfile.read(path1)
        sr_2, audio_2 = wavfile.read(path2)

        audio_1 = self.handle_audio(audio_1, sr_1)
        audio_2 = self.handle_audio(audio_2, sr_2)

        return torch.tensor(audio_1), torch.tensor(audio_2)

class InferenceCollate:
    def __init__(self) -> None:
        pass

    def padding(self, signals: List[torch.Tensor]) -> torch.Tensor:
        max_length = 0
        lengths = []

        for signal in signals:
            length = len(signal)
            lengths.append(length)
            if length > max_length:
                max_length = length
        
        padded_signals = []
        for index, signal in enumerate(signals):
            padded_signals.append(
                F.pad(signal, (0, max_length - lengths[index]), value=0.0)
            )

        return torch.stack(padded_signals)

    def __call__(self, batch: Tuple[List[torch.Tensor], List[torch.Tensor]]):
        audio_1, audio_2 = zip(*batch)
        signals = self.padding(audio_1 + audio_2)
        return signals