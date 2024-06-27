import os

import torch
from torch.utils.data import Dataset

from .processing.processor import XVectorSincNetProcessor
import pandas as pd

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