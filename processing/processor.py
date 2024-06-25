import os

import torch
import torch.nn.functional as F
import numpy as np
import json
import librosa
from typing import Optional, List, Dict

class XVectorSincNetProcessor:
    def __init__(self,
                 sampling_rate: int = 16000,
                 speaker_dict: Optional[Dict[str, int]] = None,
                 speaker_list: Optional[List[str]] = None,
                 speaker_path: Optional[str] = None
                ) -> None:
        self.sampling_rate = sampling_rate

        if speaker_dict is not None:
            self.speaker_dict = speaker_dict
        elif speaker_list is not None:
            self.speaker_dict = self.create_dict_by_list(speaker_list)
        elif speaker_path is not None:
            self.speaker_dict = self.load_speaker(speaker_path)

        assert max(self.speaker_dict.values()) == len(self.speaker_dict) - 1, "Invalid Speaker Config"
        self.speaker_list = list(self.speaker_dict.keys())

    def get_speaker_by_id(self, id: int):
        return self.speaker_list[id]

    def get_id_speaker(self, speaker: str):
        return self.speaker_list.index(speaker)
    
    def create_dict_by_list(self, speakers: List[str]):
        count = 0
        dictionary = dict()
        for speaker in speakers:
            dictionary[speaker] = count
            count += 1
        return dictionary
    
    def load_speaker(self, path: str):
        assert os.path.exists(path), "File is Not Found"
        assert os.path.isfile(path) and path.lower().endswith(".json"), "Invalid Configuration Speaker File, We need JSON file"

        with open(path, 'r', encoding='utf8') as file:
            return json.load(file)

    def gaussian_normalize(self, signal: torch.Tensor):
        return (signal - signal.mean()) / torch.sqrt(signal.var() + 1e-7)

    def load_audio(self, path: str):
        signal, _ = librosa.load(path, sr=self.sampling_rate)
        signal = torch.tensor(signal)
        return signal
    
    def get_num_speakers(self):
        return len(self.speaker_list)
    
    def as_target(self, speakers: List[str]) -> torch.Tensor:
        items = []
        for speaker in speakers:
            items.append(self.get_id_speaker(speaker))
        
        return torch.tensor(items)
    
    def __call__(self, signals: List[torch.Tensor]) -> torch.Tensor:
        max_length = torch.max([len(item) for item in signals])

        padded_signals = []
        for item in signals:
            padded_signals.append(
                F.pad(item, pad=(0, max_length - len(item)), mode='constant', value=0.)
            )

        return torch.stack(padded_signals)