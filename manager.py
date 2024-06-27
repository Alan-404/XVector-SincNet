import os
import shutil

import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from typing import Optional, Tuple

class CheckpointManager:
    def __init__(self, saved_folder: Optional[str] = None, n_saved: int = 3) -> None:
        self.saved_folder = saved_folder
        if os.path.exists(saved_folder):
            os.makedirs(saved_folder)

        self.n_saved = n_saved
        self.saved_checkpoints = []

    def save_checkpoint(self, model: Module, optimizer: Optimizer, scheduler: LRScheduler, n_steps: int, n_epochs: int) -> None:
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'n_steps': n_steps,
            'n_epochs': n_epochs
        }

        if len(self.saved_checkpoints) == self.n_saved:
            os.remove(f"{self.saved_folder}/{self.saved_checkpoints[0]}.pt")
            self.saved_checkpoints.pop(0)

        torch.save(checkpoint, f"{self.saved_folder}/{n_steps}.pt")
        self.saved_checkpoints.append(n_steps)

    def load_checkpoint(self, path: str, model: Module, optimizer: Optimizer, scheduler: LRScheduler) -> Tuple[int, int]:
        checkpoint = torch.load(path, map_location='cpu')

        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])

        n_steps = checkpoint['n_steps']
        n_epochs = checkpoint['n_epochs']

        return n_steps, n_epochs