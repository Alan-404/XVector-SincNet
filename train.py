import os
import shutil

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import GradScaler, autocast
import torch.multiprocessing as mp

from .processing.processor import XVectorSincNetProcessor
from .model.xvector_sincnet import XVectorSincNet
from .evaluation import XVectorSincNetCriterion
from .dataset import XVectorSincNetDataset, XVectorSincNetCollate
from .manager import CheckpointManager

from tqdm import tqdm
import statistics
from typing import Optional
import fire

import wandb

def setup(rank: int, world_size: int):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = 12355
    dist.init_process_group('nccl', world_size=world_size, rank=rank)
    print(f"Initialize {rank + 1} / {world_size}")

def cleanup():
    dist.destroy_process_group()

def train(rank: int,
          world_size: int,
          # Train Config
          train_path: str,
          train_batch_size: int = 1,
          num_train_samples: Optional[int] = None,
          num_epochs: int = 1,
          fp16: bool = True,
          lr: float = 9e-4,
          set_lr: bool = False,
          # Validation Config
          val_path: Optional[str] = None,
          val_batch_size: int = 1,
          num_val_samples: Optional[int] = None,
          # Checkpoint Config
          checkpoint: Optional[str] = None,
          saved_folder: str = "./checkpoints",
          n_saved_checkpoints: int = 3,
          save_checkpoints_after_epoch: int = 3,
          # Processor Config
          sampling_rate: int = 16000,
          speaker_config_path: Optional[str] = None,
          # Model Config
          embedding_dim: int = 512,
          # (Optional) Pre-trained model
          weight_path: Optional[str] = None,
          # Logging config
          logging: bool = False,
          logging_project: str = "XVector_SincNet",
          logging_name: Optional[str] = None
        ) -> None:
    if world_size > 1:
        setup(rank, world_size)
    if rank == 0:
        if logging:
            wandb.init(
                project=logging_project,
                name=logging_name
            )

    processor = XVectorSincNetProcessor(sampling_rate=sampling_rate)
    model = XVectorSincNet(sample_rate=sampling_rate, dimension=embedding_dim).to(rank)

    if weight_path is not None and os.path.exists(weight_path):
        model.load_state_dict(torch.load(weight_path, map_location='cpu'))

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=1000)

    checkpoint_manager = CheckpointManager(saved_folder, n_saved_checkpoints)
    if checkpoint is not None and os.path.exists(checkpoint):
        n_steps, n_epochs = checkpoint_manager.load_checkpoint(checkpoint, model, optimizer, scheduler)

    if set_lr:
        optimizer.param_groups[0]['lr'] = lr

    if world_size > 1:
        model = DDP(model, device_ids=[rank])

    train_dataset = XVectorSincNetDataset(train_path, num_examples=num_train_samples)
    train_sampler = DistributedSampler(dataset=train_dataset, num_replicas=world_size, rank=rank) if world_size > 1 else RandomSampler(train_dataset)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=train_batch_size, sampler=train_sampler, collate_fn=XVectorSincNetCollate(processor=processor, training=True))

    is_validation = val_path is not None and os.path.exists(val_path)
    if is_validation:
        val_dataset = XVectorSincNetDataset(val_path, um_examples=num_val_samples)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank) if world_size > 1 else None
        val_dataloader = DataLoader(dataset=val_dataset, batch_size=val_batch_size, shuffle=False, sampler=val_sampler, collate_fn=XVectorSincNetCollate(processor=processor))

    criterion = XVectorSincNetCriterion(n_speakers=processor.get_num_speakers(), embedding_size=embedding_dim)
    scaler = GradScaler(enabled=fp16)

    for epoch in range(num_epochs):
        if rank == 0:
            print(f"Epoch {epoch + 1}")
        
        train_losses = []

        model.train()
        for (x, y) in tqdm(train_dataloader, leave=False):
            with autocast(enabled=fp16):
                outputs = model(x)
            loss = criterion.addictive_softmax_margin_loss(outputs, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            scaler.step(optimizer)

            scaler.update()

            train_losses.append(loss.item())
            n_steps += 1
        
        if is_validation:
            val_losses = []

            model.eval()
            print("Validating")
            for (x, y) in tqdm(val_dataloader, leave=False):
                with torch.no_grad():
                    with autocast(enabled=fp16):
                        outputs = model(x)
                loss = criterion.addictive_softmax_margin_loss(outputs, y).item()
                val_losses.append(loss)
                

        n_epochs += 1
        scheduler.step()

        if rank == 0:
            train_loss = statistics.mean(train_losses)
            current_lr = optimizer.param_groups[0]['lr']

            print(f'Train Loss: {(train_loss):.4f}')

            if is_validation:
                val_loss = statistics.mean(val_losses)
                print(f"Val Loss: {(val_loss):.4f}")
                wandb.log({'val_loss': val_loss}, n_steps)

            if logging:
                wandb.log({
                    'train_loss': train_loss,
                    'learning_rate': current_lr
                }, n_steps)
                

            print(f"Current Learning Rate: {(current_lr)}")
            print("\n")

            if epoch % save_checkpoints_after_epoch == save_checkpoints_after_epoch - 1 or epoch == num_epochs - 1:
                checkpoint_manager.save_checkpoint(model, optimizer, scheduler, n_steps, n_epochs)

    if world_size > 1:
        cleanup()

def main( # Train Config
          train_path: str,
          train_batch_size: int = 1,
          num_train_samples: Optional[int] = None,
          n_epochs: int = 1,
          fp16: bool = True,
          lr: float = 9e-4,
          set_lr: bool = False,
          # Validation Config
          val_path: Optional[str] = None,
          val_batch_size: int = 1,
          num_val_samples: Optional[int] = None,
          # Checkpoint Config
          checkpoint: Optional[str] = None,
          saved_folder: str = "./checkpoints",
          n_saved_checkpoints: int = 3,
          save_checkpoints_after_epoch: int = 3,
          # Processor Config
          sampling_rate: int = 16000,
          speaker_config_path: Optional[str] = None,
          # Model Config
          embedding_dim: int = 512,
          # 
          weight_path: Optional[str] = None,
          # Logging config
          logging: bool = False,
          logging_project: str = "XVector-SincNet",
          logging_name: Optional[str] = None) -> None:
    
    n_gpus = torch.cuda.device_count()
    
    if n_gpus <= 1:
        train(
            0, n_gpus, train_path, train_batch_size, num_train_samples, n_epochs, bool(fp16), lr, bool(set_lr),
            val_path, val_batch_size, num_val_samples,
            checkpoint, saved_folder, n_saved_checkpoints, save_checkpoints_after_epoch,
            sampling_rate, speaker_config_path, embedding_dim,
            weight_path,
            bool(logging), logging_project, logging_name
        )
    else:
        mp.spawn(
            train, 
            args=(
                n_gpus, train_path, train_batch_size, num_train_samples, n_epochs, bool(fp16), lr, bool(set_lr),
                val_path, val_batch_size, num_val_samples,
                checkpoint, saved_folder, n_saved_checkpoints, save_checkpoints_after_epoch,
                sampling_rate, speaker_config_path, embedding_dim,
                weight_path,
                bool(logging), logging_project, logging_name
            ),
            nprocs=n_gpus,
            join=True
        )

if __name__ == '__main__':
    fire.Fire(main)