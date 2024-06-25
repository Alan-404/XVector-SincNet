import warnings
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _pool(sequences: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Helper function to compute statistics pooling

    Assumes that weights are already interpolated to match the number of frames
    in sequences and that they encode the activation of only one speaker.

    Parameters
    ----------
    sequences : (batch, features, frames) torch.Tensor
        Sequences of features.
    weights : (batch, frames) torch.Tensor
        (Already interpolated) weights.

    Returns
    -------
    output : (batch, 2 * features) torch.Tensor
        Concatenation of mean and (unbiased) standard deviation.
    """

    weights = weights.unsqueeze(dim=1)
    # (batch, 1, frames)

    v1 = weights.sum(dim=2) + 1e-8
    mean = torch.sum(sequences * weights, dim=2) / v1

    dx2 = torch.square(sequences - mean.unsqueeze(2))
    v2 = torch.square(weights).sum(dim=2)

    var = torch.sum(dx2 * weights, dim=2) / (v1 - v2 / v1 + 1e-8)
    std = torch.sqrt(var)

    return torch.cat([mean, std], dim=1)


class StatsPool(nn.Module):
    def forward(
        self, sequences: torch.Tensor, weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if weights is None:
            mean = sequences.mean(dim=-1)
            std = sequences.std(dim=-1, correction=1)
            return torch.cat([mean, std], dim=-1)

        if weights.dim() == 2:
            has_speaker_dimension = False
            weights = weights.unsqueeze(dim=1)
            # (batch, frames) -> (batch, 1, frames)
        else:
            has_speaker_dimension = True

        # interpolate weights if needed
        _, _, num_frames = sequences.size()
        _, num_speakers, num_weights = weights.size()
        if num_frames != num_weights:
            warnings.warn(
                f"Mismatch between frames ({num_frames}) and weights ({num_weights}) numbers."
            )
            weights = F.interpolate(weights, size=num_frames, mode="nearest")

        output = torch.stack(
            [
                _pool(sequences, weights[:, speaker, :])
                for speaker in range(num_speakers)
            ],
            dim=1,
        )

        if not has_speaker_dimension:
            return output.squeeze(dim=1)

        return output