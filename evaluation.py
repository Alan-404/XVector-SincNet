import torch
from pytorch_metric_learning.losses import ArcFaceLoss

import numpy as np
from scipy.spatial.distance import cosine
from sklearn.metrics import roc_curve

from typing import Union

class XVectorSincNetCriterion:
    def __init__(self, n_speakers: int, embedding_size: int, margin: float = 28.6, scale: int = 64,) -> None:
        self.arcface_loss = ArcFaceLoss(n_speakers, embedding_size, margin, scale)

    def addictive_softmax_margin_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.arcface_loss(logits, labels)
    
class XVectorSincNetMetric:
    def __init__(self) -> None:
        pass

    def cosine_distance_score(self, preds: np.ndarray, labels: np.ndarray) -> Union[np.ndarray, float]:
        if preds.ndim == 2:
            distances = []
            for i in range(len(preds)):
                distances.append(
                    cosine(preds[i], labels[i])
                )
            return np.array(distances)
        else:
            return cosine(preds, labels)

    def cosine_similarity_score(self, preds: np.ndarray, labels: np.ndarray) -> Union[np.ndarray, float]:
        distance = self.cosine_distance_score(preds, labels)
        return 1 - distance
    
    def equal_error_rate_score(self, scores: np.ndarray, labels: np.ndarray, type_score: str = 'similarity'):
        if type_score == 'distance':
            scores = -scores

        fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=True)
        fnr = 1 - tpr

        if type_score == 'distance':
            thresholds = -thresholds

        eer_index = np.where(fpr > fnr)[0][0]
        eer = .25 * (fpr[eer_index - 1] + fpr[eer_index] + fnr[eer_index - 1] + fnr[eer_index])

        return fpr, fnr, thresholds, eer
