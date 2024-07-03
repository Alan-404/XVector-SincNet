import os

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from model.xvector_sincnet import XVectorSincNet

from dataset import InferenceCollate, InferenceDataset

from tqdm import tqdm
from typing import Optional
import fire

def test(test_path: str,
         checkpoint: str,
         threshold: float = 0.55,
         batch_size: int = 1,
         num_samples: Optional[int] = None):
    model = XVectorSincNet()

    model.load_state_dict(torch.load(checkpoint, map_location='cpu'))
    model.to('cuda')

    collate_fn = InferenceCollate()

    dataset = InferenceDataset(test_path, num_examples=num_samples)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    labels = dataset.prompts['label'].to_list()
    preds = []

    for signals in tqdm(dataloader):
        signals = signals.to('cuda')
        outputs = model(signals)
        audio_1, audio_2 = outputs.split(batch_size, dim=0)

        similarity = torch.cosine_similarity(audio_1, audio_2, dim=1)

        similarity = (similarity >= threshold).type(torch.bool)
        preds += similarity.cpu().numpy().tolist()

    df = dataset.prompts
    df['pred'] = preds
    
    score = accuracy_score(labels, preds)
    print(f"Accuracy Score: {score * 100}")
    df.to_csv('./data/result.csv')

if __name__ == '__main__':
    fire.Fire(test)