from torch.utils.data import Dataset
import torch
from pathlib import Path
from maestro.util import tqdm
import json

class LaionClapDataset(Dataset):
    def __init__(self, features_dir, *, json_path=None, preload=False):
        self.features_dir = Path(features_dir)


        if json_path is None:
            json_path = self.features_dir / 'data.json'

        with open(json_path, 'r') as f:
            self.data = json.load(f)
        
        self.preload = preload
        
        if self.preload:
            self._preload()

    def __len__(self):
        return len(self.data)
    
    def _load_features(self, filename):
        return torch.load((self.features_dir / filename).with_suffix('.pt'))
    
    def __getitem__(self, idx):

        data = self.data[idx].copy()
        
        if self.preload:
            features = self.features[idx]
        else:
            features = self._load_features(data['filename'])
        
        data['features'] = features

        return data

    def _preload(self):
        pbar = tqdm(self.data, desc='Preloading features')
        self.features = [self._load_features(item['filename']) for item in pbar]
    
    