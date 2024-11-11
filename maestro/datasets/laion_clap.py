from torch.utils.data import Dataset
import torch
from pathlib import Path
from maestro.util import tqdm
import json

class LaionClapDataset(Dataset):
    def __init__(self, audio_dir, *, json_path=None, preload=False, waveforms=False):
        self.audio_dir = Path(audio_dir)
        self.waveforms = waveforms

        if json_path is None:
            json_path = self.audio_dir / 'data.json'

        with open(json_path, 'r') as f:
            self.data = json.load(f)
        
        self.preload = preload
        
        if self.preload:
            self._preload()

    def __len__(self):
        return len(self.data)
    
    def _load_audio(self, filename):
        return torch.load((self.audio_dir / (filename + ".pt") ))
    
    def __getitem__(self, idx):

        data = self.data[idx].copy()

        if self.waveforms:
            key = 'waveform'
        else:
            key = 'features'
    

        if self.preload:
            audio = self.audios[idx]
        else:
            audio = self._load_audio(data['filename'])
        
        data[key] = audio

        return data

    def _preload(self):
        pbar = tqdm(self.data, desc='Preloading features')
        self.audios = [self._load_audio(item['filename']) for item in pbar]