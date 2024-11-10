from torchaudio.transforms import Resample
import torchaudio
from pathlib import Path
from multiprocessing import Pool
import numpy as np
import torch
from transformers import ClapProcessor
from maestro.util import tqdm
import json
from maestro.constants.gtzan import ordered_genres, genre_to_label

ORDERED_GENRES = ordered_genres()

# Helper class for schema consistency
# only used in preprocessing, flattened to dict before saving

class LaionClapGTZANData:
    def __init__(self, filename, genre, source):
        self.filename = filename
        self.genre = genre
        self.source = source
        self.label = genre_to_label(genre)
    
    def to_dict(self):
        return {
            'filename': self.filename,
            'genre': self.genre,
            'source': self.source,
            'label': self.label
        }


# General for all workers
def _init_worker(target_sample_rate_, resample_only_):
    global resamplers, target_sample_rate, resample_only
    target_sample_rate = target_sample_rate_
    resample_only = resample_only_
    resamplers = {}

    if not resample_only:
        global processor
        processor = ClapProcessor.from_pretrained("laion/larger_clap_music_and_speech")

def _worker(args):
    # Get the input and output files
    in_file, out_file = args

    # Get worker processor and resampler    
    global processor, resamplers, target_sample_rate
    
    # Load the audio file
    try:
        waveform, orig_rate = torchaudio.load(in_file)
    except Exception as e:
        print(f"Error loading {in_file}: {e}")
        return False
    
    # Get the resampler for this sample rate
    if orig_rate not in resamplers:
        resamplers[orig_rate] = Resample(orig_rate, target_sample_rate)
    resampler = resamplers[orig_rate]

    # Resample the audio file
    waveform = resampler(waveform)
    
    # Convert to mono if necessary
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Now, convert to NumPy
    audios_np = [waveform.squeeze(0).numpy()]

    if resample_only:
        # Save the resampled audio file
        torch.save(audios_np, out_file)
    else:
        # Apply the processor
        input_features = processor(
            audios = audios_np,
            sampling_rate=target_sample_rate,
            return_tensors="pt",
            padding="repeatpad",
            truncation="rand_trunc"
        )['input_features']

        # Save the processed audio file
        torch.save(input_features, out_file)

    return True

def _preprocess_audio_files(data_path, out_path, *, target_sample_rate, num_workers=1, resample_only=False):

    # Get all wav files in the directory
    path = Path(data_path)
    files = path.glob('*.wav')

    # Create the output directory if it doesn't exist
    out_path = Path(out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    # Create pairs of in and out files
    in_outs = [(f, out_path / f"{f.stem}.pt") for f in files]

    total_errors = 0

    # Initialize the worker pool
    with Pool(num_workers, initializer=_init_worker, initargs=(target_sample_rate, resample_only)) as pool:
        for success in tqdm(pool.imap_unordered(_worker, in_outs, chunksize=25), desc="Preprocessing", total=len(in_outs)):
            if not success:
                total_errors += 1

    return total_errors

#GTZAN preprocessing
# Expects input formatted from kaggle GTZAN dataset
def preprocess_gtzan(input_dir, output_dir, target_sample_rate, num_workers=1, resample_only=False):
    input_path = Path(input_dir)

    genres_path = input_path / 'genres_original'

    data = []

    # go thru all genre folders:
    for genre in ORDERED_GENRES:
        genre_path = genres_path / genre
        
        # Accumulate data
        for file in genre_path.glob('*.wav'):
            data.append(LaionClapGTZANData(filename=file.name, genre=genre, source='gtzan').to_dict())
        
        
        print(f"Processing {genre}")
        # Actually apply processing
        num_errors = _preprocess_audio_files(genre_path, output_dir, target_sample_rate=target_sample_rate, num_workers=num_workers, resample_only=resample_only)

    # Save the data
    with open(Path(output_dir) / 'data.json', 'w') as f:
        json.dump(data, f)

    print(f"Finished processing GTZAN dataset with {num_errors} errors")
    if num_errors > 0:
        print("These errors were not accounted for in the data.json file. You should remove the corresponding records from the JSON manually.")

# Synth preprocessing
def preprocess_synth(data_path, out_path, target_sample_rate, num_workers=1, resample_only=False):
    num_errors = _preprocess_audio_files(data_path, out_path, target_sample_rate=target_sample_rate, num_workers=num_workers, resample_only=resample_only)

    print(f"Finished processing synth dataset with {num_errors} errors")
    if num_errors > 0:
        print("These errors were not accounted for in the data.json file. You should remove the corresponding records from the JSON manually.")