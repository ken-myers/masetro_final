from audiomentations import Compose, PitchShift, PolarityInversion, AddGaussianNoise, TimeStretch, RepeatPart, TimeMask, HighPassFilter, LowPassFilter, OneOf, GainTransition
from pathlib import Path
import torch
import json
from multiprocessing import Pool
from maestro.util import tqdm
from transformers import ClapProcessor


#TODO: Generalize this a bit more
def get_default_augs():
    return Compose([
        PitchShift(min_semitones=-3, max_semitones=3),
        PolarityInversion(),
        AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.01),
        TimeStretch(min_rate=0.9, max_rate=1.1),
        RepeatPart(p=0.05),
        TimeMask(p=0.05),
        OneOf([
            HighPassFilter(),
            LowPassFilter()
        ], p = 0.1),
        GainTransition(p=0.11)
    ])




def _init_augment_worker(sample_rate_):
    global augment, sample_rate

    global seen_sizes
    seen_sizes = set()

    global processor
    processor = ClapProcessor.from_pretrained("laion/larger_clap_music_and_speech")

    sample_rate = sample_rate_
    augment = get_default_augs()

def _augment_worker(args):
    global augment, sample_rate, processor

    global seen_sizes

    in_file, out_file = args
    
    waveform = torch.load(in_file).numpy()
    new_waveform = augment(waveform, sample_rate=sample_rate)


    #TODO: somehow make this more modular so that this code isnt duplicated
    input_features = processor(audios=[new_waveform], 
                               sampling_rate=sample_rate, 
                               return_tensors="pt", 
                               padding='repeatpad', 
                               truncation='rand_trunc',
                               max_length_s=10)['input_features'].squeeze(0)


    torch.save(input_features, out_file)

# TODO: keep track of augmentations applied with monkey patch
def augment_audio(data_path, *, sample_rate, length=None, aug_factor=None):
    if length is None and aug_factor is None:
        raise ValueError("Either length or aug_factor must be provided")
    if length is not None and aug_factor is not None:
        raise ValueError("Only one of length or aug_factor must be provided")
    

    data_path = Path(data_path)

    # create augmented subfolder
    output_dir = data_path / "augmented"
    output_dir.mkdir(exist_ok=True)


    input_files = list(data_path.glob("*.pt"))

    if length is None:
        length = len(input_files) * aug_factor

    in_outs = []
    aug_counts = {}
    for i in range(length - len(input_files)):
        file_index = i % len(input_files)
        aug_number = i // len(input_files)
        aug_counts[file_index] = aug_counts.get(file_index, 0) + 1

        input_file = input_files[file_index]
        output_file = output_dir / f"{input_file.stem}_aug_{aug_number}.pt"
        in_outs.append((input_file, output_file))


    #Update the metadata
    json_path = data_path / "data.json"
    with json_path.open("r") as f:
        data = json.load(f)

    for i, (input_file, output_file) in tqdm(enumerate(in_outs), desc="Updating metadata", total=len(in_outs)):
        input_stem = input_file.stem
        original = next((x for x in data if x["filename"] == input_stem), None)
        if original is None:
            raise ValueError("Original file not found in metadata")
        original = original.copy()
        original["filename"] = f"augmented/{output_file.stem}"

        data.append(original)
   
    with Pool(initializer=_init_augment_worker, initargs=(sample_rate,)) as p:
        for _ in tqdm(p.imap_unordered(_augment_worker, in_outs, chunksize=25), desc="Augmenting", total=len(in_outs)):
            pass

    with json_path.open("w") as f:
            json.dump(data, f)

    print("Finished augmenting")