import os
import os.path
import random
from pathlib import Path
from tqdm import tqdm
import pypianoroll
import torch
from settings import *


dataset_root = Path("data/lpd_5/lpd_5_cleansed/")
id_list = []
for path in os.listdir("data/amg"):
    filepath = os.path.join("data/amg", path)
    if os.path.isfile(filepath):
        with open(filepath) as f:
            id_list.extend([line.rstrip() for line in f])
id_list = list(set(id_list))

def msd_id_to_dirs(msd_id):
    """Given an MSD ID, generate the path prefix.
    E.g. TRABCD12345678 -> A/B/C/TRABCD12345678"""
    return os.path.join(msd_id[2], msd_id[3], msd_id[4], msd_id)

def load_data():
	data = []
	# Iterate over all the songs in the ID list
	for msd_id in tqdm(id_list):
	    # Load the multitrack as a pypianoroll.Multitrack instance
	    song_dir = dataset_root / msd_id_to_dirs(msd_id)
	    multitrack = pypianoroll.load(song_dir / os.listdir(song_dir)[0])
	    # Binarize the pianorolls
	    multitrack.binarize()
	    # Downsample the pianorolls (shape: n_timesteps x n_pitches)
	    multitrack.set_resolution(beat_resolution)
	    # Stack the pianoroll (shape: n_tracks x n_timesteps x n_pitches)
	    pianoroll = (multitrack.stack() > 0)
	    # Get the target pitch range only
	    pianoroll = pianoroll[:, :, lowest_pitch:lowest_pitch + n_pitches]
	    # Calculate the total measures
	    n_total_measures = multitrack.get_max_length() // measure_resolution
	    candidate = n_total_measures - n_measures
	    target_n_samples = min(n_total_measures // n_measures, n_samples_per_song)
	    # Randomly select a number of phrases from the multitrack pianoroll
	    for idx in np.random.choice(candidate, target_n_samples, False):
	        start = idx * measure_resolution
	        end = (idx + n_measures) * measure_resolution
	        # Skip the samples where some track(s) has too few notes
	        if (pianoroll.sum(axis=(1, 2)) < 10).any():
	            continue
	        data.append(pianoroll[:, start:end])
	# Stack all the collected pianoroll segments into one big array
	random.shuffle(data)
	data = np.stack(data)
	print(f"Successfully collect {len(data)} samples from {len(id_list)} songs")
	print(f"Data shape : {data.shape}")

	data = torch.as_tensor(data, dtype=torch.float32)
	dataset = torch.utils.data.TensorDataset(data)
	return torch.utils.data.DataLoader(
	    dataset, batch_size=batch_size, drop_last=True, shuffle=True)