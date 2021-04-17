import numpy as np

# Data
number_of_trakcs = 5  # number of tracks
lowest_pitch = 24  # lowest pitch or musciala note
is_drums = [True, False, False, False, False] 
track_names = ['Drums', 'Piano', 'Guitar', 'Bass', 'Strings']
tempo = 100
number_of_measures = 4  # measures per sample
samples_per_song = 8  # number of samples to extract from the datset
beat = 4  # temporal resolution
programs = [0, 0, 25, 33, 48]  # program number for each track, check pypianoroll
number_of_pitches = 72  # number of pitches or musical notes

# Sampling
interval = 5000  # interval where a result will be saved
number_of_samples = 4

# Training
batch = 16
latent_dim = 128
number_steps = 1000000

measure_resolution = 4 * beat
tempo_array = np.full((4 * 4 * measure_resolution, 1), tempo)
assert 24 % beat == 0, (
    "beat must be a factor of 24 "
)
assert len(programs) == len(is_drums) and len(programs) == len(track_names), (
    "is_drums and track_names must be the same."
)    