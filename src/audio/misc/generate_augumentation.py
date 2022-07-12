from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift


from scipy import signal
from scipy.io import wavfile

from tqdm import tqdm
import numpy as np
import os
import sys

import librosa

sys.path.append('constants/')
from path import DATASET, AUGUMENTATION_PATH

classes = ['Bark','Crash','Door','Doorbell','Drill','Other','Speech']

augment = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
    TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
    PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
    Shift(min_fraction=-0.5, max_fraction=0.5, p=0.5),
])

def convert_float_samples_to_int16(y):
    """
    Convert floating-point numpy array of audio samples to int16.
    :param y:
    :param clamp_values: Clip extreme values to the range [-1.0, 1.0]. This can be done to avoid
        integer overflow or underflow, which results in wrap distortion, which sounds worse than
        clipping distortion.
    :return:
    """
    if not issubclass(y.dtype.type, np.floating):
        raise ValueError("input samples not floating-point")

    return (y * np.iinfo(np.int16).max).astype(np.int16)

if not os.path.exists(AUGUMENTATION_PATH):
    os.mkdir(AUGUMENTATION_PATH)

for c in classes:

    print("Class: {}".format(c))

    folder_path = AUGUMENTATION_PATH + c

    if not os.path.exists(folder_path):
      os.mkdir(folder_path)

    for filename in tqdm(os.listdir(DATASET + c)):
        audio, rate = librosa.load(DATASET + c + '/' + filename)
        augmented_samples = augment(samples=audio, sample_rate=rate)
        augumented_int = convert_float_samples_to_int16(augmented_samples)
        wavfile.write(folder_path + '/' + filename, rate, augumented_int)

