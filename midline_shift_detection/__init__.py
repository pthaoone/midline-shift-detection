import os

# Fix for Windows compatibility in imops (imops uses os.sched_getaffinity which is Unix-only)
if not hasattr(os, 'sched_getaffinity'):
    os.sched_getaffinity = lambda pid=0: set(range(os.cpu_count() or 1))

from .data import gather_nifty, load_pair, gather_train, normalize_image
from .model import Network
from .predict import rescale_nii, crop_background
from .training import random_flip, get_random_slice, combiner, train_step

