import h5py
import numpy as np

def load():
    """
    Loads all logos into an array.

    Returns:
        numpy.array: an array of shape (486377, 3, 32, 32)
    """
    hdf5_file = h5py.File('data/LLD-icon.hdf5', 'r')
    images = np.array(hdf5_file['data'][:]).transpose((0, 2, 3, 1))
    return images
