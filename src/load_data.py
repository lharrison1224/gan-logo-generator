import h5py


def load():
    """
    Loads all logos into an array.

    Returns:
        numpy.array: an array of shape (486377, 3, 32, 32)
    """
    hdf5_file = h5py.File('data/LLD-icon.hdf5', 'r')
    images = hdf5_file['data'][:]
    return images
