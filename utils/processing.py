import numpy as np
import torch

def preprocess_numpy(Data, dim=256, low=0.1, hi=1.0):
    """Normalize and rescale (and optionally invert) images.

    Parameters
    ----------
    Data : hdf5
        Data array.
    dim : integer, optional
        Dimensions of images, assumes square.
    low : float, optional
        Minimum rescale value. Default is 0.1 since background pixels are 0.
    hi : float, optional
        Maximum rescale value.
    """
    for key in Data:
        Data[key][0] = Data[key][0].reshape(len(Data[key][0]), dim, dim, 1)
        for i, img in enumerate(Data[key][0]):
            img = img / 255.

            minn, maxx = np.min(img[img > 0]), np.max(img[img > 0])
            img[img > 0] = low + (img[img > 0] - minn) * (hi - low) / (maxx - minn)
            Data[key][0][i] = img

def preprocess(Data, dim=256, low=0.1, hi=1.0):
    """Normalize and rescale (and optionally invert) images.

    Parameters
    ----------
    Data : hdf5
        Data array.
    dim : integer, optional
        Dimensions of images, assumes square.
    low : float, optional
        Minimum rescale value. Default is 0.1 since background pixels are 0.
    hi : float, optional
        Maximum rescale value.
    """
    for key in Data:
        Data[key][0] = Data[key][0].reshape(len(Data[key][0]), dim, dim, 1)
        for i, img in enumerate(Data[key][0]):
            img = img / 255.  

            
            mask = img > 0
            if torch.any(mask):
                masked_img = img[mask]
                minn, maxx = torch.min(masked_img), torch.max(masked_img)
                img[mask] = low + (img[mask] - minn) * (hi - low) / (maxx - minn)
                Data[key][0][i] = img.to(dtype=torch.float32)
            else:
                raise ValueError(f"No positive values in the image array for key {key} at index {i}")


def get_id(i, zeropad=5):
    """Properly indexes hdf5 files.

    Parameters
    ----------
    i : int
        Image number to be indexed.
    zeropad : integer, optional
        Number of zeros to pad string.

    Returns
    -------
    String of hdf5 index.
    """
    return 'img_{i:0{zp}d}'.format(i=i, zp=zeropad)
