"""Run Convolutional Neural Network Training

Execute the training of a (UNET) Convolutional Neural Network on
images of the Moon and binary ring targets.
"""


import model_train_transform_v2 as mt
import torch
# torch.cuda.empty_cache()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

print(device) #check if cuda is being used

# Model Parameters
MP = {}

# Directory of train/dev/test image and crater hdf5 files.
MP['dir'] = 'catalogues/'

# Image width/height, assuming square images.
MP['dim'] = 256

# Batch size: smaller values = less memory but less accurate gradient estimate
MP['bs'] = 8

# Number of training epochs.
MP['epochs'] = 4

# Number of train/valid/test samples, needs to be a multiple of batch size.
MP['n_train'] = 30000
MP['n_dev'] = 5000
MP['n_test'] = 5000


# Save model (binary flag) and directory.
MP['save_models'] = 1
MP['save_dir'] = 'models/model_default_params.pt'
MP['save_dir_2'] = 'models/model_default_params.pth'

# Model Parameters (to potentially iterate over, keep in lists).
MP['N_runs'] = 1                # Number of runs
MP['filter_length'] = [3]       # Filter length
MP['lr'] = [0.0001]             # Learning rate
MP['n_filters'] = [112]         # Number of filters
MP['init'] = ['he_normal']      # Weight initialization
MP['lambda'] = [1e-6]           # Weight regularization
MP['dropout'] = [0.15]          # Dropout fraction


if __name__ == '__main__':
    mt.get_models(MP)
    def print_memory_usage():
        print(f"Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"Cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

    print_memory_usage()
