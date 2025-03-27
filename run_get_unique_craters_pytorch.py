"""Run/Obtain Unique Crater Distribution

    Execute extracting craters from model target predictions and filtering
    out duplicates.
    """
import get_unique_craters_pytorch as guc
import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Crater Parameters
CP = {}

# Image width/height, assuming square images.
CP['dim'] = 256

# Data type - train, dev, test
CP['datatype'] = 'test'

# Number of images to extract craters from
CP['n_imgs'] = 30000

# Hyperparameters
CP['llt2'] = 2.6    # D_{L,L} from Silburt et. al (2017)
CP['rt2'] = 1.8     # D_{R} from Silburt et. al (2017)

# Location of model to generate predictions (if they don't exist yet)
CP['dir_model'] = 'models/model_default_params.pth'

# Location of where hdf5 data images are stored
CP['dir_data'] = 'catalogues/%s_images.hdf5' % CP['datatype']

# Location of where model predictions are/will be stored
CP['dir_preds'] = 'catalogues/%s_preds_n%d.hdf5' % (CP['datatype'],
                                                    CP['n_imgs'])

# Location of where final unique crater distribution will be stored
CP['dir_result'] = 'catalogues/%s_craterdist.npy' % (CP['datatype'])

CP['run_template_matching'] = False

if __name__ == '__main__':
    craters_unique = np.empty([0, 3])
    craters_unique = guc.extract_unique_craters(CP, craters_unique)
