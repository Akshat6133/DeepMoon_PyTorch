#!/usr/bin/env python
"""Convolutional Neural Network Training Functions

Functions for building and training a (UNET) Convolutional Neural Network on
images of the Moon and binary ring targets.
"""
import numpy as np
import pandas as pd
import h5py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2
from torchvision import datapoints

import utils.template_match_target as tmt
import utils.processing as proc

import torch
import torch.nn as nn
import torch.nn.functional as F

k2 = True  # Assuming Keras 2 is being used

if k2:
    from torch import cat as merge
    from torch.nn import Conv2d as Convolution2D
    from torch.nn import MaxPool2d as MaxPooling2D
    from torch.nn import Upsample as UpSampling2D
else:
    raise ValueError("Only PyTorch 2 syntax is supported.")

# No need to define a wrapper function for merging layers in PyTorch, as it's directly available as `torch.cat`.
# Also, there's no need to define a wrapper function for Conv2D, MaxPooling2D, and UpSampling2D in PyTorch.
# We directly use the corresponding modules from torch.nn.

# You can adjust the values of k2 and the imports according to your actual use case and PyTorch version.


#Using transforms.v2 instead of torchvision.transforms to randomly transform both image and mask in sync.

########################
def get_param_i(param, i):
    """Gets correct parameter for iteration i.

    Parameters
    ----------
    param : list
        List of model hyperparameters to be iterated over.
    i : integer
        Hyperparameter iteration.

    Returns
    -------
    Correct hyperparameter for iteration i.
    """
    if len(param) > i:
        return param[i]
    else:
        return param[0]

########################
class CraterDataset(Dataset):
    def __init__(self, data, target, transform=None):
        self.data = data
        self.target = target
        self.transform = transform
        self.counter = 0
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx].permute(2,0,1)
        mask = self.target[idx]
        if self.transform:
            mask = mask.squeeze(0)
            img = datapoints.Image(image)
            msk = datapoints.Mask(mask)
            image,mask = self.transform(img,msk)
            image = image.to(device)
            mask = mask.unsqueeze(0).to(device)
        return image, mask
########################
def check_random_transform(self,image,mask,transform):

    mask = mask.squeeze(0)
    print(image.shape)
    print(mask.shape)
    print(type(image))
    print(type(mask))
    img = datapoints.Image(image)
    msk = datapoints.Mask(mask)
    image,mask = self.transform(img,msk)
    image = image.to(device)
    mask = mask.to(device)
    image_cpu = image.cpu()
    mask_cpu = mask.cpu()
    print(image_cpu.shape)
    print(mask_cpu.shape)
    numpy_image= image_cpu.detach().numpy()
    numpy_mask = mask_cpu.detach().numpy()

    
    plt.imsave(image_and_mask_path+'_new_image.png',numpy_image.squeeze(0),cmap='gray')
    plt.imsave(image_and_mask_path+'_new_mask.png',numpy_mask,cmap='gray')

########################
def get_metrics(data, craters, dim, model, beta=1):
    """Function that prints pertinent metrics at the end of each epoch. 

    Parameters
    ----------
    data : hdf5
        Input images.
    craters : hdf5
        Pandas arrays of human-counted crater data. 
    dim : int
        Dimension of input images (assumes square).
    model : PyTorch model object
        PyTorch model
    beta : int, optional
        Beta value when calculating F-beta score. Defaults to 1.
    """
    X, Y = data[0], data[1]

    # Get csvs of human-counted craters
    csvs = []
    minrad, maxrad, cutrad, n_csvs = 3, 50, 0.8, len(X)
    diam = 'Diameter (pix)'
    for i in range(n_csvs):
        csv = craters[proc.get_id(i)]
        # remove small/large/half craters
        csv = csv[(csv[diam] < 2 * maxrad) & (csv[diam] > 2 * minrad)]
        csv = csv[(csv['x'] + cutrad * csv[diam] / 2 <= dim)]
        csv = csv[(csv['y'] + cutrad * csv[diam] / 2 <= dim)]
        csv = csv[(csv['x'] - cutrad * csv[diam] / 2 > 0)]
        csv = csv[(csv['y'] - cutrad * csv[diam] / 2 > 0)]
        if len(csv) < 3:    # Exclude csvs with few craters
            csvs.append([-1])
        else:
            csv_coords = np.asarray((csv['x'], csv['y'], csv[diam] / 2)).T
            csvs.append(csv_coords)

    # Calculate custom metrics
    print("")
    print("*********Custom Loss*********")
    recall, precision, fscore = [], [], []
    frac_new, frac_new2, maxrad = [], [], []
    err_lo, err_la, err_r = [], [], []
    frac_duplicates = []
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():  # Disable gradient calculation
        preds = model(torch.tensor(X).float())
    for i in range(n_csvs):
        if len(csvs[i]) < 3:
            continue
        (N_match, N_csv, N_detect, maxr,
         elo, ela, er, frac_dupes) = tmt.template_match_t2c(preds[i].numpy(), csvs[i],
                                                            rmv_oor_csvs=0)
        if N_match > 0:
            p = float(N_match) / float(N_match + (N_detect - N_match))
            r = float(N_match) / float(N_csv)
            f = (1 + beta**2) * (r * p) / (p * beta**2 + r)
            diff = float(N_detect - N_match)
            fn = diff / (float(N_detect) + diff)
            fn2 = diff / (float(N_csv) + diff)
            recall.append(r)
            precision.append(p)
            fscore.append(f)
            frac_new.append(fn)
            frac_new2.append(fn2)
            maxrad.append(maxr)
            err_lo.append(elo)
            err_la.append(ela)
            err_r.append(er)
            frac_duplicates.append(frac_dupes)
        else:
            print("skipping iteration %d,N_csv=%d,N_detect=%d,N_match=%d" %
                  (i, N_csv, N_detect, N_match))

    # Convert numpy arrays to PyTorch tensors
    X_tensor = torch.tensor(X).float()
    Y_tensor = torch.tensor(Y).float()

    # Calculate binary cross-entropy loss
    criterion = nn.BCELoss()
    loss = criterion(model(X_tensor), Y_tensor)
    print("binary XE score = %f" % loss.item())

    if len(recall) > 3:
        print("mean and std of N_match/N_csv (recall) = %f, %f" %
              (np.mean(recall), np.std(recall)))
        print("""mean and std of N_match/(N_match + (N_detect-N_match))
              (precision) = %f, %f""" % (np.mean(precision), np.std(precision)))
        print("mean and std of F_%d score = %f, %f" %
              (beta, np.mean(fscore), np.std(fscore)))
        print("""mean and std of (N_detect - N_match)/N_detect (fraction
              of craters that are new) = %f, %f""" %
              (np.mean(frac_new), np.std(frac_new)))
        print("""mean and std of (N_detect - N_match)/N_csv (fraction of
              "craters that are new, 2) = %f, %f""" %
              (np.mean(frac_new2), np.std(frac_new2)))
        print("median and IQR fractional longitude diff = %f, 25:%f, 75:%f" %
              (np.median(err_lo), np.percentile(err_lo, 25),
               np.percentile(err_lo, 75)))
        print("median and IQR fractional latitude diff = %f, 25:%f, 75:%f" %
              (np.median(err_la), np.percentile(err_la, 25),
               np.percentile(err_la, 75)))
        print("median and IQR fractional radius diff = %f, 25:%f, 75:%f" %
              (np.median(err_r), np.percentile(err_r, 25),
               np.percentile(err_r, 75)))
        print("mean and std of frac_duplicates: %f, %f" %
              (np.mean(frac_duplicates), np.std(frac_duplicates)))
        print("""mean and std of maximum detected pixel radius in an image =
              %f, %f""" % (np.mean(maxrad), np.std(maxrad)))
        print("""absolute maximum detected pixel radius over all images =
              %f""" % np.max(maxrad))
        print("")


########################
class UNet(nn.Module):
    def __init__(self, dim, n_filters, FL, init, drop, lmbda):
        super(UNet, self).__init__()

        # Define layers
        self.conv1 = nn.Conv2d(1, n_filters, FL, padding=FL // 2)
        self.conv2 = nn.Conv2d(n_filters, n_filters, FL, padding=FL // 2)
        self.conv3 = nn.Conv2d(n_filters, n_filters * 2, FL, padding=FL // 2)
        self.conv4 = nn.Conv2d(n_filters * 2, n_filters * 2, FL, padding=FL // 2)
        self.conv5 = nn.Conv2d(n_filters * 2, n_filters * 4, FL, padding=FL // 2)
        self.conv6 = nn.Conv2d(n_filters * 4, n_filters * 4, FL, padding=FL // 2)
        self.conv7 = nn.Conv2d(n_filters * 4, n_filters * 4, FL, padding=FL // 2)
        self.conv8 = nn.Conv2d(n_filters * 4, n_filters * 4, FL, padding=FL // 2)
        self.conv9 = nn.Conv2d(n_filters * 4, n_filters * 2, FL, padding=FL // 2)
        self.conv10 = nn.Conv2d(n_filters * 2, n_filters * 2, FL, padding=FL // 2)
        self.conv11 = nn.Conv2d(n_filters * 2, n_filters, FL, padding=FL // 2)
        self.conv12 = nn.Conv2d(n_filters, n_filters, FL, padding=FL // 2)
        self.conv13 = nn.Conv2d(n_filters, 1, 1, padding=0)

        # Define pooling layers
        self.maxpool = nn.MaxPool2d(2, 2)

        # Define upsampling layers
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # Define dropout layer
        self.dropout = nn.Dropout(drop)

        # Initialization
        if init == 'he_normal':
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
        elif init == 'glorot_uniform':
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
        else:
            raise ValueError("Unknown initialization: " + str(init))

    def forward(self, x):
        a1 = F.relu(self.conv1(x))
        a1 = F.relu(self.conv2(a1))
        a1P = self.maxpool(a1)

        a2 = F.relu(self.conv3(a1P))
        a2 = F.relu(self.conv4(a2))
        a2P = self.maxpool(a2)

        a3 = F.relu(self.conv5(a2P))
        a3 = F.relu(self.conv6(a3))
        a3P = self.maxpool(a3)

        u = F.relu(self.conv7(a3P))
        u = F.relu(self.conv8(u))

        u = self.upsample(u)
        u = torch.cat((a3, u), dim=1)
        u = self.dropout(u)
        u = F.relu(self.conv9(u))
        u = F.relu(self.conv10(u))

        u = self.upsample(u)
        u = torch.cat((a2, u), dim=1)
        u = self.dropout(u)
        u = F.relu(self.conv11(u))
        u = F.relu(self.conv12(u))

        u = self.upsample(u)
        u = torch.cat((a1, u), dim=1)
        u = self.dropout(u)
        u = F.relu(self.conv11(u))
        u = F.relu(self.conv12(u))

        u = self.conv13(u)
        return torch.sigmoid(u)


########################
def train_and_test_model(Data, Craters, MP, i_MP):
    """Function that trains, tests and saves the model, printing out metrics
    after each model. 

    Parameters
    ----------
    Data : dict
        Inputs and Target Moon data.
    Craters : dict
        Human-counted crater data.
    MP : dict
        Contains all relevant parameters.
    i_MP : int
        Iteration number (when iterating over hypers).
    """
    # Static params
    dim, nb_epoch, bs = MP['dim'], MP['epochs'], MP['bs']

    # Iterating params
    FL = get_param_i(MP['filter_length'], i_MP)
    learn_rate = get_param_i(MP['lr'], i_MP)
    n_filters = get_param_i(MP['n_filters'], i_MP)
    init = get_param_i(MP['init'], i_MP)
    lmbda = get_param_i(MP['lambda'], i_MP)
    drop = get_param_i(MP['dropout'], i_MP)

    # Build model
    model = UNet(dim, n_filters, FL, init, drop, lmbda)

    # Define loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learn_rate)

    # Main loop
    n_samples = MP['n_train']
    for nb in range(nb_epoch):
        model.train()  # Set model to training mode
        running_loss = 0.0
        for i, data in enumerate(custom_image_generator(Data['train'][0], Data['train'][1], batch_size=bs), 0):
            inputs, labels = data
            optimizer.zero_grad()  # Zero the parameter gradients
            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, labels)  # Compute the loss
            loss.backward()  # Backward pass
            optimizer.step()  # Optimize
            running_loss += loss.item()

        # Print average loss
        print('[Epoch %d] loss: %.3f' % (nb + 1, running_loss / n_samples))

        # Evaluate model on validation set
        model.eval()  # Set model to evaluation mode
        with torch.no_grad():
            dev_loss = 0.0
            for data in custom_image_generator(Data['dev'][0], Data['dev'][1], batch_size=bs):
                inputs, labels = data
                outputs = model(inputs)
                dev_loss += criterion(outputs, labels).item()

            # Print validation loss
            print('[Validation] loss: %.3f' % (dev_loss / n_samples))

    if MP['save_models'] == 1:
        torch.save(model.state_dict(), MP['save_dir'])

    print("###################################")
    print("##########END_OF_RUN_INFO##########")
    print("""learning_rate=%e, batch_size=%d, filter_length=%e, n_epoch=%d
          n_train=%d, img_dimensions=%d, init=%s, n_filters=%d, lambda=%e
          dropout=%f""" % (learn_rate, bs, FL, nb_epoch, MP['n_train'],
                           MP['dim'], init, n_filters, lmbda, drop))
    get_metrics(Data['test'], Craters['test'], dim, model)
    print("###################################")
    print("###################################")


########################
def get_models(MP):
    """Top-level function that loads data files and calls train_and_test_model.

    Parameters
    ----------
    MP : dict
        Model Parameters.
    """
    dir = MP['dir']
    n_train, n_dev, n_test = MP['n_train'], MP['n_dev'], MP['n_test']

    # Load data
    train = h5py.File('%strain_images.hdf5' % dir, 'r')
    dev = h5py.File('%sdev_images.hdf5' % dir, 'r')
    test = h5py.File('%stest_images.hdf5' % dir, 'r')
    Data = {
        'train': [torch.tensor(train['input_images'][:n_train]).float(),
                  torch.tensor(train['target_masks'][:n_train]).float()],
        'dev': [torch.tensor(dev['input_images'][:n_dev]).float(),
                torch.tensor(dev['target_masks'][:n_dev]).float()],
        'test': [torch.tensor(test['input_images'][:n_test]).float(),
                 torch.tensor(test['target_masks'][:n_test]).float()]
    }
    train.close()
    dev.close()
    test.close()

    # Rescale, normalize, add extra dim
    proc.preprocess(Data)

    # Load ground-truth craters
    Craters = {
        'train': pd.HDFStore('%strain_craters.hdf5' % dir, 'r'),
        'dev': pd.HDFStore('%sdev_craters.hdf5' % dir, 'r'),
        'test': pd.HDFStore('%stest_craters.hdf5' % dir, 'r')
    }

    # Iterate over parameters
    for i in range(MP['N_runs']):
        train_and_test_model(Data, Craters, MP, i)




# from keras.callbacks import Callback

# class NaNCheck(Callback):
#     def on_batch_end(self, batch, logs=None):
#         loss = logs.get('loss')
#         if loss is not None and np.isnan(loss):
#             print("Training halted due to NaN loss.")
#             raise ValueError("Training halted due to NaN loss.")

# from tensorflow.keras.callbacks import TensorBoard

# tensorboard_callback = TensorBoard(log_dir='./logs')

# model.fit(X_train, y_train, batch_size=32, epochs=50, validation_data=(X_val, y_val), callbacks=[tensorboard_callback])

