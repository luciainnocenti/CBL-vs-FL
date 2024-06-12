import pickle
import sys
from typing import List

import pandas as pd
import torch.nn.functional as F
import torch
import torch.nn as nn
from monai.networks.nets import AutoEncoder
from monai.data import Dataset, DataLoader
from monai import transforms
import os
import sys
from pathlib import Path
import torchio as tio

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HOME = str(Path.home())


def visualization(saving_folder, image=None, target=None, prediction=None, ref=""):
    if image is not None:
        tio_image = tio.ScalarImage(tensor=image)
        tio_image.save(f"{saving_folder}/image.nii")
    if target is not None:
        tio_target = tio.ScalarImage(tensor=target)
        tio_target.save(f"{saving_folder}/target.nii")
    if prediction is not None:
        tio_pred = tio.ScalarImage(tensor=prediction)
        tio_pred.save(f"{saving_folder}/prediction_{ref}.nii")
    return


def autoencoder_training(ds: str, saving_folder: str,
                         id_reference: str):
    common_shape = (240, 240, 128)
    transformations = transforms.Compose(
        [
            transforms.LoadImaged(keys=['T1']),
            transforms.AddChanneld(keys=['T1']),
            transforms.CenterSpatialCropd(keys=['T1'], roi_size=common_shape),
            transforms.NormalizeIntensityd(keys=['T1']),
        ]
    )

    model = AutoEncoder(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(16,),
        strides=(2,),
        inter_channels=[8, 8, 8],
        inter_dilations=[1, 2, 4],
        num_inter_units=2
    )

    dataset_folder = os.path.join(DATA_DIR, ds)

    if os.path.exists(os.path.join(HOME, f'/splits', ds, f'participants_train_{id_reference}.csv')):
        df = pd.read_csv(os.path.join(HOME, f'/splits', ds, f'participants_train_{id_reference}.csv'))
        test_elements = list(df["Subject_ID"])
    else:
        test_elements = [f for f in os.listdir(dataset_folder) if os.path.isdir(os.path.join(dataset_folder, f))]
    folders = [os.path.join(dataset_folder, f) for f in os.listdir(dataset_folder) if
               f in test_elements]
    data = []
    ref = []
    for folder in folders:
        image = os.path.join(f"{folder}/T1/", os.listdir(f"{folder}/T1/")[0])
        data.append({'T1': image})
        ref.append(folder.split('/')[-1])
    loader = DataLoader(Dataset(data, transform=transformations), 1, batch_size=4)

    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), 0.1)

    epoch_loss_values = []
    model.to(device)
    model.train()

    N = 50
    for epoch in range(N):
        epoch_loss = 0
        step = 0
        for idx, data in enumerate(loader):
            inputs = data['T1'].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, inputs)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            step = + 1
        epoch_loss /= step
        print(f"Loss at epoch {epoch} = {epoch_loss}")
        epoch_loss_values.append(epoch_loss)
    torch.save(model.state_dict(), f"{saving_folder}/trained_ae")
    return epoch_loss_values


def autoencoder_visualize(autoencoder_path: str, images_path: List[str], typology: str = 'label'):
    common_shape = (240, 240, 128)
    transformations = transforms.Compose(
        [
            transforms.LoadImaged(keys=['T1']),
            transforms.AddChanneld(keys=['T1']),
            transforms.CenterSpatialCropd(keys=['T1'], roi_size=common_shape),
            transforms.NormalizeIntensityd(keys=['T1']),
        ]
    )
    data = []
    ref = []
    for folder in images_path:
        image = os.path.join(f"{folder}/image/", os.listdir(f"{folder}/image/")[0])
        data.append({'T1': image})
        ref.append(folder.split('/')[-1])

    loader = DataLoader(Dataset(data, transform=transformations), 1, batch_size=1)

    model = AutoEncoder(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(16,),
        strides=(2,),
        inter_channels=[8, 8, 8],
        inter_dilations=[1, 2, 4],
        num_inter_units=2
    )

    model.load_state_dict(torch.load(autoencoder_path))
    model.eval()
    encoded_imgs = {}
    for idx, instance in enumerate(loader):
        tmp = model(instance[typology].to(device))
        encoded_imgs[ref[idx]] = tmp
    return encoded_imgs


def autoencoder_testing(
        ae_model_path: str,
        id_reference: str,
        ae_losses: dict
):
    common_shape = (240, 240, 128)
    transformations = transforms.Compose(
        [
            transforms.LoadImaged(keys=['T1']),
            transforms.AddChanneld(keys=['T1']),
            transforms.CenterSpatialCropd(keys=['T1'], roi_size=common_shape),
            transforms.NormalizeIntensityd(keys=['T1']),
        ]
    )

    model = AutoEncoder(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(16,),
        strides=(2,),
        inter_channels=[8, 8, 8],
        inter_dilations=[1, 2, 4],
        num_inter_units=2
    )

    model.load_state_dict(torch.load(f"{ae_model_path}/trained_ae"))
    model.to(device)

    for testing_set in range(23):
        testing_folder = os.path.join(DATA_DIR, f"Site_{testing_set + 1}")
        testing_set_ref = f"Site_{testing_set + 1}"
        if os.path.exists(
                os.path.join(HOME, f'/splits', testing_set_ref, f'participants_test_{id_reference}.csv')):
            df = pd.read_csv(
                os.path.join(HOME, f'/splits', testing_set_ref, f'participants_test_{id_reference}.csv'))
            test_elements = list(df["Subject_ID"])
        else:
            test_elements = [f for f in os.listdir(testing_folder) if os.path.isdir(os.path.join(testing_folder, f))]
        folders = [os.path.join(testing_folder, f) for f in os.listdir(testing_folder) if
                   f in test_elements]
        data = []
        ref = []
        for folder in folders:
            image = os.path.join(f"{folder}/T1/", os.listdir(f"{folder}/T1/")[0])
            data.append({'T1': image, 'ref': folder.split('/')[-1]})
            ref.append(folder.split('/')[-1])
            if folder.split('/')[-1] not in ae_losses.keys():
                ae_losses[folder.split('/')[-1]] = {}

        loader = DataLoader(Dataset(data, transform=transformations), 1, batch_size=1, shuffle=False)

        loss_function = torch.nn.MSELoss()

        for idx, data in enumerate(loader):
            print(f"data ref = {data['ref']}")
            inputs = data['T1'].to(device)
            outputs = model(inputs)
            loss = loss_function(outputs, inputs)
            m = ae_model_path.split('/')[-1]
            print(f"m = {m}")
            ae_losses[data['ref'][0]][m] = loss.item()
    return ae_losses

DATA_DIR = sys.argv[1]
flag_training = True

saved_models_folder = sys.argv[2]
id_ref = sys.argv[3]  # "kfold_0"
models_training = os.listdir(saved_models_folder)  # contains ['Site_1', 'Site_2',...]
datasets = [x for x in os.listdir(saved_models_folder) if x.startswith('Site')]
if flag_training:
    for site in datasets:
        if os.path.exists(os.path.join(saved_models_folder, site, "trained_ae")):
            print(f'autoencoder for {site} available')
        else:
            autoencoder_training(ds=site,
                                 saving_folder=os.path.join(saved_models_folder, site),
                                 id_reference=id_ref)
else:
    ae_losses = {}
    datasets = ['Site_9', 'Site_12', 'Site_23']
    for ae_model in datasets:
        ae_model_path = os.path.join(saved_models_folder, ae_model)
        ae_losses = autoencoder_testing(ae_model_path=ae_model_path,
                                        id_reference=id_ref,
                                        ae_losses=ae_losses)
    with open('ae_losses_fets_n.pt', 'wb') as f:
        pickle.dump(ae_losses, f)
print('Done')
