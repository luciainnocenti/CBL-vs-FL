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

def autoencoder_training(dataset_folder: str, saving_folder: str, id_reference: str, typology: str = 'image'):
    assert typology in ['label', 'image'], f"The train typology is not applicable"
    common_shape = (320, 320, 16)
    transformations = transforms.Compose(
        [
            transforms.LoadImaged(keys=['image', 'label']),
            transforms.AddChanneld(keys=['image', 'label']),
            transforms.CenterSpatialCropd(keys=['image', 'label'], roi_size=common_shape),
            transforms.SpatialPadd(keys=['image', 'label'], spatial_size=common_shape),
            transforms.NormalizeIntensityd(keys=['image']),
            transforms.Lambdad(keys=['label'], func=lambda x: torch.where(x != 0, 1., 0.)),
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

    if os.path.exists(os.path.join(dataset_folder, f'participants_test_{id_reference}.csv')):
        df = pd.read_csv(os.path.join(dataset_folder, f'participants_test_{id_reference}.csv'))
        test_elements = list(df["FOLDER_NAME"])
    else:
        test_elements = [f for f in os.listdir(dataset_folder) if os.path.isdir(os.path.join(dataset_folder, f))]
    folders = [os.path.join(dataset_folder, f) for f in os.listdir(dataset_folder) if
               f in test_elements]
    data = []
    ref = []
    for folder in folders:
        image = os.path.join(f"{folder}/image/", os.listdir(f"{folder}/image/")[0])
        label = os.path.join(f"{folder}/label/", os.listdir(f"{folder}/label/")[0])
        data.append({'image': image, 'label': label})
        ref.append(folder.split('/')[-1])
    loader = DataLoader(Dataset(data, transform=transformations), 1, batch_size=4)

    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), 0.1)

    epoch_loss_values = []
    model.train()
    N = 50 if typology == 'image' else 30
    for epoch in range(N):
        epoch_loss = 0
        step = 0
        for idx, data in enumerate(loader):
            inputs = data[typology].to(device)
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
    fd = os.path.join(saving_folder, dataset_folder.split('/')[-1])
    os.makedirs(fd, exist_ok=True)
    torch.save(model.state_dict(), f"{fd}/{typology}")
    return epoch_loss_values


def autoencoder_visualize(autoencoder_path: str, images_path: List[str], typology: str = 'label'):
    common_shape = (320, 320, 16)
    transformations = transforms.Compose(
        [
            transforms.LoadImaged(keys=['image', 'label']),
            transforms.AddChanneld(keys=['image', 'label']),
            transforms.CenterSpatialCropd(keys=['image', 'label'], roi_size=common_shape),
            transforms.SpatialPadd(keys=['image', 'label'], spatial_size=common_shape),
            transforms.NormalizeIntensityd(keys=['image']),
            transforms.Lambdad(keys=['label'], func=lambda x: torch.where(torch.from_numpy(x) != 0, 1., 0.)),
        ]
    )
    data = []
    ref = []
    for folder in images_path:
        image = os.path.join(f"{folder}/image/", os.listdir(f"{folder}/image/")[0])
        label = os.path.join(f"{folder}/label/", os.listdir(f"{folder}/label/")[0])
        data.append({'image': image, 'label': label})
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

PROJECT_DIR = sys.argv[1]
mode = sys.argv[2]
saving_path = sys.argv[3]
typology_modes = ['image']
id_ref = sys.argv[4]

for typology_mode in typology_modes:
    print(f"typology = {typology_mode}")
    if mode == 'train':
        print(f'saving path: {saving_path}')
        os.makedirs(saving_path, exist_ok=True)
        datasets = ["skyra", "decathlon", "promise_coil", "promise_no_coil"]
        train_ds = f"{PROJECT_DIR}/Data/datasets_pp_nv/"
        for ds in datasets:
            autoencoder_training(dataset_folder=os.path.join(train_ds, ds),
                                 saving_folder=saving_path,
                                 id_reference=id_ref,
                                 typology=typology_mode)
        print('Done')

