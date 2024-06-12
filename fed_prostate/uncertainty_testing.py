from typing import List
import torch
import torch.nn.functional as F
from monai.networks.nets import UNet
from monai.data import Dataset, DataLoader
from monai.losses.dice import DiceLoss
from monai import transforms
import pickle
import os
# import torchio as tio
from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
from tqdm import tqdm

HOME = str(Path.home())
N = 50


def uncertainty_testing(dataset_folder: str, model_folder: str, ts_ref: str) \
        -> (List[str], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]):
    common_shape = (320, 320, 16)
    transformations = transforms.Compose(
        [
            transforms.LoadImaged(keys=['image', 'label']),
            transforms.AddChanneld(keys=['image', 'label']),
            transforms.CenterSpatialCropd(keys=['image', 'label'], roi_size=common_shape),
            transforms.SpatialPadd(keys=['image', 'label'], spatial_size=common_shape),
            transforms.NormalizeIntensityd(keys=['image']),
            transforms.Lambdad(keys=['label'], func=lambda x: torch.where(x != 0, 1, 0)),
            transforms.AsDiscreted(keys=['label'], to_onehot=2)
        ]
    )

    model_json = open(f"{model_folder}/saved_dictionary.pkl", "rb")
    model_json = pickle.load(model_json)
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=model_json['num_res_units'],
        norm="batch",
        dropout=model_json['dropout']
    )

    model.load_state_dict(torch.load(f"{model_folder}/unet"))
 
    if os.path.exists(os.path.join(dataset_folder, f'participants_test_{ts_ref}.csv')):
        df = pd.read_csv(os.path.join(dataset_folder, f'participants_test_{ts_ref}.csv'))
        test_elements = list(df["FOLDER_NAME"])
    else:
        test_elements = [f for f in os.listdir(dataset_folder) if os.path.isdir(os.path.join(dataset_folder, f))]
    folders = [os.path.join(dataset_folder, f) for f in os.listdir(dataset_folder) if
               f in test_elements]
    data = []
    data_ref = []
    for folder in folders:
        image = os.path.join(f"{folder}/image/", os.listdir(f"{folder}/image/")[0])
        label = os.path.join(f"{folder}/label/", os.listdir(f"{folder}/label/")[0])
        data.append({'image': image, 'label': label})
        data_ref.append(folder.split('/')[-1])

    loader = DataLoader(Dataset(data, transform=transformations), 1)
    std_results = []
    mean_results = []
    labels = []
    for idx, data in enumerate(loader):
        pred_sums = torch.zeros(data['image'].size())
        pred_concs = None
        for _ in tqdm(range(N), desc=f"Element {idx}/{len(loader)}"):
            prediction = model(data['image'])
            pred_sums = torch.add(pred_sums, prediction.detach())
            if pred_concs is None:
                pred_concs = prediction.detach()
            else:
                pred_concs = torch.cat((pred_concs, prediction.detach()))
        mean_results.append(torch.divide(pred_sums, N))
        std_results.append(torch.std(pred_concs, 0))
        labels.append(data['label'].detach())
    return data_ref, mean_results, std_results, labels


def visualize_uncertainty_hm(data: pd.DataFrame, csv_path: str, saving_folder: str):
    mat_count = np.empty([11, 11])
    for i in np.linspace(0, 1, 11):
        i = round(i, 1)
        for j in np.linspace(0, 1, 11):
            j = round(j, 1)
            tmp_cnt = len(data[(data['Unc_Media'] == i) & (data['Loss'] == j)])
            mat_count[int(i * 10)][int(j * 10)] = tmp_cnt

    fig = plt.figure()
    ax = sns.heatmap(mat_count, cmap="YlGnBu")
    fig.suptitle(f"Analysis dataset {csv_path.split('/')[-1]}", fontsize=15)
    plt.xlabel('Uncertainty', fontsize=11)
    plt.ylabel('Loss', fontsize=11)
    plt.savefig(os.path.join(saving_folder, f"{csv_path.split('/')[-1]}.png"))
    return