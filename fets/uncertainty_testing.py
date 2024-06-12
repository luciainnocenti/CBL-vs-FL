from typing import List
import torch
import torch.nn.functional as F
from monai.networks.nets import UNet
from monai.data import Dataset, DataLoader
from monai.losses.dice import DiceLoss
from monai import transforms
import pickle
import os
import torchio as tio
from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
from tqdm import tqdm

HOME = str(Path.home())
N = 50


def uncertainty_testing(dataset_split: str, model_folder: str, id_ref: str) \
        -> (List[str], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]):
    common_shape = (240, 240, 128)
    transformations = transforms.Compose(
        [
            transforms.LoadImaged(keys=['T1', 'T1CE', 'T2', 'FLAIR', 'SEG']),
            transforms.AddChanneld(keys=['T1', 'T1CE', 'T2', 'FLAIR', 'SEG']),
            transforms.Resized(['T1', 'T1CE', 'T2', 'FLAIR', 'SEG'], spatial_size=common_shape),
            transforms.NormalizeIntensityd(keys=['T1', 'T1CE', 'T2', 'FLAIR']),
            transforms.AsDiscreted(keys=['SEG'], to_onehot=5)
        ]
    )

    model = UNet(
        spatial_dims=3,
        in_channels=4,
        out_channels=5,
        channels=(30, 30 * 2, 30 * 4, 30 * 8, 30 * 16),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        kernel_size=3,
        dropout=0.3,
    )
    model.load_state_dict(torch.load(f"{model_folder}/unet"))
    dataset_name = dataset_split.split(sep='/')[-1]
    dataset_folder = os.path.join(DATA_DIR, dataset_name)
    if os.path.exists(os.path.join(dataset_split, f'participants_test_{id_ref}.csv')):
        df = pd.read_csv(os.path.join(dataset_split, f'participants_test_{id_ref}.csv'))
        test_elements = list(df["Subject_ID"])
    else:
        test_elements = [f for f in os.listdir(dataset_folder) if os.path.isdir(os.path.join(dataset_folder, f))]
    folders = [os.path.join(dataset_folder, f) for f in os.listdir(dataset_folder) if
               f in test_elements]
    data = []
    data_ref = []
    for folder in folders:
        T1 = os.path.join(f"{folder}/T1/", os.listdir(f"{folder}/T1/")[0])
        T1CE = os.path.join(f"{folder}/T1CE/", os.listdir(f"{folder}/T1CE/")[0])
        T2 = os.path.join(f"{folder}/T2/", os.listdir(f"{folder}/T2/")[0])
        FLAIR = os.path.join(f"{folder}/FLAIR/", os.listdir(f"{folder}/FLAIR/")[0])
        SEG = os.path.join(f"{folder}/SEG/", os.listdir(f"{folder}/SEG/")[0])
        data.append({'T1': T1, 'T1CE': T1CE, 'T2': T2, 'FLAIR': FLAIR, 'SEG': SEG})
        data_ref.append(folder.split('/')[-1])

    loader = DataLoader(Dataset(data, transform=transformations), 1)
    std_results = []
    mean_results = []
    labels = []
    for idx, instance in enumerate(loader):
        pred_sums = torch.zeros(instance['SEG'].size())
        pred_concs = None
        for _ in tqdm(range(N), desc=f"Element {idx}/{len(loader)}"):
            t1 = instance['T1']
            t1ce = instance['T1CE']
            t2 = instance['T2']
            flair = instance['FLAIR']
            prediction = model(torch.cat((t1, t1ce, flair, t2), dim=1))
            pred_sums = torch.add(pred_sums, prediction.detach())
            if pred_concs is None:
                pred_concs = prediction.detach()
            else:
                pred_concs = torch.cat((pred_concs, prediction.detach()))
        mean_results.append(torch.divide(pred_sums, N))
        std_results.append(torch.std(pred_concs, 0))
        labels.append(instance['SEG'].detach())
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
