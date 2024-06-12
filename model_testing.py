import sys

# import torchio as tio
import torch
import numpy as np
import json
from torch.utils.data import DataLoader
from monai.losses.dice import DiceLoss
from monai.networks.nets import UNet
from monai.data import Dataset, DataLoader
from monai import transforms
import pickle
import os
from tqdm import tqdm
import torch.nn.functional as F
import pandas as pd
import surface_distance
from monai.metrics import compute_hausdorff_distance
from pathlib import Path

HOME = str(Path.home())

def metrics_evaluation(y_pred, y_true):
    loss = DiceLoss(include_background=False, sigmoid=False)
    dice = loss(y_pred[:, 1, :], y_true[:, 1, :]).data.item()
    hausdorff = compute_hausdorff_distance(y_pred, y_true).numpy()[0][0]
    spacing_mm = (1, 1, 1)
    mask_gt = np.array(y_true[0, 1, :].numpy(), dtype=bool)
    mask_pred = np.array(y_pred[0, 1, :].numpy(), dtype=bool)
    surface_distances = surface_distance.compute_surface_distances(mask_gt, mask_pred, spacing_mm)
    surface = 0 #surface_distance.compute_surface_dice_at_tolerance(surface_distances, 4)
    return {'dice': dice, 'hausdorff': hausdorff, 'surface': surface}


def model_testing(dataset_folder: str, model_folder: str, id_ref: str):
    print(f"dataset = {dataset_folder.split('/')[-1]}, model_folder = {model_folder.split('/')[-1]}")
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
    if os.path.exists(os.path.join(dataset_folder, f'participants_test_{id_ref}.csv')):
        df = pd.read_csv(os.path.join(dataset_folder, f'participants_test_{id_ref}.csv'))
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

    loader = DataLoader(Dataset(data, transform=transformations), 1)
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
    sd = torch.load(f"{model_folder}/unet")
    # sd = {k.replace('unet.', ''): v for k, v in sd.items()}
    sd = {k: v for k, v in sd.items() if 'running_mean' not in k and 'running_var' not in k}
    model.load_state_dict(sd, strict=False)
    model.eval()

    test_loss = 0
    test_hausdorff = 0
    test_surface = 0
    for idx, instance in tqdm(enumerate(loader)):
        prediction = model(instance['image'])
        prediction = F.softmax(prediction, dim=1)
        prediction = prediction.detach().apply_(lambda x: int(x > 0.5))
        scores = metrics_evaluation(prediction, instance['label'])
        test_loss += scores['dice']
        test_hausdorff += scores['hausdorff']
        test_surface += scores['surface']
    test_loss /= len(loader)
    test_hausdorff /= len(loader)
    test_surface /= len(loader)
    return test_loss, test_hausdorff, test_surface

PROJECT_DIR = sys.argv[1]
saved_models_folder = sys.argv[2]
results_folder = sys.argv[3]
datasets = sys.argv[4:-1]
id_ref = sys.argv[-1]

test_ds = f"{PROJECT_DIR}/Data/datasets_pp_nv/"


dfs_results_dice = {k: pd.DataFrame(index=[id_ref]) for k in datasets}
dfs_results_hau = {k: pd.DataFrame(index=[id_ref]) for k in datasets}
dfs_results_surface = {k: pd.DataFrame(index=[id_ref]) for k in datasets}

for ds in datasets:
    for model in os.listdir(saved_models_folder):
        model_path = os.path.join(saved_models_folder, model)
        loss_value, hausdorff_value, surface_value = model_testing(dataset_folder=f"{test_ds}{ds}",
                                                                   model_folder=model_path,
                                                                   id_ref=id_ref)
        configuration_file = os.path.join(model_path, "configuration_file.json")
        with open(configuration_file) as json_file:
            data = json.load(json_file)
        train_mode = data['used_datasets']
        if train_mode not in dfs_results_dice[ds].columns:
            dfs_results_dice[ds].insert(0, train_mode, f"{loss_value:.2f}")
        if train_mode not in dfs_results_hau[ds].columns:
            dfs_results_hau[ds].insert(0, train_mode, f"{hausdorff_value:.2f}")
        if train_mode not in dfs_results_surface[ds].columns:
            dfs_results_surface[ds].insert(0, train_mode, f"{surface_value:.2f}")

for k, dataframe in dfs_results_dice.items():
    os.makedirs(os.path.join(results_folder, 'dice'), exist_ok=True)
    csv_path = os.path.join(results_folder, 'dice', f'{k}_test.csv')
    header = not(os.path.exists(csv_path))
    dataframe.to_csv(csv_path, mode='a', header=header, index=True)

for k, dataframe in dfs_results_hau.items():
    os.makedirs(os.path.join(results_folder, 'hausdorff'), exist_ok=True)
    csv_path = os.path.join(results_folder, 'hausdorff', f'{k}_test.csv')
    header = not(os.path.exists(csv_path))
    dataframe.to_csv(csv_path, mode='a', header=header, index=True)

for k, dataframe in dfs_results_surface.items():
    os.makedirs(os.path.join(results_folder, 'surface'), exist_ok=True)
    csv_path = os.path.join(results_folder, 'surface', f'{k}_test.csv')
    header = not(os.path.exists(csv_path))
    dataframe.to_csv(csv_path, mode='a', header=header, index=True)

