import json
import sys
from typing import List, Any
from datetime import datetime
import pytz
from statistics import mean
from monai.networks.nets import AutoEncoder
import pandas as pd
import torch
import torchio as tio
from torch import Tensor
from torch.utils.data import DataLoader
from scipy.special import softmax
from monai.losses.dice import DiceLoss
from monai.networks.nets import UNet
from monai.data import Dataset, DataLoader
from monai import transforms
import pickle
import os
import torch.nn.functional as F
import SimpleITK as sitk
from numpy import average
from pathlib import Path
from uncertainty_testing import uncertainty_testing
from tqdm import tqdm

HOME = str(Path.home())
PROJECT_DIR = f'{HOME}/distributed_analysis/distributed_analysis'


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


def autoencoder_combiner(dataset_folder: str, model_folders: List[str], mode: str = 'label', ts_ref: str = '',
                         visualize: bool = False):
    unet_loss = DiceLoss(include_background=False, sigmoid=False)
    ae_loss = torch.nn.MSELoss()
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

    if os.path.exists(os.path.join(dataset_folder, f'participants_test_{ts_ref}.csv')):
        df = pd.read_csv(os.path.join(dataset_folder, f'participants_test_{ts_ref}.csv'))
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

    pred_concs = {}
    ae_losses_concs = {}
    gt_concs = {}
    losses_models = {}
    for model_folder in model_folders:
        losses_models[model_folder] = {}
        loader = DataLoader(Dataset(data, transform=transformations), 1)
        model_json = open(f"{model_folder}/saved_dictionary.pkl", "rb")
        model_json = pickle.load(model_json)
        trained_unet = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=model_json['num_res_units'],
            norm="batch",
            dropout=model_json['dropout']
        )
        trained_unet.load_state_dict(torch.load(f"{model_folder}/unet"))
        trained_unet.eval()

        trained_ae = AutoEncoder(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            channels=(16,),
            strides=(2,),
            inter_channels=[8, 8, 8],
            inter_dilations=[1, 2, 4],
            num_inter_units=2)

        trained_ae.load_state_dict(torch.load(f"{model_folder}/{mode}"))
        trained_ae.eval()

        for idx, instance in enumerate(loader):
            prediction = trained_unet(instance['image'])

            prediction = F.softmax(prediction, dim=1)
            losses_models[model_folder][ref[idx]] = float(unet_loss(prediction, instance['label']))
            if ref[idx] not in pred_concs.keys():
                pred_concs[ref[idx]] = []
                ae_losses_concs[ref[idx]] = []
            mask_prediction = prediction.detach()[0, 1, :].apply_(lambda x: int(x > 0.5))
            ae_input = mask_prediction[None, None, :] if mode == 'label' else instance['image']
            ae_losses_concs[ref[idx]].append(1/float(ae_loss(trained_ae(ae_input), ae_input)))
            pred_concs[ref[idx]].append(mask_prediction)
            gt_concs[ref[idx]] = instance['label']

    averaged_results = {}
    losses_models['autoencoder_avg'] = {}
    for key, images in pred_concs.items():
        ae_losses_concs[key] = softmax(ae_losses_concs[key])
        prediction = average(images, weights=ae_losses_concs[key]).apply_(lambda x: int(x > 0.5))
        averaged_results[key] = prediction
        target = gt_concs[key][:, 1, :]
        staple_loss = unet_loss(averaged_results[key][None, :], target)
        losses_models['autoencoder_avg'][key] = float(staple_loss)
        if visualize:
            s = f"{PROJECT_DIR}/Results/visualizations"
            saving_path = os.path.join(s, key, f"visualization_autoencoder_{mode}")
            os.makedirs(saving_path, exist_ok=True)
            visualization(saving_folder=saving_path, target=target, prediction=prediction[None, :], ref=f"ae_{mode}")
    return averaged_results, losses_models


def majority_voting(dataset_folder: str, model_folders: List[str], ts_ref: str, visualize: bool = False) \
        -> tuple[dict[str, Tensor], dict[str, dict[str, float]]]:
    loss = DiceLoss(include_background=False, sigmoid=False)
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
    if os.path.exists(os.path.join(dataset_folder, f'participants_test_{ts_ref}.csv')):
        df = pd.read_csv(os.path.join(dataset_folder, f'participants_test_{ts_ref}.csv'))
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

    pred_sums = [torch.zeros(common_shape)] * len(data)
    losses_models = {}
    for model_folder in model_folders:
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
        model.load_state_dict(torch.load(f"{model_folder}/unet"))
        model.eval()
        losses_models[model_folder] = {}
        for idx, instance in enumerate(loader):
            prediction = model(instance['image'])
            prediction = F.softmax(prediction, dim=1)
            losses_models[model_folder][ref[idx]] = float(loss(prediction, instance['label']))
            pred_sums[idx] = torch.add(pred_sums[idx], prediction.detach()[0, 1, :, :, :])

    n = len(model_folders)
    averaged_results = {}
    loader = iter(DataLoader(Dataset(data, transform=transformations), 1))
    losses_models['averaged'] = {}
    for idx, pred_sum in enumerate(pred_sums):
        tmp = torch.divide(pred_sum, n)
        tmp.apply_(lambda x: int(x >= 0.5))
        averaged_results[ref[idx]] = tmp
        target = next(loader)['label']
        losses_models['averaged'][ref[idx]] = float(loss(target[:, 1, :], tmp[None, :]))
        if visualize:
            s = f"{PROJECT_DIR}/Results/visualizations"
            saving_path = os.path.join(s, ref[idx], 'visualization_mv')
            os.makedirs(saving_path, exist_ok=True)
            visualization(saving_folder=saving_path, target=target[0, :], prediction=tmp[None, :], ref='mv')
    return averaged_results, losses_models


def staple(dataset_folder: str, model_folders: List[str], ts_ref: str, visualize: bool = False) \
        -> tuple[dict[str, Tensor], dict[str, dict[Any, Any]]]:
    loss = DiceLoss(include_background=False, sigmoid=False)
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

    if os.path.exists(os.path.join(dataset_folder, f'participants_test_{ts_ref}.csv')):
        df = pd.read_csv(os.path.join(dataset_folder, f'participants_test_{ts_ref}.csv'))
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

    pred_concs = {}
    gt_concs = {}
    losses_models = {}
    for model_folder in model_folders:
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
        losses_models[model_folder] = {}
        model.load_state_dict(torch.load(f"{model_folder}/unet"))
        model.eval()
        for idx, instance in enumerate(loader):
            prediction = model(instance['image'])
            prediction = F.softmax(prediction, dim=1)
            losses_models[model_folder][ref[idx]] = float(loss(prediction, instance['label']))
            if ref[idx] not in pred_concs.keys():
                pred_concs[ref[idx]] = []

            mask_prediction = prediction.detach()[0, 1, :].apply_(lambda x: int(x > 0.5))
            tmp = sitk.GetImageFromArray(mask_prediction.type(torch.int))
            pred_concs[ref[idx]].append(tmp)
            gt_concs[ref[idx]] = instance['label']

    averaged_results = {}
    losses_models['stapled'] = {}
    for key, images in pred_concs.items():
        tmp = sitk.STAPLE(images, 1.0)
        tmp = tmp > 0.5
        averaged_results[key] = torch.from_numpy(sitk.GetArrayFromImage(tmp))
        target = gt_concs[key][:, 1, :]
        staple_loss = loss(averaged_results[key][None, :], target)
        losses_models['stapled'][key] = float(staple_loss)
        if visualize:
            s = f"{PROJECT_DIR}/Results/visualizations"
            saving_path = os.path.join(s, key, 'visualization_staple')
            os.makedirs(saving_path, exist_ok=True)
            visualization(saving_folder=saving_path, target=target, prediction=averaged_results[key][None, :],
                          ref='staple')
    return averaged_results, losses_models


def uncertainty_combiner(dataset_folder: str, model_folders: List[str], ts_ref: str, visualize: bool = False):
    unet_loss = DiceLoss(include_background=False, sigmoid=False)
    pred_concs = {}
    unc_maps_concs = {}
    gt_concs = []
    losses_models = {}
    for model_folder in model_folders:
        print(f"Model folder: {model_folder}")
        data_ref, mean_results, std_results, labels = uncertainty_testing(dataset_folder, model_folder, ts_ref)
        losses_models[model_folder] = {}
        for i in range(len(data_ref)):
            if data_ref[i] not in pred_concs.keys():
                pred_concs[data_ref[i]] = []
                unc_maps_concs[data_ref[i]] = []
                gt_concs = labels
            tmp = mean_results[i].apply_(lambda x: int(x > 0.5))
            losses_models[model_folder][data_ref[i]] = float(unet_loss(tmp[0, :], labels[i][:, 1, :]))
            pred_concs[data_ref[i]].append(tmp)
            unc_maps_concs[data_ref[i]].append(std_results[i])

    res = []
    losses_models['average_unc'] = {}
    for index, participant in enumerate(pred_concs.keys()):
        weights = softmax(1/torch.mean(torch.stack(unc_maps_concs[participant]), dim=[1, 2, 3]))
        tmp = average(pred_concs[participant], weights=weights).apply_(lambda x: int(x > 0.5))
        res.append(tmp)
        losses_models['average_unc'][participant] = float(unet_loss(tmp[0, :], gt_concs[index][:, 1, :]))
        if visualize:
            s = f"{PROJECT_DIR}/Results/visualizations"
            saving_path = os.path.join(s, participant, 'visualization_staple')
            os.makedirs(saving_path, exist_ok=True)
            visualization(saving_folder=saving_path, target=gt_concs[index], prediction=tmp[None, :], ref='staple')

    return res, losses_models


def average_combiner(dataset_folder: str, model_folders: List[str], ts_ref: str = ''):
    unet_loss = DiceLoss(include_background=False, sigmoid=False)
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

    if os.path.exists(os.path.join(dataset_folder, f'participants_test_{ts_ref}.csv')):
        df = pd.read_csv(os.path.join(dataset_folder, f'participants_test_{ts_ref}.csv'))
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

    pred_concs = {}
    ae_losses_concs = {}
    gt_concs = {}
    losses_models = {}
    for model_folder in model_folders:
        print(f"model {model_folder}")
        losses_models[model_folder] = {}
        loader = DataLoader(Dataset(data, transform=transformations), 1)
        model_json = open(f"{model_folder}/saved_dictionary.pkl", "rb")
        model_json = pickle.load(model_json)
        trained_unet = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=model_json['num_res_units'],
            norm="batch",
            dropout=model_json['dropout']
        )
        trained_unet.load_state_dict(torch.load(f"{model_folder}/unet"))
        trained_unet.eval()

        for idx, instance in enumerate(loader):
            prediction = trained_unet(instance['image'])

            prediction = F.softmax(prediction, dim=1)
            losses_models[model_folder][ref[idx]] = float(unet_loss(prediction, instance['label']))
            if ref[idx] not in pred_concs.keys():
                pred_concs[ref[idx]] = []
                ae_losses_concs[ref[idx]] = []
            mask_prediction = prediction.detach()[0, 1, :].apply_(lambda x: int(x > 0.5))
            pred_concs[ref[idx]].append(mask_prediction)
            gt_concs[ref[idx]] = instance['label']
    averaged_results = {}
    losses_models['avg'] = {}
    for key, images in tqdm(pred_concs.items(), desc="averaging elements"):
        prediction = torch.mean(torch.stack(images), dim=0).apply_(lambda x: int(x > 0.5))
        averaged_results[key] = prediction
        target = gt_concs[key][:, 1, :]
        staple_loss = unet_loss(averaged_results[key][None, :], target)
        losses_models['avg'][key] = float(staple_loss)
    return averaged_results, losses_models


def aggregation(models_path, datasets, modalities, ts):
    models = os.listdir(models_path)
    models = [os.path.join(models_path, m) for m in models if m in datasets]
    print(f"models = {models}")
    test_ds = f"{PROJECT_DIR}/Data/datasets_pp_nv/"

    dfs_results = {k: pd.DataFrame(index=[ts]) for k in datasets}
    if 'avg' in modalities:
        print('average')
        for ds in t_datasets:
            results, losses = average_combiner(dataset_folder=f"{test_ds}{ds}", model_folders=models, ts_ref=ts)
            for k, v in losses.items():
                if os.path.isdir(k):
                    configuration_file = os.path.join(k, "configuration_file.json")
                    with open(configuration_file) as json_file:
                        data = json.load(json_file)
                    train_method = data['used_datasets']
                else:
                    train_method = "avg"
                if train_method not in dfs_results[ds].columns:
                    dfs_results[ds].insert(0, train_method, f"{mean(v.values()):.2f}")

    if 'mav' in modalities:
        print('majority voting')
        for ds in datasets:
            results, losses = majority_voting(dataset_folder=f"{test_ds}{ds}", model_folders=models, ts_ref=ts)
            for k, v in losses.items():
                if os.path.isdir(k):
                    configuration_file = os.path.join(k, "configuration_file.json")
                    with open(configuration_file) as json_file:
                        data = json.load(json_file)
                    train_method = data['used_datasets']
                else:
                    train_method = "mav"
                if train_method not in dfs_results[ds].columns:
                    dfs_results[ds].insert(0, train_method, f"{mean(v.values()):.2f}")

    if 'staple' in modalities:
        print('staple')
        for ds in datasets:
            results, losses = staple(dataset_folder=f"{test_ds}{ds}", model_folders=models, ts_ref=ts)
            for k, v in losses.items():
                if os.path.isdir(k):
                    configuration_file = os.path.join(k, "configuration_file.json")
                    with open(configuration_file) as json_file:
                        data = json.load(json_file)
                    train_method = data['used_datasets']
                else:
                    train_method = "staple"
                if train_method not in dfs_results[ds].columns:
                    dfs_results[ds].insert(0, train_method, f"{mean(v.values()):.2f}")

    if 'ae_image' in modalities:
        print('auto encoder image')
        for ds in datasets:
            results, losses = autoencoder_combiner(dataset_folder=f"{test_ds}{ds}", model_folders=models, mode='image',
                                                   ts_ref=ts)
            for k, v in losses.items():
                if os.path.isdir(k):
                    configuration_file = os.path.join(k, "configuration_file.json")
                    with open(configuration_file) as json_file:
                        data = json.load(json_file)
                    train_method = data['used_datasets']
                else:
                    train_method = "ae_image"
                if train_method not in dfs_results[ds].columns:
                    dfs_results[ds].insert(0, train_method, f"{mean(v.values()):.2f}")

    if 'uncertainty' in modalities:
        print('uncertainty')
        for ds in datasets:
            results, losses = uncertainty_combiner(dataset_folder=f"{test_ds}{ds}", model_folders=models, ts_ref=ts)
            for k, v in losses.items():
                if os.path.isdir(k):
                    configuration_file = os.path.join(k, "configuration_file.json")
                    with open(configuration_file) as json_file:
                        data = json.load(json_file)
                    train_method = data['used_datasets']
                else:
                    train_method = "uncertainty"
                if train_method not in dfs_results[ds].columns:
                    dfs_results[ds].insert(0, train_method, f"{mean(v.values()):.2f}")

    os.makedirs(results_folder, exist_ok=True)
    for k, dataframe in dfs_results.items():
        csv_path = os.path.join(results_folder, f'{k}_aggregation_ENS.csv')
        header = not (os.path.exists(csv_path))
        dataframe.to_csv(csv_path, mode='a', header=header, index=True)


saved_models_folder = sys.argv[1]
modalities = ["avg"]
results_folder = sys.argv[2]
datasets = sys.argv[3:-1]
id_ref = sys.argv[-1]

aggregation(models_path=saved_models_folder, datasets=datasets, modalities=modalities, ts=id_ref)
