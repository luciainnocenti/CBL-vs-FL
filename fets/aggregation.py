import glob
import sys
import torch

from torch.utils.data import DataLoader
from monai.data import Dataset, DataLoader
from monai.networks.nets import UNet, SegResNet, AutoEncoder
from monai import transforms
from tqdm import tqdm
import os
import torch.nn.functional as F
import pandas as pd
import statistics
from pathlib import Path
from utils import staple, majority_voting, ae_cons, metrics_evaluation, plain_avg
import re
import multiprocessing as mp

HOME = str(Path.home())
# DATA_DIR = '/user/linnocen/home/workspaces/data'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def agg_computation(dataset_split: str, model_folders, id_ref: str):
    common_shape = (240, 240, 128)
    transformations = transforms.Compose(
        [
            transforms.LoadImaged(keys=['T1', 'T1CE', 'T2', 'FLAIR', 'SEG']),
            transforms.AddChanneld(keys=['T1', 'T1CE', 'T2', 'FLAIR', 'SEG']),
            transforms.CenterSpatialCropd(['T1', 'T1CE', 'T2', 'FLAIR', 'SEG'], roi_size=common_shape),
            transforms.NormalizeIntensityd(keys=['T1', 'T1CE', 'T2', 'FLAIR']),
            transforms.Lambdad(keys=['SEG'], func=lambda x: torch.where(x == 4, 3, x)),
            transforms.AsDiscreted(keys=['SEG'], to_onehot=4)
        ]
    )

    dataset_name = dataset_split.split(sep='/')[-1]
    dataset_folder = os.path.join(DATA_DIR, dataset_name)
    if os.path.exists(os.path.join(dataset_split, f'participants_test_{id_ref}.csv')):
        print("csv file found!")
        df = pd.read_csv(os.path.join(dataset_split, f'participants_test_{id_ref}.csv'))
        test_elements = list(df["Subject_ID"])
    else:
        test_elements = [f for f in os.listdir(dataset_folder) if os.path.isdir(os.path.join(dataset_folder, f))]
    folders = [os.path.join(dataset_folder, f) for f in os.listdir(dataset_folder) if
               f in test_elements]
    data = []
    ref = []
    for folder in folders:
        if folder.split('/')[-1] != 'FeTS2022_01064':
            T1 = os.path.join(f"{folder}/T1/", os.listdir(f"{folder}/T1/")[0])
            T1CE = os.path.join(f"{folder}/T1CE/", os.listdir(f"{folder}/T1CE/")[0])
            T2 = os.path.join(f"{folder}/T2/", os.listdir(f"{folder}/T2/")[0])
            FLAIR = os.path.join(f"{folder}/FLAIR/", os.listdir(f"{folder}/FLAIR/")[0])
            SEG = os.path.join(f"{folder}/SEG/", os.listdir(f"{folder}/SEG/")[0])
            data.append({'T1': T1, 'T1CE': T1CE, 'T2': T2, 'FLAIR': FLAIR, 'SEG': SEG})
            ref.append(folder.split('/')[-1])

    loader = DataLoader(Dataset(data, transform=transformations), batch_size=1, num_workers=0)
    losses_model = {"mv": {}, "staple": {}, "ae": {}, 'avg': {}}
    hausdorff_model = {"mv": {}, "staple": {}, "ae": {}, 'avg': {}}
    surface_model = {"mv": {}, "staple": {}, "ae": {}, 'avg': {}}
    with torch.no_grad():
        for idx, instance in enumerate(tqdm(loader, total=len(loader))):
            pred_concs = {}
            ae_losses = {}
            for i, model_folder in enumerate(model_folders):
                print(f"model = {model_folder}")
                net = SegResNet(spatial_dims=3,
                                init_filters=16,
                                in_channels=4,
                                out_channels=4,

                                dropout_prob=0.2,
                                act=('RELU', {'inplace': True}),
                                norm=('GROUP', {'num_groups': 8}),
                                norm_name='',
                                num_groups=8,
                                use_conv_final=True,
                                blocks_down=(1, 2, 2, 4),
                                blocks_up=(1, 1, 1)
                                )
                net.load_state_dict(torch.load(os.path.join(saved_models_folder, model_folder, "net"),
                                               map_location=device))
                net.to("cpu")
                net.eval()

                trained_ae = AutoEncoder(
                    spatial_dims=3,
                    in_channels=1,
                    out_channels=1,
                    channels=(16,),
                    strides=(2,),
                    inter_channels=[8, 8, 8],
                    inter_dilations=[1, 2, 4],
                    num_inter_units=2
                )

                trained_ae.load_state_dict(torch.load((os.path.join(saved_models_folder, model_folder, "trained_ae")),
                                                      map_location=device))
                trained_ae.eval()
                ae_loss = torch.nn.MSELoss()
                t1 = instance['T1']
                t1ce = instance['T1CE']
                t2 = instance['T2']
                flair = instance['FLAIR']
                prediction = net(torch.cat((t1ce, t1, t2, flair), dim=1))
                prediction = F.softmax(prediction, dim=1)
                prediction = prediction.detach().apply_(lambda x: int(x > 0.5)).numpy()
                pred_concs[i] = prediction
                ae_losses[i] = 1 / float(ae_loss(trained_ae(t1), t1))
                label = instance['SEG'].to('cpu')
            scores_mv = majority_voting(pred_concs, label)
            scores_staple = staple(pred_concs, label)
            scores_ae = ae_cons(pred_concs, label, ae_losses.values())
            scores_avg = plain_avg(pred_concs, label)

            losses_model['mv'][idx] = scores_mv['dice']
            hausdorff_model['mv'][idx] = scores_mv['hausdorff']
            surface_model['mv'][idx] = scores_mv['surface']
            losses_model['staple'][idx] = scores_staple['dice']
            hausdorff_model['staple'][idx] = scores_staple['hausdorff']
            surface_model['staple'][idx] = scores_staple['surface']
            losses_model['ae'][idx] = scores_ae['dice']
            hausdorff_model['ae'][idx] = scores_ae['hausdorff']
            surface_model['ae'][idx] = scores_ae['surface']
            losses_model['avg'][idx] = scores_avg['dice']
            hausdorff_model['avg'][idx] = scores_avg['hausdorff']
            surface_model['avg'][idx] = scores_avg['surface']
    return losses_model, hausdorff_model, surface_model


def ube_computation(dataset_split: str, model_folders, id_ref: str):
    common_shape = (240, 240, 128)
    transformations = transforms.Compose(
        [
            transforms.LoadImaged(keys=['T1', 'T1CE', 'T2', 'FLAIR', 'SEG'], image_only=False),
            transforms.EnsureChannelFirstd(keys=['T1', 'T1CE', 'T2', 'FLAIR', 'SEG'], channel_dim='no_channel'),
            transforms.CenterSpatialCropd(['T1', 'T1CE', 'T2', 'FLAIR', 'SEG'], roi_size=common_shape),
            transforms.NormalizeIntensityd(keys=['T1', 'T1CE', 'T2', 'FLAIR']),
            transforms.Lambdad(keys=['SEG'], func=lambda x: torch.where(x == 4, 3, x)),
            transforms.AsDiscreted(keys=['SEG'], to_onehot=4)
        ]
    )

    dataset_name = dataset_split.split(sep='/')[-1]
    dataset_folder = os.path.join(DATA_DIR, dataset_name)
    if os.path.exists(os.path.join(dataset_split, f'participants_test_{id_ref}.csv')):
        print("csv file found!")
        df = pd.read_csv(os.path.join(dataset_split, f'participants_test_{id_ref}.csv'))
        test_elements = list(df["Subject_ID"])
    else:
        test_elements = [f for f in os.listdir(dataset_folder) if os.path.isdir(os.path.join(dataset_folder, f))]
    folders = [os.path.join(dataset_folder, f) for f in os.listdir(dataset_folder) if
               f in test_elements]
    data = []
    ref = []
    for folder in folders:
        if folder.split('/')[-1] != 'FeTS2022_01064':
            T1 = os.path.join(f"{folder}/T1/", os.listdir(f"{folder}/T1/")[0])
            T1CE = os.path.join(f"{folder}/T1CE/", os.listdir(f"{folder}/T1CE/")[0])
            T2 = os.path.join(f"{folder}/T2/", os.listdir(f"{folder}/T2/")[0])
            FLAIR = os.path.join(f"{folder}/FLAIR/", os.listdir(f"{folder}/FLAIR/")[0])
            SEG = os.path.join(f"{folder}/SEG/", os.listdir(f"{folder}/SEG/")[0])
            data.append({'T1': T1, 'T1CE': T1CE, 'T2': T2, 'FLAIR': FLAIR, 'SEG': SEG})
            ref.append(folder.split('/')[-1])

    loader = DataLoader(Dataset(data, transform=transformations), batch_size=1, num_workers=0)
    losses_model = {"mv": {}, "staple": {}, "ae": {}, 'avg': {}, 'ube': {}}
    hausdorff_model = {"mv": {}, "staple": {}, "ae": {}, 'avg': {}, 'ube': {}}
    surface_model = {"mv": {}, "staple": {}, "ae": {}, 'avg': {}, 'ube': {}}
    with torch.no_grad():
        for idx, instance in enumerate(tqdm(loader, total=len(loader))):
            pred_concs = {}
            ube_weights = {}
            for i, model_folder in enumerate(model_folders):
                print(f"model = {model_folder}")
                net = SegResNet(spatial_dims=3,
                                init_filters=16,
                                in_channels=4,
                                out_channels=4,

                                dropout_prob=0.2,
                                act=('RELU', {'inplace': True}),
                                norm=('GROUP', {'num_groups': 8}),
                                norm_name='',
                                num_groups=8,
                                use_conv_final=True,
                                blocks_down=(1, 2, 2, 4),
                                blocks_up=(1, 1, 1)
                                )
                net.load_state_dict(torch.load(os.path.join(saved_models_folder, model_folder, "net"),
                                               map_location=device))
                net.to("cpu")
                net.eval()

                t1 = instance['T1']
                t1ce = instance['T1CE']
                t2 = instance['T2']
                flair = instance['FLAIR']
                tmp = []
                for _ in range(10):
                    prediction = net(torch.cat((t1ce, t1, t2, flair), dim=1))
                    prediction = F.softmax(prediction, dim=1).detach()
                    tmp.append(prediction)
                prediction = prediction.apply_(lambda x: int(x > 0.5)).numpy()
                pred_concs[i] = prediction
                ube_weights[i] = torch.mean(torch.std(torch.stack(tmp), dim=0))
                label = instance['SEG'].to('cpu')
            scores_ube = ae_cons(pred_concs, label, ube_weights.values())
            losses_model['ube'][idx] = scores_ube['dice']
            hausdorff_model['ube'][idx] = scores_ube['hausdorff']
            surface_model['ube'][idx] = scores_ube['surface']

    return losses_model, hausdorff_model, surface_model


mp.set_start_method('spawn')


DATA_DIR = sys.argv[1]
saved_models_folder = sys.argv[2]
results_folder = sys.argv[3]
datasets = sys.argv[4:-1]

id_ref = sys.argv[-1]

print(f"id_ref = {id_ref}")
mv_results = {}
staple_results = {}
ae_results = {}
avg_results = {}
models = [x for x in os.listdir(saved_models_folder) if x.startswith('Site')]
for test_set in datasets:
    print("test set = ", test_set)
    mv_results[test_set] = {}
    staple_results[test_set] = {}
    ae_results[test_set] = {}
    avg_results[test_set] = {}
    ae_results[test_set] = {}
    losses_models, hausdorff_models, surface_models = agg_computation(dataset_split=test_set,
                                                                        model_folders=models,
                                                                        id_ref=id_ref)

    mv_results[test_set]['dice'] = f"{statistics.mean(losses_models['mv'].values()):.2f}"
    mv_results[test_set]['hausdorff'] = f"{statistics.mean(hausdorff_models['mv'].values()):.2f}"
    mv_results[test_set]['surface'] = f"{statistics.mean(surface_models['mv'].values()):.2f}"
    staple_results[test_set]['dice'] = f"{statistics.mean(losses_models['staple'].values()):.2f}"
    staple_results[test_set]['hausdorff'] = f"{statistics.mean(hausdorff_models['staple'].values()):.2f}"
    staple_results[test_set]['surface'] = f"{statistics.mean(surface_models['staple'].values()):.2f}"
    ae_results[test_set]['dice'] = f"{statistics.mean(losses_models['ae'].values()):.2f}"
    avg_results[test_set]['dice'] = f"{statistics.mean(losses_models['avg'].values()):.2f}"

    losses_models, hausdorff_models, surface_models = ube_computation(dataset_split=test_set,
                                                                      model_folders=models,
                                                                      id_ref=id_ref)
    ae_results[test_set]['dice'] = f"{statistics.mean(losses_models['ube'].values()):.2f}"
pd.DataFrame(ae_results).to_csv(os.path.join(results_folder, 'ube.csv'))
pd.DataFrame(staple_results).to_csv(os.path.join(results_folder, 'staple.csv'))
pd.DataFrame(ae_results).to_csv(os.path.join(results_folder, 'ae.csv'))
pd.DataFrame(avg_results).to_csv(os.path.join(results_folder, 'avg.csv'))
pd.DataFrame(mv_results).to_csv(os.path.join(results_folder, 'mv.csv'))
