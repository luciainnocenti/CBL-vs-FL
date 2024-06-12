import sys
import torch
import numpy as np
import glob
from torch.utils.data import DataLoader
from monai.data import Dataset, DataLoader
from monai.losses.dice import DiceLoss
from monai.networks.nets import UNet, SegResNet
from monai import transforms
from tqdm import tqdm
import os
import torch.nn.functional as F
import pandas as pd
# import surface_distance
from monai.metrics import compute_hausdorff_distance
from pathlib import Path
from utils import write_output, metrics_evaluation

HOME = str(Path.home())


def model_testing(dataset_split: str, model_folder: str, id_ref: str):
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
    net.load_state_dict(torch.load(f"{model_folder}/net"))
    net.eval()

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
    ref = []
    for folder in folders:
        T1 = os.path.join(f"{folder}/T1/", os.listdir(f"{folder}/T1/")[0])
        T1CE = os.path.join(f"{folder}/T1CE/", os.listdir(f"{folder}/T1CE/")[0])
        T2 = os.path.join(f"{folder}/T2/", os.listdir(f"{folder}/T2/")[0])
        FLAIR = os.path.join(f"{folder}/FLAIR/", os.listdir(f"{folder}/FLAIR/")[0])
        SEG = os.path.join(f"{folder}/SEG/", os.listdir(f"{folder}/SEG/")[0])
        data.append({'T1': T1, 'T1CE': T1CE, 'T2': T2, 'FLAIR': FLAIR, 'SEG': SEG})
        ref.append(folder.split('/')[-1])

    loader = DataLoader(Dataset(data, transform=transformations), 1)

    test_loss = 0
    test_hausdorff = 0
    test_surface = 0
    for cnt, instance in enumerate(tqdm(loader, total=len(loader))):
        t1 = instance['T1']
        t1ce = instance['T1CE']
        t2 = instance['T2']
        flair = instance['FLAIR']
        prediction = net(torch.cat((t1ce, t1, t2, flair), dim=1))  # Concatenate input modalities
        prediction = F.softmax(prediction, dim=1)
        prediction = prediction.detach().apply_(lambda x: int(x > 0.5))
        scores = metrics_evaluation(prediction, instance['SEG'])
        test_loss += scores['dice']
        test_hausdorff += scores['hausdorff']
        test_surface += scores['surface']

    test_loss /= len(loader)
    test_hausdorff /= len(loader)
    test_surface /= len(loader)
    return test_loss, test_hausdorff, test_surface

PROJECT_DIR = sys.argv[1]
DATA_DIR = sys.argv[2]
saved_models_folder = sys.argv[3]
results_folder = sys.argv[4]
datasets = sys.argv[5:-1]
# datasets = glob.glob(f"/user/linnocen/home/workspaces/nef/brats_analysis/splits/*")

id_ref = sys.argv[-1]

print(f"saved models folder = {saved_models_folder}")
print(f"results folder = {results_folder}")
print(f"datasets = {datasets}")
print(f"id_ref = {id_ref}")
results_dice = pd.DataFrame(columns=os.listdir(saved_models_folder), index=datasets)
results_haus = pd.DataFrame(columns=os.listdir(saved_models_folder), index=datasets)
results_surface = pd.DataFrame(columns=os.listdir(saved_models_folder), index=datasets)
models_training = os.listdir(saved_models_folder)
datasets_testing = [f"testset_{x.split('/')[-1]}" for x in datasets]
if os.path.exists(os.path.join(results_folder, "testing_dice.csv")):
    processed = pd.read_csv(os.path.join(results_folder, "testing_dice.csv"), sep=',', index_col=0)
    for index, row in processed.iterrows():
        if not row.isnull().values.any():
            datasets_testing.remove(index)

for ds in datasets:
    print(f"dataset test = {ds}")
    loss_values = {}
    hausdorff_values = {}
    surface_values = {}
    models = sorted(os.listdir(saved_models_folder))

    for model in models:
        print(f"test on model {model}")
        model_path = os.path.join(saved_models_folder, model)
        loss_value, hausdorff_value, surface_value = model_testing(dataset_split=ds,
                                                                   model_folder=model_path,
                                                                   id_ref=id_ref)
        loss_values[model] = loss_value
        hausdorff_values[model] = hausdorff_value
        surface_values[model] = surface_value
        print("[", model, " | ", ds, "] ---> ", loss_values)
    output_dice = os.path.join(results_folder, "testing_dice.csv")
    output_haus = os.path.join(results_folder, "testing_haus.csv")
    output_surf = os.path.join(results_folder, "testing_surface.csv")
    write_output(output_dice, loss_values, f"testset_{ds.split('/')[-1]}", datasets) #filename, data, ts, train_models
    write_output(output_haus, hausdorff_values, f"testset_{ds.split('/')[-1]}", datasets)
    write_output(output_surf, surface_values, f"testset_{ds.split('/')[-1]}", datasets)
