import sys
import torch
from pathlib import Path
import numpy as np
from monai.losses.dice import DiceLoss
import os
import pickle
import pandas as pd
# import surface_distance
# from monai.metrics import compute_hausdorff_distance
from torchmetrics.functional.classification import dice
from numpy import average
from scipy.special import softmax
import SimpleITK as sitk


def write_output(filename, data, ts, train_models):
    if os.path.exists(filename):
        results = pd.read_csv(filename, index_col=0, header=0, sep=',')
    else:
        results = pd.DataFrame(columns=train_models)
    results.loc[ts] = pd.Series(data)
    results.to_csv(filename)
    return


def metrics_evaluation(y_pred, y_true):
    loss = DiceLoss(include_background=True, sigmoid=False)
    dice_score = loss(y_pred, y_true).data.item()
    dice_score_torch = dice(y_pred, y_true.int(), average='micro')
    hausdorff = 0
    surface = 0
    return {'dice': dice_score, 'hausdorff': dice_score_torch, 'surface': surface}


def staple(pred_concs, label):
    staple_results = []  # ToDo: make it tensor
    for label_index in range(4):
        tmp = []
        for i in pred_concs.keys():
            label_channel = pred_concs[i][0, label_index, :, :, :]  # Extract the label channel
            label_image = sitk.GetImageFromArray(label_channel.astype('int32'))
            tmp.append(label_image)
        result = sitk.STAPLE(tmp, 1.0)
        result = result > 0.5
        result = torch.from_numpy(sitk.GetArrayFromImage(result))
        staple_results.append(result)
    tensor_result = torch.Tensor([np.stack(staple_results)])
    scores = metrics_evaluation(tensor_result, label)
    return scores


def majority_voting(pred_concs, label):
    preds = {}
    for k in pred_concs.keys():
        preds[k] = softmax(pred_concs[k], axis=1)
    preds = np.array(list(pred_concs.values()))
    predicted_classes = np.argmax(preds, axis=2)
    mv = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis=0, arr=predicted_classes)
    mv = np.eye(4)[mv]
    mv = torch.from_numpy(mv)
    mv = mv.movedim(-1, 1)
    scores = metrics_evaluation(mv, label)
    return scores


def plain_avg(pred_concs, label):
    preds_concs_np = np.array(list(pred_concs.values())).astype(int)
    # print(f"avg -> preds concs shape = {preds_concs_np.shape}, label = {label.shape}")
    tmp = average(preds_concs_np, axis=0)
    # print(f"avg -> 1: {tmp.shape}")
    tmp = np.argmax(softmax(tmp, axis=1), axis=1)
    # print(f"avg -> 2: {tmp.shape}")
    tmp = np.eye(4)[tmp]
    # print(f"avg -> 3: {tmp.shape}")
    tmp = torch.from_numpy(tmp)
    # print(f"avg -> 4: {tmp.shape}")
    tmp = tmp.movedim(-1, 1)
    # print(f"avg -> 5: {tmp.shape}")
    scores = metrics_evaluation(tmp, label)
    return scores


def ae_cons(pred_concs, label, weights):
    preds_concs_np = np.array(list(pred_concs.values())).astype(int)
    weights = list(weights)
    weights = softmax([weights])[0]
    ae = average(preds_concs_np, weights=weights, axis=0)
    tmp = np.argmax(softmax(ae, axis=1), axis=1)
    #  print(f"ae -> 2: {tmp.shape}")
    tmp = np.eye(4)[tmp]
    # print(f"ae -> 3: {tmp.shape}")
    tmp = torch.from_numpy(tmp)
    tmp = tmp.movedim(-1, 1)
    # print(f"ae -> 4: {tmp.shape}")
    scores = metrics_evaluation(tmp, label)
    return scores
