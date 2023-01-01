import numpy as np
import torch
import PIL.Image as Image
import time

def IOU(prediction, target):
    prediction = prediction.squeeze(1).cpu().detach().numpy()
    target = target.squeeze(1).cpu().detach().numpy()
    batch, _, _ = prediction.shape
    count = 0
    for i in range(batch):
        pred = prediction[i]
        tar = target[i]
        pred[pred >= 0.5] = 1
        pred[pred < 0.5] = 0
        # tar[tar>=0.5] = 1
        # tar[tar<0.5] = 0
        delta = 1e-10
        IoU = ((pred * tar).sum() + delta) / (pred.sum() + tar.sum() - (pred * tar).sum() + delta)
        count = count + IoU
    return count / batch

def Dice(prediction, target):
    prediction = prediction.squeeze(1).cpu().detach().numpy()
    target = target.squeeze(1).cpu().detach().numpy()
    batch, _, _ = prediction.shape
    count = 0
    for i in range(batch):
        pred = prediction[i]
        tar = target[i]
        pred[pred >= 0.5] = 1
        pred[pred < 0.5] = 0
        # tar[tar >= 0.5] = 1
        # tar[tar < 0.5] = 0
        delta = 1e-10
        dice = (((pred * tar).sum()) * 2 + delta) / (pred.sum() + tar.sum() + delta)
        count = count + dice
    return count / batch

def ACC(prediction, target):
    prediction = prediction.squeeze(1).cpu().detach().numpy()
    target = target.squeeze(1).cpu().detach().numpy()
    batch, row, col = prediction.shape
    count = 0

    for i in range(batch):
        pred = prediction[i]
        tar = target[i]
        pred[pred >= 0.5] = 1
        pred[pred < 0.5] = 0
        # tar[tar >= 0.5] = 1
        # tar[tar < 0.5] = 0

        TP = (pred * tar).sum()
        pred1 = pred
        tar1 = tar
        pred1[pred1 == 0] = 2
        pred1[pred1 == 1] = 0
        pred1[pred1 == 2] = 1
        tar1[tar1 == 0] = 2
        tar1[tar1 == 1] = 0
        tar1[tar1 == 2] = 1
        TN = (pred1 * tar1).sum()

        delta = 1e-10
        acc = (TP + TN) / ((row * col) + delta)
        count = count + acc
    return count / batch

def SE(prediction, target):
    prediction = prediction.squeeze(1).cpu().detach().numpy()
    target = target.squeeze(1).cpu().detach().numpy()
    batch, row, col = prediction.shape
    count = 0
    TP = 0
    TN = 0
    for i in range(batch):
        pred1 = prediction[i]
        tar1 = target[i]
        pred1[pred1 >= 0.5] = 1
        pred1[pred1 < 0.5] = 0
        # tar1[tar1 >= 0.5] = 1
        # tar1[tar1 < 0.5] = 0
        TP = (pred1 * tar1).sum()

        delta = 1e-10
        se = TP / (tar1.sum() + delta)
        count = count + se

    return count / batch

def SP(prediction, target):
    prediction = prediction.squeeze(1).cpu().detach().numpy()
    target = target.squeeze(1).cpu().detach().numpy()
    batch, row, col = prediction.shape
    count = 0

    for i in range(batch):
        pred = prediction[i]
        tar = target[i]
        pred[pred >= 0.5] = 1
        pred[pred < 0.5] = 0
        # tar[tar >= 0.5] = 1
        # tar[tar < 0.5] = 0
        TP = (pred * tar).sum()

        pred1 = pred
        tar1 = tar
        pred1[pred1 == 0] = 2
        pred1[pred1 == 1] = 0
        pred1[pred1 == 2] = 1
        tar1[tar1 == 0] = 2
        tar1[tar1 == 1] = 0
        tar1[tar1 == 2] = 1
        TN = (pred1 * tar1).sum()

        delta = 1e-10
        sp = TN / (tar1.sum() + delta)
        count = count + sp

    return count / batch

def Precision(prediction, target):
    prediction = prediction.squeeze(1).cpu().detach().numpy()
    target = target.squeeze(1).cpu().detach().numpy()
    batch, _, _ = prediction.shape
    count = 0
    for i in range(batch):
        predPC = prediction[i]
        tarPC = target[i]
        predPC[predPC >= 0.5] = 1
        predPC[predPC < 0.5] = 0
        # tarPC[tarPC >= 0.5] = 1
        # tarPC[tarPC < 0.5] = 0

        delta = 1e-10
        pre = (predPC * tarPC).sum() / (predPC.sum() + delta)
        count = count + pre
    Pre_result = count / batch
    return Pre_result

def FPR(prediction, target):
    prediction = prediction.squeeze(1).cpu().detach().numpy()
    target = target.squeeze(1).cpu().detach().numpy()
    batch, _, _ = prediction.shape
    count = 0
    for i in range(batch):
        predF = prediction[i]
        tarF = target[i]
        predF[predF >= 0.5] = 1
        predF[predF < 0.5] = 0
        # tarF[tarF >= 0.5] = 1
        # tarF[tarF < 0.5] = 0

        predF1 = predF
        tarF1 = tarF
        predF1[predF1 == 0] = 2
        predF1[predF1 == 1] = 0
        predF1[predF1 == 2] = 1
        tarF1[tarF1 == 0] = 2
        tarF1[tarF1 == 1] = 0
        tarF1[tarF1 == 2] = 1
        TN = (predF1 * tarF1).sum()

        delta = 1e-10
        pre = (tarF1.sum() - TN) / (tarF1.sum() + delta)
        count = count + pre
    Pre_result = count / batch
    return Pre_result
