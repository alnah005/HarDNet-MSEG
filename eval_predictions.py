# -*- coding: utf-8 -*-
"""
file: eval_predictions.py

@author: Suhail.Alnahari

@description: 

@created: 2021-03-14T16:06:41.017Z-05:00

@last-modified: 2021-03-14T17:09:33.172Z-05:00
"""

# standard library
# 3rd party packages
# local source
import cv2
import numpy as np
import os
import json 

true_label_path = "./haveLabel_test/masks_old"
prediction_path = "./resultsThreshold/HarDMSEG/reconstructed_haveLabel_test"

def calculate_accuracy(img1,img2):
    positive = img1*img2
    negative = (1-img1)*(1-img2)
    return (positive.sum()+negative.sum())/(img1.sum()+(1-img1).sum())

def calculate_dice(img1,img2):
    positive = img1*img2
    smooth = 1
    dice = (2 * positive.sum() + smooth) / (img1.sum() + img2.sum() + smooth)
    return dice

def calculate_precision(img1,img2):
    positive = img1*img2
    return positive.sum()/img2.sum()

def calculate_recall(img1,img2):
    positive = img1*img2
    return positive.sum()/img1.sum()

metric = {"accuracy": calculate_accuracy,"Dice":calculate_dice,"Precision": calculate_precision,"Recall": calculate_recall}

true_image_names = {i:None for i in os.listdir(true_label_path)}
predicted_image_names = {i:None for i in os.listdir(prediction_path)}

assert len(true_image_names) == len(predicted_image_names)
for i in true_image_names.keys():
    assert i in predicted_image_names.keys()

result_sum = {m:0.0 for m in metric.keys()}
for k in true_image_names.keys():
    img1 = cv2.imread(true_label_path+os.path.sep+k)
    img2 = cv2.imread(prediction_path+os.path.sep+k)
    # img2 = cv2.imread(prediction_path+os.path.sep+k.split('_mask.jpg')[0]+'.png')

    img1 = img1/img1.max()
    img2 = img2/img2.max()

    for m in metric.keys():
        result_sum[m] += metric[m](img1,img2)

for m in metric.keys():
    result_sum[m] /= len(true_image_names.keys())

print(result_sum)