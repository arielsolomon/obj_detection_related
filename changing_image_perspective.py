#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 12:57:07 2020

@author: ariels

The flow:
    1. reading img file
    2. reading json anotation file
    3. reading json image bounding box from the file
    4. transofming the image
    5. transforming the bounding box according to image transformation matrix
    6. updating the json boundinbox list
    7. saving the image
    8. repeating it till last image
    9. saving the updated json"""


import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import os


import json

def json_dat(file_path):
    with open(file_path) as f:
        data = json.load(f)
        anot = data['annotations']
        return anot

def img_plot(img, dst,save_path, num):
    plt.subplot(121),plt.imshow(img),plt.title('Input')
    plt.subplot(122),plt.imshow(dst),plt.title('Output')
    plt.savefig(save_path+num+'.png')
    plt.show()

def bbox_transform(bbox_list):
    xA, yA, w, h = bbox_list
    xB, yB = xA + w, yA + h
    print(xA, yA, xB, yB)
    rect_pts = np.array([[[xA, yA]], [[xB, yA]], [[xA, yB]], [[xB, yB]]], dtype=np.float32)
    four_cor = np.squeeze(cv2.perspectiveTransform(rect_pts, M))
    [x,y,w,l] = (np.min(four_cor[:,0]),np.min(four_cor[:,1]), np.abs(np.max(four_cor[0,:])-np.min(four_cor[0,:])),np.abs(np.max(four_cor[1,:])-np.min(four_cor[1,:])) )
    return [x,y,w,l], np.squeeze(rect_pts)

## check the bb on related image


def img_plot(img, dst):
    plt.subplot(121),plt.imshow(img),plt.title('Input')
    plt.subplot(122),plt.imshow(dst),plt.title('Output')
    plt.show()




def json_dat(file_path):
    with open(file_path) as f:
        data = json.load(f)
        anot = data['annotations']
        return anot

def calc_trans_bbox(anot):
    for item in anot:
        trans_bbox = bbox_transform(item['bbox'])
        return trans_bbox

img_root = '/Data/new_carmel/small_dataset/images/'
img_file_list = glob.glob(os.path.join(img_root, '*.jpeg'))
img_file_list.sort()
save_root = '/Data/new_carmel/small_dataset/trans_images/'
json_path = '/app/disk2/carmel/thermal_sig/thermal_cnn_occluded/coco/FLIR_Dataset/FLIR_ADAS_1_3/train/thermal_annotations.json'

        ##prespective##
# n1, n2 determines the supposebly view angles. I create 3 different options
n1, n2 = 180, 460

    
    
for file in img_file_list:
    img = cv2.imread(file)
    pts1 = np.float32([[10,70],[630,70],[n1,510],[n2,510]])
    pts2 = np.float32([[0,0],[640,0],[0,512],[640,512]])
    M = cv2.getPerspectiveTransform(pts1,pts2)
    dst = cv2.warpPerspective(img,M,(640,512))


bbox_list = json_dat(json_path)[7]['bbox']
[x,y,w,l], points = bbox_transform(bbox_list)
#testing file
test_file = img_root+'FLIR_00009.jpeg'
img = cv2.imread(test_file)
dst = cv2.rectangle(img.copy(), (x, y), (np.max(points[:,0]), np.max(points[:,1])), (255,0,0), 2)
img_plot(img, dst)
img_plot(img, dst)
