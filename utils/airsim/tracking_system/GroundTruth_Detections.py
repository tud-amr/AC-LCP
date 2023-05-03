import os
import numpy as np
import json
import cv2


def get_pixels(img, color_ped):
    x_px = []
    y_px = []
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            pixel = img[i][j]
            if pixel[0] == color_ped[0] and pixel[1] == color_ped[1] and pixel[2] == color_ped[2]:
                y_px.append(i)
                x_px.append(j)
    return x_px, y_px


cameras = ['Drone1','Drone2','Drone3','Drone4']
init_path = '/home/scasao/Documents/PedestrianSystem/Records_20_11/'
colors_id_seg_path = '/home/scasao/pytorch/multi-target_tracking/color_to_pedestrian.json'

with open(colors_id_seg_path) as json_file:
    pedTOcolors = json.load(json_file)

print('Obtaining ground truth detection from segmentation....')
for cam in cameras:
    print(cam)
    seg_names = sorted([f for f in os.listdir(init_path + cam) if f.endswith('_seg.png')])
    file_names = [n[:19] for n in seg_names]

    gt_bbox = {}
    for i, name in enumerate(seg_names):
        timestamp = file_names[i]
        gt_bbox[timestamp] = {}
        img_seg = cv2.imread(init_path + cam + '/' + name)
        img_seg = cv2.cvtColor(img_seg, cv2.COLOR_BGR2RGB)
        for pedestrian in pedTOcolors.keys():
            name = pedestrian
            color = pedTOcolors[pedestrian]
            px_x, px_y = get_pixels(img_seg, color)
            if len(px_x) > 0:
                x_min = min(px_x)
                y_min = min(px_y)
                x_max = max(px_x)
                y_max = max(px_y)
                gt_bbox[timestamp][pedestrian] = {'xmin': x_min, 'y_min': y_min, 'x_max': x_max, 'y_max': y_max}
    save_path = init_path + cam + '/gt2d_pedestrians.json'
    with open(save_path, 'w') as f:
        json.dump(gt_bbox, f)
