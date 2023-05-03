import json

import numpy as np
#from PIL import Image
import cv2
import os, os.path
import json
import time
import scipy
from scipy import spatial
from bisect import bisect_left
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from utils.yolo_model import create_trained_yolo
from yolov3.configs import *
from yolov3.utils import detect_number
import csv




def segmentRed(picture_orig):
    picture = picture_orig.copy()
    picture = cv2.resize(picture, (YOLO_INPUT_SIZE, YOLO_INPUT_SIZE))
    # plt.figure()
    # plt.imshow(picture)
    fchannel_ub = picture[:, :, 0] < 100
    fchannel_ub = np.expand_dims(fchannel_ub, axis=2)
    fchannel_lb = 20 < picture[:, :, 0]
    fchannel_lb = np.expand_dims(fchannel_lb, axis=2)

    schannel_ub = picture[:, :, 1] < 90
    schannel_ub = np.expand_dims(schannel_ub, axis=2)
    schannel_lb = 0 <= picture[:, :, 1]
    schannel_lb = np.expand_dims(schannel_lb, axis=2)

    tchannel_ub = picture[:, :, 2] < 255
    tchannel_ub = np.expand_dims(tchannel_ub, axis=2)
    tchannel_lb = 110 < picture[:, :, 2]
    tchannel_lb = np.expand_dims(tchannel_lb, axis=2)

    cond_picture = np.concatenate([fchannel_ub, schannel_ub, tchannel_ub, fchannel_lb, schannel_lb, tchannel_lb],
                                  axis=2)
    cond_picture = np.all(cond_picture, axis=2)
    true_number = 255
    false_number = 0
    surropicture = np.ones_like(picture) * false_number
    surropicture[:, :, 0][cond_picture] = true_number
    surropicture[:, :, 1][cond_picture] = true_number
    surropicture[:, :, 2][cond_picture] = true_number
    return surropicture

def obtain_class_probabilities(picture, yolo, nclasses, real_class, show_image = False):
    bboxes, aux_image = detect_number(yolo, picture, "", input_size=YOLO_INPUT_SIZE, CLASSES=TRAIN_CLASSES,
                           rectangle_colors=(255, 0, 0),show_image = show_image)
    if show_image:
        plt.figure()
        plt.imshow(aux_image)
    sorted_bboxes = sorted(bboxes, key=lambda x:x[4])
    # if
    # probabilities = np
    if bboxes == []:
        p_measurement = np.ones(nclasses) * 1 / (nclasses)
    else:
        highest_probability = sorted_bboxes[-1][-2:]
        if highest_probability[-2] > 1/nclasses and int(highest_probability[-1]) == real_class:
            p_measurement = np.ones(nclasses)*(1-highest_probability[-2])/(nclasses-1)
            p_measurement[int(highest_probability[-1])-1] = highest_probability[-2]
        else:
            p_measurement = np.ones(nclasses) * 1 / (nclasses)

    return p_measurement

class Pedestrian:
    def __init__(self, ped, pedclass, nclasses, loaded = False):
        self.pedestrian_class = pedclass
        self.total_nclasses = nclasses
        self.database = []
        self.probas = []
        self.path_ped = '/home/amr/projects/alvaro/gym_target_ig/utils/library/ped'+str(ped)+'_class'+str(pedclass)+'/'
        path_images = self.path_ped+'cam1/Frames'
        valid_images = [".png"]
        if not loaded:
            for f in sorted(os.listdir(path_images)):
                ext = os.path.splitext(f)[1]
                if ext.lower() not in valid_images:
                    continue
                self.database.append(cv2.imread(os.path.join(path_images, f)))
                # self.database[-1][:, :, 0:2] = np.zeros_like(self.database[-1][:, :, 0:2])
                # self.database[-1] = cv2.cvtColor(self.database[-1], cv2.COLOR_BGR2GRAY)

        f = open(self.path_ped+'gt3d_pedestrians.json')
        self.rel_poses = json.load(f)
        f.close()

        tree_data = [np.array([x[1][0]["pos_x"],x[1][0]["pos_y"],-x[1][0]["orient_z"]*np.pi]) for x in self.rel_poses.items()]
        self.tree = spatial.KDTree(tree_data)
        self.max_x = max(tree_data, key = lambda x: x[0])[0]
        self.max_y = max(tree_data, key=lambda x: x[1])[1]
        self.min_y = min(tree_data, key=lambda x: x[1])[1]

    def getObservation(self, relPose):
        aux_t1 = time.time()
        #min_pose_idx = min(self.rel_poses.items(), key=lambda x: np.linalg.norm(np.array([x[1][0]["pos_x"],x[1][0]["pos_y"],x[1][0]["orient_z"]*np.pi])-relPose))
        idxPic = self.tree.query(relPose)[1]
        min_pose_idx = list(self.rel_poses.items())[idxPic]
        min_pose_pic = self.database[idxPic]
        return min_pose_pic#, min_pose_idx, time.time()-aux_t1

    def getProb(self, relPose):
        #aux_t1 = time.time()
        #min_pose_idx = min(self.rel_poses.items(), key=lambda x: np.linalg.norm(np.array([x[1][0]["pos_x"],x[1][0]["pos_y"],x[1][0]["orient_z"]*np.pi])-relPose))
        idxPic = self.tree.query(relPose)[1]
        min_pose_idx = list(self.rel_poses.items())[idxPic]
        #print(min_pose_idx)
        #1,8.5, ###  -np.tan(np.pi/4)*x,np.tan(np.pi/4)*x
        if 1.0<=relPose[0]<=8.5 and -np.tan(np.pi/4)*relPose[0]<=relPose[1]<=np.tan(np.pi/4)*relPose[0]\
                and (relPose[2]<=-np.pi/2 or relPose[2]>=np.pi/2):
            probabilities = self.probas[idxPic]
        else:
            probabilities = np.ones(self.total_nclasses) * 1 / (self.total_nclasses)
        # segmentedpic = segmentRed(self.database[idxPic])
        # probabilities = obtain_class_probabilities(segmentedpic, yolo, self.total_nclasses,self.pedestrian_class)
        #print(time.time()-aux_t1)
        return probabilities

    def save_probas(self, yolo):
        t1 = time.time()
        for image in self.database:
            segmentedpic = segmentRed(image)
            self.probas.append(obtain_class_probabilities(segmentedpic, yolo, self.total_nclasses, self.pedestrian_class))
            print(time.time() - t1)

        with open(self.path_ped+'csvPed.csv','w') as f:
            for proba in self.probas:
                writer = csv.writer(f)
                writer.writerow(proba)

    def load_probas(self):
        with open(self.path_ped+'csvPed.csv', newline='') as f:
            reader = csv.reader(f,delimiter=',',quoting=csv.QUOTE_NONNUMERIC)
            i=0
            for row in reader:
                self.probas.append(np.array(row))
                # print(i, self.probas[-1])
                # i+=1
                # if i==7273:
                #     continue



if __name__ == '__main__':
    # real_class = 1
    # nclasses = 2
    # query_rel_pose = np.array([7.43590, 0.25641, np.pi])
    # yolo = create_trained_yolo()
    # p1c1 = Pedestrian(1,real_class,nclasses)
    # p1c1.load_probas()
    # min_pose_pic=p1c1.getObservation(query_rel_pose)
    # plt.imshow(min_pose_pic)
    # #p1c1.save_probas(yolo)
    # # p1c2 = Pedestrian(1, 2, nclasses)
    # # p1c2.save_probas(yolo)
    # # for x in [1,2,3,4,5,6,7,8]:
    # #     rel_pose = np.array([x,0,np.pi])
    # #     print(p1c1.getProb(rel_pose,yolo))
    #
    #
    # # picture,pose,taux = p1c1.getObservation(rel_pose)
    # # print(taux)
    # print(p1c1.max_x, p1c1.max_y, p1c1.min_y)
    # print(p1c1.getProb(query_rel_pose))
    # # print(pose)
    # plt.figure()
    # segmentedpic = segmentRed(min_pose_pic)
    # plt.imshow(segmentedpic)
    # probas = obtain_class_probabilities(segmentedpic, yolo, nclasses,real_class)
    # print(probas)

    real_class = 1
    nclasses = 2
    yolo = create_trained_yolo()
    ped = Pedestrian(1, real_class, nclasses)
    ped.load_probas()

    res = 40
    rel_orient_degrees_res = 360
    for i in range(int(360 / rel_orient_degrees_res) + 1):
        plt.figure()
        rel_orient_degrees = -180  # +i*rel_orient_degrees_res
        rel_orient = np.deg2rad(rel_orient_degrees)
        x = np.linspace(10, 0, res)
        y = np.linspace(-10, 10, res)
        heatmap = np.zeros((res, res))

        for xi in range(res):
            for yi in range(res):
                min_pose_pic = ped.getObservation(np.array([x[xi], y[yi], rel_orient]))
                segmentedpic = segmentRed(min_pose_pic)
                element = obtain_class_probabilities(segmentedpic, yolo, nclasses,real_class,show_image=False)
                heatmap[xi][yi] = element[real_class - 1]
                #ped.getProb(np.array([x[xi], y[yi], rel_orient]))[real_class - 1]

        plt.imshow(heatmap, cmap='hot', interpolation='nearest')
        plt.show()
        filedir = './heatmap_figures/n' + str(nclasses) + '_class_' + str(real_class)
        if not os.path.isdir(filedir):
            os.makedirs(filedir)
        # plt.savefig(filedir+'/'+str(rel_orient_degrees)+'.png')

