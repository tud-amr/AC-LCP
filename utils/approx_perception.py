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
tf.compat.v1.disable_eager_execution()

from tensorflow import keras
from tensorflow.keras import layers

from utils.yolo_model import create_trained_yolo
from yolov3.configs import *
from yolov3.utils import detect_number
import csv
import matplotlib.pyplot as plt




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

def segmentRedHSV(picture_orig):
    picture = picture_orig.copy()
    picture = cv2.resize(picture, (YOLO_INPUT_SIZE, YOLO_INPUT_SIZE))
    picture = cv2.cvtColor(picture, cv2.COLOR_BGR2HSV)

    # plt.figure()
    # plt.imshow(picture)

    fchannel_ub = picture[:, :, 0] <= 255
    fchannel_ub = np.expand_dims(fchannel_ub, axis=2)
    fchannel_lb = 0 <= picture[:, :, 0]
    fchannel_lb = np.expand_dims(fchannel_lb, axis=2)

    schannel_ub = picture[:, :, 1] <= 255
    schannel_ub = np.expand_dims(schannel_ub, axis=2)
    schannel_lb = 100 <= picture[:, :, 1] # Jason 150 - 255 # Damian 100
    schannel_lb = np.expand_dims(schannel_lb, axis=2)

    tchannel_ub = picture[:, :, 2] <= 255
    tchannel_ub = np.expand_dims(tchannel_ub, axis=2)
    tchannel_lb = 175 <= picture[:, :, 2]
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

def obtain_class_probabilities(picture, yolo, nclasses, original_image = [], show_image = False):
    # show_image=False
    bboxes, aux_image = detect_number(yolo, picture, "", input_size=YOLO_INPUT_SIZE, CLASSES=TRAIN_CLASSES,
                           rectangle_colors=(255, 0, 0),show_image = show_image, human_image=original_image)
    if show_image:
        plt.figure()
        plt.axis('off')
        plt.imshow(aux_image)
    sorted_bboxes = sorted(bboxes, key=lambda x:x[4])
    # if
    # probabilities = np
    if bboxes == []:
        p_measurement = np.ones(nclasses) * 1 / (nclasses)
    else:
        highest_probability = sorted_bboxes[-1][-2:]
        if highest_probability[-2] > 1/nclasses: # and int(highest_probability[-1]) == real_class:
            p_measurement = np.ones(nclasses)*(1-highest_probability[-2])/(nclasses-1)
            p_measurement[int(highest_probability[-1])] = highest_probability[-2]
        else:
            p_measurement = np.ones(nclasses) * 1 / (nclasses)

    return p_measurement

class Pedestrian:
    def __init__(self, ped, pedclass, nclasses, load_imgs = True, load_poses = True):
        self.pedestrian_class = pedclass
        self.total_nclasses = nclasses
        # if nclasses!=10:
        #     self.pedestrian_class -= 1
        self.database = []
        self.probas = []
        self.path_ped = '/home/amr/projects/alvaro/gym_target_ig/utils/library/ped'+str(ped)+'_class'+str(pedclass)+'/'
        path_images = self.path_ped+'cam1/Frames'
        valid_images = [".png"]
        if load_imgs:
            for f in sorted(os.listdir(path_images)):
                ext = os.path.splitext(f)[1]
                if ext.lower() not in valid_images:
                    continue
                self.database.append(cv2.imread(os.path.join(path_images, f)))
                # self.database[-1][:, :, 0:2] = np.zeros_like(self.database[-1][:, :, 0:2])
                # self.database[-1] = cv2.cvtColor(self.database[-1], cv2.COLOR_BGR2GRAY)

        if load_poses:
            f = open(self.path_ped+'gt3d_pedestrians.json')
            self.rel_poses = json.load(f)
            f.close()

            tree_data = [np.array([x[1][0]["pos_x"],x[1][0]["pos_y"],np.cos(-x[1][0]["orient_z"]*np.pi),np.sin(-x[1][0]["orient_z"]*np.pi)]) for x in self.rel_poses.items()]
            self.tree = spatial.KDTree(tree_data)
            self.max_x = max(tree_data, key = lambda x: x[0])[0]
            self.max_y = max(tree_data, key=lambda x: x[1])[1]
            self.min_y = min(tree_data, key=lambda x: x[1])[1]

    def getObservation(self, relPose_orient):
        aux_t1 = time.time()
        relPose = np.array([relPose_orient[0],relPose_orient[1],np.cos(relPose_orient[2]), np.sin(relPose_orient[2])])
        #min_pose_idx = min(self.rel_poses.items(), key=lambda x: np.linalg.norm(np.array([x[1][0]["pos_x"],x[1][0]["pos_y"],x[1][0]["orient_z"]*np.pi])-relPose))
        idxPic = self.tree.query(relPose)[1]
        min_pose_idx = list(self.rel_poses.items())[idxPic]
        min_pose_pic = self.database[idxPic]
        return min_pose_pic, min_pose_idx, time.time()-aux_t1

    def getProb(self, relPose_orient):
        #aux_t1 = time.time()
        #min_pose_idx = min(self.rel_poses.items(), key=lambda x: np.linalg.norm(np.array([x[1][0]["pos_x"],x[1][0]["pos_y"],x[1][0]["orient_z"]*np.pi])-relPose))
        relPose = np.array([relPose_orient[0], relPose_orient[1], np.cos(relPose_orient[2]), np.sin(relPose_orient[2])])
        idxPic = self.tree.query(relPose)[1]
        min_pose_idx = list(self.rel_poses.items())[idxPic]
        #print(min_pose_idx)
        #1,8.5, ###  -np.tan(np.pi/4)*x,np.tan(np.pi/4)*x
        # FOV check!!
        if 1.0<=relPose[0]<=8.5 and -np.tan(np.pi/4)*relPose[0]<=relPose[1]<=np.tan(np.pi/4)*relPose[0]\
                and (relPose_orient[2]<=-np.pi/2 or relPose_orient[2]>=np.pi/2):
            probabilities = self.probas[idxPic]
        else:
            probabilities = np.ones(self.total_nclasses) * 1 / (self.total_nclasses)
        # segmentedpic = segmentRed(self.database[idxPic])
        # probabilities = obtain_class_probabilities(segmentedpic, yolo, self.total_nclasses,self.pedestrian_class)
        #print(time.time()-aux_t1)
        # print(self.total_nclasses)
        probabilities = np.concatenate([probabilities[1:self.total_nclasses + 1], probabilities[0:1]], axis=-1)
        if self.total_nclasses < 10:
            probabilities = probabilities[0:self.total_nclasses]

        maxprob = np.max(probabilities)
        # print(probabilities)
        if maxprob>1./self.total_nclasses and probabilities[self.pedestrian_class-1]==maxprob:
            probabilities[probabilities != maxprob] = (1-maxprob)/(self.total_nclasses-1)
        else:
            probabilities = np.ones(self.total_nclasses)/self.total_nclasses
        return probabilities

    def save_probas(self, yolo):
        t1 = time.time()
        i=0
        for image in self.database:
            # segmentedpic = segmentRed(image)
            segmentedpic = segmentRedHSV(image)
            self.probas.append(obtain_class_probabilities(segmentedpic, yolo, self.total_nclasses, self.pedestrian_class))
            print(i,time.time() - t1)
            i+=1

        with open(self.path_ped+'csvPed.csv','w') as f:
            for proba in self.probas:
                writer = csv.writer(f)
                writer.writerow(proba)

    def getProb_yolo(self, relPose_orient, yolo, oracle = True):
        min_pose_pic, pose, taux = self.getObservation(relPose_orient)
        segmentedpic = segmentRedHSV(min_pose_pic)

        if 1.0<=relPose_orient[0]<=8.5 and -np.tan(np.pi/4)*relPose_orient[0]<=relPose_orient[1]<=np.tan(np.pi/4)*relPose_orient[0]\
                and (relPose_orient[2]<=-np.pi/2 or relPose_orient[2]>=np.pi/2):
            probabilities = obtain_class_probabilities(segmentedpic, yolo, 10)

        else:
            probabilities = np.ones(self.total_nclasses) * 1 / (self.total_nclasses)

        probabilities = np.concatenate([probabilities[1:self.total_nclasses + 1], probabilities[0:1]], axis=-1)
        if self.total_nclasses < 10:
            probabilities = probabilities[0:self.total_nclasses]

        maxprob = np.max(probabilities)
        if oracle:
            if maxprob>1./self.total_nclasses and probabilities[self.pedestrian_class-1]==maxprob:
                probabilities[probabilities != maxprob] = (1-maxprob)/(self.total_nclasses-1)
            else:
                probabilities = np.ones(self.total_nclasses)/self.total_nclasses
        else:
            if maxprob>1./self.total_nclasses:
                probabilities[probabilities != maxprob] = (1-maxprob)/(self.total_nclasses-1)
            else:
                probabilities = np.ones(self.total_nclasses)/self.total_nclasses
        return probabilities

    def getProbFromImg_yolo(self, relPose_orient, pic, yolo, oracle = True):
        if pic.size==0:
            probabilities = np.ones(self.total_nclasses) * 1 / (self.total_nclasses)
        else:
            # print(pic)
            # print(len(pic))
            # print(pic.size)
            segmentedpic = segmentRedHSV(pic)
            if (relPose_orient[2]<=-np.pi/2 or relPose_orient[2]>=np.pi/2):
                probabilities = obtain_class_probabilities(segmentedpic, yolo, 10)
            else:
                probabilities = np.ones(self.total_nclasses) * 1 / (self.total_nclasses)

        # print(probabilities)
        probabilities = np.concatenate([probabilities[1:self.total_nclasses + 1], probabilities[0:1]], axis=-1)
        if self.total_nclasses<10:
            probabilities = probabilities[0:self.total_nclasses]

        # print(probabilities)
        maxprob = np.max(probabilities)
        argmaxprob = np.argmax(probabilities)
        if oracle:
            if maxprob > 1. / self.total_nclasses and probabilities[self.pedestrian_class-1] == maxprob:
                probabilities[probabilities != maxprob] = (1 - maxprob) / (self.total_nclasses - 1)
            else:
                probabilities = np.ones(self.total_nclasses) / self.total_nclasses
        else:
            if maxprob > 1. / self.total_nclasses and argmaxprob < self.total_nclasses:
                probabilities[probabilities != maxprob] = (1 - maxprob) / (self.total_nclasses - 1)
            else:
                probabilities = np.ones(self.total_nclasses) / self.total_nclasses
        # print(self.pedestrian_class)
        # print(probabilities)
        return probabilities

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

def generate_probas(nclasses = 10):
    # query_rel_pose = np.array([2, 0, -np.pi])
    yolo = create_trained_yolo()

    p1c1 = Pedestrian(1, 1, nclasses)
    p1c1.save_probas(yolo)

    p1c2 = Pedestrian(1, 2, nclasses)
    p1c2.save_probas(yolo)

    p1c3 = Pedestrian(1, 3, nclasses)
    p1c3.save_probas(yolo)


def test_graph_function(real_class = 1, nclasses = 10):
    query_rel_pose = np.array([7, 0, -np.pi])
    yolo = create_trained_yolo()
    p1c1 = Pedestrian(1, real_class, nclasses)
    p1c1.load_probas()
    min_pose_pic, pose, taux = p1c1.getObservation(query_rel_pose)
    segmentedpic = segmentRedHSV(min_pose_pic)


    probas = obtain_class_probabilities(segmentedpic, yolo, 10)
    print(probas)

def test_function(real_class = 1, nclasses = 10):
    # real_class = 1
    # nclasses = 10
    query_rel_pose = np.array([5, 0, -np.pi])
    yolo = create_trained_yolo()
    p1c1 = Pedestrian(1, real_class, nclasses)
    # p1c1.load_probas()
    min_pose_pic, pose, taux = p1c1.getObservation(query_rel_pose)
    # min_pose_pic = p1c1.getObservation(query_rel_pose)
    human_image = cv2.resize(cv2.cvtColor(min_pose_pic, cv2.COLOR_BGR2RGB), (YOLO_INPUT_SIZE, YOLO_INPUT_SIZE))
    plt.imshow(human_image)
    # p1c1.save_probas(yolo)
    # p1c2 = Pedestrian(1, 2, nclasses)
    # p1c2.save_probas(yolo)
    # for x in [1,2,3,4,5,6,7,8]:
    #     rel_pose = np.array([x,0,np.pi])
    #     print(p1c1.getProb(rel_pose,yolo))

    # picture,pose,taux = p1c1.getObservation(rel_pose)
    # print(taux)
    print(p1c1.max_x, p1c1.max_y, p1c1.min_y)
    print([pose[1][0]["pos_x"], pose[1][0]["pos_y"], pose[1][0]["orient_z"]])
    plt.figure()
    # segmentedpic = segmentRed(min_pose_pic)
    segmentedpic = segmentRedHSV(min_pose_pic)
    plt.imshow(segmentedpic)
    probas1 = obtain_class_probabilities(segmentedpic, yolo, 10, human_image)
    print("1",probas1)
    probas2 = p1c1.getProb_yolo(query_rel_pose, yolo)
    print("2",probas2)
    # probas3 = p1c1.getProb(query_rel_pose)
    # print("3",probas3)


from yolov3.utils import image_preprocess
def prep_image(image):
    segmentedpic = segmentRedHSV(image)
    original_image = cv2.cvtColor(segmentedpic, cv2.COLOR_BGR2RGB)
    image_data = image_preprocess(np.copy(original_image), [416, 416])
    image_data = image_data[np.newaxis, ...].astype(np.float32)
    return image_data

def test_yolo(real_class = 1, nclasses = 10):
    yolo = create_trained_yolo()
    p1c1 = Pedestrian(1, real_class, nclasses)
    query_rel_pose = np.array([7, 0, -np.pi])
    min_pose_pic, pose, taux = p1c1.getObservation(query_rel_pose)
    image_data = prep_image(min_pose_pic)
    auxt1 = time.time()
    predictions = yolo.predict(image_data)
    print(time.time()-auxt1)

    query_rel_pose = np.array([5, 0, -np.pi])
    min_pose_pic, pose, taux = p1c1.getObservation(query_rel_pose)
    image_data = prep_image(min_pose_pic)
    auxt1 = time.time()
    predictions = yolo.predict(image_data)
    print(time.time() - auxt1)

    uery_rel_pose = np.array([4, 0, -np.pi])
    min_pose_pic, pose, taux = p1c1.getObservation(query_rel_pose)
    image_data = prep_image(min_pose_pic)
    auxt1 = time.time()
    predictions = yolo.predict(image_data)
    print(time.time() - auxt1)

    uery_rel_pose = np.array([3, 0, -np.pi])
    min_pose_pic, pose, taux = p1c1.getObservation(query_rel_pose)
    image_data = prep_image(min_pose_pic)
    auxt1 = time.time()
    predictions = yolo.predict(image_data)
    print(time.time() - auxt1)
    #print(predictions)

def testSegmentation(real_class = 1, nclasses = 10):
    query_rel_pose = np.array([5, 0, -np.pi])
    #yolo = create_trained_yolo()
    p1c1 = Pedestrian(1, real_class, nclasses)

    min_pose_pic, pose, taux = p1c1.getObservation(query_rel_pose)
    min_pose_pic_rgb = cv2.cvtColor(min_pose_pic, cv2.COLOR_BGR2RGB)
    plt.imshow(min_pose_pic_rgb)

    print(p1c1.max_x, p1c1.max_y, p1c1.min_y)
    print([pose[1][0]["pos_x"], pose[1][0]["pos_y"], pose[1][0]["orient_z"]])

    plt.figure()
    segmentedpic = segmentRed(min_pose_pic)
    plt.imshow(segmentedpic)

    segmentedpic = segmentRedHSV(min_pose_pic)
    plt.figure()
    plt.imshow(segmentedpic)
    #probas = obtain_class_probabilities(segmentedpic, yolo, nclasses, real_class)
    #print(probas)
    # print(p1c1.getProb(query_rel_pose))


if __name__ == '__main__':
    real_class = 1
    nclasses = 10
    # generate_probas(nclasses=nclasses)
    test_function(real_class=real_class,nclasses=nclasses)
    # test_graph_function(real_class=real_class, nclasses=nclasses)
    # testSegmentation(real_class=real_class,nclasses=nclasses)
    # test_yolo(real_class=1, nclasses=2)
