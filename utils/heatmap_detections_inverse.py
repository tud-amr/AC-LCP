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


from approx_perception import Pedestrian, segmentRed, obtain_class_probabilities


if __name__ == '__main__':
    real_class = 1
    nclasses = 10
    #yolo = create_trained_yolo()
    ped = Pedestrian(1,real_class,nclasses)
    ped.load_probas()

    res = 40
    rel_orient_degrees_res = 20
    rel_orient_list = [180]#[-100, -140, -180, 140]
    # fig, (ax1, ax2, ax3, ax4)
    for i in range(len(rel_orient_list)):#int(360/rel_orient_degrees_res)+1):for i in range(int(360/rel_orient_degrees_res)+1):
        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])  # [0.1, 0.1, 0.8, 0.8])
        rel_orient_degrees = rel_orient_list[i]#-180+i*rel_orient_degrees_res
        rel_orient = np.deg2rad(rel_orient_degrees)

        x_min = 0
        x_max = 9
        y_min = -8
        y_max = 8
        x = np.linspace(x_max, x_min, res)
        y = np.linspace(y_min, y_max, res)
        heatmap = np.zeros((res, res))

        for xi in range(res):
            for yi in range(res):
                nsamples = 1
                prob_dist = 0
                for samp in range(nsamples):
                    prob_dist += ped.getProb(np.array([x[xi]+np.random.normal(0,0.1),y[yi]+np.random.normal(0,0.1),rel_orient+np.random.normal(0,0.1)]))

                prob_dist /= nsamples
                if prob_dist[real_class-1] < 1/nclasses:
                    if prob_dist.max() > 0.5:
                        heatmap[xi][yi] = prob_dist.max()
                else:
                    heatmap[xi][yi] = 0.5
        radius = 5.
        triangle_points = np.array([[20., 39.5],
                                    [20. - radius * 1. / np.sqrt(2), 39.5 - radius * 1. / np.sqrt(2.)],
                                    [20. + radius * 1. / np.sqrt(2), 39.5 - radius * 1. / np.sqrt(2.)]])
        triangle = plt.Polygon(triangle_points, color='white')
        circle = plt.Circle(triangle_points[0], 1, color='cyan')
        plt.gca().add_patch(triangle)
        plt.gca().add_patch(circle)
        plt.axis('off')
        plt.imshow(heatmap, cmap='hot', interpolation='nearest',vmin=0.5,vmax=1.0)
        plt.show()

        # xi = list(range(x_max,x_min-1,-1))
        # yi = list(range(y_min,y_max+1))
        # ticks = np.arange(0, len(x) + 1, 10)
        # ax.set_yticks(ticks)
        # ax.set_yticklabels(np.linspace(x_max, x_min, len(ticks)))
        #
        # ax.set_xticks(ticks)
        # ax.set_xticklabels(np.linspace(y_min, y_max, len(ticks)))
        #
        # plt.ylabel('x')
        # plt.xlabel('y')

        # cax = plt.axes([0.85, 0.1, 0.075, 0.8])
        # plt.colorbar(cax=cax)
        if i==len(rel_orient_list)-1:
            cax = plt.axes([0.85, 0.1, 0.075, 0.8])
            plt.colorbar(cax=cax)
        filedir = './heatmap_figures/HSVSeg/n'+str(nclasses)+'_class_'+str(real_class)+'_inverse'
        if not os.path.isdir(filedir):
         os.makedirs(filedir)
        plt.savefig(filedir+'/'+str(rel_orient_degrees)+'.png')


