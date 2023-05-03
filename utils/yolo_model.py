import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2
import numpy as np
import random
import time
import tensorflow as tf
from yolov3.yolov4 import Create_Yolo
from yolov3.utils import detect_image
from yolov3.configs import *

def create_trained_yolo():
    yolo = Create_Yolo(input_size=YOLO_INPUT_SIZE, CLASSES=TRAIN_CLASSES)
    filepath = "/home/amr/projects/alvaro/gym_target_ig/utils/checkpoints" #TODO make it cleaner
    yolo.load_weights(filepath+f"/{TRAIN_MODEL_NAME}")  # use keras weights

    # detect_image(yolo, image_path, "mnist_test.jpg", input_size=YOLO_INPUT_SIZE, show=True, CLASSES=TRAIN_CLASSES,
    #              rectangle_colors=(255, 0, 0))
    return yolo

# while True:
#     ID = random.randint(0, 200)
#     label_txt = "mnist/mnist_test.txt"
#     image_info = open(label_txt).readlines()[ID].split()
#
#     image_path = image_info[0]
#
#     yolo = Create_Yolo(input_size=YOLO_INPUT_SIZE, CLASSES=TRAIN_CLASSES)
#     yolo.load_weights(f"./checkpoints/{TRAIN_MODEL_NAME}") # use keras weights
#
#     detect_image(yolo, image_path, "mnist_test.jpg", input_size=YOLO_INPUT_SIZE, show=True, CLASSES=TRAIN_CLASSES, rectangle_colors=(255,0,0))

if __name__ == '__main__':
    yolo = create_trained_yolo()
    # detect_image(yolo, image_path, "mnist_test.jpg", input_size=YOLO_INPUT_SIZE, show=True, CLASSES=TRAIN_CLASSES,
    #             rectangle_colors=(255, 0, 0))