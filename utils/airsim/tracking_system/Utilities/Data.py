import os

import cv2
import sys
import json
import numpy as np
import sys

sys.path.append('/home/scasao/pytorch/multi-target_tracking/')

from Utils import imageTOdepth

video_folder = '/home/scasao/Documents/PedestrianSystem/Records_rename_19_19/'


def getVideo(cam):
    """
    Returns the video of the camera provided
    :param camera: the camera as returned by getCameras()
    :return: cv2.VideoCapture
    """
    return cv2.VideoCapture(video_folder + cam + "/Frames/%04d.png")


def initVideos(cam):
    video = getVideo(cam)
    if not video.isOpened():
        print("Could not open video for camera", cam)
        sys.exit()

    nframes = max(0, int(video.get(cv2.CAP_PROP_FRAME_COUNT)))
    print('frames del video', nframes)
    video_ok, frame = video.read()
    if not video_ok:
        print("Cannot read video file for camera", cam)
        sys.exit()

    return video, frame, video_ok


def loadCamStates(cam):
    path_state = video_folder + cam + '/state_info.json'
    with open(path_state, 'r') as f:
        state_info = json.load(f)
    return state_info


def getDepth(cam, frame_index):
    name_file = str(frame_index)
    name_file = name_file.zfill(4)

    depth_img = cv2.imread(video_folder + cam + '/' + name_file + '.png', 0)
    depth = imageTOdepth(depth_img)
    return depth


def getDepth_matrix(cam, frame_index):
    name_file = str(frame_index)
    name_file = name_file.zfill(4)

    depth = np.load(video_folder + cam + '/Depth/' + name_file + '.npy')
    return depth




