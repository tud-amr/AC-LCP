import cv2
import numpy as np


def draw_bboxes(info_pedestrians, images, cam):
    # color_bbox = (255,255,255)
    color_bbox = (0, 0, 255)
    name = (0,0, 255)
    for ped in info_pedestrians:
        identity = ped['id']
        xmin = int(ped['xmin'])
        ymin = int(ped['ymin'])
        xmax = int(ped['xmax'])
        ymax = int(ped['ymax'])

        width = xmax - xmin
        height = ymax - ymin

        cv2.rectangle(images[cam]['rgb'], (xmin, ymin), (xmax, ymax), color_bbox, 1)
        # cv2.putText(images[cam]['rgb'], str(identity), ((xmin + np.int0(width / 2)), (ymin + np.int0(height / 2))),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, name, 1)
    # cv2.putText(images[cam]['rgb'], str(cam), (30, images[cam]['rgb'].shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return images


def crop_bboxes(info_pedestrians, images, cam):
    color_bbox = (255, 255, 255)
    name = (0, 0, 255)
    image_ped = {}
    for ped in info_pedestrians:
        identity = ped['id']
        xmin = int(ped['xmin'])
        ymin = int(ped['ymin'])
        xmax = int(ped['xmax'])
        ymax = int(ped['ymax'])

        image_ped[identity] = images[cam]['rgb'][ymin:ymax, xmin:xmax].copy()
        #images[cam]['rgb'] = images[cam]['rgb'][ymin:ymax, xmin:xmax]
    #return images
    return image_ped

def visualize(cameras, images, frame_index, gt2d_pedestrians=None):
    if len(cameras) == 4:
        row_frames1, row_frames2 = [], []
        for i, cam in enumerate(cameras):
            if gt2d_pedestrians is not None:
                frame_index_key = str(frame_index)
                frame_index_key = frame_index_key.zfill(4)

                info_pedestrians = gt2d_pedestrians[cam][frame_index_key]
                images = draw_bboxes(info_pedestrians, images, cam)
            if cam == cameras[0] or cam == cameras[1]:
                row_frames1.append(images[cam]['rgb'])
            else:
                row_frames2.append(images[cam]['rgb'])
        img_list_2h = [row_frames1, row_frames2]
        concatenatedFrames = cv2.vconcat([cv2.hconcat(im_list_h) for im_list_h in img_list_2h])
    elif 1 < len(cameras) < 4:
        if gt2d_pedestrians is not None:
            for i, cam in enumerate(cameras):
                frame_index_key = str(frame_index)
                frame_index_key = frame_index_key.zfill(4)

                info_pedestrians = gt2d_pedestrians[cam][frame_index_key]
                images = draw_bboxes(info_pedestrians, images, cam)
        img_list_2h = [images[cam]['rgb'] for cam in cameras]
        concatenatedFrames = cv2.hconcat(img_list_2h)
    else:
        frame_index_key = str(frame_index)
        frame_index_key = frame_index_key.zfill(4)
        if gt2d_pedestrians is not None:
            info_pedestrians = gt2d_pedestrians[cameras[0]][frame_index_key]
            #images = draw_bboxes(info_pedestrians, images, cameras[0])
            #images = crop_bboxes(info_pedestrians, images, cameras[0])
            ped_images = crop_bboxes(info_pedestrians, images, cameras[0])
        #concatenatedFrames = images[cameras[0]]['rgb']
    # cv2.putText(concatenatedFrames, str(frame_index), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    #cv2.imshow('Tracking', concatenatedFrames) # MODIFIED

    k = cv2.waitKey(33)
    if k == 32:
        cv2.destroyAllWindows()
    elif k == ord('p'):
        cv2.waitKey(0)
    #return concatenatedFrames
    return ped_images

def visualize_4Plots(cameras, images, frame_index, gt2d_pedestrians=None, print_bbox = True):
    if len(cameras) == 4:
        row_frames1, row_frames2 = [], []
        for i, cam in enumerate(cameras):
            if gt2d_pedestrians is not None:
                frame_index_key = str(frame_index)
                frame_index_key = frame_index_key.zfill(4)

                info_pedestrians = gt2d_pedestrians[cam][frame_index_key]
                images = draw_bboxes(info_pedestrians, images, cam)
            if cam == cameras[0] or cam == cameras[1]:
                row_frames1.append(images[cam]['rgb'])
            else:
                row_frames2.append(images[cam]['rgb'])
        img_list_2h = [row_frames1, row_frames2]
        concatenatedFrames = cv2.vconcat([cv2.hconcat(im_list_h) for im_list_h in img_list_2h])
    elif 1 < len(cameras) < 4:
        if gt2d_pedestrians is not None:
            for i, cam in enumerate(cameras):
                frame_index_key = str(frame_index)
                frame_index_key = frame_index_key.zfill(4)

                info_pedestrians = gt2d_pedestrians[cam][frame_index_key]
                images = draw_bboxes(info_pedestrians, images, cam)
        img_list_2h = [images[cam]['rgb'] for cam in cameras]
        concatenatedFrames = cv2.hconcat(img_list_2h)
    else:
        frame_index_key = str(frame_index)
        frame_index_key = frame_index_key.zfill(4)
        if gt2d_pedestrians is not None:
            info_pedestrians = gt2d_pedestrians[cameras[0]][frame_index_key]
            ped_images = crop_bboxes(info_pedestrians, images, cameras[0])
            if print_bbox:
                images = draw_bboxes(info_pedestrians, images, cameras[0])
            #images = crop_bboxes(info_pedestrians, images, cameras[0])
        #concatenatedFrames = images[cameras[0]]['rgb']
    # cv2.putText(concatenatedFrames, str(frame_index), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    #cv2.imshow('Tracking', concatenatedFrames) # MODIFIED

    k = cv2.waitKey(33)
    if k == 32:
        cv2.destroyAllWindows()
    elif k == ord('p'):
        cv2.waitKey(0)
    #return concatenatedFrames
    return ped_images, images