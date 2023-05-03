import cv2
import numpy as np
import copy

from tracking_system.detection_module import Detector
from tracking_system.Utilities.Data import initVideos, loadCamStates, getDepth_matrix
from tracking_system.TMsingleView import getDetectorInfo
from tracking_system.CrossCameras import DKF_singleView, communications, DA_crossCameras_Hungarian, DKF
from tracking_system.Trackers.camerasTracker import SingleCameraTracker0, SingleCameraTracker1, SingleCameraTracker2, SingleCameraTracker3
from tracking_system.saveResultsTracker import SaveResults_pymotFormat
from tracking_system.associationSingleView import DA_singleView_Hungarian
from tracking_system.reId import ReId
from tracking_system.Gallery import updateTemporalGallery, updateScoreGallery
from tracking_system.TMsingleView import initTrackers, cleanTrackers, define_environment_limits
from tracking_system.Utilities.geometryCam import from3dCylinder, cropBbox
from Utils import getImagesResponse, getRGB_D

C_WHITE = (255, 255, 255)

GALLERY_SIZE = 20
SAVE_APPEARANCE_PATCHES = 2
N_DETECTIONS = 1


def InitTrackingSystem(cameras):
    detector = Detector()
    reid = ReId()
    results_pymot = {}
    trackers, old_trackers, LUT_delete, LUT_delete = {}, {}, {}, {}
    for cam in cameras:
        trackers[cam], old_trackers[cam], LUT_delete[cam] = [], [], []
        results_pymot[cam] = SaveResults_pymotFormat()
    return trackers, old_trackers, LUT_delete, detector, reid, results_pymot


def getTrackers(groupCameras):
    """
    :param groupCameras: str, list of camera names
    :return:
    """
    singleTrackers = {}
    for camera in groupCameras:
        if camera == groupCameras[0]:
            singleTrackers[camera] = SingleCameraTracker0
            singleTrackers[camera].NEXTID = 0
        elif camera == groupCameras[1]:
            singleTrackers[camera] = SingleCameraTracker1
            singleTrackers[camera].NEXTID = 0
        elif camera == groupCameras[2]:
            singleTrackers[camera] = SingleCameraTracker2
            singleTrackers[camera].NEXTID = 0
        elif camera == groupCameras[3]:
            singleTrackers[camera] = SingleCameraTracker3
            singleTrackers[camera].NEXTID = 0

    return singleTrackers


def visualize(cameras, frames, frame_index, trackers, states_cameras, depth=None, online=False):
    for i, cam in enumerate(cameras):
        if online:
            cam_state = states_cameras[cam]
            depth_matrix = depth[cam]
        else:
            cam_state = states_cameras[cam][frame_index]
            depth_matrix = getDepth_matrix(cam, frame_index)

        for track in trackers[cam]:
            # if track.id is not None:
            # Get bbox from cylinder
            cylinder_e = track.cylinder
            bbox_e = from3dCylinder(cam_state, depth_matrix, cylinder_e, online)
            bbox_e = cropBbox(bbox_e, frames[cam])
            # Get bbox parameters
            xmin, ymin, xmax, ymax = np.int0(bbox_e.getAsXmYmXMYM())
            width = xmax - xmin
            height = ymax - ymin
            # Draw bbox
            cv2.rectangle(frames[cam], (xmin, ymin), (xmax, ymax), track.colorTOshow, 1)
            cv2.putText(frames[cam], str(track.id), ((xmin + np.int0(width / 2)), (ymin + np.int0(height / 2))),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, track.colorTOshow, 1)
        cv2.putText(frames[cam], str(cam), (30, frames[cam].shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    if len(frames) == 4:
        row_frames1, row_frames2, final_frames= [],  [], []
        for cam in cameras:
            if cam == 'Drone1' or cam == 'Drone2':
                row_frames1.append(frames[cam])
            else:
                row_frames2.append(frames[cam])
        img_list_2h = [row_frames1, row_frames2]
        concatenatedFrames = cv2.vconcat([cv2.hconcat(im_list_h) for im_list_h in img_list_2h])
        cv2.putText(concatenatedFrames, str(frame_index), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    else:
        concatenatedFrames = frames[cameras[0]]
    cv2.imshow('Tracking', concatenatedFrames)

    k = cv2.waitKey(33)
    # cv2.waitKey(0)
    if k == 32:
        cv2.destroyAllWindows()
    elif k == ord('p'):
        cv2.waitKey(0)
    return concatenatedFrames


def MCMT_DTM(trackers, cameras, frames, depth, frame_index,  detector, reid, old_trackers, LUT_delete, states_cameras, singleTrackers, coefficients=None):
    frames_aux = copy.deepcopy(frames)
    detector_results = getDetectorInfo(detector, frames, cameras,frame_index, N_DETECTIONS, states_cameras, depth, coefficients, online=True)

    for cam in cameras:
        for bbox in detector_results[cam]:
            xmin, ymin, xmax, ymax = bbox.getAsXmYmXMYM()
            cv2.rectangle(frames[cam], (int(xmin), int(ymin)), (int(xmax), int(ymax)), C_WHITE, 1)

    detectors_unused = {}
    for cam in cameras:
        depth_matrix = depth[cam]
        states_cameras_frame = states_cameras[cam]
        trackers[cam], detectors_unused[cam] = DA_singleView_Hungarian(trackers[cam], states_cameras_frame, detector_results[cam], depth_matrix, reid, frames_aux[cam], online=True)

    trackers, old_trackers, LUT_delete = initTrackers(states_cameras, trackers, detectors_unused, old_trackers,
                                                      LUT_delete, cameras, singleTrackers, GALLERY_SIZE, frame_index, depth, online=True)

    for cam in cameras:
        trackers[cam] = updateTemporalGallery(trackers[cam], states_cameras[cam], frames_aux[cam], depth[cam], reid,
                                          frame_index, GALLERY_SIZE, SAVE_APPEARANCE_PATCHES, online=True)

    trackers, old_trackers, LUT_delete = DA_crossCameras_Hungarian(trackers, cameras, singleTrackers, old_trackers, LUT_delete)
    trackers = DKF(trackers, cameras)
    trackers, old_trackers, LUT_delete = cleanTrackers(trackers, cameras, old_trackers, LUT_delete)

    # stop = False
    # for cam in cameras:
    #     lut = [track.LUT for track in trackers[cam] if track.id is not None]
    #     if len(lut) > 0:
    #         print('cam', cam)
    #         print(lut)
    #         print('\n')
    #         stop = True
    visualizeResults = visualize(cameras, frames, frame_index, trackers, states_cameras, depth, online=True)
    # if stop:
    #     input('p')
    return trackers, old_trackers, LUT_delete


def PerceptionModuleInitialization(cameras):
    trackers, old_trackers, LUT_delete, detector, reid, results_pymot = InitTrackingSystem(cameras)

    coefficients = define_environment_limits()
    communications(len(cameras))
    singleTrackers = getTrackers(cameras)
    print('Perception module initialized')
    return singleTrackers, trackers, old_trackers, LUT_delete, detector, reid, coefficients, results_pymot


def PerceptionModule(client, cameras, drones, singleTrackers, trackers, old_trackers, LUT_delete, detector, reid, coefficients, results_pymot, frame_index):
    frames, depth, states_cameras = {}, {}, {}
    for d in drones:
        responses = getImagesResponse(client, [d.name])
        frames[d.name], depth[d.name] = getRGB_D(responses, [d.name])
        # frames[d.name], depth[d.name] = d.get_rgb_d(client)
        # print(d.get_extrinsic(client))
        states_cameras[d.name] = d.get_extrinsic(client)

    trackers, old_trackers, LUT_delete = MCMT_DTM(trackers, cameras, frames, depth, frame_index,  detector, reid, old_trackers, LUT_delete, states_cameras, singleTrackers, coefficients)
    for i, cam in enumerate(cameras):
        # Save results
        depth_matrix = depth[cam]
        states_cam_frame = states_cameras[cam]
        results_pymot[cam].save_results(frame_index, trackers[cam], states_cam_frame, depth_matrix, frames[cam], online=True)

    return trackers, old_trackers, LUT_delete, frames, depth, states_cameras

