import cv2
import numpy as np
import copy

from detection_module import Detector
from Utilities.Data import initVideos, loadCamStates, getDepth_matrix
from TMsingleView import getDetectorInfo
from tracking_system.CrossCameras import DKF_singleView, communications, DA_crossCameras_Hungarian, DKF
from tracking_system.Trackers.camerasTracker import SingleCameraTracker0, SingleCameraTracker1, SingleCameraTracker2, SingleCameraTracker3
from tracking_system.saveResultsTracker import SaveResults_pymotFormat
from tracking_system.associationSingleView import DA_singleView_Hungarian
from tracking_system.reId import ReId
from tracking_system.Gallery import updateTemporalGallery, updateScoreGallery
from tracking_system.TMsingleView import initTrackers, cleanTrackers, define_environment_limits
from tracking_system.Utilities.geometryCam import from3dCylinder, cropBbox
from check_projectiosn import test_projections

C_WHITE = (255, 255, 255)

GALLERY_SIZE = 20
SAVE_APPEARANCE_PATCHES = 2
N_DETECTIONS = 10


def InitCameraSystem(cameras, mode_data):
    video_ok = False
    videos, frames, state_cameras = None, None, None
    if mode_data == 'load_from_folder':
        nframes = 0
        videos, frames, state_cameras = {}, {}, {}
        for cam in cameras:
            video, frame, video_ok = initVideos(cam)
            nframes = max(nframes, int(video.get(cv2.CAP_PROP_FRAME_COUNT)))

            videos[cam] = video
            frames[cam] = frame
            state_cameras[cam] = loadCamStates(cam)
            # assert len(state_cameras[cam]) == nframes, 'Error len, state camera {} while num frames {}'.format(len(state_cameras[cam]), nframes)

    return videos, frames, state_cameras, video_ok


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


def visualize(cameras, frames, frame_index, trackers, states_cameras):
    if len(frames) == 4:
        row_frames1, row_frames2, final_frames= [],  [], []
        for i, cam in enumerate(cameras):
            cam_state = states_cameras[cam][frame_index]
            depth_matrix = getDepth_matrix(cam, frame_index)
            for track in trackers[cam]:
                # if track.id is not None:
                # Get bbox from cylinder
                cylinder_e = track.cylinder
                bbox_e = from3dCylinder(cam_state, depth_matrix, cylinder_e)
                bbox_e = cropBbox(bbox_e,frames[cam])
                # Get bbox parameters
                xmin, ymin, xmax, ymax = np.int0(bbox_e.getAsXmYmXMYM())
                width = xmax - xmin
                height = ymax - ymin
                #Draw bbox
                cv2.rectangle(frames[cam], (xmin, ymin), (xmax, ymax), track.colorTOshow, 1)
                cv2.putText(frames[cam], str(track.id), ((xmin + np.int0(width / 2)), (ymin + np.int0(height / 2))), cv2.FONT_HERSHEY_SIMPLEX, 0.5, track.colorTOshow, 1)
            cv2.putText(frames[cam], str(cam), (30, frames[cam].shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
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


def MCMT_DTM(trackers, cameras, frames, frame_index,  detector, reid, old_trackers, LUT_delete, states_cameras, singleTrackers, out, coefficients = None):
    frames_aux = copy.deepcopy(frames)
    detector_results = getDetectorInfo(detector, frames, cameras,frame_index, N_DETECTIONS, states_cameras, coefficients)

    for cam in cameras:
        for bbox in detector_results[cam]:
            xmin, ymin, xmax, ymax = bbox.getAsXmYmXMYM()
            cv2.rectangle(frames[cam], (int(xmin), int(ymin)), (int(xmax), int(ymax)), C_WHITE, 1)

    detectors_unused = {}
    for cam in cameras:
        # print(cam)
        depth_matrix = getDepth_matrix(cam, frame_index)
        states_cameras_frame = states_cameras[cam][frame_index]
        trackers[cam], detectors_unused[cam] = DA_singleView_Hungarian(trackers[cam], states_cameras_frame, detector_results[cam], depth_matrix, reid, frames_aux[cam])

    trackers, old_trackers, LUT_delete = initTrackers(states_cameras, trackers, detectors_unused, old_trackers,
                                                      LUT_delete, cameras, singleTrackers, GALLERY_SIZE, frame_index)

    for cam in cameras:
        depth_matrix = getDepth_matrix(cam, frame_index)
        states_cameras_frame = states_cameras[cam][frame_index]
        trackers[cam] = updateTemporalGallery(trackers[cam], states_cameras_frame, frames_aux[cam], depth_matrix, reid,
                                          frame_index, GALLERY_SIZE, SAVE_APPEARANCE_PATCHES)

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
    visualizeResults = visualize(cameras, frames, frame_index, trackers, states_cameras)
    out.write(visualizeResults)
    # if stop:
    #     input('p')
    return trackers, old_trackers, LUT_delete, out


if __name__ == "__main__":
    cameras = ['Drone1', 'Drone2', 'Drone3', 'Drone4']

    videos, frames, states_cameras, video_ok = InitCameraSystem(cameras, mode_data='load_from_folder')
    trackers, old_trackers, LUT_delete, detector, reid, results_pymot = InitTrackingSystem(cameras)

    coefficients = define_environment_limits()
    communications(len(cameras))
    singleTrackers = getTrackers(cameras)

    out = cv2.VideoWriter('TrackingStaticCameras.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 20.0, (2*frames[cameras[0]].shape[1], 2*frames[cameras[0]].shape[0]))

    frame_index = 0
    while video_ok:
        print('frame index', frame_index)
        trackers, old_trackers, LUT_delete, out = MCMT_DTM(trackers, cameras, frames, frame_index,  detector, reid, old_trackers, LUT_delete, states_cameras, singleTrackers, out, coefficients)
        for cam in cameras:
            # Read a new frame
            video_ok, frames[cam] = videos[cam].read()

            # Save results
            depth_matrix = getDepth_matrix(cam, frame_index)
            states_cam_frame = states_cameras[cam][frame_index]
            results_pymot[cam].save_results(frame_index, trackers[cam], states_cam_frame, depth_matrix, frames[cam])
        frame_index += 1

    for cam in cameras:
        folder = 'Baseline_Slow_DEFK/Rx1.5'
        results_pymot[cam].close_saveFile(cam, folder)
    cv2.destroyAllWindows()
    out.release()
