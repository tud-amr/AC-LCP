import numpy as np
from scipy.spatial import distance
from tracking_system.Utilities.geometry2D_utils import Bbox, f_noise
from tracking_system.Utilities.geometryCam import to3dCylinder
from tracking_system.Trackers.cylinderModelEKF import CylinderModel
from tracking_system.Utilities.Data import getDepth_matrix


FRAMES_DETECTED = 2
FRAMES_LOST = 15
MIN_LEN_GALLERY = 2


def define_environment_limits():
    """
    :param limit_coordinates: list of coordinates that define the limit of the valid space: [(x0,y0), (x1,y1)]
    """
    global SLOPE, B

    limit_coordinates = [(-18.7, -10.4), (10.8, -10.4)]

    x_coords, y_coords = zip(*limit_coordinates)
    coefficients = np.polyfit(x_coords, y_coords, 1)
    SLOPE = coefficients[0]
    B = coefficients[1]


def getDetectorInfo(detector, frames, cameras, frame_index, N_DETECTIONS, states_cameras, depth_info=None, coefficients=None, online=False):
    """
    Obtain people bounding box and clean unlikely person detections (noise) 
    """

    detector_results = {}
    if frame_index % N_DETECTIONS == 0:
        frames_cameras = [frames[cam] for cam in cameras]
        detections = detector.evaluateImages(frames_cameras)

        for i, cam in enumerate(cameras):
            bboxes_img = [list(map(int, bbox)) for bbox in detections[i]]
            detector_results[cam] = [(Bbox.XmYmXMYM(result[0], result[1], result[2], result[3])) for result in bboxes_img if result[5] == 0]

        if coefficients is not None:
            detect_valid_pos = {}
            for cam in cameras:
                if online:
                    depth = depth_info[cam]
                    cam_state = states_cameras[cam]
                else:
                    depth = getDepth_matrix(cam, frame_index)
                    cam_state = states_cameras[cam][frame_index]
                detect_valid_pos[cam] = []
                for bbox in detector_results[cam]:
                    x, y = to3dCylinder(cam_state, depth, bbox, online=online).getFeet().getAsXY()
                    y_check = x * coefficients[0] + coefficients[1]
                    if y > y_check:
                        detect_valid_pos[cam].append(bbox)
        else:
            detect_valid_pos = detector_results

        clean_noise = {}
        for cam in cameras:
            main_detections = detect_valid_pos[cam].copy()
            for bbox in detector_results[cam]:
                for bbox_2 in detector_results[cam]:
                    if bbox != bbox_2:
                        small_box, noise = f_noise(bbox, bbox_2)
                        if noise and small_box in main_detections:
                            main_detections.remove(small_box)
            clean_noise[cam] = main_detections
        detector_results = clean_noise
        # clean_detects = {}
        # for cam in cameras:
        #     clean_detects[cam] = []
        #     main_detections = detector_results[cam].copy()
        #     for bbox in clean_noise[cam]:
        #         x,y,w,h = bbox.getAsXmYmWH()
        #         if 1.5 < h/w < 5:
        #             clean_detects[cam].append(bbox)
        #     clean_detects[cam] = main_detections
        # detector_results = clean_detects
    else:
        for cam in cameras:
            detector_results[cam] = []

    return detector_results


def cleanTrackers(trackers, cameras, old_trackers, LUT_delete):
    """
    Clean overlapping trackers and lost trackers
    If there are two trackers of the same person delete the worst and clean overlap trackers
    """
    trackers_updated = {}
    for cam in cameras:
        trackers_updated[cam] = []
        for track in trackers[cam]:
            if track.framesLost < FRAMES_LOST:
                trackers_updated[cam].append(track)
            elif track.framesLost >= FRAMES_LOST and track.id is not None and len(track.appearance) >= MIN_LEN_GALLERY:
                LUT_delete[cam].append(track.LUT)
                old_trackers[cam].append(track)
    return trackers_updated, old_trackers, LUT_delete


def merge_appearance(track, old_track, GALLERY_SIZE):
    # Info tracker
    app1 = track.appearance
    weight_app1 = track.weight_app
    app_frames1 = track.app_model_frame_index

    # Info old tracker
    app2 = old_track.appearance
    weight_app2 = old_track.weight_app
    app_frames2 = old_track.app_model_frame_index

    if (len(app1) + len(app2)) <= GALLERY_SIZE:
        merge_app = app2 + app1
        merge_weights = weight_app2 + weight_app1
        merge_app_frames = app_frames1 + app_frames2
    else:
        dif = (len(app1) + len(app2)) - GALLERY_SIZE
        to_delete = np.int0(np.round(dif / 2)) + 1
        del app1[0:to_delete]
        del app2[0:to_delete]
        del weight_app1[0:to_delete]
        del weight_app2[0:to_delete]
        del app_frames1[0:to_delete]
        del app_frames2[0:to_delete]

        merge_app = app1 + app2
        merge_weights = weight_app1 + weight_app2
        merge_app_frames = app_frames1 + app_frames2

    return merge_app, merge_weights, merge_app_frames


def recover_oldTrack(track, old_track, camera, old_trackers, LUT_delete, GALLERY_SIZE):
    # Merge data
    track.id = old_track.id
    track.color = old_track.color
    track.LUT[camera] = [track.id, track.color]

    track.idTOshow = track.id
    track.colorTOshow = track.color

    merge_app, merge_weights, merge_app_frames = merge_appearance(track, old_track, GALLERY_SIZE)

    track.appearance = merge_app
    track.weight_app = merge_weights
    track.app_model_frame_index = merge_app_frames

    old_trackers[camera].remove(old_track)
    LUT_delete[camera].remove(old_track.LUT)
    return track, old_trackers, LUT_delete


def initTrackers(states_cameras, trackers, detectors_unused, old_trackers, LUT_delete, groupCameras, singleTrackers, GALLERY_SIZE, frame_index, depth_info=None, online=False):
    for camera in groupCameras:
        if online:
            depth = depth_info[camera]
            cam_state = states_cameras[camera]
        else:
            depth = getDepth_matrix(camera, frame_index)
            cam_state = states_cameras[camera][frame_index]

        for bbox in detectors_unused[camera]:
            cylinder = to3dCylinder(cam_state, depth, bbox, online)
            motionModel = CylinderModel(camera)
            motionModel.init(cylinder)

            track = singleTrackers[camera](motionModel, camera, cylinder)
            track.cylinder = cylinder
            track.detectorFound = True
            track.computeInfo(cylinder)
            trackers[camera].append(track)

    for camera in groupCameras:
        for i, track in enumerate(trackers[camera]):
            if track.id is None and track.framesDetected > FRAMES_DETECTED and len(track.appearance) >= MIN_LEN_GALLERY:
                if len(old_trackers[camera]) > 0:
                    app1 = track.appearance
                    min_distances = []
                    for old_track in old_trackers[camera]:
                        app2 = old_track.appearance
                        distances = distance.cdist(app1, app2, 'cosine')
                        min_distances.append(np.amin(distances))
                    global_min = np.amin(min_distances)
                    if global_min <= 0.25:  # recovering old tracker
                        ind = min_distances.index(global_min)
                        old_track = old_trackers[camera][ind]
                        track, old_trackers, LUT_delete = recover_oldTrack(track, old_track, camera, old_trackers, LUT_delete, GALLERY_SIZE)
                    else:
                        track.setID()
                        track.LUT[camera] = [track.id, track.color]
                        # print('init new track in', camera, 'with id', track.id)
                else:
                    track.setID()
                    # print('init new track in', camera, 'with id', track.id)
                    track.LUT[camera] = [track.id, track.color]

    return trackers, old_trackers, LUT_delete

