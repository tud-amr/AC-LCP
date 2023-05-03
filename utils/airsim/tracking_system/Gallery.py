import numpy as np
import scipy
from tracking_system.Utilities.geometryCam import from3dCylinder, cutImage


def updateTemporalGallery(trackers, cam_state, frame, depth_matrix, reid, frame_index, GALLERY_SIZE, SAVE_APPEARANCE_PATCHES, online=False):
    """
    Save n patches of each tracker to compare in re-identification
    """
    for track in trackers:
        if len(track.appearance) == 0:
            cylinder = track.cylinder
            bbox = from3dCylinder(cam_state, depth_matrix, cylinder, online)
            patch = cutImage(frame, bbox)
            if patch is not None and patch.shape[0]/patch.shape[1] > 1 and patch.shape[0]/patch.shape[1] < 5: #5
                features = reid.extractFeatures(patch)[0]
                track.appearance.append(features)
                track.weight_app.append(0)
    if frame_index % SAVE_APPEARANCE_PATCHES == 0:
        for track in trackers:
            if track.detectorFound:
                cylinder = track.cylinder
                bbox = from3dCylinder(cam_state, depth_matrix, cylinder, online)
                patch = cutImage(frame, bbox)
                if patch is not None and patch.shape[0]/patch.shape[1] > 1.5 and patch.shape[0]/patch.shape[1] < 5: #5
                    features = reid.extractFeatures(patch)[0]
                    if len(track.appearance) < GALLERY_SIZE:
                        track.appearance.append(features)
                        track.weight_app.append(0)
                    elif len(track.appearance) == GALLERY_SIZE:
                        track.appearance.pop(0)
                        track.weight_app.pop(0)
                        
                        track.appearance.append(features)
                        track.weight_app.append(0)
    return trackers


def updateScoreGallery(trackers, cam_state, frame, depth_matrix, reid, frame_index, GALLERY_SIZE, SAVE_APPEARANCE_PATCHES, online):
    """
    Save n patches of each tracker to compare in re-identification
    """
    for track in trackers:
        if len(track.appearance) == 0:
            cylinder = track.cylinder
            bbox = from3dCylinder(cam_state, depth_matrix, cylinder, online)
            patch = cutImage(frame, bbox)
            if patch is not None and patch.shape[0]/patch.shape[1] >= 1.5 and patch.shape[0]/patch.shape[1] < 5: #5
                features = reid.extractFeatures(patch)[0]
                track.appearance.append(features)
                track.weight_app.append(0)
                track.app_model_frame_index.append(frame_index)
    if frame_index % SAVE_APPEARANCE_PATCHES == 0:
        for track in trackers:
            if track.detectorFound:
                cylinder = track.cylinder
                bbox = from3dCylinder(cam_state, depth_matrix, cylinder, online)
                patch = cutImage(frame,bbox)
                if patch is not None and patch.shape[0]/patch.shape[1] >= 1.5 and patch.shape[0]/patch.shape[1] < 5: #5
                    features = reid.extractFeatures(patch)[0]
                    if len(track.appearance) < GALLERY_SIZE:
                        track.appearance.append(features)
                        track.weight_app.append(0)
                        track.app_model_frame_index.append(frame_index)
                    elif len(track.appearance) == GALLERY_SIZE:
                        track = update_score_track(track)
                        track.appearance.append(features)
                        track.weight_app.append(0)
                        track.app_model_frame_index.append(frame_index)
                        
    return trackers


def update_score_track(track):
    mean = np.mean(track.appearance, axis = 0)
    dist = scipy.spatial.distance.cdist(track.appearance,[mean],'cosine')
    closest = np.argmin(dist)

    farest = np.argmax(dist)
    track.weight_app[closest] += 1
    track.weight_app[farest] -= 1
    min_score = np.argmin(track.weight_app)
    track.appearance.pop(min_score)
    track.weight_app.pop(min_score)
    track.app_model_frame_index.pop(min_score)

    return track