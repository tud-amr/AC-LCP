import cv2
import numpy as np
import copy

from scipy.optimize import linear_sum_assignment
from tracking_system.reId import checkAPP_crossTrackers, applyAppThreshold
from tracking_system.Trackers.cylinderModelEKF import CylinderModel
from scipy.spatial import distance
from tracking_system.Utilities.geometry3D_utils import Cylinder, f_euclidian_ground

EUCL_DIST = 5#50
gamma = 0.5


def communications(ncameras):
    global ADJ
    ADJ = np.zeros((ncameras,ncameras))
    np.fill_diagonal(ADJ, 0)


def currentIDS(trackers):
    """
    Give the current trackers in one camera
    input: trackers in 1 camera
    output: current ids
    """
    ids = []
    colors = []
    current_trackers = []
    for track in trackers:
        if track.id is not None:
            ids.append(track.id)
            colors.append(track.color)
            current_trackers.append(track)
    return ids, colors, current_trackers


def LuT(trackers):
    """
    Give the Look Up Table of the current trackers in one camera
    input: trackers in 1 camera
    output: Look Up Table of trackers
    """
    LUT = []
    current_trackers = []
    for track in trackers:
        if track.id is not None:
            LUT.append(track.LUT)
            current_trackers.append(track)
    return LUT, current_trackers


def checkLUT_Delete(LUT_delete, camera2, old_trackers):
    old_tracks_candidates = []
    for i, lut in enumerate(LUT_delete):
        if camera2 not in lut.keys():
            old_tracks_candidates.append(old_trackers[i])
    return old_tracks_candidates


def lostConsensus(groupTrackers):
    nframes = []
    for track in groupTrackers:
        nframes.append(track.framesLost)
    minimum = min(nframes)
    return minimum


def checkNeighbour(groupCameras, node1, node2):
    """
    Check if both nodes are neigbours
    """
    out = 0
    for i, camera in enumerate(groupCameras):
        for j, camera2 in enumerate(groupCameras):
            if camera == node1 and camera2 == node2:
                out = ADJ[i][j]
                break
    if out == 1:
        return True
    else:
        return False


def unrelatedTrackers(aux_trackers, camera, camera2):
    camera_LUT, current_trackers = LuT(aux_trackers[camera])
    camera2_IDS, _, current_trackers2 = currentIDS(aux_trackers[camera2])
    
    # Identities in camera2 that none tracker of the current camera has related
    ids2_related = [f[camera2][0] for f in camera_LUT if camera2 in f.keys()]
    unrelated_trackers2 = [f for i,f in enumerate(current_trackers2) if (camera2_IDS[i] not in ids2_related) and (camera2_IDS[i] is not None)]
    
    # Trackers in the current camera unrealted with camera2
    unrelated_trackers1 = [f for i,f in enumerate(current_trackers) if camera2 not in camera_LUT[i].keys()]

    return unrelated_trackers1, unrelated_trackers2


def DA_trackers_Hungarian(aux_trackers, unrelated_trackers1, unrelated_trackers2, camera, camera2, alg):
    cost_matrix = np.ones((len(unrelated_trackers1), len(unrelated_trackers2)))
    
    for i, track1 in enumerate(unrelated_trackers1):
        x1_p = track1.motionModel.mean
        cylinder1_p = Cylinder.XYZWH(*x1_p[:5].copy())
        score_geom, trackers_candidates = [], []
        # Check geometry to obtain candidates to association
        for track2 in unrelated_trackers2:                      
            x2_p = track2.motionModel.mean
            cylinder2_p = Cylinder.XYZWH(*x2_p[:5].copy())
            dist = f_euclidian_ground(cylinder1_p.getCenter(), cylinder2_p.getCenter())
            # print('eucl dist',  dist)
            s_dist = dist/EUCL_DIST
            if s_dist < 1:
                score_geom.append(s_dist)
                trackers_candidates.append(track2)
            
        # Check appearance of candidates
        if len(trackers_candidates) > 0:
            if 'G' in alg:
                best_patches = np.argsort(track1.weight_app)[::-1]
                app1 = [track1.appearance[best_patches[0]], track1.appearance[best_patches[1]]]
            else:
                app1 = track1.appearance[0:2]
                
            score_app = []
            for track2 in trackers_candidates: 
                if 'G' in alg:
                    best_patches = np.argsort(track2.weight_app)[::-1]
                    app2 = [track2.appearance[best_patches[0]], track2.appearance[best_patches[1]]]
                else:
                    app2 = track2.appearance[0:2]
                    
                distances = distance.cdist(app1, app2, 'cosine')
                score_app.append(np.amin(distances))
            
            assert len(score_app) == len(score_geom)
            # Obtain global scores
            trackers_idx2 = [unrelated_trackers2.index(track2) for track2 in trackers_candidates]
            global_scores = [(1 - gamma) * a + gamma * b for a, b in zip(score_geom, score_app)]
            assert len(global_scores) == len(trackers_idx2)
            
            for s, j in enumerate(trackers_idx2):
                cost_matrix[i][j] = global_scores[s]

    # Clean detections without candidates
    trackers_none_candidates1, trackers_to_associate1 = [], []
    for i in range(cost_matrix.shape[0]):
        if np.all(cost_matrix[i] == 1):
            trackers_none_candidates1.append(i)
        else:
            trackers_to_associate1.append(i)
            
    cost_matrix = np.delete(cost_matrix, trackers_none_candidates1, axis = 0)
    assert cost_matrix.shape[0] == (len(unrelated_trackers1) - len(trackers_none_candidates1)) == len(trackers_to_associate1)    
    
    # Assigment optimization
    row_ind, col_ind = linear_sum_assignment(cost_matrix)            
    for i, r_idx in enumerate(row_ind):
        track1_idx = trackers_to_associate1[r_idx]
        track1 = unrelated_trackers1[track1_idx]

        track2_idx = col_ind[i]
        track2 = unrelated_trackers2[track2_idx]
        # print('track', track1.camera, track1.id, 'realted with track', track2.camera, track2.id)
        track1.LUT[camera2] = [track2.id, track2.color]
        global_idx_track1 = aux_trackers[camera].index(track1)
        aux_trackers[camera][global_idx_track1] = track1
    return aux_trackers


def DA_crossCameras_Hungarian(trackers, groupCameras, singleTrackers, old_trackers, LUT_delete, alg='Temporal'):
    next_trackers = {}
    for camera in groupCameras:
        # print('camera', camera)
        aux_trackers = copy.deepcopy(trackers)
        for camera2 in groupCameras:
            if checkNeighbour(groupCameras, camera, camera2):
                # print('camera2', camera2)
                # Get unrelated trackers from current camera and unrelated trackers received
                unrelated_trackers1, unrelated_trackers2 = unrelatedTrackers(aux_trackers, camera, camera2)
                unrelated_trackers1_id = [track.id for track in unrelated_trackers1]
                unrelated_trackers2_id = [track.id for track in unrelated_trackers2]
                # print('INIT')
                # print('ids', camera, 'unrealted', unrelated_trackers1_id)
                # print('ids', camera2, 'unrealted', unrelated_trackers2_id)

                # CHECK APPEARANCE
                aux_trackers = checkAPP_crossTrackers(aux_trackers, unrelated_trackers1, unrelated_trackers2, camera, camera2)
                unrelated_trackers1, unrelated_trackers2 = unrelatedTrackers(aux_trackers, camera, camera2)
                unrelated_trackers1_id = [track.id for track in unrelated_trackers1]
                unrelated_trackers2_id = [track.id for track in unrelated_trackers2]
                # print('AFTER APP')
                # print('ids', camera, 'unrealted', unrelated_trackers1_id)
                # print('ids', camera2, 'unrealted', unrelated_trackers2_id)

                # DA TRACKERS
                aux_trackers = DA_trackers_Hungarian(aux_trackers, unrelated_trackers1, unrelated_trackers2, camera, camera2, alg)
                unrelated_trackers1, unrelated_trackers2 = unrelatedTrackers(aux_trackers, camera, camera2)
                unrelated_trackers1_id = [track.id for track in unrelated_trackers1]
                unrelated_trackers2_id = [track.id for track in unrelated_trackers2]
                # print('AFTER HUNG')
                # print('ids', camera, 'unrealted', unrelated_trackers1_id)
                # print('ids', camera2, 'unrealted', unrelated_trackers2_id)

                # INIT NOT RELATED TRACKERS
                for unrelated_track2 in unrelated_trackers2:
                    if unrelated_track2.framesLost < 5:
                        cylinder = unrelated_track2.motionModel.getCylinder()
                        motionModel = CylinderModel(camera)
                        motionModel.init(cylinder)

                        track_new = singleTrackers[camera](motionModel, camera, cylinder)
                        track_new.cylinder = cylinder
                        track_new.motionModel.mean = unrelated_track2.motionModel.mean
                        
                        if 'G' in alg:
                            best_patches = np.argsort(unrelated_track2.weight_app)[::-1]
                            app_selected = [unrelated_track2.appearance[best_patches[0]], unrelated_track2.appearance[best_patches[1]]]
                            patches_selected = [unrelated_track2.app_model_frame_index[best_patches[0]], unrelated_track2.app_model_frame_index[best_patches[1]]]
                        else:
                            app_selected = unrelated_track2.appearance[0:2]    
                            patches_selected = unrelated_track2.app_model_frame_index[0:2]
                            
                        track_new.appearance = app_selected
                        track_new.weight_app = [0,0]
                        track_new.app_model_frame_index = patches_selected
                        
                        track_new.detectorFound = False 
                        track_new.computeInfo(None)
                        track_new.LUT[camera2] = [unrelated_track2.id, unrelated_track2.color]
                        
                        # Check if tracker's 2 identity had been related before with a tracker to assign the same identity in the current camera
                        LUT_delRelatedCam2 = [f for f in LUT_delete[camera] if camera2 in f.keys()]
                        ids_delete = [f[camera2][0] for f in LUT_delRelatedCam2]
                        if unrelated_track2.id in ids_delete:
                            LUT_i = LUT_delRelatedCam2[ids_delete.index(unrelated_track2.id)]
                            track_new.LUT[camera] = LUT_i[camera]

                            track_new.id = LUT_i[camera][0]
                            track_new.color = LUT_i[camera][1]
                            
                            # Remove data reinitialized
                            LUT_delete[camera].remove(LUT_i)
                            for old_track in old_trackers[camera]:
                                if old_track.id == track_new.id:
                                    track_new.appearance = track_new.appearance + old_track.appearance[2:len(old_track.appearance)]
                                    track_new.weight_app = track_new.weight_app + old_track.weight_app[2:len(old_track.weight_app)] 
                                    track_new.app_model_frame_index = track_new.app_model_frame_index + old_track.app_model_frame_index[2:len(old_track.app_model_frame_index)]
                                    
                                    assert len(track_new.appearance) == len(track_new.weight_app)
                                    
                                    old_trackers[camera].remove(old_track)
                        else:
                            track_new.setID()
                            track_new.LUT[camera] = [track_new.id, track_new.color]
                        # print('init new track with id', track_new.id,' in cam', camera, 'from cam', unrelated_track2.camera, 'with id', unrelated_track2.id)
                        # input('p')
                        aux_trackers[camera].append(track_new)
        next_trackers[camera] = aux_trackers[camera]

    return next_trackers, old_trackers, LUT_delete


def DKF_singleView(trackers, groupCameras):
    next_trackers = trackers.copy()
    for camera in groupCameras:
        for i, track in enumerate(trackers[camera]):
            next_track = next_trackers[camera][i]
            next_track.update_SV(track.S, track.y)
            next_trackers[camera][i] = next_track
    return next_trackers


def update_tracker(camera, track, next_trackers, track_index):
    next_track = next_trackers[camera][track_index]
    next_track.update(track.S, track.y, np.zeros((1, 8))[0])
    next_trackers[camera][track_index] = next_track
    return next_trackers


def get_consensus(next_trackers, groupTrackers, track, tracker_index, camera):
    # print('inside consensus')
    minimum = lostConsensus(groupTrackers)
    # print(camera, 'track id', track.id, 'neighbors', len(groupTrackers))
    sum_y = np.zeros((1, 8))
    sum_S = np.zeros((8, 8))
    for track2 in groupTrackers:
        sum_S = sum_S + track2.S
        sum_y = sum_y + track2.y
    sum_y = sum_y[0]

    difference = np.zeros((1, 8))[0]
    # print('state track', track.id, camera, track.motionModel.mean)
    for track2 in groupTrackers:
        if track != track2:
            # print('state track2', track2.id, track2.camera, track2.motionModel.mean)
            # print('S', track2.S, 'y', track2.y)
            dif_pre = track2.motionModel.mean - track.motionModel.mean
            difference = difference + dif_pre
    # print('dif', difference)
    next_track = next_trackers[camera][tracker_index]
    # print('track id pre update', next_track.id)
    next_track.update(sum_S, sum_y, difference)
    # print('state track', next_track.id, camera, next_track.x)
    # input('p')
    next_track.framesLost = minimum
    next_trackers[camera][tracker_index] = next_track
    return next_trackers


def DKF(trackers, groupCameras):
    next_trackers = copy.deepcopy(trackers)
    for camera in groupCameras:
        # print('inside DKF CAMERA', camera)
        # print('camera1', camera)
        for i, track in enumerate(trackers[camera]):
            # print('inside dkf')
            # print('track id', track.id)
            if track.id is None and track.S is not None:
                next_trackers = update_tracker(camera, track, next_trackers, i)
            elif track.id is not None:
                groupTrackers = []
                if track.detectorFound:
                    groupTrackers.append(track)
                for camera2 in groupCameras:
                    # print('camera2', camera2)
                    if checkNeighbour(groupCameras, camera, camera2) and camera2 in track.LUT.keys():
                        for track2 in trackers[camera2]:
                            if track.LUT[camera2][0] == track2.id and track2.detectorFound:
                                # print('track2 id', track2.id)
                                groupTrackers.append(track2)
                if len(groupTrackers) == 1:
                    next_trackers = update_tracker(camera, groupTrackers[0], next_trackers, i)
                if len(groupTrackers) > 1:
                    next_trackers = get_consensus(next_trackers, groupTrackers, track, i, camera)
                else:
                    next_trackers = update_tracker(camera, track, next_trackers, i)
    return next_trackers
