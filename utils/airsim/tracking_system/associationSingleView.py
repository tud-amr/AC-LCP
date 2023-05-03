import numpy as np
from scipy.optimize import linear_sum_assignment
from tracking_system.Utilities.geometryCam import from3dCylinder, to3dCylinder, compute_geometry_scores, cropBbox, cutImage
from tracking_system.reId import compute_appearance_score, applyAppThreshold
from tracking_system.Utilities.geometry2D_utils import f_iou


GEOM_THRESHOLD = 1
APP_THRESHOLD = 0.5
gamma = 0.5


def saveNoneTrack(none_bbox, trackers_candidates, frame, cam_state, depth_matrix):
    # globa_index = none_index[index_best_NoneTrack]
    # noneTrack = trackers_candidates[globa_index]
    #
    # none_bbox = from3dCylinder(cam_state, depth_matrix, noneTrack.motionModel.getCylinder())
    # none_bbox = cropBbox(none_bbox,fame)
    keep = True
    for track in trackers_candidates:
        if track.id is not None:
            ID_bbox = from3dCylinder(cam_state, depth_matrix, track.motionModel.getCylinder())
            ID_bbox = cropBbox(ID_bbox, frame)
            if f_iou(none_bbox, ID_bbox) >= 0.3:
                keep = False
                break
    return keep


def cleanOverlapNoneTrackers(trackers_candidates, none_index, index_best_NoneTrack, frame, cam_state, depth_matrix):
    global_index = none_index[index_best_NoneTrack]
    bestNoneTrack = trackers_candidates[global_index]

    best_none_bbox = from3dCylinder(cam_state, depth_matrix, bestNoneTrack.motionModel.getCylinder())
    best_none_bbox = cropBbox(best_none_bbox, frame)

    none_index_to_delete = []
    for i, track in enumerate(trackers_candidates):
        if track.id is None and i in none_index and track != bestNoneTrack:
            none_bbox = from3dCylinder(cam_state, depth_matrix, bestNoneTrack.motionModel.getCylinder())
            none_bbox = cropBbox(none_bbox, frame)
            if f_iou(best_none_bbox, none_bbox) >= 0.5:
                none_index_to_delete.append(i)
                break
    # keep = saveNoneTrack(best_none_bbox, trackers_candidates, frame, cam_state, depth_matrix)
    # if not keep:
    #     none_index_to_delete.append(global_index)
    return none_index_to_delete


def cleanNoneTrackers(trackers_candidates, double_trackers, scores_geom, frames, cam_state, depth_matrix, best_none_tracks):
    none_index = [i for i,f in enumerate(trackers_candidates) if f.id is None]
    # print('len trackers candidates pre', len(trackers_candidates))
    # print('len none index', len(none_index))
    # if len(trackers_candidates) > len(none_index) > 1:
    if len(none_index) > 1:
        track_candidates_aux = trackers_candidates.copy()
        detect = [track_candidates_aux[none_i].framesDetected for none_i in none_index]
        # print('frames detected', detect)
        detect = np.array(detect)
        index_best_NoneTracks = np.where(detect == np.max(detect))[0]
        index_best_NoneTracks = index_best_NoneTracks.tolist()
        # print('best track index', index_best_NoneTracks)
        none_index_aux = none_index.copy()
        for best_idx in index_best_NoneTracks:
            best_none_tracks.append(trackers_candidates[none_index[best_idx]])
            none_index_aux.remove(none_index[best_idx])
        none_index = none_index_aux
        # none_index = cleanOverlapNoneTrackers(trackers_candidates, none_index, index_best_NoneTrack, frames, cam_state, depth_matrix)

        scores_geom = [f for i,f in enumerate(scores_geom) if i not in none_index]
        # print('len tracker candidates pre', len(trackers_candidates))
        # print('len none index', len(none_index))
        for none_i in none_index:
            track_none = track_candidates_aux[none_i]
            double_trackers.append(track_none)
            trackers_candidates.remove(track_none)
        # print('len double_trackers candidates post', len(double_trackers))
        # input('p')
    return trackers_candidates, double_trackers, scores_geom, best_none_tracks


def updateMeasurement(cylinder_d, bbox, detections_used, trackers, indexTrack_selected, trackers_associated):
    track = trackers[indexTrack_selected]
    # print('update measurement of tracker', track.id, 'cylinder', cylinder_d.getXYZWH())
    track.detectorFound = True
    track.computeInfo(cylinder_d)
    
    trackers_associated.append(track)
    detections_used.append(bbox)
    
    return detections_used, trackers_associated, track


def hierarchical_association(trackers, detector_results, cam_state, depth, cost_matrix, trackers_to_associate, local_index_trackers, detect_to_associate, detections_used, trackers_associated, online):
    # print('cost matrix to assign \n', cost_matrix)
    detection_associated = []
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    for i, r_idx in enumerate(row_ind):
        detect_idx = detect_to_associate[r_idx]
        # print('index detection', detect_idx)
        detection_associated.append(detect_idx)
        bbox = detector_results[detect_idx]
        cylinder_d = to3dCylinder(cam_state, depth, bbox, online)
        local_track_index = local_index_trackers[col_ind[i]]
        track_index = trackers_to_associate[local_track_index]
        # print('global track index', track_index)
        # print('track associated in single view', trackers[track_index].id)
        # input('p')
        detections_used, trackers_associated, track = updateMeasurement(cylinder_d, bbox, detections_used, trackers, track_index, trackers_associated)
        # if best_patches[r_idx][col_ind[i]] is not None and len(track.weight_app) > 0:
        #     track.weight_app[best_patches[r_idx][col_ind[i]]] += 1
        trackers[track_index] = track
    return trackers, detections_used, trackers_associated, detection_associated


def get_detections_to_associate(cost_matrix, extra_detects_to_delete = None):
    detect_none_candidates, detect_to_associate = [], []
    for i in range(cost_matrix.shape[0]):
        if np.all(cost_matrix[i] == 1):
            detect_none_candidates.append(i)
        else:
            detect_to_associate.append(i)

    if extra_detects_to_delete is not None:
        detect_none_candidates = detect_none_candidates + extra_detects_to_delete
    cost_matrix = np.delete(cost_matrix, detect_none_candidates, axis=0)
    # best_patches = np.delete(best_patches, detect_none_candidates, axis=0)
    # assert cost_matrix.shape[0] == (len(detector_results) - len(detect_none_candidates)) == len(detect_to_associate)
    return cost_matrix, detect_to_associate


def optimal_association(trackers, cam_state, depth, cost_matrix, detector_results, trackers_associated, detections_used, best_patches):
    # print('cost matrix initial \n', cost_matrix)
    # Clean trackers without candidates
    detect_none_candidates, detect_to_associate = [], []
    for i in range(cost_matrix.shape[0]):
        if np.all(cost_matrix[i] == 1):
            detect_none_candidates.append(i)
        else:
            detect_to_associate.append(i)
    cost_matrix = np.delete(cost_matrix, detect_none_candidates, axis=0)

    # Clean trackers without candidates
    trackers_non_valid, trackers_to_associate, trackers_id = [], [], []
    cost_matrix_arr = np.array(cost_matrix)
    for j in range(cost_matrix.shape[1]):
        if np.all(cost_matrix_arr[:,j] == 1):
            trackers_non_valid.append(j)
        else:
            trackers_to_associate.append(j) # idx global
            trackers_id.append(trackers[j].id)
    cost_matrix = np.delete(cost_matrix, trackers_non_valid, axis=1)
    # best_patches = np.delete(best_patches, trackers_to_associate, axis=1)
    assert cost_matrix.shape[1] == (len(trackers) - len(trackers_non_valid)) == len(trackers_to_associate)
    # print('cost matrix after clean detections and trackers \n', cost_matrix)

    # local_id_index = [i for i, track_id in enumerate(trackers_id) if track_id is not None]
    # # local_none_index = [i for i, track_id in enumerate(trackers_id) if track_id is None]
    # for idx in local_id_index:
    #     cost_matrix[:,idx] = 0.1*cost_matrix[:,idx]
    # print('deetect to assiciate', detect_to_associate)
    # print('cost matrix to assign \n', cost_matrix)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    for i, r_idx in enumerate(row_ind):
        detect_idx = detect_to_associate[r_idx]
        bbox = detector_results[detect_idx]
        cylinder_d = to3dCylinder(cam_state, depth, bbox)
        track_index = trackers_to_associate[col_ind[i]]
        detections_used, trackers_associated, track = updateMeasurement(cylinder_d, bbox, detections_used, trackers, track_index, trackers_associated)
        # if best_patches[r_idx][col_ind[i]] is not None and len(track.weight_app) > 0:
        #     track.weight_app[best_patches[r_idx][col_ind[i]]] += 1
        trackers[track_index] = track
    return trackers, detections_used, trackers_associated


def optimal_association_hierarchy(trackers, cam_state, depth, cost_matrix, detector_results, trackers_associated, detections_used, best_patches, online):
    # Clean trackers without candidates
    # print('cost matrix initial \n', cost_matrix)
    trackers_non_valid, trackers_to_associate, trackers_id = [], [], []
    cost_matrix_arr = np.array(cost_matrix)
    for j in range(cost_matrix.shape[1]):
        # print('cost function column', cost_matrix_arr[:,j])
        if np.all(cost_matrix_arr[:,j] == 1):
            trackers_non_valid.append(j)
            # print('id of non valid trackers', trackers[j].id)
        else:
            trackers_to_associate.append(j) # idx global
            trackers_id.append(trackers[j].id)
    cost_matrix = np.delete(cost_matrix, trackers_non_valid, axis=1)
    # best_patches = np.delete(best_patches, trackers_to_associate, axis=1)
    assert cost_matrix.shape[1] == (len(trackers) - len(trackers_non_valid)) == len(trackers_to_associate)
    # print('cost matrix after clean detections and trackers \n', cost_matrix)

    local_id_index = [i for i, track_id in enumerate(trackers_id) if track_id is not None]
    local_none_index = [i for i, track_id in enumerate(trackers_id) if track_id is None]
    #cost_matrix.shape[0] < cost_matrix.shape[1] and
    if len(trackers_to_associate) > len(local_none_index) > 0:
        # 1. Assignment of trackers with identity
        # print('TRACKER WITH IDENTITY ASSOCIATION')
        # print('none index', local_none_index)
        cost_matrix_id = np.delete(cost_matrix, local_none_index, axis=1)
        cost_matrix_id, detect_to_associate = get_detections_to_associate(cost_matrix_id)
        # print('detect_to_associate', detect_to_associate)
        # best_patches = np.delete(best_patches, local_none_index, axis=1)
        assert cost_matrix_id.shape[1] == (len(trackers) - len(trackers_non_valid) - len(local_none_index)) == len(local_id_index)

        trackers, detections_used, trackers_associated, detection_associated = hierarchical_association(trackers, detector_results, cam_state, depth, cost_matrix_id, trackers_to_associate, local_id_index, detect_to_associate, detections_used, trackers_associated, online)
        if len(detection_associated) < len(detect_to_associate):
            # print('TRACKER NONE ASSOCIATION')
            # 2. Assignment of trackers without identity
            cost_matrix_none = np.delete(cost_matrix, local_id_index, axis=1)
            cost_matrix_none, detect_to_associate = get_detections_to_associate(cost_matrix_none, detection_associated)

            global_index_detections = [i for i in detect_to_associate if i not in detection_associated]
            trackers, detections_used, trackers_associated, detection_associated = hierarchical_association(trackers, detector_results, cam_state,depth,cost_matrix_none,trackers_to_associate, local_none_index, global_index_detections, detections_used, trackers_associated, online)
    else:
        # print('DONT CLEANING COST MATRIX')
        # Assigment optimization
        cost_matrix, detect_to_associate = get_detections_to_associate(cost_matrix)
        # print('deetect to assiciate', detect_to_associate)
        # print('cost matrix to assign \n', cost_matrix)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        for i, r_idx in enumerate(row_ind):
            detect_idx = detect_to_associate[r_idx]
            bbox = detector_results[detect_idx]
            cylinder_d = to3dCylinder(cam_state, depth, bbox)
            track_index = trackers_to_associate[col_ind[i]]
            detections_used, trackers_associated, track = updateMeasurement(cylinder_d, bbox, detections_used, trackers, track_index, trackers_associated)
            # if best_patches[r_idx][col_ind[i]] is not None and len(track.weight_app) > 0:
            #     track.weight_app[best_patches[r_idx][col_ind[i]]] += 1
            trackers[track_index] = track
    return trackers, detections_used, trackers_associated


def DA_singleView_Hungarian(trackers, cam_state, detector_results, depth, reid, frames_aux, online=False):
    trackers_associated, detections_used, double_trackers, best_none_trackers = [],[],[], []
    # print('len trackers', len(trackers), 'len detections', len(detector_results))
    # If only exists one detection and one tracker -> check geometry threshold and perform association if it accomplishes
    if detector_results is not None and len(detector_results) == 1 and len(trackers) == 1:
        track = trackers[0]
        bbox = detector_results[0]
        
        cylinder_d = to3dCylinder(cam_state, depth, bbox, online)
        score_geom = compute_geometry_scores(track, cylinder_d)
        if score_geom < GEOM_THRESHOLD:
            track_index = 0
            detections_used, trackers_associated, track = updateMeasurement(cylinder_d, bbox, detections_used, trackers, track_index, trackers_associated)
            trackers[track_index] = track

    # More than one tracker or detection to match
    elif detector_results is not None and len(detector_results) > 0:
        cost_matrix = np.ones((len(detector_results), len(trackers)))
        best_patches = np.full((len(detector_results), len(trackers)), None)
        for i, bbox in enumerate(detector_results):
            cylinder_d = to3dCylinder(cam_state, depth, bbox, online)
            # Relation between trackers and their detection by geometry and appearance
            scores_geom, trackers_candidates = [],[]
            for j, track in enumerate(trackers):
                score_geom = compute_geometry_scores(track, cylinder_d)
                # print('score geom', score_geom, 'track id', track.id)
                if score_geom < GEOM_THRESHOLD: 
                    scores_geom.append(score_geom)
                    trackers_candidates.append(track)

            # There are trackers that are closer to the detection than GEOM_THRESHOLD
            if len(trackers_candidates) > 0:
                trackers_candidates, double_trackers, scores_geom, best_none_tracks = cleanNoneTrackers(trackers_candidates, double_trackers, scores_geom, frames_aux, cam_state, depth, best_none_trackers)
                # print('len track candidates after clean', len(trackers_candidates))
                # Only one clean tracker are closer to the detection than GEOM_THRESHOLD -> avoid use appearance
                # if len(trackers_candidates) == 1:
                #     assert len(scores_geom) == 1, "Error in length score geom"
                #     track_candidate = trackers_candidates[0]
                #     global_tracker_idx = trackers.index(track_candidate)
                #     cost_matrix[i][global_tracker_idx] = scores_geom[0]
                # More than one clean tracker are closer to the detection than GEOM_THRESHOLD -> use of appearance
                # if len(trackers_candidates) > 1:
                scores_app, bestPatches_index = compute_appearance_score(reid, trackers_candidates, frames_aux, bbox)
                # print('scores app', scores_app)
                # scores_app, scores_geom, trackers_candidates, bestPatches_index = applyAppThreshold(scores_app, scores_geom, trackers_candidates, APP_THRESHOLD,  bestPatches_index)
                # print('score_app', scores_app)
                # if len(trackers_candidates) > 0:
                assert len(scores_geom) == len(scores_app)
                # Global scores
                global_trackers_idx = [trackers.index(track) for track in trackers_candidates]
                global_scores = [(1-gamma)*a + gamma*b for a, b in zip(scores_geom, scores_app)]
                # global_scores = [a * b for a, b in zip(scores_geom, scores_app)]
                for s, j in enumerate(global_trackers_idx):
                    cost_matrix[i][j] = global_scores[s]
                    best_patches[i][j] = bestPatches_index[s]
        # Perform optimal association minimizing the cost matrix
        trackers, detections_used, trackers_associated = optimal_association_hierarchy(trackers, cam_state, depth, cost_matrix, detector_results, trackers_associated, detections_used, best_patches, online)

    final_trackers = trackers.copy()
    for track in trackers:
        if track in double_trackers and track not in best_none_trackers:
            final_trackers.remove(track)

    for track in final_trackers:
        if track not in trackers_associated:
            track.detectorFound = False
            track.computeInfo(None)

    detections_unused = []
    for bbox in detector_results:
        if bbox not in detections_used:
            detections_unused.append(bbox)
    return final_trackers, detections_unused