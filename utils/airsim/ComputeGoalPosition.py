import numpy as np
import math

from scipy.spatial.transform import Rotation as R
from tracking_system.Utilities.geometryCam import from3dCylinder, cropBbox

THRESHOLD_POS = 2
STEP_TRANSLATION = 1
STEP_ORIENTATION = 15


def get_new_3dpos(pos_3d, cam_state):
    x_p = pos_3d[0]
    y_p = pos_3d[1]

    translation, rot_matrix, E = cam_state
    print('drone position', translation)
    x_diff = x_p - translation[0][0]
    y_diff = y_p - translation[1][0]
    x_diff_abs = abs(x_diff)
    y_diff_abs = abs(y_diff)

    step_x = float(np.sign(x_diff) * STEP_TRANSLATION)
    step_y = float(np.sign(y_diff) * STEP_TRANSLATION)

    final_x = translation[0][0] + step_x
    final_y = translation[1][0] + step_y
    new_pos = None
    if x_diff_abs < THRESHOLD_POS < y_diff_abs:
        print('movement in y')
        new_pos = [translation[0][0], final_y, -3]
    elif y_diff_abs < THRESHOLD_POS < x_diff_abs:
        print('movement in x')
        new_pos = [final_x, translation[1][0], -3]
    elif y_diff_abs > THRESHOLD_POS and x_diff_abs > THRESHOLD_POS:
        print('movement in x and y')
        new_pos = [final_x, final_y, -3]
    else:
        print('not move')
    # new_pos = [5,-5,-3]
    return new_pos


def get_new_yaw(mean_points, frame, cam_state):
    fx = 320
    central_point = int(frame.shape[1] / 2)
    px_variations = central_point - mean_points


    translation, rot_matrix, E = cam_state
    r = R.from_matrix(rot_matrix)
    yaw = r.as_euler('xyz', degrees=True)[2]

    yaw_variation = math.atan(px_variations / fx)
    yaw_variation = yaw_variation*(180/math.pi)

    step_orientation = float(np.sign(px_variations) * STEP_ORIENTATION)

    if yaw_variation < 10:
        new_yaw = yaw
    else:
        new_yaw = yaw + step_orientation
    # new_yaw = 150
    return new_yaw


def get_next_position_and_orientation(camera_names, trackers, cam_states, frames, depth_matrix):
    new_goals = {}
    for cam in camera_names:
        frame = frames[cam]
        cam_state = cam_states[cam]
        depth = depth_matrix[cam]

        mean_points, pos_3d = [], []
        for track in trackers[cam]:
            if track.id is not None:
                cylinder = track.cylinder
                x3d, y3d, width, height = cylinder.getXYWH()
                print('global pos', x3d, y3d)

                bbox_e = from3dCylinder(cam_state, depth, cylinder, online=True)
                bbox_e = cropBbox(bbox_e, frame)
                # Get bbox parameters
                xmin, ymin, xmax, ymax = np.int0(bbox_e.getAsXmYmXMYM())
                mean_point = (xmax - xmin) / 2

                mean_points.append(mean_point)
                pos_3d.append([x3d, y3d])
        if len(pos_3d) > 0 and len(mean_points) > 0:
            mpoint = np.mean(mean_points)
            new_yaw = get_new_yaw(mpoint, frame, cam_state)

            mean_3dpos = np.mean(pos_3d, axis=0)
            print('mean pos', mean_3dpos)
            new_pos = get_new_3dpos(mean_3dpos, cam_state)
            new_goals[cam] = (new_pos, new_yaw)
            # new_goals[cam] = None
            print('nueva posicion', new_pos)
            print('nuevo angulo', new_yaw)
        else:
            new_goals[cam] = None
    return new_goals


