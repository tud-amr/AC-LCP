import json
import sys

import airsim
import numpy as np

OVERPECENT = 0.05
PEDESTRIAN_WIDTH = 1.0
PEDESTRIAN_HEIGHT = 1.75
ORIGIN_COORD_UNREAL = [0, 0, -1.7]


def define_environment_limits_line(limit_coordinates):
    """
    :param limit_coordinates: list of coordinates that define the limit of the valid space: [(x0,y0), (x1,y1)]
    """
    global coefficients, coefficients2

    x_coords, y_coords = zip(*limit_coordinates)
    coefficients = np.polyfit(x_coords, y_coords, 1)
    coefficients2 = None


def define_environment_limits_x_value(x_value, x2_value=None):
    """
    :param limit_coordinates: list of coordinates that define the limit of the valid space: [(x0,y0), (x1,y1)]
    """
    global coefficients, coefficients2
    if x2_value is None:
        coefficients = [0, x_value]
        coefficients2 = None
    else:
        coefficients = [0, x_value]
        coefficients2 = [0, x2_value]


def define_environment_limits_y_value(y_value, y2_value=None):
    """
    :param limit_coordinates: list of coordinates that define the limit of the valid space: [(x0,y0), (x1,y1)]
    """
    global coefficients, coefficients2
    if y2_value is None:
        coefficients = [0, y_value]
        coefficients2 = None
    else:
        coefficients = [0, y_value]
        coefficients2 = [0, y2_value]


def get_dict_colors():
    file = '/home/scasao/pytorch/multi-target_tracking/segmentation_rgb.txt'
    color_dict = {}
    with open(file) as file:
        for line in file:
            key, value = line.split('\t')
            value = value.replace('[', ' ')
            value = value.replace(']', ' ')
            r, g, b = value.split(',')

            color_dict[int(key)] = [int(r), int(g), int(b)]
    return color_dict


def get_name_pedestrians(client, ref_struct, segmentation=False):
    name_pedestrians = client.simListSceneObjects(ref_struct + '.*')
    if segmentation:
        colors = get_dict_colors()
        color_to_pedestrian = {}
        for i, name_ped in enumerate(name_pedestrians):
            client.simSetSegmentationObjectID(name_ped, i + 1, True)
            color_to_pedestrian[name_ped] = colors[i + 1]

        with open('color_to_pedestrian.json', 'w') as f:
            json.dump(color_to_pedestrian, f)
    return name_pedestrians


def check_save(config, frames, global_3dpos):
    save = False
    camera_names = config.camera_names
    for cam in camera_names:
        translation, rot_matrix = config.get_pose(cam)
        f_px = config.get_focal_length_px(cam, frames[cam]['rgb'])
        bbox_2d = get_gt_bboxes(global_3dpos, frames[cam]['rgb'], translation, rot_matrix, f_px)
        if bbox_2d is not None:
            xmin, ymin, xmax, ymax = cropBbox(bbox_2d, frames[cam]['rgb'])
            width = xmax - xmin
            height = ymax - ymin
            if width > 0 and height > 0 and height / width >= 0.75:
                save = True
    return save


def check_position(config, frames, pos_x, pos_y, pos_z):
    save = False
    global_3dpos = [[pos_x],
                    [pos_y],
                    [pos_z]]

    y_check = pos_x * coefficients[0] + coefficients[1]
    if coefficients2 is not None:
        y2_check = pos_x * coefficients2[0] + coefficients2[1]
        if y_check < pos_x < y2_check:
            save = check_save(config, frames, global_3dpos)
    # elif pos_x > y_check:
    elif pos_y > y_check:
        save = check_save(config, frames, global_3dpos)
    return save


def set_position(name):
    pose = airsim.Pose(airsim.Vector3r(position[0], position[1], position[2]),
                       airsim.to_quaternion(orientation[0], orientation[1], orientation[2]))
    client.simSetObjectPose(name, pose, True)


def update_gt3d_pedestrian(client, name_pedestrians, info_pedestrians_3d, frame_index, frames, config):
    save = True
    frame_index = str(frame_index)
    frame_index = frame_index.zfill(4)

    info_pedestrians_3d[frame_index] = []
    for name_ped in name_pedestrians:
        pose = client.simGetObjectPose(name_ped)
        pos_x = pose.position.x_val
        pos_y = pose.position.y_val
        pos_z = pose.position.z_val

        orient_w = pose.orientation.w_val
        orient_x = pose.orientation.x_val
        orient_y = pose.orientation.y_val
        orient_z = pose.orientation.z_val
        # Check if the object is inside the valid space
        # save = check_position(config, frames, pos_x, pos_y, pos_z)
        if save:
            info = {'id': name_ped,
                    'pos_x': pos_x,
                    'pos_y': pos_y,
                    'pos_z': pos_z,
                    'orient_w': orient_w,
                    'orient_x': orient_x,
                    'orient_y': orient_y,
                    'orient_z': orient_z}
            info_pedestrians_3d[frame_index].append(info)
    return info_pedestrians_3d


def update_gt2d_pedestrian(client, cam_name, info_pedestrians, frame_index, config):
    captured_pedestrians = []
    if config.external:
        bboxes = client.simGetDetections(camera_name=cam_name, image_type=airsim.ImageType.Scene, external=True)
    else:
        bboxes = client.simGetDetections(camera_name='0', vehicle_name=cam_name, image_type=airsim.ImageType.Scene)

    for bbox in bboxes:
        name_ped = bbox.name
        xmin = bbox.box2D.min.x_val
        ymin = bbox.box2D.min.y_val
        xmax = bbox.box2D.max.x_val
        ymax = bbox.box2D.max.y_val
        info = {'id': name_ped,
                'xmin': xmin,
                'ymin': ymin,
                'xmax': xmax,
                'ymax': ymax}
        info_pedestrians[cam_name][frame_index].append(info)
        captured_pedestrians.append(name_ped)
    return info_pedestrians, captured_pedestrians


def save_3dgroundTruth(info_pedestrians, path_save_info):
    name_file = 'gt3d_pedestrians.json'

    final_path = path_save_info + '/' + name_file
    with open(final_path, 'w') as f:
        json.dump(info_pedestrians, f)


def save_2dgroundTruth(info_pedestrians, camera_names, path_save_info):
    name_file = 'gt2d_pedestrians.json'

    for folder in camera_names:
        final_path = path_save_info + '/' + folder + '/' + name_file
        with open(final_path, 'w') as f:
            json.dump(info_pedestrians[folder], f)

# #####Compute 2D image ground truth through the projection of the 3D ground truth#################################


def get_3dhead_coord(global_3dpos):
    coord_head0 = [[global_3dpos[0][0]],
                   [global_3dpos[1][0]],
                   [-PEDESTRIAN_HEIGHT]]
    coord_head1 = [[global_3dpos[0][0] - PEDESTRIAN_WIDTH / 2.],
                   [global_3dpos[1][0]],
                   [-PEDESTRIAN_HEIGHT]]
    coord_head2 = [[global_3dpos[0][0]],
                   [global_3dpos[1][0] - PEDESTRIAN_WIDTH / 2.],
                   [-PEDESTRIAN_HEIGHT]]
    coord_head3 = [[global_3dpos[0][0] - PEDESTRIAN_WIDTH / 2.],
                   [global_3dpos[1][0] - PEDESTRIAN_WIDTH / 2.],
                   [-PEDESTRIAN_HEIGHT]]
    coord_head4 = [[global_3dpos[0][0] + PEDESTRIAN_WIDTH / 2.],
                   [global_3dpos[1][0]],
                   [-PEDESTRIAN_HEIGHT]]
    coord_head5 = [[global_3dpos[0][0]],
                   [global_3dpos[1][0] + PEDESTRIAN_WIDTH / 2.],
                   [-PEDESTRIAN_HEIGHT]]
    coord_head6 = [[global_3dpos[0][0] + PEDESTRIAN_WIDTH / 2.],
                   [global_3dpos[1][0] + PEDESTRIAN_WIDTH / 2.],
                   [-PEDESTRIAN_HEIGHT]]
    return coord_head0, coord_head1, coord_head2, coord_head3, coord_head4, coord_head5, coord_head6


def get_3dbase_coord(global_3dpos):
    feet = OVERPECENT * PEDESTRIAN_HEIGHT
    coord_base0 = [[global_3dpos[0][0]],
                   [global_3dpos[1][0]],
                   [feet]]
    coord_base1 = [[global_3dpos[0][0] - PEDESTRIAN_WIDTH / 2.],
                   [global_3dpos[1][0]],
                   [feet]]
    coord_base2 = [[global_3dpos[0][0]],
                   [global_3dpos[1][0] - PEDESTRIAN_WIDTH / 2.],
                   [feet]]
    coord_base3 = [[global_3dpos[0][0] - PEDESTRIAN_WIDTH / 2.],
                   [global_3dpos[1][0] - PEDESTRIAN_WIDTH / 2.],
                   [feet]]
    coord_base4 = [[global_3dpos[0][0] + PEDESTRIAN_WIDTH / 2.],
                   [global_3dpos[1][0]],
                   [feet]]
    coord_base5 = [[global_3dpos[0][0]],
                   [global_3dpos[1][0] + PEDESTRIAN_WIDTH / 2.],
                   [feet]]
    coord_base6 = [[global_3dpos[0][0] + PEDESTRIAN_WIDTH / 2.],
                   [global_3dpos[1][0] + PEDESTRIAN_WIDTH / 2.],
                   [feet]]

    return coord_base0, coord_base1, coord_base2, coord_base3, coord_base4, coord_base5, coord_base6


def to_2dcamera(frame, translation, rot_matrix, global_pos, f_px):
    height, width = frame.shape[0], frame.shape[1]

    # From global coordinates to local coordinates
    R_inv = np.linalg.inv(rot_matrix)
    tranf_t = np.array([[global_pos[0][0] - translation[0][0]],
                        [global_pos[1][0] - translation[1][0]],
                        [global_pos[2][0] - translation[2][0]]])

    relative_camera_pos = np.dot(R_inv, tranf_t)
    px_coord_x = relative_camera_pos[1][0] / relative_camera_pos[0][0]
    px_coord_x = int(px_coord_x * f_px + width / 2.)

    px_coord_y = relative_camera_pos[2][0] / relative_camera_pos[0][0]
    px_coord_y = int(px_coord_y * f_px + height / 2.)
    px_coord = (px_coord_x, px_coord_y)

    return px_coord


def get_gt_bboxes(global_3dpos, frame, translation, rot_matrix, f_px):
    coord_base = get_3dbase_coord(global_3dpos)
    coord_head = get_3dhead_coord(global_3dpos)
    coord_3d = coord_base + coord_head

    bbox_2d = None
    ymax, xmax = 0,0
    ymin, xmin = frame.shape[0], frame.shape[1]
    for coord in coord_3d:
        x, y = to_2dcamera(frame, translation, rot_matrix, coord, f_px)
        if x < xmin:
            xmin = x
        if y < ymin:
            ymin = y
        if x > xmax:
            xmax = x
        if y > ymax:
            ymax = y
    width = xmax - xmin
    height = ymax - ymin
    if width > 0 and height > 0 and height / width >= 0.75:
        bbox_2d = (xmin, ymin, xmax, ymax)
    return bbox_2d


def cropBbox(bbox_coord, frame):
    xmin, ymin, xmax, ymax = bbox_coord
    frame_width = frame.shape[1]
    frame_height = frame.shape[0]
    if xmin < xmax > frame.shape[1]:
        xmax = frame_width
    if ymin < ymax > frame.shape[0]:
        ymax = frame_height
    if xmin < 0:
        xmin = 0
    if ymin < 0:
        ymin = 0
    return xmin, ymin, xmax, ymax


def update_gt2d_pedestrian_proj(client, name_pedestrians, info_pedestrians, config, frames, frame_index):
    camera_names = config.camera_names
    for name_ped in name_pedestrians:
        pose = client.simGetObjectPose(name_ped)
        pos_x = pose.position.x_val
        pos_y = pose.position.y_val
        pos_z = pose.position.z_val
        global_3dpos = [[pos_x],
                        [pos_y],
                        [pos_z]]
        # Check if the object is inside the valid space
        y_check = pos_x * coefficients[0] + coefficients[1]
        if pos_x > y_check:
            for cam in camera_names:
                translation, rot_matrix = config.get_pose(cam)
                f_px = config.get_focal_length_px(cam, frames[cam]['rgb'])
                bbox_2d = get_gt_bboxes(global_3dpos, frames[cam]['rgb'], translation, rot_matrix, f_px)
                if bbox_2d is not None:
                    xmin, ymin, xmax, ymax = cropBbox(bbox_2d, frames[cam]['rgb'])
                    width = xmax - xmin
                    height = ymax - ymin

                    if width > 0 and height > 0 and height / width >= 0.75:
                        info = {'id': name_ped,
                                'xmin': bbox_2d[0],
                                'ymin': bbox_2d[1],
                                'xmax': bbox_2d[2],
                                'ymax': bbox_2d[3]}
                        info_pedestrians[cam][frame_index].append(info)
    return info_pedestrians


