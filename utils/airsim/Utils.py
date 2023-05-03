import cv2
import json
import airsim
import time
import numpy as np
import os
import rope.base.utils.pycompat
from tabulate import tabulate
from joblib import Parallel, delayed
from scipy.spatial.transform import Rotation as R

MIN_DEPTH_METERS = 0
MAX_DEPTH_METERS = 100


def getImagesResponse(clients, drone_names):
    if len(drone_names) > 1:
        responses = Parallel(n_jobs=4, backend='threading')(
            delayed(get_responses)(cl, name) for cl, name in zip(clients, drone_names))
    else:
        responses = get_responses(clients[0], drone_names[0])
    return responses


def getRGB_D(responses, drone_name, frame_index, cam_index=None, save=False, path_save_info=None):
    if cam_index is not None:
        response_rgb = responses[cam_index][0]
        response_depth = responses[cam_index][1]
    else:
        response_rgb = responses[0]
        response_depth = responses[1]

    img_rgb = reponseTOrgb(response_rgb)
    depth_matrix = responseTOdepth(response_depth)[1]

    if save:
        images = [img_rgb, depth_matrix]
        save_images(images, frame_index, path_save_info)
    return img_rgb, depth_matrix


def get_responses(client, vehicle_name):
    responses = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene,  False, False),
                                     airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, True, False)], vehicle_name=vehicle_name)
    return responses


def get_responses_seg(client, vehicle_name):
    responses = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene,  False, False),
                                     airsim.ImageRequest("0", airsim.ImageType.Segmentation, False, False),
                                     airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, True, False)], vehicle_name=vehicle_name)
    return responses


def get_images_modes(client, vehicle_name, image_type='RGB-D'):
    images = []
    if image_type == 'RGB':
        # scene vision image in uncompressed RGBA array
        responses = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)],
                                        vehicle_name=vehicle_name)
        # from response to rgb format
        img_rgb = reponseTOrgb(responses[0])
        images = [img_rgb, None, None, None]

    elif image_type == 'RGB-D':
        responses = client.simGetImages([
            airsim.ImageRequest("0", airsim.ImageType.Scene, False, False),
            airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, True, False)], vehicle_name=vehicle_name)
        img_rgb = reponseTOrgb(responses[0])
        depth_img_in_meters, depth_img_in_mm, depth_img = responseTOdepth(responses[1])

        images = [img_rgb, depth_img_in_meters, depth_img_in_mm, depth_img]
    return images


def reponseTOrgb(response):
    # get numpy array and reshape array to 4 channel image array H X W X 3
    img_rgb = np.frombuffer(response.image_data_uint8, dtype=np.uint8).reshape(response.height, response.width, 3)
    return img_rgb


def responseTOpng(response):
    png = cv2.imdecode(airsim.string_to_uint8_array(response), cv2.IMREAD_UNCHANGED)
    return png


def responseTOdepth(response):
    # Reshape to a 2d array with correct width and height
    depth_img_in_meters = airsim.list_to_2d_float_array(response.image_data_float, response.width, response.height)
    depth_img_in_meters = depth_img_in_meters.reshape(response.height, response.width, 1)

    # Lerp 0..100m to 0..255 gray values
    depth_img = np.interp(depth_img_in_meters, (MIN_DEPTH_METERS, MAX_DEPTH_METERS), (0, 255))

    # Convert depth_img to millimeters to fill out 16bit unsigned int space (0..65535). Also clamp large values (e.g. SkyDome) to 65535
    depth_img_in_millimeters = depth_img_in_meters * 1000
    depth_img_in_mm = np.clip(depth_img_in_millimeters, 0, 65535)
    return depth_img_in_meters, depth_img_in_mm, depth_img


def depthTOimage(depth):
    depth_m = depth / 1000
    depth_img = np.interp(depth_m, (MIN_DEPTH_METERS, MAX_DEPTH_METERS), (0, 255))
    depth_img = depth_img.astype(np.uint8)
    depth_img = np.squeeze(depth_img, axis=2)
    return depth_img


def imageTOdepth(image):
    depth_img_in_meters = np.interp(image, (0, 255), (MIN_DEPTH_METERS, MAX_DEPTH_METERS))
    depth_img_in_millimeters = depth_img_in_meters * 1000
    depth_img_in_mm = np.clip(depth_img_in_millimeters, 0, 65535)
    return depth_img_in_mm


def save_images(images, frame_index, path):
    if not os.path.exists(path):
        os.makedirs(path)
    # timestamp = images[0].timestamp
    frame_index = str(frame_index)
    frame_index = frame_index.zfill(4)

    name_rgb = str(frame_index) + '.png'
    name_depth = str(frame_index) + '.npy'

    assert type(images[1]) == np.ndarray

    cv2.imwrite(path + '/' + name_rgb, images[0])
    np.save(path + '/' + name_depth, images[1])


def save_images_seg(images, timestamp, path):
    name_rgb = str(timestamp) + '.png'
    name_seg = str(timestamp) + '_seg.png'
    name_depth = str(timestamp) + '.npy'

    cv2.imwrite(path + '/' + name_rgb, images[0])
    cv2.imwrite(path + '/' + name_seg, images[1])
    np.save(path + '/' + name_depth, images[1])


def get_camera_matrix(client, image, drone):
    #Intrisics camera
    f = drone.get_focal_length(client)/1000
    height = image.shape[0]
    width = image.shape[1]

    K = np.array([[f/width, 0, width/2],
                 [0, f/height, height/2],
                 [0, 0, 1]])

    #Get rotation matrix
    state = drone.get_state(client)
    rotation = state.kinematics_estimated.orientation
    quaternions = [rotation.x_val, rotation.y_val, rotation.z_val, rotation.w_val]
    rot = R.from_quat(quaternions)
    rot_matrix = rot.as_matrix()

    #Get translation
    t_x, t_y, t_z = drone.get_position(state.kinematics_estimated.position)
    # t_x = state.kinematics_estimated.position.x_val
    # t_y = state.kinematics_estimated.position.y_val
    # t_z = state.kinematics_estimated.position.z_val
    translation = [[t_x], [t_y], [t_z]]
    E = np.append(rot_matrix, translation, axis=1)

    #Final camera matrix
    camera_matrix = np.dot(K, E)

    assert np.shape(camera_matrix)[0] == 3 and np.shape(camera_matrix)[1] == 4

    return K, E, camera_matrix


def relTOglobal(extrinsics_camera, pos_rel):
    rel_point = [[pos_rel.x_val],
                 [pos_rel.y_val],
                 [pos_rel.z_val],
                 [1]]
    pos_global = np.dot(extrinsics_camera, rel_point)
    return pos_global


def project_to_image(pts_3d, camera_matrix):
    """ Projects a 3D point cloud to 2D points for plotting

    :param pts_3d: 3D point cloud (3, N)
    :param camera_matrix: Camera matrix (3, 4)

    :return: pts_2d: the image coordinates of the 3D points in the shape (2, N)
    """
    pts_3d = np.append(pts_3d, [[1]], axis=0)
    assert pts_3d.shape[0] == 4

    pts_2d = np.dot(camera_matrix, pts_3d)
    pts_2d[0, :] = (pts_2d[0, :] / pts_2d[2, :])
    pts_2d[1, :] = (pts_2d[1, :] / pts_2d[2, :])
    pts_2d = np.delete(pts_2d, 2, 0)
    return pts_2d




