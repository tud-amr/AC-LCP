"""
Functions and classes math and camera-geometry related
"""
import numpy as np

from scipy.spatial import distance
from tracking_system.Utilities.geometry2D_utils import f_euclidian_image, f_add, f_subtract, Point2D, Bbox, f_area, \
    f_intersection
from tracking_system.Utilities.geometry3D_utils import Cylinder, Point3D, f_euclidian_ground, f_subtract_ground, \
    f_add_ground
from scipy.spatial.transform import Rotation as R

DIST_MAHA_HIGH_THRESHOLD = 2000


def get_camera_extrinsics_offline(cam_state):
    t_x = cam_state['pos_x']
    t_y = cam_state['pos_y']
    t_z = cam_state['pos_z']
    rot_x = cam_state['orient_x']
    rot_y = cam_state['orient_y']
    rot_z = cam_state['orient_z']
    rot_w = cam_state['orient_w']

    # Get rotation matrix
    quaternions = [rot_x, rot_y, rot_z, rot_w]
    rot = R.from_quat(quaternions)
    rot_matrix = rot.as_matrix()

    # Get translation
    translation = [[t_x], [t_y], [t_z]]
    E = np.append(rot_matrix, translation, axis=1)

    assert np.shape(E)[0] == 3 and np.shape(E)[1] == 4

    return translation, rot_matrix, E


def get_camera_extrinsics_online(client, drone):
    # Get rotation matrix
    state = drone.get_state(client)
    rotation = state.kinematics_estimated.orientation
    quaternions = [rotation.x_val, rotation.y_val, rotation.z_val, rotation.w_val]
    rot = R.from_quat(quaternions)
    rot_matrix = rot.as_matrix()

    # Get translation
    t_x, t_y, t_z = drone.get_position(state.kinematics_estimated.position)
    # t_x = state.kinematics_estimated.position.x_val
    # t_y = state.kinematics_estimated.position.y_val
    # t_z = state.kinematics_estimated.position.z_val
    translation = [[t_x], [t_y], [t_z]]
    E = np.append(rot_matrix, translation, axis=1)

    assert np.shape(E)[0] == 3 and np.shape(E)[1] == 4

    return translation, rot_matrix, E


def to3dCylinder(cam_state, depth, bbox, online=False):
    """
    Converts a bbox from the specified camera image coordinates to a cylinder in floor plane
    """
    xmin, ymin, width, height = bbox.getAsXmYmWH()
    c_bbox = [xmin + int(width / 2), ymin + int(height / 2)]

    if online:
        translation, rot_matrix, E = cam_state
    else:
        translation, rot_matrix, E = get_camera_extrinsics_offline(cam_state)

    height_img, width_img = depth.shape[0], depth.shape[1]

    # Relative position camera-pedestrian
    depth_px = float(depth[c_bbox[1]][c_bbox[0]][0] / 1000)
    groundP = to3d(depth_px, bbox.getFeet(), width_img, height_img, rot_matrix, translation)

    # Compute width
    groundP_left = to3d(depth_px, bbox.getFeet(1), width_img, height_img, rot_matrix, translation)
    groundP_right = to3d(depth_px, bbox.getFeet(-1), width_img, height_img, rot_matrix, translation)

    width = f_euclidian_ground(groundP, groundP_left) / 2. + f_euclidian_ground(groundP, groundP_right) / 2.

    headZ = to3d(depth_px, bbox.getHair(), width_img, height_img, rot_matrix, translation).getXYZ()[2]
    groundPZ = groundP.getXYZ()[2]
    height = groundPZ - headZ

    return Cylinder(groundP, width, height)


def to3d(depth_px, point, width, height, rot_matrix, translation):
    f_px = 320
    # From pixel image coordinates to relative 3d camera coordinates
    coord_img_x, coord_img_y = point.getAsXY()

    coord_3d_y = (coord_img_x - width / 2.) * depth_px / f_px
    coord_3d_z = (coord_img_y - height / 2.) * depth_px / f_px
    coord_3d_x = depth_px

    coord = [[coord_3d_x],
             [coord_3d_y],
             [coord_3d_z]]

    # From relative coordiantes to global
    global_coord3d = np.dot(rot_matrix, coord)

    coord_3d_x = global_coord3d[0][0] + translation[0][0]
    coord_3d_y = global_coord3d[1][0] + translation[1][0]
    coord_3d_z = global_coord3d[2][0] + translation[2][0]
    return Point3D(coord_3d_x, coord_3d_y, coord_3d_z)


def from3dCylinder(cam_state, depth_matrix, cylinder, online=False):
    """
    Converts a cylinder in floor plane to a bbox from the specified camera image coordinates
    """
    height_img, width_img = depth_matrix.shape[0], depth_matrix.shape[1]
    if online:
        translation, rot_matrix, E = cam_state
    else:
        translation, rot_matrix, E = get_camera_extrinsics_offline(cam_state)
    center = cylinder.getCenter()

    # 1. Projection of cylinder center
    cx, cy, cz, cwidth, cheight = cylinder.getXYZWH()
    bottom, depth = from3d(cx, cy, cz, rot_matrix, translation, width_img, height_img)

    # 2. Add a horizontal vector in the image plane and reproject to 3d
    vector_2d = f_add(bottom, Point2D(10, 0))
    vector_3d = to3d(depth, vector_2d, width_img, height_img, rot_matrix, translation)

    # 3. Compute 3d normalized vector to the width of the cylinder
    vector_3d_norm = f_subtract_ground(vector_3d, center).normalize(cwidth)

    # 4. Add vector 3d to cylinder center and reproject to image
    wx, wy, wz = f_add_ground(vector_3d_norm, center).getXYZ()
    feet_edge2d, depth = from3d(wx, wy, wz, rot_matrix, translation, width_img, height_img)

    # 5. Compute distance between bbox bottom and the edge
    width = f_euclidian_image(bottom, feet_edge2d) * 2

    # Compute of the height
    hx, hy, hheight = cylinder.getHair().getXYZ()
    head_point, depth = from3d(hx, hy, hheight, rot_matrix, translation, width_img, height_img)
    height = f_subtract(head_point, bottom).getAsXY()[1]

    return Bbox.FeetWH(bottom, width, height, heightReduced=True)


def from3d(cx, cy, cz, rot_matrix, translation, width, height):
    f_px = 320
    # From 3d global coordinates to relative 3d coordiantes
    R_inv = np.linalg.inv(rot_matrix)
    transf_t = np.array([[cx - translation[0][0]],
                         [cy - translation[1][0]],
                         [cz - translation[2][0]]])

    rel_3dpos = np.dot(R_inv, transf_t)

    # From relative 3d coordinates to image pixel coordinates
    px_coord_x = rel_3dpos[1][0] / rel_3dpos[0][0]
    px_coord_x = int(px_coord_x * f_px + width / 2.)

    px_coord_y = rel_3dpos[2][0] / rel_3dpos[0][0]
    px_coord_y = int(px_coord_y * f_px + height / 2.)
    return Point2D(px_coord_x, px_coord_y), rel_3dpos[0][0]


def compute_geometry_scores(tracker, cylinder_d):
    """
    Obtain the geometry score between one detection, several trackers from other views
    Input: Detection and tracker from other views and
    Output: identity: id more likely, ids: list of ids evaluates,scores: scores of similarity geometry of ids evaluates
    """
    x_p = tracker.motionModel.mean
    cylinder_p = Cylinder.XYZWH(*x_p[:5].copy())

    track_cov = tracker.motionModel.covariance
    measurement_cov = tracker.motionModel.getR()

    total_cov = np.linalg.inv(track_cov[:2, :2] + measurement_cov[:2, :2])
    dist = distance.mahalanobis(cylinder_d.getCenter().getAsXY(), cylinder_p.getCenter().getAsXY(), total_cov)

    if dist < DIST_MAHA_HIGH_THRESHOLD:
        score = (1 / DIST_MAHA_HIGH_THRESHOLD) * dist
    else:
        score = 1
    return score


def toInt(v):
    """
    rounds to int all elements of the tuple
    """
    return tuple(int(round(e)) for e in v)


def cropBbox(bbox, frame):
    """
    Crops the bounding box for displaying purposes
    (otherwise opencv gives error if outside frame)
    """
    if bbox is None: return None

    height, width = frame.shape[:2]
    return Bbox.XmYmXMYM(*toInt(f_intersection(bbox, Bbox.XmYmWH(0, 0, width, height)).getAsXmYmXMYM()))


def cutImage(image, bbox):
    """
    Returns the path of the image under the rounded bbox
    """
    bboxC = cropBbox(bbox, image)
    if f_area(bboxC) > 0:
        return image[bboxC.ymin:bboxC.ymax + 1, bboxC.xmin:bboxC.xmax + 1]
    else:
        return None


def get_camera_matrix(client, image, drone):
    # Intrisics camera
    f = drone.get_focal_length(client) / 1000
    height = image.shape[0]
    width = image.shape[1]

    K = np.array([[f / width, 0, width / 2],
                  [0, f / height, height / 2],
                  [0, 0, 1]])

    # Get rotation matrix
    state = drone.get_state(client)
    rotation = state.kinematics_estimated.orientation
    quaternions = [rotation.x_val, rotation.y_val, rotation.z_val, rotation.w_val]
    rot = R.from_quat(quaternions)
    rot_matrix = rot.as_matrix()

    # Get translation
    t_x, t_y, t_z = drone.get_position(state.kinematics_estimated.position)
    # t_x = state.kinematics_estimated.position.x_val
    # t_y = state.kinematics_estimated.position.y_val
    # t_z = state.kinematics_estimated.position.z_val
    translation = [[t_x], [t_y], [t_z]]
    E = np.append(rot_matrix, translation, axis=1)

    # Final camera matrix
    camera_matrix = np.dot(K, E)

    assert np.shape(camera_matrix)[0] == 3 and np.shape(camera_matrix)[1] == 4

    return K, E, camera_matrix
