# import airsim
# import os
# import time
# import datetime
# import cv2
# from drone import MultiRotor
# from Utils import save_images, get_responses, reponseTOrgb, responseTOdepth
# from pedestrians import define_environment_limits, update_gt_pedestrian_pose, save_3dgroundTruth, get_name_pedestrians
# #from tracking_system.main_tracking import PerceptionModuleInitialization, PerceptionModule
# from Utils import getImagesResponse, getRGB_D
# from ComputeGoalPosition import get_next_position_and_orientation
import time


def AgentsManager(uavs, clients):
    cl1 = clients[0]
    drone_names = [drone.name for drone in uavs]
    task = None

    # -------Connecting--------------
    for drone in uavs:
        drone.arm(cl1)

    # ----Initialization of Perception-------
    # singleTrackers, trackers, old_trackers, LUT_delete, detector, reid, coefficients, results_pymot = PerceptionModuleInitialization(drone_names)
    # coefficients = None

    # -----Taking off----------------
    print('Taking off...')
    for drone in uavs:
        task = drone.take_off(cl1)
    task.join()

    # --------MoveTO------------------
    for drone in uavs:
        next_pos, next_yaw = drone.get_init_position_and_orientation()
        airsim_client_def, task = drone.moveTOpos(cl1, next_pos, next_yaw)
    task.join()
    print('Drone achive goal')

    # -------------Take Image and Save Info-----------------
    # now = datetime.datetime.now()
    # folder_date = 'Records_' + str(now.day) + '_' + str(now.hour)
    # path_save_info = '/home/scasao/Documents/PedestrianSystem/' + folder_date
    #
    # for drone in uavs:
    #     final_path = path_save_info + '/' + drone.name
    #     if not os.path.exists(final_path):
    #         os.makedirs(final_path)

    gt_pedestrians = {}
    name_pedestrians = get_name_pedestrians(cl1, 'BP_P_')
    print(name_pedestrians)

    print('Taking data...')
    i = 0
    while i < 1500:
        print(i)
        # trackers, old_trackers, LUT_delete, frames, depth, states_cameras = PerceptionModule(clients[0], drone_names, uavs, singleTrackers, trackers, old_trackers, LUT_delete, detector, reid, coefficients, results_pymot, i)
        # timestamp = None
        responses = getImagesResponse(clients, drone_names)
        if len(drone_names) >= 1:
            for j, drone in enumerate(uavs):
                img_rgb, depth_matrix = getRGB_D(responses, drone.name, i, j, save=False)
                # Update state
                drone.update_state_info(cl1, i)
                cv2.imshow(drone.name, img_rgb)
        else:
            drone = uavs[0]
            img_rgb, depth_matrix = getRGB_D(responses, drone.name, i, save=False)
            cv2.imshow(drone.name, img_rgb)
        gt_pedestrians = update_gt_pedestrian_pose(cl1, name_pedestrians, gt_pedestrians, i)
        timestamp = str(i)
        timestamp = timestamp.zfill(4)
        actual_ped_state = gt_pedestrians[timestamp]
        print(actual_ped_state)
        # new_goals = get_next_position_and_orientation(drone_names, trackers, states_cameras, frames, depth)
        # for drone in uavs:
        #     if new_goals[drone.name] is not None:
        #         new_pos, new_yaw = new_goals[drone.name]
        #         client, task = drone.moveTOpos(cl1, new_pos, new_yaw)
        #         task.join()
        i += 1

        k = cv2.waitKey(33)
        if k == 32:
            cv2.destroyAllWindows()
        elif k == ord('p'):
            cv2.waitKey(0)
    # Save drone states
    # for drone in uavs:
    #     drone.save_info_state(path_save_info)
    #
    # # Save gt pedestrians
    # save_3dgroundTruth(gt_pedestrians, path_save_info)

def set_pose_drone(client, drone, position, orientation,offset, degrees=False):
    _, task = drone.moveTOpos(client, [position[0]-offset[0],(-position[1]+offset[1]),position[2]], -orientation[2])
    if degrees:
        orientation = [np.deg2rad(angle) for angle in orientation]
    # set orientation
    pose = airsim.Pose(airsim.Vector3r(position[0]-offset[0], (-position[1]+offset[1]), position[2]),
                       airsim.to_quaternion(orientation[0], orientation[1], -orientation[2]))
    client.simSetVehiclePose(pose, True, vehicle_name=drone.name)
    return client

def quaternion_to_euler(x, y, z, w):

    import math
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    X = math.degrees(math.atan2(t0, t1))

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    Y = math.degrees(math.asin(t2))

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    Z = math.degrees(math.atan2(t3, t4))

    return X, Y, Z

def get_3ddrone_pose(client, drone):
    state = drone.get_state(client)
    pos_x = state.position.x_val
    pos_y = state.position.y_val
    pos_z = state.position.z_val
    _, _, yaw = quaternion_to_euler(state.orientation.x_val,
                                    state.orientation.y_val,
                                    state.orientation.z_val,
                                    state.orientation.w_val)
    return np.array([pos_x,pos_y,pos_z,-yaw])

def get_2ddrone_pose(client, drone):
    state = drone.get_state(client)
    pos_x = state.position.x_val
    pos_y = state.position.y_val
    pos_z = state.position.z_val
    _, _, yaw = quaternion_to_euler(state.orientation.x_val,
                                    state.orientation.y_val,
                                    state.orientation.z_val,
                                    state.orientation.w_val)
    return np.array([pos_x,pos_y,-yaw])

def detectedPedestriansImg(client, config, drone):
    images = getResponseImages([client], config)
    cameras_names = config.camera_names
    gt2d_pedestrians = {}
    for cam_name in cameras_names:
        gt2d_pedestrians[cam_name] = {}
        client.simAddDetectionFilterMeshName(camera_name="0", image_type=airsim.ImageType.Scene, mesh_name="BP_P*",
                                          vehicle_name=drone.name)
    frame_index = 0
    frame_index_key = str(frame_index)
    frame_index_key = frame_index_key.zfill(4)
    for cam in cameras_names:
        # Update 2d pedestrians bbox obtained
        gt2d_pedestrians[cam][frame_index_key] = []
        gt2d_pedestrians, _ = update_gt2d_pedestrian(client, cam, gt2d_pedestrians, frame_index_key, config)
    ped_images = visualize(cameras_names, images, frame_index, gt2d_pedestrians)
    return ped_images


if __name__ == "__main__":
    ## Setting up the client
    import airsim
    from drone import MultiRotor
    import numpy as np
    from utils.airsim.settings_airsim import Configuration
    from utils.airsim.image_utils import getResponseImages, save_images
    from utils.airsim.pedestrians import update_gt2d_pedestrian
    import cv2
    import matplotlib.pyplot as plt
    from utils.airsim.Visualization import visualize

    velocity = 2
    from drone import environment_limits
    env_c_offset = np.array([-4.2,2.5])
    drone1 = MultiRotor("Drone1",[0,0,0], velocity)
    dronelist = [drone1]
    name_experiment = 'airsim_images'
    img_types = 'RGB'  # 'RGB' , 'RGB-D', 'RGB-DS'
    save_mode = 'start'  # 'start', 'wait'
    config = Configuration(img_types, None, save_mode, visualize_images=True, vis_pedestrian_2dGT=True,
                           name_experiment=name_experiment, external=False, uavs=dronelist)
    ## set up the client
    clients = [airsim.MultirotorClient() for i in range(len(dronelist))]
    assert len(clients) == len(dronelist)
    #limit_valid_coordinates = [(-16.4, -8.0), (8.5, -8.2)]
    #define_environment_limits(limit_valid_coordinates)
    #AgentsManager(dronelist, clients)

    ## Everything done for one drone
    cl1 = clients[0]
    drone_names = [drone.name for drone in dronelist]
    task = None
    drone = dronelist[0]
    drone.arm(cl1)

    # take off
    task = drone.take_off(cl1)
    task.join()
    z_ref = -2.4
    set_pose_drone(cl1,drone,[0,0,z_ref],[0,0,0],np.array([0.,0.]),degrees=False)

    #
    # next_poses = [[0.0, 0.0, 0,0.],[2.0, 0.0, 1,0.0],[-2.0, 0.0, 0,0.0],[0.0, 2.0, 0,0.0],
    #               [0.0, -2.0, 0.,0]]#,[0.0, 2.0, z_ref,-0],[-2.0, 0.0, z_ref,0],[0.,0.,z_ref,-0.]] # 8 - -19 //

    # for next_pose in next_poses:
    #     print(drone.get_state(cl1))
    #     next_pos = next_pose[0:3]
    #     next_yaw = next_pose[3]
    #     airsim_client_def, task = drone.moveTOrelpos(cl1, next_pos, next_yaw)
    #     task.join()
    #     time.sleep(0.25)
    # print(get_3ddrone_pose(cl1,drone))
    # for next_pose in next_poses:
    #     next_pos = next_pose[0:3]
    #     next_yaw = next_pose[3]
    #     # _, task = drone.moveByVel(cl1, next_pos, next_yaw)
    #     #_, task = drone.moveTOpos(cl1, next_pos, next_yaw)
    #     # set_pose_drone(cl1, drone, next_pos,[0,0,next_yaw],np.array([0.,0.]), degrees=True)#[0,0,z_ref], [0.,0.,0],np.array([0.,0.]))
    #     # print("right after command:", get_3ddrone_pose(cl1, drone))
    #     # print("next pose: ", next_pose)
    #     # _, task = drone.moveTOpos(cl1, next_pos, next_yaw)
    #     _, task = drone.moveByVel(cl1, next_pos, yaw=-next_yaw, duration=1.00)
    #     #task.join()
    #     time.sleep(10)
    #     print("after 10 secs:", get_3ddrone_pose(cl1, drone))

    # Get images from external cameras

    ped_images = detectedPedestriansImg(cl1,config)
    for key in ped_images.keys():
        plt.figure()
        plt.imshow(ped_images[key])
    #plt.imshow(images["Drone1"]["rgb"])

    print('Drone achive goal')

    # while True:
    #     print(drone.get_state(cl1))
    #     state = drone
    #     next_pose = next_poses[-1]
    #     next_pos = next_pose[0:3]
    #     next_yaw = next_pose[3]
    #     airsim_client_def, task = drone.moveTOrelpos(cl1, next_pos, next_yaw)
    #     task.join()
    #     time.sleep(0.25)

    # velocity = 1
    # drone1 = MultiRotor("Drone1", [0, 0, 0], velocity)
    # # drone2 = MultiRotor("Drone2", [1.5, 1.5, 0], velocity)
    # # drone3 = MultiRotor("Drone3", [-1.5, 1.5, 0], velocity)
    # # drone4 = MultiRotor("Drone4", [-1.5, -1.5, 0], velocity)
    #
    # # drone1 = MultiRotor("Drone1", [0, 0, 0], velocity)
    # uavs = [drone1]
    # # , drone2, drone3, drone4]
    #
    # clients = [airsim.MultirotorClient() for i in range(len(uavs))]
    # assert len(clients) == len(uavs)
    #
    # # Environment limits
    # limit_valid_coordinates = [(-16.4, -8.0), (8.5, -8.2)]
    # define_environment_limits(limit_valid_coordinates)
    #
    # AgentsManager(uavs, clients)
