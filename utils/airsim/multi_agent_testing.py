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


if __name__ == "__main__":
    ## Setting up the client
    import airsim
    from drone import MultiRotor
    import numpy as np
    from pedestrians import define_environment_limits, update_gt_pedestrian_pose, save_3dgroundTruth, get_name_pedestrians

    velocity = 2
    from drone import environment_limits
    env_c_offset = np.array([-4.2,2.5])
    drone1 = MultiRotor("Drone1",[0,0,0], velocity)
    dronelist = [drone1]

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

    next_poses = [[0.0, 0.0, -3.0,0],[0.5, 0.0, -3.0,0],
                  [1.0, 0.0, -3.0,0],[11, 0.0, -3.0,0],
                  [-19.0, 0.0, -3.0,0],[0.0, 0.0, -3.0,0]] # 8 - -19 //

    for next_pose in next_poses:
        print(drone.get_position(cl1))
        next_pos = next_pose[0:3]
        next_yaw = next_pose[3]
        airsim_client_def, task = drone.moveTOpos(cl1, next_pos, next_yaw)
        task.join()
        time.sleep(0.25)


        print('Drone achive goal')

    while True:
        print(drone.get_state(cl1))
        state = drone
        next_pose = next_poses[-1]
        next_pos = next_pose[0:3]
        next_yaw = next_pose[3]
        airsim_client_def, task = drone.moveTOrelpos(cl1, next_pos, next_yaw)
        task.join()
        time.sleep(0.25)

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
