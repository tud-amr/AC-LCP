"""
Multicamera tracker for cylinders based on detections in multiple cameras
"""
import numpy as np
import sympy as sp
from tracking_system.Utilities.geometry3D_utils import Cylinder
from tracking_system.Utilities.geometryCam import from3dCylinder


class CylinderModel:
    """
    Implementation of the tracker
    """

    def __init__(self, camera): 
        """
        Empty tracker for the current cameras list
        """
        self.mean = None
        self.covariance = None
        self.camera = camera

        self._dt = 0.01

        # Create Kalman filter model matrices.
        self._F = np.eye(7, 7)
        self._F[0, 5] = self._dt
        self._F[1, 6] = self._dt

        self._H = np.eye(5, 7)  # H

        # Motion and observation uncertainty are chosen relative to the current
        # state estimate. These weights control the amount of uncertainty in
        # the model. This is a bit hacky.
        self._std_weight_measurable = 1. / 40  # 1/40
        self._std_weight_non_measurable = 1. / 160  # 1/160

        # Epsilon parameter ponderate different measurement between nodes
        self._epsilon = 0.01
        
    def init(self, cylinder):
        """
        Initializes the tracker with the bboxes provided (one or more)
        Create track from unassociated measurement.

        Parameters
        ----------
        measurement : ndarray
            Cylinder coordinates (x, y, w, h) with center position (x, y),
            width w, and height h.

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector (8 dimensional) and covariance matrix (8x8
            dimensional) of the new track. Unobserved velocities are initialized
            to 0 mean.
        """
        cylinder = cylinder.getXYZWH()

        mean_pos = cylinder
        mean_orient = np.zeros(1)
        mean_vel = np.ones(2)

        self.mean = np.r_[mean_pos, mean_vel]  # X

        std = [
            2 * self._std_weight_measurable * cylinder[3],
            2 * self._std_weight_measurable * cylinder[3],
            2 * self._std_weight_measurable * cylinder[3],
            2 * self._std_weight_measurable * cylinder[3],
            2 * self._std_weight_measurable * cylinder[3],
            10 * self._std_weight_non_measurable * cylinder[3],
            10 * self._std_weight_non_measurable * cylinder[3]]
        self.covariance = np.diag(np.square(std))  # P

    def computeInfo(self, cylinder):
        """Calculate S,y of each person"""
        if cylinder is not None:
            cylinder = cylinder.getXYZWH()
            S = self.getS()
            y = self.gety(cylinder)
        else:
            S = 0
            y = 0
        return S, y

    def update(self, S, y, difference):
        """
        Runs the prediction and update steps of the tracker with the specified cylinder as measure step
        """
        M = self.updateM(S)

        abs_M = (np.trace(np.dot(M.T, M))) ** 2
        epsilon = 1 / (abs_M + 1)
        innovation = y - np.dot(S, self.mean)
        # print('innovation', np.dot(innovation, M.T))
        # print('consensus',epsilon * np.dot(difference, M.T))
        self.mean = self.mean + np.dot(innovation, M.T) + epsilon * np.dot(difference, M.T)
        x = self.mean
        cov = self.covariance

        # Predict
        motion_cov = self.getQ()
        self.covariance = np.linalg.multi_dot((self._F, M, self._F.T)) + motion_cov
        self.mean = np.dot(self._F, self.mean)

        return x, cov

    def getCylinder(self):
        """
        Returns the current cylinder
        :return:
        """
        cylinder = Cylinder.XYZWH(*self.mean[:5].copy())
        return cylinder

    def getBboxes(self, cam_state, depth_matrix):
        """
        Returns the list of bboxes (the cylinder translated to each camera)
        :return:
        """
        cylinder = self.getCylinder()
        bbox = from3dCylinder(cam_state, depth_matrix, cylinder)
        return bbox
    
    def getS(self):
        
        measurement_cov_mat = self.getR()
        measurement_cov_mat = np.linalg.inv(measurement_cov_mat)

        inv_measur_cov_mat = np.linalg.multi_dot((self._H.T, measurement_cov_mat, self._H)) #S
        
        return inv_measur_cov_mat

    def gety(self, measurement):

        measurement_cov_mat = self.getR()
        measurement_cov_mat = np.linalg.inv(measurement_cov_mat)

        sensor_data = np.linalg.multi_dot((self._H.T, measurement_cov_mat, measurement)) #y

        return sensor_data
    
    def updateM(self, S):
        covariance_inv = np.linalg.inv(self.covariance)
        M = np.linalg.inv(covariance_inv + S)
        return M

    def getQ(self):

        std_pos = [
            self._std_weight_measurable * self.mean[3],
            self._std_weight_measurable * self.mean[3],
            self._std_weight_measurable * self.mean[3],
            self._std_weight_measurable * self.mean[3],
            self._std_weight_measurable * self.mean[3]]
        std_vel = [
            self._std_weight_non_measurable * self.mean[3],
            self._std_weight_non_measurable * self.mean[3]]
        std_ori = [
            self._std_weight_non_measurable * self.mean[3]]

        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))  # Q

        return motion_cov

    def getR(self):

        std = [
            self._std_weight_measurable * self.mean[3],
            self._std_weight_measurable * self.mean[3],
            self._std_weight_measurable * self.mean[3],
            self._std_weight_measurable * self.mean[3],
            self._std_weight_measurable * self.mean[3]]
        measurement_cov_mat = np.diag(np.square(std))  # R

        return measurement_cov_mat
