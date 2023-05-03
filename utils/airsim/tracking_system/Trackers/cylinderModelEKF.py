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

        self._dt = 0.08

        # Create Kalman filter model matrices.
        x, y, z, w, h, vx, vy, theta = sp.symbols('x, y, z, w, h, vx, vy, theta')
        self._fx = sp.Matrix([[x + vx * self._dt],
                              [y + vy * self._dt],
                              [z],
                              [w],
                              [h],
                              [vx],
                              [vy],
                              [sp.atan(vy / vx)]])

        self._F = self._fx.jacobian(sp.Matrix([x, y, z, w, h, vx, vy, theta]))

        self.subs = {x: 0, y: 0, z: 0, w: 0, h: 0, vx: 0, vy: 0, theta: 0}
        self.x_x, self.x_y, self.x_z, self.x_w, self.x_h = x, y, z, w, h
        self.vx, self.vy, self.theta = vx, vy, theta

        self._H = np.eye(5, 8)  # H

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

        self.mean = np.r_[mean_pos, mean_vel, mean_orient]  # X

        std = [
            2 * self._std_weight_measurable * cylinder[3],
            2 * self._std_weight_measurable * cylinder[3],
            2 * self._std_weight_measurable * cylinder[3],
            2 * self._std_weight_measurable * cylinder[3],
            2 * self._std_weight_measurable * cylinder[3],
            10 * self._std_weight_non_measurable * cylinder[3],
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
        # Update M and get the new epsilon
        M = self.updateM(S)
        abs_M = (np.trace(np.dot(M.T, M))) ** (1/2)
        epsilon = 1 / (abs_M + 1)
        # print('epsilon', epsilon)
        # print('difference', difference)
        # Update estimation
        innovation = y - np.dot(S, self.mean)
        epsilon_M = epsilon * M
        # print('innovation', innovation)

        # print('innovarion', np.dot(M, innovation))
        # print('diff', np.dot(epsilon_M, difference))
        self.mean = self.mean + np.dot(M, innovation) + np.dot(epsilon_M, difference)

        x_estimate = self.mean
        cov_estimate = self.covariance

        # Predict
        F_i = self.getF()
        motion_cov = self.getQ()

        self.mean = np.dot(F_i, self.mean)
        self.covariance = np.linalg.multi_dot((F_i, M, F_i.T)) + motion_cov
        # print('cov', self.covariance)
        return x_estimate, cov_estimate

    def getF(self):
        self.subs[self.vx] = self.mean[5]
        self.subs[self.vy] = self.mean[6]

        Fi = np.array(self._F.evalf(subs=self.subs)).astype(float)
        return Fi

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

        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel, std_ori]))  # Q

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
