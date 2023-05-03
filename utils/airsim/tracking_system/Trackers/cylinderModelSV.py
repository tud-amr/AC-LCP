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
        self._std_weight_measurable = 1. / 40  # 1/50
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
        mean_vel = 0.5*np.ones(2)

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
        S = self.getS()
        if cylinder is not None:
            cylinder = cylinder.getXYZWH()
            y = self.gety(cylinder)
        else:
            y = 0
        return S, y

    def update(self, S, y):
        """
        Runs the prediction and update steps of the tracker with the specified cylinder as measure step
        """
        K = self.updateK(S)
        if len(np.shape(y)) == 0:
            self.mean = self.mean
        else:
            self.mean = self.mean + np.dot(K, y)

        self.covariance = (np.identity(8) - np.dot(K, self._H)) * self.covariance

        x_estimate = self.mean
        cov_estimate = self.covariance

        # Predict
        F_i = self.getF()
        motion_cov = self.getQ()

        self.mean = np.dot(F_i, self.mean)
        self.covariance = np.linalg.multi_dot((F_i, self.covariance, F_i.T)) + motion_cov
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
        measur_cov_mat = np.linalg.multi_dot((self._H, self.covariance, self._H.T)) + self.getR()  # S
        return measur_cov_mat

    def gety(self, measurement):
        sensor_data = measurement - np.dot(self._H, self.mean)
        return sensor_data

    def updateK(self, S):
        S_inv = np.linalg.inv(S)
        K = np.linalg.multi_dot((self.covariance, self._H.T, S_inv))# S
        return K

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
