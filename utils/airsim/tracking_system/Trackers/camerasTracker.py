import random
from tracking_system.Utilities.geometry3D_utils import Cylinder


class SingleCameraTracker0:
    NEXTID = 0

    def __init__(self, motionModel, camera, cylinder):
        self.motionModel = motionModel
        self.camera = camera
        self.appearance = []
        self.weight_app = []
        self.app_model_frame_index = []
        self.LUT = {}
        self.id = None
        self.color = None
        self.idTOshow = None
        self.colorTOshow = None

        self.cylinder = cylinder
        self.x = None
        self.covariance = None
        self.y = None
        self.S = None

        self.detectorFound = False
        self.framesDetected = 0
        self.framesLost = 0
        
    def computeInfo(self, cylinder):
        self.S, self.y = self.motionModel.computeInfo(cylinder)
        if self.detectorFound:
            self.framesDetected += 1
            # self.framesLost -= 1
            self.framesLost = 0
        else:
            # self.framesDetected -= 1
            self.framesLost += 1

    def update_SV(self, S, y):
        self.x, self.covariance = self.motionModel.update(S, y)
        self.cylinder = Cylinder.XYZWH(*self.x[:5].copy())

    def update(self, S, y, difference):
        self.x, self.covariance = self.motionModel.update(S, y, difference)
        self.cylinder = Cylinder.XYZWH(*self.x[:5].copy())

    def setID(self):
        self.id = SingleCameraTracker0.NEXTID
        self.color = getColors()
        
        self.idTOshow = self.id
        self.colorTOshow = self.color

        SingleCameraTracker0.NEXTID += 1


class SingleCameraTracker1:
    NEXTID = 0

    def __init__(self, motionModel, camera, cylinder):
        self.motionModel = motionModel
        self.camera = camera
        self.appearance = []
        self.weight_app = []
        self.app_model_frame_index = []
        self.LUT = {}
        self.id = None
        self.color = None
        self.idTOshow = None
        self.colorTOshow = None

        self.cylinder = cylinder
        self.x = None
        self.covariance = None
        self.y = None
        self.S = None

        self.detectorFound = False
        self.framesDetected = 0
        self.framesLost = 0

    def computeInfo(self, cylinder):
        self.S, self.y = self.motionModel.computeInfo(cylinder)
        if self.detectorFound:
            self.framesDetected += 1
            # self.framesLost -= 1
            self.framesLost = 0
        else:
            # self.framesDetected -= 1
            self.framesLost += 1

    def update_SV(self, S, y):
        self.x, self.covariance = self.motionModel.update(S, y)
        self.cylinder = Cylinder.XYZWH(*self.x[:5].copy())

    def update(self, S, y, difference):
        self.x, self.covariance = self.motionModel.update(S, y, difference)
        self.cylinder = Cylinder.XYZWH(*self.x[:5].copy())

    def setID(self):
        self.id = SingleCameraTracker1.NEXTID
        self.color = getColors()

        self.idTOshow = self.id
        self.colorTOshow = self.color

        SingleCameraTracker1.NEXTID += 1


class SingleCameraTracker2:
    NEXTID = 0

    def __init__(self, motionModel, camera, cylinder):
        self.motionModel = motionModel
        self.camera = camera
        self.appearance = []
        self.weight_app = []
        self.app_model_frame_index = []
        self.LUT = {}
        self.id = None
        self.color = None
        self.idTOshow = None
        self.colorTOshow = None

        self.cylinder = cylinder
        self.x = None
        self.covariance = None
        self.y = None
        self.S = None

        self.detectorFound = False
        self.framesDetected = 0
        self.framesLost = 0

    def computeInfo(self, cylinder):
        self.S, self.y = self.motionModel.computeInfo(cylinder)
        if self.detectorFound:
            self.framesDetected += 1
            # self.framesLost -= 1
            self.framesLost = 0
        else:
            # self.framesDetected -= 1
            self.framesLost += 1

    def update_SV(self, S, y):
        self.x, self.covariance = self.motionModel.update(S, y)
        self.cylinder = Cylinder.XYZWH(*self.x[:5].copy())

    def update(self, S, y, difference):
        # print(self.camera)
        # print('track id pre', self.id, 'cylinder', self.cylinder.getXYZWH())
        self.x, self.covariance = self.motionModel.update(S, y, difference)
        self.cylinder = Cylinder.XYZWH(*self.x[:5].copy())
        # print('track id post', self.id, 'cylinder', self.cylinder.getXYZWH())

    def setID(self):
        self.id = SingleCameraTracker2.NEXTID
        self.color = getColors()

        self.idTOshow = self.id
        self.colorTOshow = self.color

        SingleCameraTracker2.NEXTID += 1


class SingleCameraTracker3:
    NEXTID = 0

    def __init__(self, motionModel, camera, cylinder):
        self.motionModel = motionModel
        self.camera = camera
        self.appearance = []
        self.weight_app = []
        self.app_model_frame_index = []
        self.LUT = {}
        self.id = None
        self.color = None
        self.idTOshow = None
        self.colorTOshow = None

        self.cylinder = cylinder
        self.x = None
        self.covariance = None
        self.y = None
        self.S = None

        self.detectorFound = False
        self.framesDetected = 0
        self.framesLost = 0

    def computeInfo(self, cylinder):
        self.S, self.y = self.motionModel.computeInfo(cylinder)
        if self.detectorFound:
            self.framesDetected += 1
            # self.framesLost -= 1
            self.framesLost = 0
        else:
            # self.framesDetected -= 1
            self.framesLost += 1

    def update_SV(self, S, y):
        self.x, self.covariance = self.motionModel.update(S, y)
        self.cylinder = Cylinder.XYZWH(*self.x[:5].copy())

    def update(self, S, y, difference):
        self.x, self.covariance = self.motionModel.update(S, y, difference)
        self.cylinder = Cylinder.XYZWH(*self.x[:5].copy())

    def setID(self):
        self.id = SingleCameraTracker3.NEXTID
        self.color = getColors()

        self.idTOshow = self.id
        self.colorTOshow = self.color

        SingleCameraTracker3.NEXTID += 1


def getColors():
    color = [random.randint(0,255),random.randint(0,255), random.randint(0,255)]

    return color



