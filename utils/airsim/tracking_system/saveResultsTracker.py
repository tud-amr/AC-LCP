import json
import os
from tracking_system.Utilities.geometryCam import from3dCylinder, cropBbox


class SaveResults_pymotFormat:
    def __init__(self):
        self.tracker_results = []

    def save_results(self, frame_index, trackers, cam_state, depth_matrix, frame, online=False):
        hypotheses = []
        for track in trackers:
            if track.id is not None:
                identity = track.id
                x3d, y3d, z3d, width3d, height3d = track.cylinder.getXYZWH()
                bbox = from3dCylinder(cam_state, depth_matrix, track.cylinder, online)
                bbox = cropBbox(bbox, frame)
                xmin2d, ymin2d, width2d, height2d = bbox.getAsXmYmWH()
                pred_results = {'id': identity,
                                'x3d': x3d,
                                'y3d': y3d,
                                'height3d': height3d,
                                'width3d': width3d,
                                'xmin2d': xmin2d,
                                'ymin2d':ymin2d,
                                'width2d': width2d,
                                'height2d': height2d}
                hypotheses.append(pred_results)
                
        data = {'timestamp': frame_index, 'num': frame_index, 'hypotheses': hypotheses}
        self.tracker_results.append(data)

    def close_saveFile(self, camera, alg):
        folder = alg
        save_path = '/home/scasao/pytorch/Results/' + folder
        
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            
        with open(save_path + '/' + camera + '.json', 'w') as outfile:
            json.dump(self.tracker_results, outfile)







