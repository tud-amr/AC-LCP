import torch


class Detector:
    def __init__(self):
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True, device=0)

    def evaluateImages(self, frames):
        predictions = self.model(frames)
        bboxes = predictions.xyxy
        bboxes = [bbox.cpu().tolist() for bbox in bboxes]
        return bboxes



