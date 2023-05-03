import cv2
from pytorchyolo import detect, models

# Load the YOLO model
model = models.load_model(
  "/home/scasao/pytorch/multi-target_tracking/PyTorch-YOLOv3/config/yolov3.cfg",
  "/home/scasao/pytorch/multi-target_tracking/PyTorch-YOLOv3/weights/darknet53.conv.74")

# Load the image as a numpy array
img = cv2.imread('/home/scasao/Documents/PedestrianSystem/Records_15_21/Drone1/1655321450871531264.png')

# Convert OpenCV bgr to rgb
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Runs the YOLO model on the image
boxes = detect.detect_image(model, img)

print(boxes)