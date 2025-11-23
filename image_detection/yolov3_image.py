# USAGE
# python image_detection/yolov3_image.py --image input/images/baggage_claim.jpg

# import the necessary packages
import cv2
import numpy as np
import os
import time
import argparse

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, type=str,
                help="path to input image file")
args = vars(ap.parse_args())

# define the root and paths for model files
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
output_folder = os.path.join(ROOT_DIR, "output/yolov3/images")
yolo_dir = os.path.join(ROOT_DIR, "yolo-coco")

# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.join(yolo_dir, "coco.names")
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
                           dtype="uint8")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.join(yolo_dir, "yolov3.weights")
configPath = os.path.join(yolo_dir, "yolov3.cfg")

# create output directory if not already present
os.makedirs(output_folder, exist_ok=True)

# load our YOLO object detector trained on COCO dataset (80 classes)
print("[INFO] loading YOLOv3 model from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# determine only the *output* layer names that we need from YOLO
ln = net.getLayerNames()
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
print("[INFO] model loaded successfully.")

# load our input image and grab its spatial dimensions
image_path = args["image"]
image_file = os.path.basename(image_path)
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"[ERROR] Could not read {image_file}")
(H, W) = image.shape[:2]

# construct a blob from the input image and perform a forward pass
# giving us our bounding boxes and associated probabilities
blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
                             swapRB=True, crop=False)
net.setInput(blob)
start = time.time()
layerOutputs = net.forward(ln)
end = time.time()

# show timing information on YOLO
print("[INFO] YOLOv3 took {:.6f} seconds".format(end - start))

# initialize lists for detected bounding boxes, confidences, and class IDs
boxes = []
confidences = []
classIDs = []

# loop over each of the layer outputs
for output in layerOutputs:
    # loop over each of the detections
    for detection in output:
        # extract the class ID and confidence of the current object
        scores = detection[5:]
        classID = np.argmax(scores)
        confidence = scores[classID]

        # filter out weak predictions
        if confidence > 0.5:
            # scale bounding box coordinates back relative to image size
            box = detection[0:4] * np.array([W, H, W, H])
            (centerX, centerY, width, height) = box.astype("int")

            # derive top-left corner coordinates
            x = int(centerX - (width / 2))
            y = int(centerY - (height / 2))

            # update our list of boxes, confidences, and class IDs
            boxes.append([x, y, int(width), int(height)])
            confidences.append(float(confidence))
            classIDs.append(classID)

# apply non-maxima suppression to suppress overlapping boxes
idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)

# draw boxes and labels on the image
if len(idxs) > 0:
    for i in idxs.flatten():
        (x, y) = (boxes[i][0], boxes[i][1])
        (w, h) = (boxes[i][2], boxes[i][3])
        color = [int(c) for c in COLORS[classIDs[i]]]
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
        cv2.putText(image, text, (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# save the output image
out_path = os.path.join(output_folder, f"yolov3_{image_file}")
cv2.imwrite(out_path, image)
print(f"[INFO] Output saved as: {out_path}")

# display the output image
cv2.imshow("YOLOv3 Detection", image)
cv2.waitKey(0)
