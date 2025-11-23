# USAGE
# python video_detection/yolov3_video.py --input input/videos/airport.mp4

# import the necessary packages
import cv2
import numpy as np
import os
import time
import argparse

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True, type=str,
                help="path to input video file")
args = vars(ap.parse_args())

# define the root and paths for model files
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
output_folder = os.path.join(ROOT_DIR, "output/yolov3/videos")
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

# load our YOLOv3 model trained on COCO dataset
print("[INFO] loading YOLOv3 model from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# determine the output layer names we need from YOLO
ln = net.getLayerNames()
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
print("[INFO] model loaded successfully.")

# load input video
input_video = args["input"]
video_file = os.path.basename(input_video)
if not os.path.exists(input_video):
    raise FileNotFoundError(f"[ERROR] Could not find {video_file}")

# initialize video stream and writer
vs = cv2.VideoCapture(input_video)
writer = None
(W, H) = (None, None)
frame_count = 0
start_total = time.time()

# process each frame from the video stream
while True:
    grabbed, frame = vs.read()
    if not grabbed:
        break
    frame_count += 1
    if W is None or H is None:
        (H, W) = frame.shape[:2]

    # create a blob from the frame and perform a forward pass
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416),
                                 swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()

    # initialize lists for bounding boxes, confidences, and class IDs
    boxes = []
    confidences = []
    classIDs = []

    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each detection in the output
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > 0.5:
                # scale bounding box coordinates relative to image size
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # derive top-left corner coordinates
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update lists of detections
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # apply non-maxima suppression to suppress overlapping boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)
    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
            cv2.putText(frame, text, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # initialize writer once output video starts
    if writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        out_path = os.path.join(output_folder, f"yolov3_{video_file}")
        writer = cv2.VideoWriter(out_path, fourcc, 30,
                                 (frame.shape[1], frame.shape[0]), True)
        print(f"[INFO] writing output to: {out_path}")

    writer.write(frame)

    # display the frame in a window
    cv2.imshow("YOLOv3 Detection", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("[INFO] stopped manually by user.")
        break

# release file pointers and close windows
writer.release()
vs.release()
cv2.destroyAllWindows()
print(f"[INFO] processed {frame_count} frames in {time.time() - start_total:.2f}s -> saved in {output_folder}")
