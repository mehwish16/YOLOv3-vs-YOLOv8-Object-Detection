# USAGE
# python image_detection/yolov8_image.py --image input/images/baggage_claim.jpg

# import the necessary packages
from ultralytics import YOLO
import os
import time
import argparse
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, type=str,
                help="path to input image file")
args = vars(ap.parse_args())

# define the root and paths for input and output
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
output_folder = os.path.join(ROOT_DIR, "output/yolov8/images")

# create the output directory if it does not exist
os.makedirs(output_folder, exist_ok=True)

# load the YOLOv8 model
print("[INFO] loading YOLOv8 model from disk...")
model = YOLO("yolov8n.pt")
print("[INFO] model loaded successfully.")

# read the input image
image_path = args["image"]
image_file = os.path.basename(image_path)
if not os.path.exists(image_path):
    raise FileNotFoundError(f"[ERROR] Could not find {image_file}")

# run YOLOv8 detection
start = time.time()
results = model.predict(
    source=image_path,
    conf=0.5,
    save=True,
    project=output_folder,
    name="predict",
    exist_ok=True
)
end = time.time()

# show timing information
print(f"[INFO] YOLOv8 took {end - start:.6f} seconds")

# print output directory path
print(f"[INFO] Output saved in: {output_folder}")

# path to processed image inside predict folder
predict_folder = os.path.join(output_folder, "predict")
output_image_path = os.path.join(predict_folder, image_file)

# display the image if available
if os.path.exists(output_image_path):
    img = cv2.imread(output_image_path)
    cv2.imshow("YOLOv8 Detection", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print(f"[INFO] Could not find processed image at {output_image_path}")