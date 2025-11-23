# USAGE
# python video_detection/yolov8_video.py --input input/videos/airport.mp4

# import the necessary packages
from ultralytics import YOLO
import os
import time
import argparse
import shutil

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True, type=str,
                help="path to input video file")
args = vars(ap.parse_args())

# define the root and paths for input and output
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
output_folder = os.path.join(ROOT_DIR, "output/yolov8/videos")

# create the output directory if it does not exist
os.makedirs(output_folder, exist_ok=True)

# load the YOLOv8 model
print("[INFO] loading YOLOv8 model from disk...")
model = YOLO("yolov8n.pt")
print("[INFO] model loaded successfully.")

# read the input video
input_video = args["input"]
video_file = os.path.basename(input_video)
if not os.path.exists(input_video):
    raise FileNotFoundError(f"[ERROR] Could not find {video_file}")

# run the YOLOv8 detection and save the output
start = time.time()
results = model.predict(
    source=input_video,
    conf=0.5,
    save=True,
    show=True,          # enable live preview of detections
    project=output_folder,
    name="predict",
    exist_ok=True
)
end = time.time()

# move generated files from temporary folder to main output
temp_folder = os.path.join(output_folder, "predict")
for file in os.listdir(temp_folder):
    src = os.path.join(temp_folder, file)
    dst = os.path.join(output_folder, file)
    if os.path.isfile(src):
        shutil.move(src, dst)
shutil.rmtree(temp_folder, ignore_errors=True)

# show timing and save information
print(f"[INFO] YOLOv8 took {end - start:.2f}s -> saved in {output_folder}")