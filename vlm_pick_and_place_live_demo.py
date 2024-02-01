import cv2
import time
import copy
import torch
import threading
import numpy as np
import pykinect_azure as pykinect
from utils.vlm_detection_util import detect_object
from segment_anything import sam_model_registry
from calibration.models import Robot2DCalibrationNN, evaluate_model
from utils.robot_utils import *
from pymycobot.mycobot import MyCobot

color_image = None
annotated_image = None
quit_program = False
box = None

sam_checkpoint = "sam_vit_b_01ec64.pth"
model_type = "vit_b"
device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

calibration_model = Robot2DCalibrationNN(3, 3)
calibration_model.load_state_dict(torch.load("calibration_model.pth"))
calibration_model.eval()
calibration_model.to("cuda")

mc = MyCobot("COM7", 115200)

def show_image():
    cv2.namedWindow('Color Image',cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    # Modify camera configuration
    device_config = pykinect.default_configuration
    device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_1080P
    # print(device_config)

    # Start device
    device = pykinect.start_device(config=device_config)
    global color_image, annotated_image, quit_program, box
    while True:

        # Get capture
        capture = device.update()

        # Get the color image from the capture
        lock.acquire()
        ret, color_image = capture.get_color_image()
        lock.release()

        if not ret:
            continue
        
        # Plot the image
        lock.acquire()
        if box is not None:
            annotated_img = cv2.drawContours(color_image,[box],0,(0,255,0),2)
        else:
            annotated_img = color_image
        cv2.imshow("Color Image", annotated_img)
        lock.release()
        cv2.waitKey(10)

        lock.acquire()
        if quit_program:
            break
        lock.release()

def vlm_detect_object():
    global color_image, annotated_image, quit_program, box
    while True:
        query = input("Please Enter Prompt, type \"quit\" to exit:\n")
        if query == "quit":
            quit_program = True
            break
        elif query == "clear":
            box = None
            continue

        while color_image is None:
            time.sleep(0.1)
        lock.acquire()
        input_image = copy.deepcopy(color_image)
        lock.release()
        raw_annotated_image, pose, raw_box = detect_object(query, input_image, sam)
        if pose == None:
            print("Cannot get image coords")
            continue

        lock.acquire()
        box = copy.deepcopy(raw_box)
        annotated_image = copy.deepcopy(raw_annotated_image)
        lock.release()

        image_width = input_image.shape[1]
        image_height = input_image.shape[0]

        print(image_width, image_height)

        cX, cY, theta = pose
        normX, normY, normTheta = normalize_camera_coord(cX, cY, theta, image_width, image_height)
        input_array = np.array([[normX, normY, normTheta]]).astype(np.float32)

        predicted_val = evaluate_model(calibration_model, input_array)
        denorm_coords = denormalize_robot_coord(predicted_val)

        pick_and_place(mc, denorm_coords)

        lock.acquire()
        box = None
        lock.release()

if __name__ == "__main__":

    # Initialize the library, if the library is not found, add the library path as argument
    pykinect.initialize_libraries()

    global lock
    lock = threading.Lock() 

        # creating threads 
    t1 = threading.Thread(target=show_image) 
    t2 = threading.Thread(target=vlm_detect_object) 
  
    # start threads 
    t1.start() 
    t2.start() 
  
    # wait until threads finish their job
    t2.join()
    t1.join() 

